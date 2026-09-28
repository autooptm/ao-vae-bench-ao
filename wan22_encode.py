"""Wan2.2 TI2V-5B video VAE (diffusers AutoencoderKLWan, 16x spatial), fp32 stock: encode a fixed set of video clips to latents.

    python wan22_encode.py                 # 12 clips of 33 frames at 480x832
    python wan22_encode.py --clips 8 --frames 17

The VAE is loaded from the 'Wan-AI/Wan2.2-TI2V-5B-Diffusers' checkpoint (vae subfolder only). Each clip
is encoded independently; the loop over clips is the unit of work. Writes
out/wan22_encode.json with per-clip wall time and latent statistics.
"""
import argparse
import os
import time

import torch

from vae_common import HERE, VIDEO, load_frames, make_clips, write_summary

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

_AO_OPT_1 = os.environ.get("WAN22_OPT_1", "fp16").lower()
_AO_OPT_2 = os.environ.get("WAN22_OPT_2", "on").lower()
_AO_OPT_3 = os.environ.get("WAN22_OPT_3", "1") == "1"
_AO_OPT_4 = os.environ.get("WAN22_OPT_4", "0") == "1"
_AO_OPT_5 = os.environ.get("WAN22_OPT_5", "0") == "1"


def _ao_opt_norm():
    from diffusers.models.autoencoders import autoencoder_kl_wan as _m
    rms, orig = _m.WanRMS_norm, _m.WanRMS_norm.forward

    def forward(self, x):
        if not self.channel_first:
            return orig(self, x)
        g = self.gamma.reshape(-1)
        perm = (0, 2, 3, 4, 1) if x.dim() == 5 else (0, 2, 3, 1)
        back = (0, 4, 1, 2, 3) if x.dim() == 5 else (0, 3, 1, 2)
        y = torch.nn.functional.rms_norm(x.permute(*perm), (g.numel(),), g, eps=1e-12)
        y = y.permute(*back)
        return y + self.bias if isinstance(self.bias, torch.Tensor) else y

    rms.forward = forward


def _ao_accelerate(vae):
    opt_mode = torch.channels_last_3d if _AO_OPT_2 == "on" else None
    if opt_mode is not None:
        for m in vae.encoder.modules():
            w = getattr(m, "weight", None)
            if isinstance(m, torch.nn.Conv3d) and w is not None and w.dim() == 5:
                m.weight.data = w.data.contiguous(memory_format=opt_mode)
        if getattr(vae, "quant_conv", None) is not None and vae.quant_conv.weight.dim() == 5:
            vae.quant_conv.weight.data = vae.quant_conv.weight.data.contiguous(
                memory_format=opt_mode)

    opt_dt = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(_AO_OPT_1)
    if opt_dt is not None:
        vae.encoder.to(opt_dt)
        if getattr(vae, "quant_conv", None) is not None:
            vae.quant_conv.to(opt_dt)
    if _AO_OPT_4:
        _ao_opt_norm()
    if _AO_OPT_5 and getattr(vae, "decoder", None) is not None:
        vae.decoder = None
    return opt_dt, opt_mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--frames", type=int, default=33, help="4n+1 frames per clip")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "wan22_encode.json"))
    args = ap.parse_args()

    from diffusers import AutoencoderKLWan
    dtype = torch.float32

    if _AO_OPT_3:
        torch.backends.cudnn.benchmark = True

    t0 = time.perf_counter()
    frames = load_frames(args.video, args.frames + args.stride * (args.clips - 1))
    clips = make_clips(frames, args.frames, args.height, args.width, args.clips, args.stride)
    print(f"input: {frames.shape} -> {len(clips)} clips of {tuple(clips[0].shape)} "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    t0 = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(REPO, subfolder="vae", torch_dtype=dtype)
    vae.to("cuda").eval()
    if False:
        vae.enable_tiling()
    opt_dt, opt_mode = _ao_accelerate(vae)
    print(f"model: {REPO} vae loaded in {time.perf_counter() - t0:.1f}s "
          f"({sum(p.numel() for p in vae.parameters()) / 1e6:.0f}M params, {dtype})", flush=True)

    rows, walls = [], []
    with torch.no_grad():
        for i, clip in enumerate(clips):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = clip.to("cuda", dtype)
            if opt_mode is not None:
                x = x.contiguous(memory_format=opt_mode)
            with torch.autocast("cuda", dtype=opt_dt or torch.float16, enabled=opt_dt is not None):
                z = vae.encode(x).latent_dist.mode()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            walls.append(dt)
            zf = z.float()
            rows.append({"clip": i, "s": round(dt, 4), "shape": list(z.shape),
                         "mean": float(zf.mean()), "std": float(zf.std())})
            print(f"  clip {i:2d} {dt:.3f}s latent {tuple(z.shape)}", flush=True)

    write_summary(args.out, "wan22_encode", rows, walls, {"repo": REPO, "dtype": str(dtype)})


if __name__ == "__main__":
    main()
