"""Wan2.1 T2V-1.3B video VAE (diffusers AutoencoderKLWan), fp32 stock: encode a fixed set of video clips to latents.

    python wan21_encode.py                 # 12 clips of 33 frames at 480x832
    python wan21_encode.py --clips 8 --frames 17

The VAE is loaded from the 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers' checkpoint (vae subfolder only). Each clip
is encoded independently; the loop over clips is the unit of work. Writes
out/wan21_encode.json with per-clip wall time and latent statistics.
"""
import argparse
import os
import time

import torch

from vae_common import HERE, VIDEO, load_frames, make_clips, write_summary

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _on(name, default="1"):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off")


def _apply_opt_5(vae, chunk):
    def _encode(self, x):
        _, _, num_frame, _, _ = x.shape
        patch_size = getattr(self.config, "patch_size", None)
        if patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import patchify
            x = patchify(x, patch_size=patch_size)
        self.clear_cache()
        self._enc_conv_idx = [0]
        outs = [self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map,
                             feat_idx=self._enc_conv_idx)]
        i = 1
        while i < num_frame:
            j = min(i + chunk, num_frame)
            self._enc_conv_idx = [0]
            outs.append(self.encoder(x[:, :, i:j, :, :], feat_cache=self._enc_feat_map,
                                     feat_idx=self._enc_conv_idx))
            i = j
        out = outs[0] if len(outs) == 1 else torch.cat(outs, 2)
        enc = self.quant_conv(out)
        self.clear_cache()
        return enc

    vae._encode = _encode.__get__(vae, type(vae))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--frames", type=int, default=33, help="4n+1 frames per clip")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "wan21_encode.json"))
    args = ap.parse_args()

    from diffusers import AutoencoderKLWan
    dtype = torch.float16 if os.environ.get("WAN21_OPT_1", "fp16").lower() == "fp16" \
        else torch.float32

    if _on("WAN21_OPT_2"):
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

    if _on("WAN21_OPT_4") and _on("WAN21_OPT_3"):
        with torch.no_grad():
            for t in list(vae.parameters()) + list(vae.buffers()):
                if t.dim() == 5:
                    t.data = t.data.to(memory_format=torch.channels_last_3d)
                elif t.dim() == 4:
                    t.data = t.data.to(memory_format=torch.channels_last)

    if _on("WAN21_OPT_3"):
        for name in ("cache_size_limit", "recompile_limit",
                     "accumulated_cache_size_limit", "accumulated_recompile_limit"):
            if hasattr(torch._dynamo.config, name):
                setattr(torch._dynamo.config, name, 256)
        vae.encoder.forward = torch.compile(vae.encoder.forward, dynamic=False)

    _chunk = int(os.environ.get("WAN21_OPT_5", "8"))
    if _chunk > 4:
        _apply_opt_5(vae, _chunk)

    print(f"model: {REPO} vae loaded in {time.perf_counter() - t0:.1f}s "
          f"({sum(p.numel() for p in vae.parameters()) / 1e6:.0f}M params, {dtype})", flush=True)

    rows, walls = [], []
    with torch.no_grad():
        for i, clip in enumerate(clips):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = clip.to("cuda", dtype)
            z = vae.encode(x).latent_dist.mode()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            walls.append(dt)
            zf = z.float()
            rows.append({"clip": i, "s": round(dt, 4), "shape": list(z.shape),
                         "mean": float(zf.mean()), "std": float(zf.std())})
            print(f"  clip {i:2d} {dt:.3f}s latent {tuple(z.shape)}", flush=True)

    write_summary(args.out, "wan21_encode", rows, walls, {"repo": REPO, "dtype": str(dtype)})


if __name__ == "__main__":
    main()
