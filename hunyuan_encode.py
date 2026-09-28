"""HunyuanVideo VAE (diffusers AutoencoderKLHunyuanVideo), fp16 tiled stock: encode a fixed set of video clips to latents.

    python hunyuan_encode.py                 # 12 clips of 33 frames at 480x832
    python hunyuan_encode.py --clips 8 --frames 17

The VAE is loaded from the 'hunyuanvideo-community/HunyuanVideo' checkpoint (vae subfolder only). Each clip
is encoded independently; the loop over clips is the unit of work. Writes
out/hunyuan_encode.json with per-clip wall time and latent statistics.
"""
import argparse
import os
import time

import torch

from vae_common import HERE, VIDEO, load_frames, make_clips, write_summary

REPO = "hunyuanvideo-community/HunyuanVideo"

AO_DEFAULT = "ao1,ao2,ao3"


def _apply_opt(vae):
    import torch._dynamo as _dynamo
    _dynamo.config.automatic_dynamic_shapes = False
    _dynamo.config.cache_size_limit = 128
    _dynamo.config.accumulated_cache_size_limit = 1024

    cls = type(vae.encoder)
    cls.forward = torch.compile(cls.forward, dynamic=False)
    return [cls.__name__]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=VIDEO)
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--frames", type=int, default=33, help="4n+1 frames per clip")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "out", "hunyuan_encode.json"))
    ap.add_argument("--ao", default=os.environ.get("AO_OPT", AO_DEFAULT),
                    help="comma-separated switches: ao1, ao2, ao3. "
                         "Use 'none' for the original path.")
    args = ap.parse_args()
    ao = {s for s in args.ao.replace(" ", "").split(",") if s and s != "none"}

    from diffusers import AutoencoderKLHunyuanVideo
    dtype = torch.float16

    if "ao3" in ao:
        torch.backends.cudnn.benchmark = True

    t0 = time.perf_counter()
    frames = load_frames(args.video, args.frames + args.stride * (args.clips - 1))
    clips = make_clips(frames, args.frames, args.height, args.width, args.clips, args.stride)
    print(f"input: {frames.shape} -> {len(clips)} clips of {tuple(clips[0].shape)} "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    t0 = time.perf_counter()
    vae = AutoencoderKLHunyuanVideo.from_pretrained(REPO, subfolder="vae", torch_dtype=dtype)
    vae.to("cuda").eval()
    if True:
        vae.enable_tiling()
    if "ao1" in ao:
        vae.tile_sample_min_num_frames = 1 << 14
        vae.tile_sample_stride_num_frames = 1 << 14
    opt_applied = _apply_opt(vae) if "ao2" in ao else []
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

    write_summary(args.out, "hunyuan_encode", rows, walls, {"repo": REPO, "dtype": str(dtype)})


if __name__ == "__main__":
    main()
