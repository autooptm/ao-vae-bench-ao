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
    dtype = torch.float32

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
