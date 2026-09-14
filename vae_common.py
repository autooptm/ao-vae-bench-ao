"""Shared input pipeline for the video-VAE encode benchmarks.

Decodes the first frames of data/jellyfish.mp4 with PyAV, resizes them to the
benchmark canvas, and cuts a fixed set of overlapping clips (4n+1 frames each,
as the causal video VAEs require). Everything here runs once, before the timed
loop in each entry script.
"""
import os
import av
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "data", "jellyfish.mp4")


def load_frames(path, n_frames):
    """First n_frames RGB frames as uint8 [T, H, W, 3]."""
    frames = []
    with av.open(path) as c:
        s = c.streams.video[0]
        for f in c.decode(s):
            frames.append(f.to_ndarray(format="rgb24"))
            if len(frames) >= n_frames:
                break
    return np.stack(frames)


def make_clips(frames, num_frames, height, width, n_clips, stride):
    """n_clips tensors [1, 3, num_frames, height, width] in [-1, 1], fp32, on the CPU."""
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    t = torch.nn.functional.interpolate(t, size=(height, width), mode="bilinear",
                                        align_corners=False, antialias=True)
    span = max(1, t.shape[0] - num_frames + 1)
    clips = []
    for i in range(n_clips):
        s = (i * stride) % span
        clips.append(t[s:s + num_frames].permute(1, 0, 2, 3).unsqueeze(0).contiguous())
    return clips


def write_summary(path, name, rows, walls, extra=None):
    import json, statistics
    os.makedirs(os.path.dirname(path), exist_ok=True)
    med = statistics.median(walls) if walls else 0.0
    out = {"bench": name, "clips": rows, "median_s": med, "total_s": sum(walls),
           "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
    if extra:
        out.update(extra)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"{name}: {len(walls)} clips, median {med:.3f}s per clip, total {sum(walls):.2f}s, "
          f"peak {out['peak_gb']:.2f} GB")
