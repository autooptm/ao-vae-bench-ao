# ao-vae-bench

Encode benchmarks for three video VAEs, each a stand-alone script that loads
only the `vae` subfolder of a public diffusers checkpoint and encodes a fixed
set of clips cut from `data/jellyfish.mp4` (1080p, 30 fps):

| script | checkpoint | precision |
|---|---|---|
| `wan21_encode.py` | Wan-AI/Wan2.1-T2V-1.3B-Diffusers | fp32 |
| `wan22_encode.py` | Wan-AI/Wan2.2-TI2V-5B-Diffusers | fp32 |
| `hunyuan_encode.py` | hunyuanvideo-community/HunyuanVideo | fp16, tiled |

```
pip install -r requirements.txt
python wan21_encode.py            # 12 clips x 33 frames at 480x832
```

Each script prints one line per clip and writes `out/<name>.json` with the
per-clip wall time, the latent shape and its mean/std.
