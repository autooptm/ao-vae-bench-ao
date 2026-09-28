<div align="center">
  <a href="https://autooptm.com"><img src=".autooptm/logo.png" width="96" alt="AutoOptm"></a>

  <h1>ao-vae-bench · optimized by <a href="https://autooptm.com">AutoOptm</a></h1>

  <p><b>3.69x / 2.13x / 1.64x faster end to end</b> on the commands below, output verified against the stock program.</p>

  <p>
    <a href="https://autooptm.com"><img alt="speedup" src="https://img.shields.io/badge/end--to--end-3.69x-2ea44f"></a>
    <a href="https://github.com/autooptm-ai/ao-vae-bench/commit/6b500f8a98dd35a30b00ceb200ed39ed64c89eb3"><img alt="base" src="https://img.shields.io/badge/upstream-6b500f8a98dd-blue"></a>
    <img alt="card" src="https://img.shields.io/badge/measured%20on-NVIDIA%20RTX%204090-lightgrey">
  </p>
</div>

> This is a fork of [autooptm-ai/ao-vae-bench](https://github.com/autooptm-ai/ao-vae-bench) at commit
> [`6b500f8a98dd`](https://github.com/autooptm-ai/ao-vae-bench/commit/6b500f8a98dd35a30b00ceb200ed39ed64c89eb3) with the AutoOptm patch applied on top.
> The optimisation was found, measured and verified automatically by [AutoOptm](https://autooptm.com);
> the patch is also kept at [`.autooptm/autooptm.patch`](.autooptm/autooptm.patch).

Every change is on by default and the command runs unchanged — same file, same flags, same outputs. Every change is behind a switch that defaults on; see [`.autooptm/autooptm.patch`](.autooptm/autooptm.patch).

## The result — `python wan21_encode.py`

| | |
|---|---|
| **Command** | `python wan21_encode.py` |
| **Entry point** | `wan21_encode.py` |
| **Unit measured** | one 33-frame clip encoded by the Wan 2.1 VAE (wan21_encode.py) |
| **Before (stock)** | 2.195 (as reported) per unit |
| **After (this tree, all switches default ON)** | 0.595 (as reported) per unit |
| **Speedup** | **3.69x** end to end on NVIDIA RTX 4090, host noise floor 0.1% |
| **Output** | latents within rel_l2 1.3e-3 (PSNR 73.6 dB) of the stock program |

### What changed

| File | Where | Gain (alone) |
|---|---|---|
| `wan21_encode.py` | main() | 1.75x |
| `wan21_encode.py` | main() | 1.124x |
| `wan21_encode.py` | main() | 1.331x |
| `wan21_encode.py` | main() | 1.193x |
| `wan21_encode.py` | _apply_opt_5() | 1.02x |


## The result — `python wan22_encode.py`

| | |
|---|---|
| **Command** | `python wan22_encode.py` |
| **Entry point** | `wan22_encode.py` |
| **Unit measured** | one clip encoded by the Wan 2.2 VAE (wan22_encode.py) |
| **Before (stock)** | 826.5 ms per unit |
| **After (this tree, all switches default ON)** | 386.5 ms per unit |
| **Speedup** | **2.13x** end to end on NVIDIA RTX 4090, host noise floor 0.0% |
| **Output** | latents within rel_l2 1.1e-3 (PSNR 78.3 dB) of the stock program |

### What changed

| File | Where | Gain (alone) |
|---|---|---|
| `wan22_encode.py` | _ao_accelerate() | 1.8646x |
| `wan22_encode.py` | _ao_accelerate() | 1.0729x |
| `wan22_encode.py` | main() | 1.0562x |


## The result — `python hunyuan_encode.py`

| | |
|---|---|
| **Command** | `python hunyuan_encode.py` |
| **Entry point** | `hunyuan_encode.py` |
| **Unit measured** | one 33-frame clip encoded by the HunyuanVideo VAE (hunyuan_encode.py) |
| **Before (stock)** | 2.995 (as reported) per unit |
| **After (this tree, all switches default ON)** | 1.826 (as reported) per unit |
| **Speedup** | **1.64x** end to end on NVIDIA RTX 4090, host noise floor 0.1% |
| **Output** | latents within rel_l2 1.7e-2 (PSNR 50.9 dB) of the stock program |

### What changed

| File | Where | Gain (alone) |
|---|---|---|
| `hunyuan_encode.py` | main() | 1.288x |
| `hunyuan_encode.py` | _apply_opt() | 1.211x |
| `hunyuan_encode.py` | main() | 1.013x |


## Reproduce

```bash
git clone https://github.com/autooptm/ao-vae-bench-ao.git
cd ao-vae-bench-ao
# set up exactly as upstream documents, then:
python wan21_encode.py
python wan22_encode.py
python hunyuan_encode.py
```

`git diff 6b500f8a98dd` is the same change as `.autooptm/autooptm.patch`.

---

<div align="center"><sub>Optimized by <a href="https://autooptm.com">AutoOptm</a> — point it at a repository, get back a verified speedup and the patch.</sub></div>

---

The upstream README is unchanged below.

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
