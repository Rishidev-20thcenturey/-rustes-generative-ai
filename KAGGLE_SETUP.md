# Kaggle / Colab Environment Setup

Environment recipe for running the RAY-IMAGE N8 pipeline (cache build, VAE v2,
DiT v2/v3). Written for the Kaggle image (Python 3.12/3.13) and equally valid on
Colab with a T4 runtime.

> **Golden rule: after `pip install`, always restart the kernel before importing
> torch.** Kaggle/Colab keep the previously imported binary modules alive in the
> running process; a fresh install is only picked up by a new kernel. Every
> session, not just the first one.

---

## 1. Install dependencies (no pins)

Let pip resolve the newest compatible wheels for the interpreter in use. Do
**not** pin versions here — the Kaggle base image moves, and pinned wheels are
what cause "no matching distribution" errors on a new Python.

```bash
# Kaggle ships torch/torchvision built against a specific CUDA; removing the
# audio/vision companions first avoids mixed-ABI import errors after the upgrade.
pip uninstall -y torchaudio torchvision

pip install -U \
  torch torchvision \
  transformers datasets accelerate bitsandbytes \
  lpips huggingface_hub pillow tqdm numpy
```

Notes:

* `requirements.txt` in this repo lists the same packages with `>=` bounds and is
  the source of truth for CI. If you prefer it: `pip install -U -r requirements.txt`
  after the `pip uninstall` line above.
* `bitsandbytes` and `accelerate` are required at runtime by
  `BitsAndBytesConfig(load_in_8bit=True)` even though no module imports them by
  name (`ray_image/text_encoder.py`).
* `safetensors` arrives transitively with `transformers`; it is listed in
  `requirements.txt` but never imported directly by this project.

## 2. Restart the kernel

Runtime → Restart session (Colab) / Restart & clear (Kaggle), **then** verify:

```python
import torch, transformers, bitsandbytes
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("devices", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
```

Expected on Kaggle T4 x2: `cuda True`, `devices 2`, two `Tesla T4` entries.
If `torch.cuda.is_available()` is `False`, the install and the GPU wheel do not
match — reinstall on the GPU runtime, do not debug the model code.

## 3. Checkpoint / token configuration

* Credentials come from the environment only. Never paste a token into a cell and
  never write it into a file in the repo:

  ```python
  import os
  os.environ["HF_TOKEN"] = "<from Kaggle Secrets>"
  # Kaggle: Add-ons -> Secrets -> HF_TOKEN, then:
  # from kaggle_secrets import UserSecretsClient
  # os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
  ```

  `ray_image/*` reads `os.getenv("HF_TOKEN")` and never prints or logs it. Every
  `create_repo` call in this repository now passes `private=True`, so a mistyped
  or new repo name cannot leak weights.

* The Qwen text encoder is pinned to **CUDA device 0** inside
  `ray_image/text_encoder.py` (`device_map={"": 0}` — already applied). It will
  not spread across both T4s; run one training job per session, or set
  `CUDA_VISIBLE_DEVICES=0`.

* `ray_image/train_n5.py --vae-checkpoint` now **requires** `--image-size 256`
  (RB-05) so the VAE cannot be silently fed the wrong resolution.

## 4. Long runs: quiet logs + a persisted transcript

Kaggle wipes `/kaggle/working` between sessions, so the terminal scrollback is
the only record unless you tee it to a persisted location (a mounted dataset
output, Drive, or an HF upload you already have).

```bash
# Suppress the noisy warnings that flood a 10k-step run, keep everything on disk.
export PYTHONWARNINGS="ignore"
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TQDM_DISABLE=1

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
LOG=/kaggle/working/dit_v3_${RUN_ID}.log

python -u -m ray_image.train_dit_v3 \
    --cache-path /kaggle/input/monet-qwen-cache \
    --steps 200000 --batch-size 8 \
    --hf-repo Rishix500/ray-image-dit-n8-v3 \
  2>&1 | grep -v -E "Some weights of|You should probably|warnings.warn|FutureWarning" \
       | tee "$LOG"
```

Then keep only what matters:

```bash
grep -E "^step=|\[checkpoint\]|\[resume\]|\[HF\]|\[milestone\]|\[cache\]|Traceback" "$LOG" | tail -50
grep -c "loss=nan" "$LOG"          # NaN-skip counter over the whole run
grep -o "zero_pred_mse=[0-9.]*" "$LOG" | tail -5   # loss vs the zero-predictor baseline
```

Useful flags from the fixes in this branch:

| Flag | Why |
| :--- | :--- |
| `--cache-path <dir\|file>` | Bypasses live VAE/Qwen inference; requires the precomputed cache |
| `--start-chunk / --end-chunk` (`precompute_qwen_cache.py`) | Builds any 10k window, including the previously unreachable 0–5 |
| `--resume <ckpt>` | Validated against the current architecture before loading (RB-11) |
| `--no-hf-upload` | Local-only run; checkpoints still go to `--output-dir` |

## 5. Cache precompute (optional, GPU-heavy)

```bash
python -u -m ray_image.precompute_qwen_cache \
    --start-chunk 0 --end-chunk 5 \
    --hf-repo <private-dataset-repo> \
    --output-dir /kaggle/tmp \
  2>&1 | tee /kaggle/working/cache_build.log
```

Each chunk covers `[N*10000, N*10000+9999]`; the stream skip is derived from
`--start-chunk`, so chunks 0–4 are now reproducible. Every saved chunk reports
`cached=` and `skip_reasons=` (no_thumbnail / empty_caption / preprocess_error /
encoder_failure / non_finite / shape_mismatch) — a chunk with a large
`shape_mismatch` count means the padding contract broke, not that the data is bad.

## 6. Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| `RuntimeError: ... requires CUDA` | No GPU attached / kernel not restarted | Select a GPU runtime, restart kernel |
| `torch.cuda.is_available() == False` after install | Mixed wheel ABI (torchvision/torchaudio leftovers) | Re-run §1, then restart kernel |
| Import errors after `pip install` | Kernel still holds the old modules | Restart kernel (§2) |
| `[cache] WARNING: mmap load failed` | Filesystem/torch build without mmap support | Expected fallback; the chunk is read fully into RAM — watch memory |
| `[resume] refusing to resume: ... mismatch` | Checkpoint and current constants disagree | Use the matching script/constants or start a fresh run (RB-11) |
| `[resume] WARNING: ... missing latent_space` | Pre-RB-03 checkpoint | Harmless; the run proceeds |
| `NotImplementedError: generation from stage='generator_v0.3...'` | Legacy N5/N7/N8 checkpoint served by `generate.py` | Expected (RB-04); those stages are not supported for inference |

---

## 7. CFG session recipe (issue #7: caption dropout, null prompt, EMA, CFG sampling)

Kaggle notebook cells for a session that trains or samples with CFG. Run them in
this order, **then restart the kernel** before importing torch.

```python
!pip uninstall -y torchaudio torchvision
!pip install -q -U transformers bitsandbytes accelerate huggingface_hub datasets safetensors peft tokenizers einops lpips
```

* **Every session must restart the kernel after `pip install`.** Kaggle keeps the
  previously imported binary modules alive; only a new kernel picks up the new wheels.
* `ray_image/text_encoder.py` pins Qwen to `device_map={'': 0}` (already applied).
  It is not spread across both T4s.
* Note: `peft` and `einops` are listed in the cell above as the issue #7 recipe. No module
  in this repository imports them; they are harmless but not required by this code.
  The §1 install recipe above (no pins, plus `torch`/`torchvision`) remains the reference
  for `torch` itself.

Warning suppression and a persisted transcript for long runs (see §4 for the full form):

```bash
export PYTHONWARNINGS="ignore" TRANSFORMERS_VERBOSITY=error HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1
LOG=/kaggle/working/dit_v3_cfg_$(date -u +%Y%m%dT%H%M%SZ).log
python -u -m ray_image.train_dit_v3 --cache-path /kaggle/input/monet-qwen-cache \
    --steps 200000 --batch-size 8 --caption-drop 0.10 --ema-decay 0.999 \
  2>&1 | grep -v -E "Some weights of|You should probably|warnings.warn|FutureWarning" | tee "$LOG"
grep -E "^step=|\[null\]|\[checkpoint\]|\[resume\]|Traceback" "$LOG" | tail -50
```

Sampling with CFG (after kernel restart; requires the v3 route):

```bash
python -m ray_image.generate --checkpoint ray_dit_v3_ema_step_0200000.pt --use-ema \
    --vae-checkpoint <vae.pt> --prompt "a watercolor harbour at dusk" \
    --cfg-scale 3.0 --steps 40 --output out.png
```

The null prompt (`aux/null_prompt_emb.pt` in the DiT repo) is fetched from HF. If it is
absent, `train_dit_v3.py` rebuilds it once and uploads it to the private repo; `generate.py`
rebuilds it locally without uploading.
