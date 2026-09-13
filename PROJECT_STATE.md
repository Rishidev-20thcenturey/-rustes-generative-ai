# RAY-IMAGE — Project State

**Last Updated:** September 13, 2026  
**Current Phase:** N8 (Real-World Scaling)  
**Overall Status:** Core pipeline complete. Real-world VAE trained. DiT v2 architecture built.

---

## 1. Mission

Build an efficient text-to-image generation system from scratch that competes with larger models through architecture and training strategy — not brute-force scaling.

**Constraints:**
- Zero budget
- Free-tier GPU (Kaggle T4 x2, 30 hrs/week)
- Solo development
- Full pipeline ownership (no pretrained models for core components)

---

## 2. Completed Phases

### Foundation (N0–N7)

| Phase | Achievement | Deliverable |
| :--- | :--- | :--- |
| **N0** | Initial VAE (4-channel, 64×64) | `ray_vae_v0_2.pt` |
| **N1** | Latent statistics probe | `vae_latent_stats.json` |
| **N2** | Baseline DiT (8k steps, toy shapes) | `ray_image_v0_2_whiten.pt` |
| **N3** | Text conditioning diagnostics | Report |
| **N4** | Latent/velocity separability probe | Report |
| **N5** | Gated cross-attention (novel architecture) | `ray_image_v0_3_n5.pt` |
| **N5.1** | Evaluator calibration audit | `n5_1_report.json` |
| **N5.2** | VAE bottleneck diagnosis | Report |
| **N6** | 16-channel VAE + LPIPS + PatchGAN | `ray-image-vae-n6` |
| **N7** | DiT retrained on N6 latents (48k steps) | `ray-image-dit-n7` |

**Outcome:** Fully working 64×64 toy pipeline with validated architecture.

---

## 3. Current Phase: N8 — Real-World Scaling

### Completed in N8

| Sub-Phase | Status | Details |
| :--- | :--- | :--- |
| **N8.1** | ✅ Done | Qwen3-4B text encoder integrated (8-bit, frozen, ~1.9 GB VRAM) |
| **N8.2** | ✅ Done | MONET streaming via `datasets` library (104.9M images available) |
| **N8.3** | ✅ Done | Full pipeline smoke test — forward + backward pass validated |
| **N8.4** | ✅ Done | VAE v2 architecture (256×256, 16-channel latents) |
| **N8.5** | ✅ Done | **VAE v2 trained to 30,500 steps** — coherent real-world reconstruction |
| **N8.6** | ✅ Done | DiT v2 architecture built for `[B, 16, 32, 32]` latents |
| **N8.7** | ⏳ Next | Build `train_dit_v2.py` training script |
| **N8.8** | ⏳ Planned | Train DiT v2 on VAE v2 latents (~200K-1M steps) |
| **N8.9** | ⏳ Planned | First real-world image generation test |

### N8 Deliverables

| Component | Size | Location |
| :--- | :--- | :--- |
| **VAE v2** (30.5k steps) | ~10M params | `Rishix500/ray-image-vae-n8` |
| **DiT v2** (architecture) | ~180M params | `ray_image/dit_v2.py` |
| **Qwen3-4B** | 4B (frozen) | Local HF cache |

---

## 4. Architecture Overview

### Full Pipeline

```

Text Prompt → Qwen3-4B → Text Embedding [B, seq_len, 2560]
↓
Random Noise → DiT v2 → Denoised Latent [B, 16, 32, 32]
↓
VAE v2 Decoder
↓
Image [B, 3, 256, 256]

```

### Component Specifications

| Component | Architecture | Params | Status |
| :--- | :--- | :--- | :--- |
| **Qwen3-4B** | Transformer (8-bit) | 4B (frozen) | ✅ Integrated |
| **VAE v2** | Encoder-Decoder (16-ch latent) | ~10M | ✅ Trained (30.5k) |
| **DiT v2** | Transformer + Gated Cross-Attention | ~180M | ✅ Built |
| **Training pipeline** | Flow matching + whitening | — | ✅ Working |

### Key Technical Decisions

- **16-channel latent space** (vs 4-channel baseline)
- **LPIPS + PatchGAN losses** for edge sharpness
- **Gated cross-attention** (novel architectural modification)
- **Frozen Qwen3-4B** in 8-bit precision (VRAM efficient)
- **Constant learning rate** (safe for chunked training)
- **Chunked training** with `--resume` support
- **Hugging Face checkpointing** every 500 steps

---

## 5. Infrastructure

| Layer | Tool | Purpose |
| :--- | :--- | :--- |
| **Compute** | Kaggle T4 x2 (free, 30 hrs/week) | Model training |
| **Code** | GitHub | Version control, CI |
| **Model hosting** | Hugging Face | Checkpoint storage |
| **Data** | MONET (Apache 2.0) | Streaming training data |
| **Data (future)** | RicemanT Anime Background (Apache 2.0) | LoRA fine-tuning |
| **Text encoder** | Qwen/Qwen3-4B | Prompt understanding |

### Hugging Face Repositories

| Repo | Contents | Status |
| :--- | :--- | :--- |
| `Rishix500/ray-image-vae-n6` | 16-channel VAE (toy data) | Private |
| `Rishix500/ray-image-dit-n7` | DiT (toy shapes) | Private |
| `Rishix500/ray-image-vae-n8` | VAE v2 (30.5k steps) | Private |
| `Rishix500/ray-image-dit-n8` | Deprecated (deleted) | — |

---

## 6. Repository Structure

```

anime-ai-companion/
├── ray_image/
│   ├── vae.py                    # Original VAE (4-channel)
│   ├── vae_v2.py                 # VAE v2 (16-channel, 256×256)
│   ├── train_vae_v2.py           # VAE v2 training (with --resume)
│   ├── dit.py                    # Original DiT
│   ├── dit_v2.py                 # DiT v2 (32×32 latents)
│   ├── dit_v2_smoke.py            # DiT v2 smoke test
│   ├── text_encoder.py            # Qwen3-4B wrapper
│   ├── train_n5.py                # Toy dataset training
│   ├── generate.py                # Inference script
│   ├── config.py                  # Global configuration
│   └── utils.py                   # Helpers
├── tools/
│   └── make_toy_dataset.py
├── notebooks/                    # Kaggle/Colab notebooks
├── README.md                     # Public-facing description
└── PROJECT_STATE.md              # This file

```

---

## 7. Roadmap (N9 → N12)

| Phase | Goal | Status |
| :--- | :--- | :--- |
| **N9** | Progressive resolution (256 → 512 → 1024) | Planned |
| **N10** | 500M dense model + anime LoRA (public release) | Planned |
| **N11** | Sparse MoE scaling (~2B active, optional) | Optional |
| **N12** | Reasoning + GGUF deployment | Planned |

### Target Capabilities (N10+)

| Capability | Target | Technique |
| :--- | :--- | :--- |
| **Text Rendering** | 0.97 OCR | HDGlyph + glyph mask loss |
| **Prompt Alignment** | 0.89 Prism | NCPPO (perceptual preference) |
| **Spatial Reasoning** | 0.76 SpatialGenEval | SCOP data + Learn-to-Steer |
| **Layout Control** | 0.69 mIoU | JSON bounding-box conditioning |

---

## 8. Immediate Next Steps

### Session 1 (Next)
1. Build `ray_image/train_dit_v2.py`
2. Include `--resume` support from the start
3. HF checkpoint uploads every 500 steps
4. Target HF repo: `Rishix500/ray-image-dit-n8-v2`

### Session 2
1. Smoke test DiT v2 training (1,000 steps)
2. Verify loss drops and VRAM fits
3. Check first samples

### Session 3+
1. Train DiT v2 to 10k → 50k → 200k steps
2. Test generation with real prompts
3. Evaluate quality

---

## 9. Key Learnings

| Lesson | How It Was Learned |
| :--- | :--- |
| **VAE compression determines sharpness** | Diagnosed blur at 4-channel latents |
| **Chunked training is essential** | Lost 12-hour sessions to timeout |
| **HF checkpointing prevents loss** | Resumed from checkpoints after session death |
| **Constant LR is safe for chunks** | No reset issues on resume |
| **GAN weights need tuning** | Collapse at 0.05 GAN weight, fixed at 0.02 |
| **Mobile browser is unreliable** | Always verify via Hugging Face, not Kaggle |

---

## 10. Critical Notes

- **Kaggle session limit:** 12 hours maximum — always use chunked training
- **`/kaggle/working/` is wiped** between sessions — HF checkpoints are essential
- **Qwen3-4B download:** ~5 minutes per fresh session (not cached)
- **VAE v2 @ 30.5k:** Good enough to train DiT — don't over-optimize the VAE
- **Current VRAM usage:** ~4.5 GB (leaves plenty of headroom for DiT training)

---

## 11. Development Principles

1. **Diagnose before you train** — every failure gets a test, not a guess
2. **Chunked, resumable training** — 10k-step sessions with HF saves
3. **Version everything** — checkpoints, stats, reports all persisted
4. **Simplify public docs** — detailed internals stay private until release
5. **Ship incrementally** — every phase produces a usable artifact

---

## 12. Status Summary

**What's Working:**
- Complete pipeline (Qwen → DiT → VAE) validated end-to-end
- VAE v2 produces coherent real-world reconstructions
- Streaming MONET data works
- Chunked training with resume support
- HF checkpointing reliable

**What's Next:**
- DiT v2 training script (`train_dit_v2.py`)
- First DiT v2 training run
- First real-world image generation test

**What's Blocked:**
- Nothing critical
- MoE (N11) requires rented GPU (~$20-40) — deferred

---

**End of Project State**
