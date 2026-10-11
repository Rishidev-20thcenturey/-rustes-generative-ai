# REPO_AUDIT_README.md

**Independent Forensic Audit — RAY-IMAGE (GitHub: `Rishidev-20thcenturey/-rustes-generative-ai`)**

| Field | Value |
| :--- | :--- |
| **Audit type** | Read-only forensic audit (static inspection + read-only API/CLI checks) |
| **Audit performed (UTC)** | 2026-10-11, 08:12 – 08:18 UTC |
| **Repository** | `https://github.com/Rishidev-20thcenturey/-rustes-generative-ai` (private, not a fork, not archived) |
| **Default branch** | `main` |
| **Branch inspected** | `arena/fe783906-rustes-generative-ai` (this session's branch; pointer identical to `main`) |
| **Commit SHA inspected** | `7b73c86537ad980662c43ee541a9c47b50b6321d` (= `origin/main` HEAD) |
| **Commit subject** | `revert: remove accidentally merged food delivery backend` (2026-10-10 18:04:13 +0530) |
| **Total commits on `main`** | 135 (via GitHub API pagination) |
| **Tracked files** | 67 (≈ 230 KB Python, ≈ 382 KB Jupyter per GitHub language stats) |
| **Auditor access** | Local shallow clone (1 commit) + read-only GitHub API + `git` CLI |
| **Files created by this audit** | **None inside the repository.** This report lives at `/home/user/REPO_AUDIT_README.md`, outside the repo. See §7.4 and §12 for the one transient artifact I created and removed. |
| **Repository modified?** | **No.** `git status` clean, `git diff origin/main` empty, no new Git objects, refs unchanged. One *untracked, git-ignored* `__pycache__` pair was created by an accidental `compileall` invocation and deleted (full disclosure in §7.4). |

> **Scope note on the phrase "…In the repo"** in the original request: the explicit read-only rule ("Create this file outside the GitHub repository… Do not upload it to GitHub or place it inside the repository") was treated as authoritative, so the report was originally written outside the working tree. See the addendum below — the owner has since asked for it to live inside the repository.

> **Post-audit addendum (2026-10-11, owner request).** After the audit window closed, the owner instructed that this report be moved into the repository. It has therefore been placed at the repository root (`REPO_AUDIT_README.md`) and committed on the session branch `arena/fe783906-rustes-generative-ai`. Its contents are unchanged apart from this note. Every integrity statement in §6.4 and in the final response remains accurate for the audit window (08:12–08:18 UTC); the only repository change after that window is the addition of this file, made by the owner's explicit instruction, not by the audit itself. Nothing in §9–§11 (remediation, tests, checklist) has been implemented — all findings remain open.

> **Second addendum (2026-10-11, same day, owner instruction).** A second pass over the three trainers produced five further findings, **RAY-035 … RAY-039**. They are now part of this report: the priority table (§4.1), a new detail subsection (§4.3), the category analyses (§5.1, §5.5) and the fix-tracking checklist (§11). The finding total is therefore **39 (RAY-001 … RAY-039)**. Two statements above are superseded by later events: (a) "the only repository change after that window is the addition of this file" — this second addendum itself changes the file; and (b) "Nothing in §9–§11 … has been implemented" — after this report was delivered, the owner opened a follow-up remediation task, and the 17 items it defined (**RB-01 … RB-17**: latent-space/whitening wiring, refusal of the legacy N5/N7/N8 stage, HF repositories created private, checkpoint provenance keys, resume-config validation, zero-prediction baseline logging, cache chunk range and repo selection, text padding, mmap-fallback warning, per-reason skip accounting, an image-size guard, and a new CPU resume smoke test) are implemented on the session branch and proposed in PR #6. That implementation is the owner's work, not this audit's; this report records it here only so the two documents do not contradict each other. Findings **RAY-035 … RAY-039** were found by reading the **audited** tree, so their citations are to commit `7b73c86`; where the RB work has since touched the same code, the entry says so.

---

## 1. Audit scope

### 1.1 What was inspected

| Area | Detail |
| :--- | :--- |
| Branches | `main`, `arena/fe783906-rustes-generative-ai` (local); remote refs `origin/main`, `arena/0d90ac7c-rustes-generative-ai`, `arena/01a07cdc-anime-ai-companion`, `revert/food-backend-from-ray-image` |
| Working tree | All 67 tracked files: 40 Python modules, 7 notebooks, `.github/workflows/test.yml`, `requirements.txt`, `.gitignore`, `README.md`, `PROJECT_STATE.md`, `runs/N5.1/*` (report + JSON + 12 oracle PNGs) |
| Commit history | Full 135-commit log on `main` via GitHub API (the local clone is shallow: `git rev-list --all --count` = 1) |
| CI | Workflow file + 15 most recent workflow runs and the step-level result of the latest `main` run |
| PRs / issues | 5 PRs (metadata), 0 releases |
| Other branches | Tree listings + diff stats vs `main` (including an unrelated TypeScript backend that was merged then reverted) |
| Static checks | `ast.parse` on all 40 Python files; AST-based duplication analysis; import-graph analysis; pure-Python parameter accounting; PNG header parsing; notebook JSON inspection; repository-wide greps (untracked writes) |

### 1.2 What is explicitly inside scope for this project

This is an AI/ML research repository (text-to-image, flow matching, custom VAE/DiT). Applicable audit categories: training-pipeline correctness, inference/serving consistency, checkpoint format & compatibility, numerical stability, data-caching integrity, reproducibility, evaluation correctness, resource/disk budget, HF Hub integration, and CI coverage. Categories such as HTTP authorization, session handling, SQL injection, and database transactions are **not applicable** — the repository contains no server, no database, and no login surface (the only such code that ever existed, the Express/PostgreSQL food-delivery backend accidentally merged in PR #3, is no longer in `main` — see RAY-014).

### 1.3 Method statement

Everything below is grounded in **file content read from the audited commit** or **read-only API output**. Where a statement is inference rather than measurement, it is labelled. No model code was executed (PyTorch is not installed in the audit environment, and installing packages was prohibited), so *no runtime behaviour is claimed as verified* unless CI evidence exists.

---

## 2. Executive summary and overall health assessment

### 2.1 What this repository is

RAY-IMAGE is a from-scratch text-to-image research pipeline: a custom VAE → custom latent DiT (flow matching) → frozen Qwen3-4B text encoder, with a strong "diagnose before you train" methodology preserved in-tree (`runs/N5.1/REPORT.md`, six probe modules, six CI smoke tests). Development happens on free-tier cloud GPUs (Kaggle T4 ×2 / Colab), with weights stored in private Hugging Face repos and datasets streamed from the `jasperai/monet` dataset.

### 2.2 Health verdict

**The `main`-branch toy pipeline (N0–N5.1) is healthy, well-tested for its size, and honestly documented. The current-generation N6–N8 real-world pipeline is functional-looking but has a several-order-of-magnitude weaker safety net: no CI coverage, no documentation, no held-out evaluation, and at least three latent inconsistencies between training and inference.**

| Dimension | Rating | Evidence summary |
| :--- | :--- | :--- |
| Repository hygiene / structure | 🟢 Good | 67 files, clear package boundary (`ray_image/`), only 4 stale branches, CI green |
| Toy pipeline (N0–N5.1) correctness | 🟢 Good | 6 CI smoke tests green (last run 2 min 38 s), probes assert report structure, `n5_smoke` validates gate/mask/checkpoint compatibility |
| Current-generation pipeline (N6–N8) correctness | 🔴 Weak | No CI, no docs, train/serve inconsistencies (RAY-001/002/003), hardcoded lengths, hardcoded chunk range |
| Reproducibility | 🟠 Poor for N8 | Weights only in private HF/Drive; whitening stats file has no provenance; no seeds/RNG state in checkpoints; unbounded `>=` dependencies |
| Documentation accuracy | 🟠 Fair | Public README claims are ahead of evidence; parameter counts off by 1.3–1.75×; PROJECT_STATE stale by the same week it was written |
| Security / secrets | 🟢 Good | No secrets in tree or notebook outputs (verified by grep); `HF_TOKEN` read from env only. One disclosure risk: HF repos created *public by default* (RAY-006) |
| Data integrity | 🟠 Fair | Cache chunks validated (contiguity, dtype, shape, filename↔metadata) — good. But a committed statistics file is unverifiable, and no train/val split exists anywhere |
| Evaluation methodology | 🔴 Weak | The only metric in the repo is a known-miscalibrated 12-class toy classifier (0.046-wide "square" band); no real-world eval at all |
| Test pyramid | 🟠 Toy-only | 6 smoke scripts (assert-based, no framework); zero tests for inference, caching, or training scripts |
| Dependency/CI hardening | 🟠 Fair | No lockfile, no upper bounds, no Dependabot config, no `permissions:` block |
| Licence / release readiness | 🔴 Not ready | README claims Apache 2.0 but there is **no LICENSE file** (GitHub licence API returns 404); no security policy; zero releases |

**Confirmed Critical findings: 0.** No credential leakage, no remote-code-execution path in the current tree, no corruption in committed data was found. The highest-severity findings are classified **High** (RAY-001 … RAY-005).

### 2.3 Highest-impact problems in one paragraph

The single most consequential issue is **RAY-001**: `ray_image/generate.py` (the only inference entry point for the N8 models) silently applies an *un-whitening* transform (`z = z_norm·std + mean`, std ≈ 2.4–9.9, mean ≈ ±0.44) to latents produced by a diffusion transformer that — as provably visible in `ray_image/train_dit_v3.py`, `train_dit_v2.py` and `ray_image/precompute_qwen_cache.py` — was **never trained on whitened latents**. The transform was introduced deliberately on 2026-09-22 ("resolve dark image bias") as a *default on* behaviour, so today's inference output cannot be trusted to reflect the model, and a real model defect could be masked by a preprocessing bug (or vice versa). Closely behind: `--vae-checkpoint` is a hidden architecture selector that makes the newest trained model family (DiT v2) impossible to generate from (**RAY-002**), a hardcoded 47-token text length that couples the cache builder, the trainer, and inference (**RAY-003**), a 256 px VAE that silently runs at 64 px in the N7/N8 warm-start trainer (**RAY-004**), and a dataset-preparation script that cannot reproduce the first 50 % of its own cache (**RAY-005**).

---

## 3. Architecture and repository structure

### 3.1 Actual layout (audited commit)

```
-ray_image-ai/ (repo root = "-rustes-generative-ai")
├── .github/workflows/test.yml        # 1 job, 6 smoke steps, Python 3.11, no cache/matrix/permissions
├── .gitignore                         # __pycache__, checkpoints/, data/, ray_suite/, *.pt …
├── README.md                          # 99 lines, public-facing claims
├── PROJECT_STATE.md                   # 247 lines, internal status ("Last Updated: September 13, 2026")
├── notebook9e7e01df92.ipynb           # Kaggle notebook, 52 code cells, 94 embedded outputs (334 KB)
├── notebooks/                         # 6 Colab notebooks (N1, N2, N4, N5, N5.1, base)
├── requirements.txt                   # 12 unpinned ">=" dependencies
├── ray_image/                         # 34 modules (the whole product)
│   ├── config.py  flow.py  utils.py   # config dataclass, flow matching, checkpoints/seeding
│   ├── vae.py  vae_v2.py              # 4/16-ch @8× (64px) and 16-ch @8× (256px) VAEs
│   ├── dit.py  dit_v2.py  dit_v3.py   # v1 (2D-sinusoidal), v2 (768/18, 2D-sinusoidal), v3 (512/12, RoPE+SwiGLU+RMSNorm)
│   ├── text_encoder.py                # LegacyRAYTextEncoder (toy) + RAYTextEncoder (frozen Qwen3-4B, 8-bit)
│   ├── whiten.py                      # LatentNormalizer (train/serve whitening convention)
│   ├── dataset.py  toy.py  evaluate.py
│   ├── train.py  train_vae.py  train_vae_v2.py  train_generator.py  train_n5.py
│   ├── train_dit_v2.py  train_dit_v3.py  precompute_qwen_cache.py
│   ├── generate.py                    # ** the only inference CLI **
│   ├── *_smoke.py (7 files: train, vae_v2, dit_v2, dit_v3, text_encoder, probe, whiten, probe_text, n4, n5)
│   ├── probe_*.py (5 files: vae_latents, text_conditioning, latent_separability, n5_shape, smoke)
│   └── latent_stats_corrected.json    # hardcoded means/stds consumed at inference
├── runs/N5.1/                         # REPORT.md, n5_1_report.json, oracle/*.png (12 × 64×64 RGB)
└── tools/                             # make_toy_dataset.py, evaluate_toy_suite.py
```

### 3.2 Data flow (as implemented)

```
TRAINING (current generation, N8)
  jasperai/monet (HF streaming)  ──► RAYVAE_v2 (256px → [B,16,32,32], sampled z)
        │                                   │
        │                                   ├──► precompute_qwen_cache.py  ──► HF dataset chunks
        │                                   │        (latents fp16 + Qwen text_emb fp16 + mask, 47 tokens)
        └──► caption ──► RAYTextEncoder (Qwen3-4B, fp16/8-bit, frozen, L≤256)
                                            │
                          train_dit_v2.py (bf16, 768/18)  ──  or  ──  train_dit_v3.py (fp16, 512/12, cached)
                                            │
                              checkpoint {stage, step, seed, config, dit, optimizer}
                                            │
INFERENCE  generate.py  ──►  RAYDiT_v3 + RAYTextEncoder (Qwen)         [only with --vae-checkpoint]
                          ──►  euler_sample (t: 1 → 0)                         20–50 steps
                          ──►  LatentNormalizer.denormalize  ◄──── latent_stats_corrected.json  ⚠ mismatch
                          ──►  RAYVAE_v2.decode → PNG
```

### 3.3 Coupling / module-boundary observations

* `ray_image/` is a flat module namespace with no `__all__`, no sub-packages, and heavy cross-imports (`utils.py` ← `dit.py`, `text_encoder.py`, `vae.py`; `generate.py` ← `dit_v3.py`, `whiten.py`, `html utils`). **No circular imports were found** (verified by AST import graph).
* Three parallel DiT generations (`dit.py`, `dit_v2.py`, `dit_v3.py`) and two VAE generations coexist with *no version registry*; the version actually used is decided by which script and which CLI flag you invoke (see RAY-002/RAY-011).
* `generate.py` is the only module that knows how to assemble an N8 pipeline; nothing else validates that flag combinations and checkpoint families are compatible.
* Duplication is measurable: **10 byte-identical function/class bodies** across files (e.g. `seed_everything` ×3, `preprocess_thumbnail` ×2, `upload_checkpoint` ×2, `load_resume` ×2) plus near-duplicates with 0.64–0.90 similarity (`checkpointed_forward`, `load_vae`, `cleanup_hf_checkpoints`, `PatchGAN`, `RAYDiTBlock` vs `RAYDiTBlock_v2`). Divergence has already happened: only `train_dit_v3.py` has disk-window cleanup and NaN+Inf checking (RAY-007).

---

## 4. Prioritized findings

Severity scale: **Critical** (severe security/data-loss/fundamental failure) · **High** (major functional failure, substantial security weakness, serious integrity risk) · **Medium** (meaningful correctness/reliability/integration issue) · **Low** (minor defect, non-urgent quality issue) · **Informational** (improvement opportunity or risk needing more evidence).

Classification: **Confirmed Bug** · **Probable Bug** · **Potential Risk** · **Improvement**.

### 4.1 Priority table

| ID | Sev. | Class | Title | Primary location |
| :--- | :--- | :--- | :--- | :--- |
| **RAY-001** | 🔴 High | Probable Bug | Inference un-whitens latents that training never whitened (hidden default) | `ray_image/generate.py:15,36-39,68-69,126-129`; `ray_image/whiten.py:37-44`; `ray_image/latent_stats_corrected.json` |
| **RAY-002** | 🔴 High | Confirmed Bug | `--vae-checkpoint` silently selects DiT v3 + Qwen; DiT v2 (the trained N8 family) is unloadable; v3 without the flag crashes | `ray_image/generate.py:36-87,115` |
| **RAY-003** | 🔴 High | Potential Risk | Hardcoded 47-token text contract couples cache, trainer and inference (256) | `ray_image/train_dit_v3.py:186-187,305`; `ray_image/precompute_qwen_cache.py:31,108-116`; `ray_image/text_encoder.py:82-90` |
| **RAY-004** | 🔴 High | Probable Bug | N7/N8 warm-start trainer never sets `image_size=256`; 256 px VAE silently fed 64 px images | `ray_image/train_n5.py:71-112,118` vs `ray_image/vae_v2.py:1-10` |
| **RAY-005** | 🔴 High | Confirmed Bug | Cache builder hardcodes chunks 5–9 / `skip(50_000)`; cannot rebuild the first 50 % of the cache | `ray_image/precompute_qwen_cache.py:226-232` |
| **RAY-006** | 🟠 High | Potential Risk | HF repos created **public by default** on first upload (weights + optimizer state) | `ray_image/train_dit_v3.py:320,336`; `train_dit_v2.py:260`; `train_vae_v2.py:87` |
| **RAY-007** | 🟡 Medium | Confirmed Bug | `train_dit_v2.py` has no local checkpoint window/disk guard → ~2.4 GB × 400 files | `ray_image/train_dit_v2.py:505-523` |
| **RAY-008** | 🟡 Medium | Potential Risk | HF upload/milestone failures abort an expensive run (no retry, uncaught raise) | `ray_image/train_dit_v3.py:329,405,425-429` |
| **RAY-009** | 🟡 Medium | Improvement | N8 architecture has **zero** automated coverage; two smokes refuse to run without CUDA | `.github/workflows/test.yml:22-33`; `vae_v2_smoke.py:9-10`; `dit_v3_smoke.py:13-14` |
| **RAY-010** | 🟡 Medium | Confirmed Bug | Known-miscalibrated shape metric left unfixed; all shape numbers in-tree are distorted | `tools/evaluate_toy_suite.py:41-45,60-70`; `runs/N5.1/REPORT.md:48-97` |
| **RAY-011** | 🟡 Medium | Confirmed Bug | Cross-gate config key has three spellings; `getattr(...,"cross_gate",True)` always wins | `config.py:16`; `train_dit_v3.py:306`; `generate.py:64` |
| **RAY-012** | 🟡 Medium | Potential Risk | `bf16` autocast claimed/logged while the documented hardware (T4) has no native bf16 | `train_dit_v2.py:418-419,437-441`; `PROJECT_STATE.md:15,109` |
| **RAY-013** | 🟡 Medium | Confirmed Bug | Evaluator exits 0 with no metrics when the target directory is empty | `tools/evaluate_toy_suite.py:110-113` |
| **RAY-014** | 🟡 Medium | Potential Risk | Entire N8 pipeline (cache, v2/v3 training, generation) is undocumented; notebooks pin a branch 82 commits behind `main` | `README.md`, `PROJECT_STATE.md`, `notebooks/*.ipynb:cell 2` |
| **RAY-015** | 🟢 Low | Confirmed Bug | `torch.empty` used for cache batch buffers (uninitialised memory) | `ray_image/train_dit_v3.py:186-189` |
| **RAY-016** | 🟢 Low | Improvement | 10 identical + ~8 near-identical duplicated symbols; model forward re-implemented in two trainers | repo-wide (see §5.4) |
| **RAY-017** | 🟢 Low | Confirmed Bug | `probe_vae_latents` hardcodes `RAYVAE` → N1 probe cannot analyse the current VAE v2 | `ray_image/probe_vae_latents.py:47-54` |
| **RAY-018** | 🟢 Low | Confirmed Bug | `decision_boundaries()` contains dead/duplicated sweep logic; evaluator feature code is copy-pasted into the probe | `ray_image/probe_n5_shape.py:78-126,242-277` |
| **RAY-019** | 🟢 Low | Confirmed Bug | Colour/shape/shape-score code duplicated between evaluator and probe with no parity test | `tools/evaluate_toy_suite.py` ↔ `ray_image/probe_n5_shape.py` |
| **RAY-020** | 🟢 Low | Confirmed Bug | `--n 2048` cell in the Kaggle notebook does not match `--samples` in the tool | `notebook9e7e01df92.ipynb` cell 41 vs `tools/make_toy_dataset.py:24` |
| **RAY-021** | 🟢 Low | Potential Risk | 9 `torch.load(weights_only=False)` call sites (pickle deserialisation of checkpoints) | `utils.py:15` + 8 others |
| **RAY-022** | 🟢 Low | Confirmed Bug | README claims Apache 2.0; no LICENSE file exists (API 404) | `README.md:91-93` |
| **RAY-023** | 🟢 Low | Confirmed Bug | Param counts wrong in docs: DiT v2 180M→200.15M, DiT v3 "60M"→86.22M, VAE v2 ~10M→5.70M | `PROJECT_STATE.md:53,63`; `ray_image/dit_v3.py:197` |
| **RAY-024** | 🟢 Low | Confirmed Bug | PROJECT_STATE stale within 24 h of writing (says "build train_dit_v2.py" although it exists) | `PROJECT_STATE.md:54,177` vs `ray_image/train_dit_v2.py` (committed 2026-09-13 13:46) |
| **RAY-025** | 🟢 Low | Potential Risk | Cache stores fp16 latents used as fp32 flow targets; no per-row provenance | `precompute_qwen_cache.py:137-138`; `train_dit_v3.py:178-183` |
| **RAY-026** | 🟢 Low | Potential Risk | No train/validation split anywhere; no real-world metric; probe "accuracy" is measured on training-distribution images | repo-wide; `ray_image/probe_vae_latents.py:57-70,121-125` |
| **RAY-027** | 🟢 Low | Improvement | Dependency hygiene: `>=` only, `safetensors` unused, no Dependabot config, no `permissions:` in CI | `requirements.txt`; `.github/workflows/test.yml` |
| **RAY-028** | 🟢 Low | Improvement | Legacy/abandoned code kept without deprecation notes (`train.py`, `evaluate.py`, `_smoke`/`_now` variants) | `ray_image/train.py`, `ray_image/evaluate.py` |
| **RAY-029** | ⚪ Info | Potential Risk | 1-D RoPE over a flattened 2-D patch grid replaces v2's explicit 2-D sinusoidal positions | `ray_image/dit_v3.py:64-82` vs `dit_v2.py:201-229` |
| **RAY-030** | ⚪ Info | Confirmed Bug | v3 smoke test intentionally uses an out-of-range timestep (500) vs training t∈[0,1] | `ray_image/dit_v2_smoke.py:36` |
| **RAY-031** | ⚪ Info | Potential Risk | Fully-masked attention rows when a legacy caption is empty (undefined/NaN in some torch versions) | `ray_image/dataset.py:45-48`; `ray_image/dit.py:57-63` |
| **RAY-032** | ⚪ Info | Potential Risk | Kaggle notebook committed with 94 stale outputs (334 KB) including private slugs and Drive paths | `notebook9e7e01df92.ipynb` |
| **RAY-033** | ⚪ Info | Improvement | Repository contains an unrelated, fully-reverted TypeScript backend in two surviving stale branches; 4 low-value commits (`tmp`, `tmp2`, `temp`) | branches `arena/0d90ac7c…`, `revert/food-backend…`; commits `1ad6bf29`, `87f4fa32`, `f612d55b`, `fd93bb87` |
| **RAY-034** | ⚪ Info | Improvement | `--latent-stats` documented "Optional/Required" but effectively always applied | `ray_image/generate.py:96-98,128-129` |
| **RAY-035** | 🟡 Medium | Probable Bug | `next_batch` skips unreadable rows **silently and without bound** in all three trainers (no counter, no log, no attempt cap) | `ray_image/train_dit_v3.py:83-95`; `train_dit_v2.py:88-103`; `train_vae_v2.py:67-79` |
| **RAY-036** | 🟡 Medium | Confirmed Bug | The 10 000-step milestone archive is reachable **only on the happy path** — both fallbacks (disk-pressure, exception) bypass it | `ray_image/train_dit_v3.py:384-393,401-414,416-429` |
| **RAY-037** | 🟡 Medium | Potential Risk | Default `--output-dir` is container-local scratch (`/kaggle/tmp/checkpoints`), while the other two trainers default to `/kaggle/working/…` | `ray_image/train_dit_v3.py:49,441` vs `train_dit_v2.py:344`, `train_vae_v2.py:129` |
| **RAY-038** | 🟢 Low | Potential Risk | Milestone repo has **no retention rule**: ≈1.03 GB per 10 000 steps ≈ 20.7 GB for a 200k run, every file also duplicated in the main repo | `ray_image/train_dit_v3.py:33-34,401-414`; `cleanup_hf_checkpoints` `:256-267` is called only for the main repo (`:323,339`) |
| **RAY-039** | 🟢 Low | Confirmed Bug | `train_vae_v2.py` re-saves and re-uploads the final checkpoint whenever `--steps` is a multiple of 500 — the guard compares a variable that is never updated | `ray_image/train_vae_v2.py:168,183,252-265,268-273` |

### 4.2 High-severity findings in detail

---

#### RAY-001 — Inference applies an un-whitening transform that training never applied
**Severity:** High · **Classification:** Probable Bug · **Confidence:** High on the inconsistency; Medium on the visual outcome (could not execute)

**Files/lines**
* `ray_image/generate.py:15` — `DEFAULT_LATENT_STATS = Path(__file__).with_name("latent_stats_corrected.json")`
* `ray_image/generate.py:36-39` — `if latent_stats is None: latent_stats = DEFAULT_LATENT_STATS`
* `ray_image/generate.py:68-69` — `whiten = LatentNormalizer.from_stats_json(latent_stats); whiten.validate_channels(16)`
* `ray_image/generate.py:126-129` — `if whiten is not None: latent = whiten.denormalize(latent)` (immediately before `vae.decode`)
* `ray_image/whiten.py:37-44` — parser widened to accept a bare `means`/`stds` document
* `ray_image/latent_stats_corrected.json` — `{"means":[…16 values…], "stds":[3.96,2.40,5.82,…,9.89]}`
* **Absence of whitening in training:** `ray_image/train_dit_v3.py`, `ray_image/train_dit_v2.py` and `ray_image/precompute_qwen_cache.py` contain **no** reference to `whiten`, `LatentNormalizer`, `normalize` or `denormalize` (verified by repository-wide grep; the only hits are in `generate.py`, `train_generator.py`, `train_n5.py`, `whiten.py` and the smokes).

**Evidence chain**
1. `git show 3e24cf54` (2026-09-22, "fix(N8): Add corrected 16-channel latent stats and update generate.py path to resolve dark image bias") changed exactly two things: it turned a hard requirement into a silent default —
   `-            raise ValueError("--latent-stats is required when --vae-checkpoint is provided")`
   `+            latent_stats = DEFAULT_LATENT_STATS`
   — and relaxed `LatentNormalizer.from_stats_dict` to read the new compact `means`/`stds` schema instead of the probe's `per_channel_mean`/`per_channel_std`.
2. `ray_image/train_dit_v3.py` feeds the DiT `clean_latents` taken **directly** from the cache (`next_cached_batch`, lines 178-183) or **directly** from `vae(images)` (line 495-497). `precompute_qwen_cache.py` likewise stores `z` returned by `vae(image_batch)` (lines 129-131) with no normalisation.
3. Therefore, for any checkpoint produced by the v2/v3 trainers, the sampler's output lives in **raw VAE latent space**, and `denormalize` multiplies it by std≈2.4–9.9 and adds mean≈±0.44 before decoding — a transform that did not exist at training time.
4. The code comment at `generate.py:126-127` states the assumption explicitly ("The sampler operates in whitened space when the checkpoint was trained with whitening"), which is exactly the assumption that is false for the checkpoints this code path exists to serve.

**Expected vs actual**
* *Expected:* decode the exact latent the transformer was trained to produce (`decode(model_output)`).
* *Actual (as written):* `decode(model_output · std + mean)`.

**What could go wrong / under which conditions**
Any use of `python -m ray_image.generate --checkpoint <dit_v2|dit_v3 ckpt> --vae-checkpoint <vae>.pt` (i.e. the documented N7/N8 inference route) yields systematically wrong colour/contrast statistics; symptoms will look like a *model quality* failure (which is exactly the failure mode that motivated the commit), so the bug can absorb debugging effort and mask genuine model defects. Note the *fix direction* is ambiguous: either remove the transform, or add whitening to the two DiT trainers and rebuild the cache — **that decision needs the author's training logs; it is not inferable from the repository.**

**Recommended future fix (do not apply yet):** make the whitening contract explicit data, not an implicit default — e.g. store `whiten` (or `"latent_space": "raw"|"whitened"`) inside every DiT checkpoint, require a match at load time, and remove the hardcoded default. If the v3 training was in fact whitened offline (not visible in the committed code), document that step in the trainer.

**Regression risk of fixing:** High if the transform is load-bearing for the currently-deployed weights; changing it silently invalidates saved comparison images. Mitigate by rendering a fixed prompt set with a "raw off" and "off" variant.

**Safe test for a future developer:** add a unit test that builds a tiny checkpoint with `whiten=None`, runs `load_models(..., vae_checkpoint=...)` and asserts `whiten is None` unless the checkpoint declares a whitening space; run the same fixed prompt with an identity stats file (`means=0, stds=1`) and with the current default and assert the outputs differ (proving the transform is active).

---

#### RAY-002 — `--vae-checkpoint` is a hidden architecture selector; DiT v2 cannot be used for inference
**Severity:** High · **Classification:** Confirmed Bug · **Confidence:** Certain (control flow is explicit)

**Files/lines:** `ray_image/generate.py:18-87`
* lines 36-69: *only if* `vae_checkpoint is not None` does the loader construct `RAYVAE_v2`, `RAYTextEncoder` (Qwen) and **`RAYDiT_v3`** (`cond_dim=cfg.text_dim`, `cross_gate=getattr(cfg,"cross_gate",True)`).
* lines 70-77: otherwise it calls `build_models(cfg, device)` → `RAYDiT` (v1) + `LegacyRAYTextEncoder`.
* line 115: `text, text_mask = text_encoder.encode_texts([args.prompt])` — a method **only** `RAYTextEncoder` has (`ray_image/text_encoder.py:92`); `LegacyRAYTextEncoder` has no `encode_texts`.

**Consequences (both verified by reading the code paths)**
1. A checkpoint produced by `train_dit_v2.py` (which saves `{"stage":"dit_v2_n8", …, "dit": RAYDiT_v2.state_dict()}`) can be loaded **only** through the `--vae-checkpoint` branch, which builds a **v3** module → `load_pretrained` → `load_state_dict` with `strict=True` (`utils.py:79-84`) → **RuntimeError** for every key. There is no flag to select `RAYDiT_v2`. The `HF_DEFAULT_REPO` in that trainer is `Rishix500/ray-image-dit-n8-v2`, i.e. exactly the "N8 model" the project state documents (`PROJECT_STATE.md:53-55`).
2. A v3 checkpoint passed **without** `--vae-checkpoint` goes down the legacy branch (v1 DiT + legacy encoder): the DiT state dict fails to load, and even if it did, `encode_texts` on line 115 raises `AttributeError`.
3. The same flag simultaneously switches the VAE, the text encoder, the DiT architecture, **and** enables RAY-001's whitening. A single boolean-ish argument controls four independent decisions.

**Recommended future fix:** make the architecture a first-class attribute — e.g. `--arch {dit_v1,dit_v2,dit_v3}` (or an `arch` field read from the checkpoint and validated against the available classes) — and validate the flag/checkpoint combination up front with a clear error message.

**Regression risk:** medium; the legacy path is exercised by `whiten_smoke` and `n5_smoke` step 7, so any refactor must keep those green.

**Safe test:** add a table-driven test that (a) builds a tiny v2 checkpoint via `build_dit()` + `save_checkpoint`, (b) asserts `generate.load_models(...)` returns a `RAYDiT_v2`, and (c) asserts a mismatched architecture raises a *descriptive* error rather than a raw `RuntimeError` key list.

---

#### RAY-003 — Hardcoded 47-token text contract across three modules
**Severity:** High · **Classification:** Potential Risk (contains one likely Confirmed Bug) · **Confidence:** High on the code facts; **Medium on runtime impact — the actual cache was not inspectable (HF access unavailable, §8)**

**Files/lines**
* `ray_image/precompute_qwen_cache.py:31` — `CACHE_TOKENS = 47`
* `ray_image/precompute_qwen_cache.py:108-116` — each row is accepted only if `tuple(text_hidden[i].shape) == (CACHE_TOKENS, TEXT_DIM)`; otherwise `skipped += 1`; if nothing survives → `RuntimeError("No valid samples produced for chunk …")` (line 154)
* `ray_image/text_encoder.py:82-90` — `tokenize()` calls the HF tokenizer with `padding=True` (**pad to the longest sequence in the batch**, *not* to `max_length`) and `truncation=True, max_length=47`
* `ray_image/train_dit_v3.py:186-187` — the cached-batch reader allocates `torch.empty((batch_size, 47, TEXT_DIM))` and `torch.empty((batch_size, 47))`; any other sequence length raises a shape error mid-training
* `ray_image/train_dit_v3.py:305` — the checkpoint records `"text_max_tokens": 256`
* `ray_image/generate.py:51-55` — inference constructs `RAYTextEncoder(max_tokens=cfg.max_tokens)` = 256, so a prompt is tokenised to its **natural** length (e.g. 5–15 tokens for "a red circle"), never 47

**Why this is a real contract violation and not just style**
With `padding=True`, a cache row has length `L = min(47, longest caption in that batch)`. The `== 47` assertion therefore only holds if *every* MONET `caption_florence-2-large` in that batch exceeds 47 tokens and is truncated. If any batch's longest caption is shorter than 47 tokens, every sample in that batch is silently dropped (`skipped`), and if that happens for all batches, chunk building aborts with an opaque error. Whether the published cache is really 47-long is not verifiable from here (see §8) — but the coupling is verifiable and is fragile in both directions: a different `max_length`, a different caption field, or a shorter caption distribution breaks the trainer at step *k* of a multi-hour run.

**Train/serve skew (independent of the above):** the v3 model is trained with a 47-token, mostly truncated/masked text context, while inference feeds an unpadded natural-length context. `RAYDiT_v3` tolerates variable `L` (key-padding mask), so this will not crash — it will simply condition on a differently distributed context than training, which degrades or biases output.

**Recommended future fix:** store `seq_len` (and the padding policy) in the cache payload; read it from the payload in `next_cached_batch`; use `padding="max_length"` at cache-build time if a fixed length is wanted; truncate/pad prompts at inference to the value recorded in the checkpoint; and assert equality with a clear message.

**Regression risk:** low for correctness of the cache as-is (it may already be 47-wide); high if the cache is rebuilt and downstream training targets change.

**Safe test:** a unit test that feeds a deliberately short caption through `RAYTextEncoder.tokenize(..., max_length=47)` and asserts either `L == 47` (if fixed padding is chosen) or that the returned length is propagated, plus an explicit assertion message in `next_cached_batch`.

---

#### RAY-004 — N7/N8 warm-start trainer silently runs the 256 px VAE at 64 px
**Severity:** High · **Classification:** Probable Bug (silent misconfiguration) · **Confidence:** High

**Files/lines**
* `ray_image/vae_v2.py:1-10` — the module's own contract: `image: [B,3,256,256] → latent: [B,16,32,32]`
* `ray_image/train_n5.py:71-112` (`build_n7_from_n6`) sets only `latent_channels = 16`, `dit_cross_gate = True`, `text_encoder_backend = "qwen3"`, `text_dim = 2560` — **`cfg.image_size` is never set to 256**; it remains whatever the warm-start checkpoint contained or the 64 default (`config.py:6`)
* `ray_image/train_n5.py:118` — `RAYCaptionDataset(manifest, cfg.image_size)` resizes every image to `cfg.image_size`
* `ray_image/train_n5.py:159-163` — `vae.encode(images)` is called on those images; a 64×64 input yields an `[B,16,8,8]` latent, which then feeds a DiT whose positional/geometry assumptions come from `cfg.latent_size = image_size // 8 = 8`

**Impact:** the documented N7/N8 pipeline ("DiT retrained on N6 16-channel 256 px latents") can silently train a generator on 8×8 latents derived from 64 px images pushed through a decoder-side model trained only at 256 px. There is no assertion anywhere that the VAE is being used at its trained resolution. This is the kind of defect that produces "it trains but the pictures are mush" and burns GPU weeks.

**Status qualification:** `PROJECT_STATE.md:36` claims N7 produced `ray-image-dit-n7` at 48k steps and `train_dit_v3.py:31` pins `ray_vae_v2_step_0047500.pt`, implying a separate (uncommitted) 256 px route was used for the newer work. So this is a defect in the *committed* N7/N8 path of `train_n5.py`, not evidence that the author's checkpoints are necessarily wrong.

**Recommended future fix:** add explicit resolution validation in `build_n7_from_n6` (`cfg.image_size = 256`) plus a hard assert that `vae.latent_size == cfg.latent_size` before the first step.

**Regression risk:** low (adds a constraint the code already claims to satisfy); medium if some existing script relies on the 64 px behaviour — check `train_n5.py --vae-checkpoint` callers first (none are committed outside the notebooks).

**Safe test:** a CPU test that builds `build_n7_from_n6` with a dummy VAE checkpoint and asserts `cfg.image_size == 256` and that a `RAYCaptionDataset` sample is `[3,256,256]`; a second assert that `vae.encode(x).shape[2:] == (32,32)`.

---

#### RAY-005 — Cache builder cannot reproduce the first half of the cache
**Severity:** High (for reproducibility of the current training data) · **Classification:** Confirmed Bug · **Confidence:** Certain

**Files/lines:** `ray_image/precompute_qwen_cache.py`
* line 32 — `TOTAL_SAMPLES = 100_000` is defined and never used
* lines 214-220 — `create_repo(repo_id="Rishix500/monet-qwen-cache-50k", …)` hardcodes the repository, ignoring the `--hf-repo` argument
* lines 226-228 — `start_chunk = 5`, `end_chunk = 10`, `dataset.skip(50_000)` are hardcoded in `main()`
* lines 232-279 — the loop therefore only ever processes chunks 50k–100k

**Consequence:** the committed script is a *resume-completion* tool, not the generator of the 100k cache its own docstring advertises ("Build a resumable MONET VAE + Qwen cache in 10k chunks… rerun resumes from the first missing chunk"). Chunks 0–4 can never be produced by this code. Combined with RAY-003’s `skip(50_000)` being unconditional, changing `--hf-repo` or the chunk window silently misaligns the stream.

**Recommended future fix:** promote the range to CLI arguments (`--start-chunk`, `--end-chunk`, `--skip`), derive the skip from `start * CHUNK_SIZE`, and pass `args.hf_repo` to `create_repo`.

**Regression risk:** low.
**Safe test:** dry-run the argument parsing and assert that the computed `skip` equals `start_chunk * CHUNK_SIZE` for several values without touching the network or the dataset.

---

#### RAY-006 — Hugging Face repositories are created **public** by default
**Severity:** High (confidentiality) · **Classification:** Potential Risk · **Confidence:** High (API semantics), contingent on the target repo not already existing

**Files/lines**
* `ray_image/train_dit_v3.py:320` and `:336` — `api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)` — **no `private=True`**
* `ray_image/train_dit_v2.py:260` — identical pattern
* `ray_image/train_vae_v2.py:87` — identical pattern
* Contrast: `ray_image/train_dit_v3.py:407` correctly uses `private=True` for the milestone repo, and `precompute_qwen_cache.py:214-219` correctly uses `private=True` for the cache dataset. So the codebase knows the parameter — three call sites just omit it.
* `PROJECT_STATE.md:120-123` asserts all HF repos are Private; that claim is enforced by nothing in the code.

**What could go wrong:** `huggingface_hub.create_repo(exist_ok=True)` creates the repository **public** when it does not exist (privacy is a creation-time property; `exist_ok=True` does not make an existing private repo public, but it also does not protect a new one). Any typo in `--hf-repo`, any deleted repo, or any new experiment name therefore publishes weights, `optimizer.state_dict()` (which contains no raw data but does expose training scale/step), and the `config` dict to the world — silently, in a background step of a long training run.

**Recommended future fix (deferred):** always pass `private=True` on creation, or read a `--hf-private/--hf-public` flag with a private default; add a startup assertion that the target repo's `private` flag is what you expect before uploading.

**Regression risk:** none for training; a public-release workflow must opt in explicitly.
**Safe test:** unit-test the helper with a mocked `HfApi` and assert `create_repo` is called with `private=True` (or with the configured visibility).

---

### 4.3 Supplementary findings in detail (second-pass review, added 2026-10-11)

All five were found by re-reading the three N8 trainers in the **audited** tree; every citation is to commit `7b73c86`.

---

#### RAY-035 — `next_batch` skips unreadable rows silently and without bound in all three trainers
**Severity:** Medium (diagnosability + data accounting) · **Classification:** Probable Bug · **Confidence:** High that the code behaves as described (it is unambiguous); the *magnitude* needs one measurement (the invalid-row rate of the stream)

**Files/lines** (audited commit `7b73c86`)
* `ray_image/train_dit_v3.py:83-95` — the whole batch builder: `while len(images) < batch_size:` (85), `sample = next(iterator)` (86), `if image is None: continue` (88-89), `except (TypeError, ValueError): continue` (93-94), and its two call sites `:491-495` (with a `StopIteration` → rebuild-iterator retry)
* `ray_image/train_dit_v2.py:88-103` — the same structure verbatim, call site `:419-423`
* `ray_image/train_vae_v2.py:67-79` — the same structure without captions, call site `:199-203`
* `ray_image/train_dit_v3.py:90-94` puts `preprocess_thumbnail(image)` **and** `_pick_caption(sample)` inside the same `try`, so a caption-selection failure silently discards an image that was otherwise fine.

**Consequence**
* **Uncounted skips.** Nothing counts them: no per-reason counter, no log line, no `[progress]`-style report (contrast the cache builder, which after the owner's RB-17 fix does count skips per reason). A run that skipped 40 % of the stream writes exactly the same log as one that skipped none. "Step 100 000 of 200 000" therefore no longer implies "1.6 M images consumed" (batch 8) — the denominator of every progress claim becomes unknown.
* **Unbounded inner loop.** The `while` has no attempt cap. If the remaining stream is systematically unreadable — e.g. a MONET schema change makes `thumbnail` always `None`, or `preprocess_thumbnail` starts raising — the loop drains the *entire* stream at network speed producing nothing, hits `StopIteration`, and the caller's retry (`:493-495`, `train_dit_v2.py:421-423`, `train_vae_v2.py:201-203`) **rebuilds the iterator from the start and drains it again**. The observable symptom is a frozen progress bar with a live network, forever — not a crash, not a message.
* **Bias is silent.** Because `None`-thumbnail rows and unreadable rows are dropped with the same anonymous `continue`, an image/caption quality regression cannot be distinguished from a benign missing-thumbnail gap.

**Recommended future fix (same pattern now used in `precompute_qwen_cache.py` after RB-17):** count skips per reason, print the counts periodically and at the end; log `rows_consumed / images_yielded` so the effective dataset size is visible; split the `try` so caption failures do not discard good images; bound the retry loop and raise a named error instead of draining indefinitely.

**Regression risk:** low — counters and a bounded loop do not change which samples are used.
**Safe test:** feed a synthetic iterator of 100 rows (50 × `None` thumbnail, 10 × raising preprocess, 40 valid) and assert the returned tensors, the exact per-reason counts, and that a fully-invalid stream raises a named error within a bounded number of attempts rather than spinning.

---

#### RAY-036 — The milestone archive exists only on the happy path
**Severity:** Medium · **Classification:** Confirmed Bug · **Confidence:** Certain

**Files/lines** (audited commit `7b73c86`)
* `ray_image/train_dit_v3.py:401-414` — the milestone block (`if step > 0 and step % MILESTONE_INTERVAL == 0:`, interval 10 000 from `:34`) lives **inside** the `if save_checkpoint(...)` branch of the `try:` opened at `:395`
* `:384-393` — the disk-pressure branch (`local_disk_pct >= DISK_REFUSE_SAVE_PCT`) returns after `upload_checkpoint_direct(...)` (`:392-393`) **before** the `try` — no archive
* `:416-429` — the `except (RuntimeError, OSError)` fallback also re-uploads directly and returns **without** archiving
* `:403-405` — the archive raises `RuntimeError` when `HF_TOKEN` is missing, and that raise is caught by the surrounding `except` at `:416`, whose message then reads "local save failed at step=…" (`:417`) — so a milestone-token failure is mislabelled as a local-save failure.

**Consequence:** the milestone repo is the *durable* per-10 000-step record (the main repo is pruned to a rolling 4-file window, RAY-038). The two situations that most need it — a full local disk and a transient upload/exception — are exactly the two that skip it. A run that spends its late phase above the 90 % threshold can therefore end with **no milestone archive for those boundaries and only the rolling 4-file window** on the main repo: at 200 000 steps the durable record degrades to the last four checkpoints.

**Recommended future fix:** move the milestone block out of the local-save success branch (archive after either save path), or archive in a `finally`-style helper invoked by all three paths; keep `HF_TOKEN` validation outside the block that the generic `except` mislabels.

**Regression risk:** low — the archive is additive; the only behavioural change is *more* files uploaded.
**Safe test:** call `save_or_upload_checkpoint` three times with a mocked `HfApi` and a stubbed `save_checkpoint` — happy path, `DISK_REFUSE_SAVE_PCT` exceeded, and a raising save — with `step % MILESTONE_INTERVAL == 0`, and assert that all three produce a milestone upload.

---

#### RAY-037 — The default checkpoint directory is container-local scratch
**Severity:** Medium · **Classification:** Potential Risk · **Confidence:** High that the default path is as cited; the durability claim below is inference and is flagged as such

**Files/lines** (audited commit `7b73c86`)
* `ray_image/train_dit_v3.py:49` — `LOCAL_CHECKPOINT_DIR = "/kaggle/tmp/checkpoints"`; `:441` — `--output-dir` defaults to it
* Contrast: `ray_image/train_dit_v2.py:344` defaults to `/kaggle/working/ray_image_dit_n8_v2`; `ray_image/train_vae_v2.py:129` to `/kaggle/working/ray_image_vae_n8`
* `ray_image/train_dit_v3.py:314-317` — with no `HF_TOKEN` in the environment, `upload_checkpoint` prints "[HF] HF_TOKEN not set; keeping checkpoint local only" and returns

**Consequence:** with `--no-hf-upload`, or simply with an unset `HF_TOKEN`, the only copy of a 200 000-step run is written under `/kaggle/tmp/…`, which is the v3 trainer's *default* — while the two older trainers default to `/kaggle/working`. *Inference (not verified in this audit — I could not inspect Kaggle's current filesystem policy):* `/kaggle/working` is the path Kaggle persists as notebook output, whereas `/kaggle/tmp` is session-local scratch. Under that reading, the newest trainer has the least durable default, and the failure is silent (the run completes; the files vanish with the session). Even without the Kaggle-specific reading, the default points at a container path rather than at durable or versioned storage.

**Recommended future fix:** default to `/kaggle/working/ray_image_dit_n8_v3`, or make `--output-dir` required, and print the resolved output directory plus whether HF upload is enabled at startup.

**Regression risk:** none (a path string), though a resumed run must be pointed at the old directory explicitly.
**Safe test:** parse the arguments and assert the resolved default is not under `/kaggle/tmp`.

---

#### RAY-038 — The milestone repository has no retention rule
**Severity:** Low · **Classification:** Potential Risk · **Confidence:** Certain for the missing retention (grep: two `delete_file` call sites in the entire tree, both for the main repo); the size figure is computed, not measured

**Files/lines** (audited commit `7b73c86`)
* `ray_image/train_dit_v3.py:33-34` — `MILESTONE_REPO = "Rishix500/ray-image-dit-n8-v3-milestones"`, `MILESTONE_INTERVAL = 10_000`
* `:401-414` — every 10 000 steps the current checkpoint is uploaded there and nothing is ever removed
* `:256-267` — `cleanup_hf_checkpoints()` keeps `HF_CHECKPOINT_WINDOW = 4` (`:47`) files (`:260`) — but it is only called from `upload_checkpoint`, i.e. for the **main** repo (`:323`, `:339`)
* Repo-wide `grep -n delete_file ray_image/*.py` → `train_dit_v3.py:262` and `train_dit_v2.py:239` only, i.e. the two main-repo pruners. The milestone repo appears in no cleanup path.

**Consequence:** DiT v3 is 86.22 M parameters (§RAY-023), so one checkpoint (weights + two Adam moments at 4 B) is ≈ 1.03 GB; at 200 000 steps the milestone repo would hold ≈ 20 files ≈ **20.7 GB**, with every file also present in the main repo until the 4-file window evicts it. Growth is silent, unbounded by the code, and interacts with RAY-008 (a failed upload aborts a run) — the failure arrives late in a long run, when it is most expensive.

**Recommended future fix:** an explicit retention policy (keep every Nth milestone, or a max count/age), a `--milestone-interval 0` opt-out, and a size line printed next to the existing archive message.

**Regression risk:** deleting archives is irreversible — implement retention as an explicit, default-conservative policy and never prune below the last N.
**Safe test:** with a mocked `HfApi`, feed a synthetic 20-file listing to the retention function and assert exactly which names it would delete for a given policy.

---

#### RAY-039 — The v2 VAE trainer re-saves and re-uploads its final checkpoint
**Severity:** Low · **Classification:** Confirmed Bug · **Confidence:** Certain

**Files/lines** (audited commit `7b73c86`)
* `ray_image/train_vae_v2.py:168` — `last_checkpoint_step = args.start_step`; `:183` — the checkpoint's step, set **only** when resuming
* `grep -n "last_checkpoint_step" ray_image/train_vae_v2.py` → `168, 183, 184, 186, 188, 192, 198, 268` — the periodic save at `:252-265` (the actual write is `:262`) never assigns it
* `:268-273` — `if last_checkpoint_step != args.steps:` saves **and** re-uploads `ray_vae_v2_step_{args.steps:07d}.pt`

**Consequence:** because the compared variable only ever holds the *start/resume* step, the guard cannot fire: whenever `--steps` is a multiple of 500 — the normal case, and exactly how the shipped VAE was trained (`ray_vae_v2_step_0047500.pt`) — the loop has already written that exact file at `:262`, and the tail block writes it again and uploads it again (content-equivalent; the bytes simply move twice, plus one extra `save_checkpoint` on the critical path at the end of the run). The guard reads like duplicate suppression but suppresses nothing.

**Recommended future fix:** assign `last_checkpoint_step = current_step` inside the periodic-save branch (or test `checkpoint_path.exists()`).

**Regression risk:** none.
**Safe test:** drive the tail of `main()` with a stubbed `save_checkpoint` and assert the final block is skipped when the loop already wrote `args.steps`.

---

## 5. Category analyses

### 5.1 Correctness and bugs (Confirmed / Probable)

**Confirmed bugs** (behaviour provably contradicts the code's own contract):

| ID | Bug | Location |
| :--- | :--- | :--- |
| RAY-002 | Hidden architecture selection; unloadable DiT v2; crash without the flag | `generate.py:36-87` |
| RAY-005 | Hardcoded chunk window / skip | `precompute_qwen_cache.py:226-232` |
| RAY-007 | No checkpoint window in v2 trainer | `train_dit_v2.py:505-523` |
| RAY-010 | Evaluator "square" win-band is 0.954–1.001 → real squares score as circles | `tools/evaluate_toy_suite.py:41-70`, `runs/N5.1/n5_1_report.json` |
| RAY-011 | Three spellings of the gating config; hardcoded `True` default | `config.py:16`, `train_dit_v3.py:306`, `generate.py:64` |
| RAY-013 | Exit code 0 and no metrics for an empty directory | `tools/evaluate_toy_suite.py:110-113` |
| RAY-015 | `torch.empty` buffers | `train_dit_v3.py:186-189` |
| RAY-017 | N1 probe hardcodes the v1 VAE | `probe_vae_latents.py:47-54` |
| RAY-018/019 | Dead sweep logic; copy-pasted metric code without parity tests | `probe_n5_shape.py:78-126,242-277` |
| RAY-020 | `--n` vs `--samples` mismatch in the Kaggle notebook | notebook cell 41 |
| RAY-022/023/024 | Missing LICENSE; wrong param counts; stale project state | see table |
| RAY-030 | Smoke test timestep 500 vs training range [0,1] | `dit_v2_smoke.py:36` |
| RAY-034 | "Optional" stats argument that is always applied | `generate.py:96-98` |
| RAY-036 | Milestone archive bypassed by both fallback paths (disk-pressure and exception) | `train_dit_v3.py:384-393,401-414,416-429` |
| RAY-039 | Final checkpoint re-saved/re-uploaded when `--steps` is a multiple of 500 | `train_vae_v2.py:168,183,252-265,268-273` |

**Probable bugs** (logic is internally inconsistent with its own documented intent; runtime confirmation blocked):

* **RAY-001** un-whitening mismatch (above).
* **RAY-004** image-size/resolution mismatch in the N7/N8 warm-start path (above).
* **RAY-035** silent, uncounted and unbounded stream skipping in all three `next_batch` implementations (see §4.3) — the code behaviour is certain; whether it matters depends on the stream's invalid-row rate, which needs one measurement.

**Edge cases / validation gaps worth noting (not separate IDs):**
* `RAYCaptionDataset.__init__` (`dataset.py:19`) calls `read_text()` without an explicit encoding and `json.loads` without error handling: one malformed manifest line aborts construction with a raw `JSONDecodeError`. It also opens images without a context manager (`dataset.py:30`).
* `train_dit_v2.py:470` tests `torch.isnan(loss)` only, while `train_dit_v3.py:497` tests NaN **and** Inf — the v2 variant can in principle accept an `inf` loss (the subsequent gradient-clip guard catches most, but not all, cases).
* `train_vae_v2.py:252-265,268-273` always writes/compares a final checkpoint even when `args.steps` is already a multiple of 500 — the comparison variable `last_checkpoint_step` is never updated by the periodic save, so the tail block re-saves and re-uploads a content-equivalent file (**now tracked as RAY-039**, §4.3, which supersedes the earlier "unreachable-true" phrasing in this bullet).
* `n4_smoke`/`probe_text_smoke`/`whiten_smoke` all assert successfully on **randomly initialised** models; their own docstrings correctly warn that the numbers are meaningless — but nothing in CI distinguishes "code path runs" from "model works", so a *scientific* regression in the probe maths would still pass.

### 5.2 Data integrity

**Good:** `train_dit_v3.load_cache` (`lines 121-168`) is unusually careful for this codebase — it validates required keys, dtype-independent shapes, `text_emb.shape[0] == latents.shape[0]`, mask shape, non-emptiness, filename↔payload source-range agreement, strictly-increasing non-overlapping contiguous ranges, and it tries `mmap=True` first (which keeps the ~27 GB total cache from being copied into RAM). `LatentNormalizer` validates channel counts before use. `checkpoint_has_corruption`/`checkpoint_has_nonfinite_weights` refuse to write non-finite weights after first deleting a partial file, and `_checkpoint_payload` is atomic-enough for `torch.save` usage.

**Weak:**
1. **Unverifiable statistics artifact.** `ray_image/latent_stats_corrected.json` contains only `means`/`stds` — *not* the schema the N1 probe writes (`probe_vae_latents.py:136-148` writes `source`, `latent_shape`, `n_images`, `global_*`, `per_channel_*`). The file was hand-added in commit `3c821cb9` with no provenance: unknown VAE checkpoint, unknown sample count, unknown resolution, unknown whether it describes sampled or mean latents, raw or whitened space. It is nonetheless the *default input to inference*. This is the data-integrity root of RAY-001.
2. **No per-row provenance in the cache.** `build_chunk` stores only chunk-level `source_start/source_end/samples_seen/samples_cached/samples_skipped` (`precompute_qwen_cache.py:157-173`). Individual rows cannot be traced back to a MONET sample — so dedup, leakage audits, and "which image produced this latent?" questions are unanswerable, and skipped rows make row index ≠ source index.
3. **Precision downgrade with no record.** Latents are computed in fp32 and stored as **fp16** (`:137-138`), then used as fp32 flow targets (`train_dit_v3.py:178-183`). This is a legitimate memory choice (310 MB vs 620 MB per 10k chunk for latents alone; ~2.3 GB per chunk for the 47×2560 text embeddings) but it silently adds quantisation noise to the regression target and is not recorded per-chunk beyond `"latent_dtype": "float16"`.
4. **No validation split, no held-out set, no real-world metric** (RAY-026). `MONET` is streamed for training; the only evaluation is the 12-class toy suite, whose images come from the same deterministic generator that produced the training set (`ray_image/toy.py:29-53` with `seed + i` equals `tools/make_toy_dataset.py:37`). A VAE "reconstruction accuracy" of 1.000 measured on those images is an auto-encoding metric on training-distribution data — it cannot support any generalisation claim.
5. `probe_vae_latents._load_vae` loads whatever `latent_channels` the checkpoint declares into `RAYVAE` (v1 architecture) — feeding an N6/N8 `RAYVAE_v2` state dict would raise a key error rather than an explanatory diagnostic (RAY-017).

### 5.3 Security

| Check | Result |
| :--- | :--- |
| Hardcoded credentials / tokens | **None found.** Repository-wide grep for `hf_…`, `sk-…`, `ghp_…`, `AIza…`, `password`, `api_key`, `secret` returned only `os.getenv("HF_TOKEN")` uses. |
| Secrets in notebook outputs | **None found** in the 94 embedded Kaggle outputs. Informational leakage only: private Kaggle dataset slug `/kaggle/input/datasets/rishirajdev/dataset/ray_image_v0_2_whiten.pt` and Drive paths `/content/drive/MyDrive/RAY_IMAGE/…` (RAY-032). |
| Secrets management | `HF_TOKEN` exclusively via environment variables (`os.getenv("HF_TOKEN")` in 5 modules) — correct practice; no token is ever written to a checkpoint, log, or file. |
| Silent public publication | **RAY-006** — three upload helpers omit `private=True` on repo creation. |
| Deserialisation | 9 `torch.load(..., weights_only=False)` sites (RAY-021). All currently load either the author's own local files or files downloaded from the author's *private* HF repos, so there is no untrusted-input path today. The helper `utils.load_checkpoint` is, however, a project-wide choke point that would execute arbitrary pickle payloads if a checkpoint ever came from an untrusted source — including a repo accidentally made public (RAY-006) and then re-consumed. |
| File operations | No shell interpolation of user data, no `os.system`, no `eval`/`exec`. `subprocess` is used twice (`probe_vae_latents.py:158`, `probe_n5_shape.py:71-74`) with **list** arguments and `sys.executable` — no shell injection surface. Notebooks contain `rm -rf /kaggle/working/...` inside sandboxed cloud runtimes (destructive but correctly scoped, and not executed by CI). |
| Authorization | Not applicable: private repo, no service endpoints, no multi-user surface. |
| CI permissions | `.github/workflows/test.yml` declares no `permissions:` block → the job inherits repository defaults (potentially write-scoped `GITHUB_TOKEN`). Low-risk hardening item (RAY-027). |
| Vulnerability alerts / Dependabot | **Could not be inspected** (API returns 403 for this integration, §8). **No `.github/dependabot.yml` exists** (only `workflows/` is present), so automated dependency updates are not configured. |
| Release artefacts | 0 releases, 0 published artefacts. No accidental release surface. |

### 5.4 Dependencies and compatibility

* `requirements.txt`: 12 entries, all `>=` with **no upper bound and no lockfile** — CI resolves whatever is newest on the day (`torch>=2.5`, `transformers>=4.51.0`, `datasets>=3.0`, `bitsandbytes>=0.43`). This is the main build-reproducibility risk (RAY-027).
* `safetensors>=0.4` is never imported directly (it is pulled in transitively by `transformers`/`torch`) — dead requirement.
* `bitsandbytes` and `accelerate` are not imported by name but are **required at runtime** by `BitsAndBytesConfig(load_in_8bit=True)` (`text_encoder.py:49-67`) — correct to list, but nothing validates their presence: the ImportError message mentions all three packages yet the `try` only wraps the transformers import (`text_encoder.py:48-53`).
* `ray_image/dit_v2.py` and `dit_v3.py` use PEP-604/585 annotations (`torch.Tensor | None`, `tuple[...]`) → **Python ≥ 3.10 required**; CI pins 3.11 while the recorded Kaggle runs used Python 3.12.13 (notebook metadata) — no `python_requires`/`pyproject.toml` declares this anywhere (RAY-027).
* `text_encoder.py:65` hardcodes `device_map={"": 0}` → the N8 text encoder can only be constructed on **CUDA device 0**. `generate.py:105` nonetheless falls back to `cpu` when CUDA is absent, producing a confusing failure instead of a clear precondition error (also relevant to RAY-002).
* AST import scan: the only import not listed in `requirements.txt` is `matplotlib`, used solely inside the Kaggle notebook (not part of the package and not executed by CI).
* Transitive: `lpips` needs `torchvision` + `scipy`; the CI pre-download step (`test.yml:19-21`) correctly warms the VGG weights, so an offline/blocked run would fail there rather than in the smoke tests — a reasonable ordering.
* No vulnerable dependency was identified; a proper scan (pip-audit/Dependabot) **could not be executed** because installing tools was prohibited (§8).

### 5.5 Performance and resource reliability

* **Measured duplicate work:** in `train_vae.py` stage 3 the discriminator runs on `recon` twice per step (lines 141 and 174) — once to compute the adaptive weight, once after the D-step. The first graph is discarded. ~1 extra D forward/step (D = 2.77 M params per pure-Python arithmetic) — small but pure waste (RAY-016 family).
* **Checkpoint I/O budget (computed, not run):** DiT v2 ≈ 200.15 M params → ~800 MB weights, ~2.40 GB per file with Adam moments; DiT v3 ≈ 86.22 M → ~1.03 GB per file. `train_dit_v3.py` keeps a 2-file local window and refuses to save at ≥90 % disk (lines 48-50, 281-288, 384-388); **`train_dit_v2.py` does neither**, and its default output directory is `/kaggle/working/ray_image_dit_n8_v2` — Kaggle's working quota is tens of GB, so a 200k-step run at 500-step cadence would attempt ~961 GB of writes (RAY-007).
* **Cache memory:** a 10k chunk ≈ 310 MB latents + ≈2.3 GB text embeddings; `load_cache` for a directory holds all chunks (mitigated by `mmap=True`, which was verified by reading the code, not by running it).
* **VRAM/latency:** `ray_image/dit_v2.py` builds 200 M params; `dit_v2_smoke` (CPU-capable) would need ~1.6 GB RAM on CPU. `train_dit_v3.py` uses gradient checkpointing per block with `use_reentrant=False` (`lines 219-255`) — correct choice for T4-class hardware.
* **No memory-leak patterns found:** no module-level tensor caches, no growing global lists inside the step loops (the NaN-skip counters are bounded ints), no unclosed file handles except the un-context-managed `Image.open` in the dataset.
* **Logging cost:** `checkpoint_has_corruption` walks every parameter with a GPU `torch.isfinite(...).all()` per checkpoint (every 500 steps) — negligible.
* **Milestone storage growth (RAY-038, computed, not run):** ≈1.03 GB per milestone file for DiT v3 (weights + two Adam moments), one file per 10 000 steps, in a repository that nothing ever prunes (§4.3) → ≈20.7 GB for a 200 000-step run, duplicated in the main repo until the 4-file window evicts it. Compare the deliberate windows: `LOCAL_CHECKPOINT_WINDOW = 2`, `HF_CHECKPOINT_WINDOW = 4` — both apply to the main repo only.
* **Default output directory is container-local (RAY-037):** `train_dit_v3.py:49,441` defaults to `/kaggle/tmp/checkpoints`; `train_dit_v2.py:344` and `train_vae_v2.py:129` default to `/kaggle/working/…`. With HF upload disabled or `HF_TOKEN` unset (`train_dit_v3.py:314-317`), the only copy of a run lands on the v3 default path. *Labeled inference:* this is likely session scratch rather than persisted output (see §4.3 for the caveat). The fix is one path string; the exposure is a whole run.

### 5.6 Numerical stability and reproducibility (AI/ML specific)

**Good:** both DiT trainers guard NaN loss and non-finite gradient norms, clip gradients at 1.0, use AdamW `(0.9, 0.99)` + `weight_decay=0.01`, and refuse to persist non-finite weights; v3 adds warm-up (`set_learning_rate`, lines 372-377) and fp16 (not bf16) autocast, which is the correct choice for T4; `RMSNorm` accumulates in fp32 (`dit_v3.py:27-30`); RoPE casts its tables to the activation dtype (`dit_v3.py:78-82`).

**Weak / open:**
* **bf16 on T4 (RAY-012).** `train_dit_v2.py:418-419` prints `"[precision] DiT forward autocast=bfloat16"` and line 437 enables exactly that, while `PROJECT_STATE.md:15,109` states the target hardware is Kaggle T4 (the committed Kaggle notebook’s recorded `nvidia-smi` output confirms 2 × Tesla T4, 15.36 GB each, PyTorch 2.10.0+cu128). Turing (SM 7.5) has no native bfloat16 support; depending on the PyTorch version, bf16 autocast on that hardware either silently falls back for many ops or raises. The commit history ("fix(N8): prevent DiT v2 NaN divergence — correct LR, bf16, grad clip, guards") suggests bf16 was adopted as a remedy, which makes the *hardware/precision assumption* worth re-verifying rather than a style point. **Uncertainty: I could not execute on a T4, and I am not claiming it failed — only that the log line cannot be trusted to describe the actual precision.**
* **RNG state is not checkpointed.** `train_dit_v2/3` save `{"step","seed", "dit","optimizer"}`; resuming re-seeds from `--seed` at process start (`seed_everything`) and then *does not* advance RNG by `step` — so the noise/interpolation stream after a resume is a fresh sequence, not the continuation. The project documents this as acceptable ("constant LR is safe for chunks"), and for flow matching it is a defensible choice, but it means **runs are not bit-reproducible across chunk boundaries** and no two resumed chunks are statistically identical.
* **Teacher-forced data order is not preserved:** cache training samples randomly with replacement (`next_cached_batch`), streaming training samples sequentially from the beginning of MONET on every fresh Kaggle session (`load_dataset(...).skip(0)`), so the *data* seen is not reproducible across sessions either.
* **Fixed composition:** `models/dit_v2/train` use `prediction.float()` MSE; loss is computed outside autocast — fine. `sample_flow_pair` uses `torch.rand(t)` uniform — standard rectified-flow setup, and `euler_sample` integrates `x ← x − dt·v` from t=1 → 0 using the velocity at the left endpoint: **the sampler and the objective are mutually consistent** (verified by reading `flow.py:4-31` against each trainer’s target construction). This is a genuinely correct implementation and should be preserved.
* **N1 whitening statistics were computed from mean latents** (`probe_vae_latents.py:83-95` uses `vae.encode(x)` → `mean`), while the N8 trainers train on **sampled** latents (`vae(images)` includes `mean + std·ε`). Two different distributions are therefore being normalised/reasoned about in the same project (RAY-026/RB-1 below). The N2-era `train_generator.py:122-131` deliberately used the mean latent "so the tiny semantic experiment gets a deterministic image target" — a decision that was never re-evaluated when the N8 pipeline switched to sampled latents.

### 5.7 Evaluation correctness

* The only quantitative metric in the repository is `tools/evaluate_toy_suite.py`. Its shape decision reduces to a 1-D threshold on `bbox_fill` because corner features are computed at **whole-image corners** (`:41-45`) rather than bounding-box corners, making the elaborate `corner_targets` (`:64-68`) inert for centred shapes. The measured winning bands (from the committed `n5_1_report.json`): triangle < 0.626, circle 0.626–0.954, **square > 0.954** — a 4.6 %-wide band for squares. Consequence: visually correct squares are scored as circles, i.e. `shape_accuracy = 0.333` is largely a **measurement artefact**, exactly as documented in `runs/N5.1/REPORT.md §4`.
* The finding has been *known since 2026-09-08* and the recommended fix ("measure corners/quadrants inside the object bounding box") was **never implemented** (`tools/evaluate_toy_suite.py` is unchanged). Every shape-derived number in the tree (`PROJECT_STATE.md:32`, the N5 notebook outputs, the evaluator blocks inside `n5_1_report.json`) inherits that bias (RAY-010).
* The probe re-implements the evaluator's feature and score functions (`probe_n5_shape.py:78-126`) instead of importing them, with **no parity test** — the very act of fixing the evaluator will silently desynchronise the diagnostic that is supposed to audit it (RAY-018/019).
* `evaluate_toy_suite.py` prints nothing and exits **0** when the directory is empty or all files are missing (guarded by `if total:`, line 110) — a wrong `--dir` looks like success (RAY-013). The probes that parse this output only survive because `probe_smoke.py:60-62` asserts on the parsed metrics.

### 5.8 Documentation vs implementation

| Claim | Evidence | Verdict |
| :--- | :--- | :--- |
| README: "Core architecture ✅ Complete", "Foundation training ✅ Complete", "Output: … up to 256×256" | `README.md:20-21,32-36` | **Ahead of evidence.** The only *trained* 256 px component is the VAE; no committed artefact or report demonstrates a real-world generated 256 px image. Treat as roadmap, not status. |
| README: "Prompt-faithful composition" | `README.md:25` | Contradicted by the project's own N5.1 report, which explicitly declines to conclude that shape conditioning works (§5(4)). |
| README: "Reproducibility above all. Every experiment, checkpoint, and metric is versioned." | `README.md:52` | Partially false: weights live in private HF repos / Drive only; whitening stats have no provenance; resume does not restore RNG; dependencies are unpinned. |
| README: "Apache 2.0 — free for research and commercial use." | `README.md:91-93` | **No LICENSE file exists** (`gh api …/license` → 404; absent from the tree). No licence is granted; MONET's own Apache-2.0 terms still bind the data. (RAY-022) |
| README: "Contact: [rishirajc406@gmail.com]" | `README.md:99` | Personal address with no formatting/obfuscation in a repo whose plan is a public release — acceptable for a project like this, but note it is the only personal datum in the tree. |
| PROJECT_STATE: "N8.7 ⏳ Next — Build `train_dit_v2.py`" / "Immediate Next Steps: 1. Build `ray_image/train_dit_v2.py`" | `PROJECT_STATE.md:54,177` | **Stale:** `train_dit_v2.py` (17.3 KB) was committed 2026-09-13 13:46, ~4 h after the doc update (09:27). The document is also silent about everything after 2026-09-23 (DiT v3, cache extension, latent-stats change) — the newest commit that touched PROJECT_STATE is `abf9be8f`; ~40 later commits exist. (RAY-024) |
| PROJECT_STATE: component sizes (DiT v2 ~180 M, VAE v2 ~10 M) / dit_v3 docstring "~60M-parameter" | `PROJECT_STATE.md:53,63,90`; `dit_v3.py:197` | **Wrong.** Constructor arithmetic (no execution needed): DiT v2 = **200,150,080** (200.15 M), DiT v3 = **86,215,244** (86.2 M, SwiGLU hidden 1344), VAE v2 = **5,704,646** (5.70 M), v1 DiT = 7.73 M, v1 VAE = 0.387 M, PatchGAN = 2.77 M. (RAY-023) |
| PROJECT_STATE §6 layout says the root is `anime-ai-companion/` and lists `dit_v2_smoke.py` under `ray_image/` etc. | `PROJECT_STATE.md:127-150` | Stale repo name (the repository was renamed; `Rishidev-20thcenturey/anime-ai-companion` now 301-redirects to the current name — verified via the API). Relative paths still resolve because GitHub redirects, but every Colab notebook writes into `/content/anime-ai-companion`. |
| "Detailed roadmap available on request." | `README.md:76` | Roadmap actually exists in `PROJECT_STATE.md:154-171` (N9–N12, with unverifiable target metrics such as "0.97 OCR", "0.89 Prism") — i.e. internal targets presented as public capability plan. |

### 5.9 Build, deployment and CI

* Single workflow `.github/workflows/test.yml`: `push`/`pull_request` to `main` only, one job, `ubuntu-latest`, Python 3.11, `pip install -r requirements.txt`, LPIPS weight pre-download, then 6 smoke scripts (`train_smoke`, `probe_smoke`, `whiten_smoke`, `probe_text_smoke`, `n4_smoke`, `n5_smoke`).
* **Verified green:** the latest `main` push (run `38052442985`, 2026-10-10 12:34:16Z → 12:36:57Z) passed every step in **2 min 38 s**; the last 15 runs all concluded `success`. Two workflow files ran in the N8 era (`RAY-IMAGE tests` and a `Backend CI` that came with PR #3 and no longer exists in `main`).
* **Coverage gap (RAY-009):** none of the six steps touches `dit_v2`, `dit_v3`, `vae_v2`, `text_encoder`, `precompute_qwen_cache`, `train_dit_v2/v3`, `train_vae_v2`, or `generate.main`. `dit_v2_smoke` supports `--device cpu` (`dit_v2_smoke.py:10-13`) and would run in CI, yet is not wired in. `vae_v2_smoke.py:9-10` and `dit_v3_smoke.py:13-14` hard-require CUDA and therefore *cannot* run on GitHub-hosted runners at all.
* No dependency caching, no matrix, no `timeout-minutes`, no `concurrency`, no artefact upload, no `permissions:` block, no `if: github.event_name` guards. Given a 2m38s runtime, caching `~/.cache/pip` and pre-downloading LPIPS is the single cheapest CI improvement.
* Deployment: none (this is a training-research repo). Release readiness is blocked by RAY-001/002/022 and by the absence of any held-out evaluation.

---

## 6. Testing and verification results

### 6.1 Checks that passed (executed in this audit)

| # | Command / method | Result |
| :--- | :--- | :--- |
| 1 | `git status`, `git diff --stat origin/main`, `git diff --stat HEAD` | clean / empty / empty |
| 2 | `git for-each-ref` | 4 refs, all at `7b73c865…` as cloned; unchanged |
| 3 | `git reflog --all` | only the original clone/checkout entries |
| 4 | `git fsck --no-progress` | no errors; no loose objects created (`find .git/objects -type f -not -path '*/pack/*'` → empty) |
| 5 | `ast.parse` over all 40 tracked `.py` files (PYTHONDONTWRITEBYTECODE=1) | 40 parsed, **0 syntax errors** |
| 6 | AST duplicate-body hashing across `ray_image/` + `tools/` | 10 byte-identical definitions; ~8 near-duplicates at 0.46–0.90 similarity |
| 7 | AST import graph (third-party + intra-repo) | no circular imports; every `requirements.txt` package except `safetensors` is imported (bitsandbytes/accelerate indirectly by design) |
| 8 | Pure-Python parameter accounting reconstructed from the constructors | DiT v1 7,734,848 · DiT v2 200,150,080 · DiT v3 86,215,244 · VAE v2 5,704,646 · VAE v1 386,531 · PatchGAN 2,766,529 |
| 9 | PNG signature/chunk parse of `runs/N5.1/oracle/*.png` | 12 files, all **64×64, 8-bit, truecolour**, `IHDR/IDAT/IEND` only — consistent with `probe_n5_shape.build_oracle(size=64)` |
| 10 | Notebook JSON inspection (7 notebooks) | schema valid; 94 outputs only in the Kaggle notebook; **no tokens/keys/e-mails** in any source or output |
| 11 | Repository-wide greps (secrets, `torch.load`, `subprocess`, `eval/exec`, module references, dead code) | as reported in §5.1/§5.3; `ray_image/train.py` and `ray_image/evaluate.py` are unreferenced (dead) |
| 12 | GitHub API reads: repo metadata, branches, PR list, `main` commit pagination (135), workflow runs, job steps, commit patches, contents of other branches, licence endpoint (404) | all consistent with §1 and §4 |

### 6.2 Checks that were **not** run, and why

| Check | Reason | Alternative verification |
| :--- | :--- | :--- |
| `python -m ray_image.*_smoke` (all 6 CI tests) | Requires `torch`, `numpy`, `PIL`, `lpips`, `datasets` — none installed; **installing packages was prohibited**. | The same six scripts pass in GitHub Actions (evidence §5.9); re-run locally in a disposable venv if desired. |
| Any training / cache build / inference | Prohibited (expensive, mutating, requires GPU + private HF credentials). | Read the CI logs; run `--no-hf-upload --steps 2` variants on a scratch copy. |
| `tools/evaluate_toy_suite.py --dir …` | Needs numpy/PIL. | CI's `probe_smoke` exercises it end-to-end via subprocess with assertions on parsed metrics. |
| Dependency vulnerability scan (pip-audit, Dependabot alerts, `vulnerability-alerts`) | Tool installation prohibited; GitHub endpoints returned **403 for this integration**. | Enable Dependabot alerts in repository settings and re-query. |
| Branch-protection / security-feature inventory | `…/branches/main/protection` → **403**; `security_and_analysis` returned empty for this private repo. | Check in the GitHub UI as the owner. |
| Hugging Face artefacts (private repos, cache chunks, checkpoint contents) | `huggingface.co` is not reachable from this environment. | Inspect repo privacy/contents in the HF UI; verify the cache's real sequence length (RAY-003). |
| Kaggle/Colab execution of the 7 notebooks | Cloud credentials, GPU, and large downloads. | Notebooks are read-only reviewed here; their recorded outputs were inspected. |
| `git log`/`git blame` history in full detail for old files | Clone is **shallow** (1 commit); only API metadata was available. | `git fetch --unshallow` on the owner's clone for commit-level attribution. |

### 6.3 Checks that failed (i.e. produced evidence of a defect)

Only one "check" failed in the sense of contradicting an expectation, and it was accidental: `python -m compileall -q -f ray_image tools` wrote bytecode into the working tree (see §6.4). Every other automated check passed; the defects in §4 were established by reading code (and, for RAY-023, by arithmetic), not by failing executions.

### 6.4 Integrity incident disclosure (full transparency)

While performing "read-only syntax checks" I ran `python -m compileall -q -x '__pycache__' -f -d . ray_image tools`. Despite `-q`, `compileall` **creates `__pycache__/*.pyc` files** — i.e. it wrote into the repository working tree. I detected this immediately in the very next check (`find . -name __pycache__`), verified by timestamp (`2026-10-11 08:14:29` for the `.pyc` files vs `2026-10-11 08:12:38` for every checked-out file) that they were **my** artefacts and not pre-existing, deleted exactly those two directories (`ray_image/__pycache__`, `tools/__pycache__`), and confirmed restoration:

* `find . -not -path './.git/*' -newermt "2026-10-11 08:13:00"` → only `.git`, `ray_image`, `tools` (directory mtimes, not contents);
* `git status --porcelain --ignored` → **empty**;
* `git diff --stat origin/main` and `git diff --stat HEAD` → **empty**;
* `.git` object database unchanged (no loose objects), `git fsck` clean, refs/reflog unchanged.

Consequence for the audit: no tracked file was created, modified, renamed, or deleted; no commit, branch, tag, PR, issue, release, workflow or repository setting was touched; the repository content is byte-identical to the cloned commit. **The lesson for future audits: `compileall` is a write operation — use `ast.parse` (as done thereafter, with `PYTHONDONTWRITEBYTECODE=1`).**

---

## 7. Confirmed bugs, probable bugs, potential risks, improvements (grouped)

### 7.1 Confirmed bugs (behaviour contradicts the code's own documented contract) — fix these first

1. **RAY-002** hidden architecture selection in `generate.py` (DiT v2 unloadable; v3 crashes without the flag).
2. **RAY-005** hardcoded cache chunk window / `skip(50_000)` / `create_repo` ignoring `--hf-repo`.
3. **RAY-007** `train_dit_v2.py` writes unbounded checkpoints (~961 GB projected for 200k steps) with no disk guard.
4. **RAY-010 / RAY-019** known-broken shape metric left unfixed, duplicated into the diagnostic, no parity test.
5. **RAY-011** three spellings of the cross-gate flag with a hardcoded `True` default.
6. **RAY-013** evaluator returns success with zero metrics.
7. **RAY-017 / RAY-018** probes hardcoding the wrong architecture / carrying dead code.
8. **RAY-020 / RAY-022 / RAY-023 / RAY-024 / RAY-030** documentation and notebook defects (see §5.8).
9. **RAY-015** uninitialised `torch.empty` buffers used to assemble training batches.
10. **RAY-034** an "optional" statistics argument that is always applied.
11. **RAY-036** the 10 000-step milestone archive is skipped by both fallback paths (disk-pressure and exception) — see §4.3.
12. **RAY-039** `train_vae_v2.py` re-saves and re-uploads its final checkpoint whenever `--steps` is a multiple of 500.

### 7.2 Probable bugs (logic inconsistent with its own stated intent — needs one experiment to confirm)

1. **RAY-001** un-whitening at inference for checkpoints trained in raw latent space.
2. **RAY-004** 64 px inputs into a 256 px VAE on the N7/N8 warm-start path.
3. **RAY-031** empty legacy caption ⇒ all-padding row ⇒ fully-masked cross-attention (documented as 0 in modern PyTorch, historically NaN; never observed here because the toy captions are never empty).
4. **RAY-003 (the 47-token part)** — likely benign today, but only because every MONET Florence-2 caption is long enough to be truncated; unproven for the published cache.
5. **RAY-035** uncounted, unbounded stream skipping in `next_batch` (all three trainers) — the mechanism is certain, the rate is not; one measurement settles it (§4.3).

### 7.3 Potential risks (contingent, but cheap to neutralise)

* **RAY-006** public-by-default HF repo creation (disclosure).
* **RAY-008** uncaught upload failures killing a run.
* **RAY-012** bf16 claimed on non-bf16 hardware.
* **RAY-021** pickle-based checkpoint loading (9 sites).
* **RAY-025 / RAY-026** fp16 targets, no provenance, no held-out evaluation.
* **RAY-029** 1-D RoPE over a flattened 2-D grid (loses v2's explicit 2-D bias).
* **RAY-032** stale notebook outputs leaking private slugs/paths.
* **RAY-009** zero CI coverage for the current architecture generation.
* **RAY-037** the v3 trainer's default output directory is container-local scratch, so a run without HF uploads keeps its only copy somewhere that may not outlive the session.
* **RAY-038** the milestone repository grows without a retention rule (≈20.7 GB at 200 000 steps).

### 7.4 Improvements / informational

* **RAY-014 / RAY-016 / RAY-027 / RAY-028 / RAY-033** — documentation of the N8 pipeline, de-duplication (`seed_everything` ×3, `preprocess_thumbnail` ×2, `upload_checkpoint` ×2, `load_resume` ×2, `PatchGAN` ×2, model-forward copies in both trainers), dependency pinning/Dependabot/CI `permissions`, retirement notes for dead modules, and cleanup of stale branches/cosmetic commit messages.

### 7.5 Working components and sound design decisions worth preserving

| Keep | Why |
| :--- | :--- |
| `ray_image/flow.py` (flow-matching pair + Euler sampler) | Mathematically consistent with every trainer's target construction; verified by cross-reading, not assumed. |
| `ray_image/whiten.py` `LatentNormalizer` | Correct invertible per-channel normalisation with channel-count validation and a clear train/serve convention — the *convention* is good; RAY-001 is a wiring bug, **not** a reason to delete it. |
| `ray_image/dit_v3.py` internals | RMSNorm in fp32, correct RoPE half-rotation, correct SDPA mask inversion (`attn_mask = ~context_mask`, `True = keep`) — the mask convention is easy to get wrong and is right here. |
| `train_dit_v3.py` cache validation + disk guards + non-finite checkpoint refusal | Best engineering in the repository; the template the other three trainers should follow. |
| `runs/N5.1/REPORT.md` + `n5_1_report.json` + oracle PNGs | Exemplary, measured, self-critical diagnostics that explicitly refuse to fabricate results or over-claim — the project's "diagnose before you train" principle in practice. |
| `n5_smoke.py` | Genuinely strong test design: asserts gate existence/init, same-weights equivalence at gate = 1.0, mask preservation, checkpoint round-trip, warm-start, *and* the generation loader path. Preserve its shape when adding v2/v3 coverage. |
| The 6 CI smoke tests + LPIPS pre-download step | Green in 2m38s; a real safety net for the toy pipeline. |
| `tools/make_toy_dataset.py` + `ray_image/toy.py` | Deterministic, shared rasteriser used by dataset, oracle and evaluator — good use of determinism for reproducible diagnostics. |
| `git`/PR workflow discipline (revert branch + PR + green CI before merge) | The accidental backend merge was reverted properly through a PR with CI, and `main` is currently clean of it (verified by tree inspection). Keep this pattern. |

---

## 8. Areas that could not be inspected or verified

1. **Hugging Face artefacts** — `huggingface.co` is unreachable from the audit environment. Therefore: the existence/privacy of `Rishix500/ray-image-{vae-n6,vae-n8,dit-n7,dit-n8-v3,dit-n8-v3-milestones}` and `monet-qwen-cache-50k`, the *actual* sequence length stored in the cache (RAY-003), the contents of `ray_vae_v2_step_0047500.pt`, and whether the milestone repo really is private, are all **unverified**. All statements about them come from the code and `PROJECT_STATE.md`.
2. **Trained weights** — no `.pt` file is in the repository (deliberately ignored via `.gitignore:13`). No real generated image exists in the tree (only the 12 ideal-geometry oracle PNGs). **No claim about actual image quality is made anywhere in this report.**
3. **Runtime behaviour of any model code** — PyTorch/NumPy/PIL are not installed; installing them was prohibited. All correctness statements are static (control flow, shapes, masks, arithmetic).
4. **CUDA/T4-specific behaviour** — bf16 autocast fallbacks, 8-bit `bitsandbytes` behaviour, VRAM peaks, T4 kernel availability: unverified.
5. **Git history details** — the clone is **shallow**; only API metadata and the two patches quoted above were available. `git blame`, the full 135-commit diff-by-file, and the deleted branches' history could not be walked.
6. **Dependabot alerts, vulnerability alerts, branch protection, secret-scanning configuration** — HTTP 403 for this integration.
7. **Other local worktrees / private branches not pushed** — none visible; anything unpushed is out of scope by definition.
8. **The un-shallow-able question behind RAY-001** — whether the author's *actual* v3 training run applied whitening outside the committed code (e.g. in a modified copy). The committed code says no; the author's private logs are the only way to settle it.
9. **MONET dataset semantics** (caption length distribution, thumbnail resolution) — no dataset access; only code-level assumptions were audited.
10. **The Kaggle/Colab notebook executions** were inspected as artefacts (source + recorded outputs), not re-run.

---

## 9. Staged future remediation plan

> Ordering is by *risk retired per unit of effort* and by *dependency*: do not refactor what you are about to re-specify. Stages are designed to be independently shippable.

### Stage 0 — Decide the facts the code cannot tell you (hours, no code)
- [ ] **0.1** Determine whether the DiT v2/v3 training runs whitened latents (RAY-001). Check the training shell commands / notebook cells / cache-build invocation. Write the answer into `PROJECT_STATE.md` as a one-liner: *"latents at DiT input are {raw | whitened}"*.
- [ ] **0.2** Inspect the published cache: `payload["text_emb"].shape` and `payload["mask"].sum(dim=1).min()`. Record the real `L` (RAY-003).
- [ ] **0.3** In the HF UI, record the actual visibility of every repo the code can write to (RAY-006).

### Stage 1 — Stop silent wrongness (days; highest value)
- [ ] **1.1** Make the latent-space contract explicit: write `"whiten"` (or `"latent_space"`) into every DiT checkpoint's config, require it at load, delete the hardcoded default; `--latent-stats` must be an explicit opt-in (RAY-001, RAY-034).
- [ ] **1.2** Add resolution/channel validation to `build_n7_from_n6` and to the VAE/DiT handshake (RAY-004).
- [ ] **1.3** Replace the `--vae-checkpoint`-implies-v3 behaviour with an explicit `--arch` (or a checkpoint-declared arch) and a descriptive error on mismatch (RAY-002).
- [ ] **1.4** Record and enforce the text-context length contract end-to-end (cache `seq_len` → trainer → inference) (RAY-003).
- [ ] **1.5** Always `private=True` on `create_repo`, or an explicit `--hf-private` default (RAY-006).

### Stage 2 — Make long runs safe (days)
- [ ] **2.1** Port the v3 disk policy (local window + ≥90 % refusal + non-finite guard + direct-to-HF fallback) into `train_dit_v2.py` and `train_vae_v2.py` (RAY-007).
- [ ] **2.2** Wrap every HF upload in bounded retry/backoff; never let a checkpoint upload abort training (RAY-008).
- [ ] **2.3** Persist RNG + dataloader/cache cursor state in checkpoints, or document loudly that runs are only step-resumable and not stream-identical (RAY-026).
- [ ] **2.4** Replace `torch.empty` batch buffers with `zeros` + a post-assembly row-coverage assertion (RAY-015).

### Stage 3 — Restore measurement integrity (days; unlocks every scientific conclusion)
- [ ] **3.1** Fix the shape metric to use bounding-box-local corner occupancy, then re-baseline **all** N2/N5/N5.1 numbers and mark the old ones superseded (RAY-010).
- [ ] **3.2** Delete the copy in `probe_n5_shape.py` and import the evaluator's functions; add a parity assertion test (RAY-018/019).
- [ ] **3.3** Make the evaluator fail (non-zero exit) when a directory is missing/empty (RAY-013).
- [ ] **3.4** Introduce a held-out monitor set for the real pipeline (even 512 fixed MONET samples with cached embeddings) and one honest metric (reconstruction PSNR/LPIPS on held-out data + a fixed-prompt visual sheet) (RAY-026).
- [ ] **3.5** Fix `probe_vae_latents` to accept an architecture selector so the N1 probe can audit VAE v2 (RAY-017).

### Stage 4 — CI and dependency hardening (days)
- [ ] **4.1** Add CPU-capable smoke steps: `dit_v2_smoke --device cpu`, a `vae_v2` shape test that skips without CUDA, a `generate.load_models` test covering both branches with tiny synthetic checkpoints (RAY-009).
- [ ] **4.2** Pin the CI toolchain (`pip install -r requirements.txt --constraint constraints.txt` or a lockfile), add `permissions: contents: read`, add `timeout-minutes`, add pip caching (RAY-027).
- [ ] **4.3** Add `.github/dependabot.yml` for pip + GitHub Actions; enable Dependabot/secret scanning in settings.
- [ ] **4.4** Add a `pytest` wrapper so CI reports per-assertion failures instead of a single script exit code.

### Stage 5 — Hygiene and documentation truth (a day)
- [ ] **5.1** Add a LICENSE file (Apache-2.0 as the README claims) or change the README (RAY-022).
- [ ] **5.2** Correct the parameter counts and refresh `PROJECT_STATE.md`, including an explicit "current generation = N8 (DiT v2/v3, cache, VAE v2)" section **and** a short "how to run N8" recipe (RAY-023/024/014).
- [ ] **5.3** De-duplicate the measured 10 identical + 8 near-identical symbols behind `ray_image/common.py` — starting with `seed_everything`, `preprocess_thumbnail`, `upload_checkpoint`, `load_resume`, `_checkpoint_sort_key`, `cleanup_hf_checkpoints` (RAY-016).
- [ ] **5.4** Have `checkpointed_forward` call a single shared `model.forward(..., checkpoint_blocks=True)` so the model and trainer can never diverge again (RAY-016, cross-cutting).
- [ ] **5.5** Point the Colab notebooks at `main` (or a tagged revision) instead of the 82-commit-stale `arena/01a07cdc-…` branch, and delete/archive the stale branches (RAY-014/033).
- [ ] **5.6** Strip notebook outputs before committing (or use `nbstripout`) (RAY-032).
- [ ] **5.7** Mark or delete dead modules (`train.py`, `evaluate.py`) and the abandoned variants (RAY-028).

### What can wait
Stages 3.4 and 4 are quality-of-life; RAY-029 (RoPE dimensionality) is a research question, not a defect — it deserves an ablation *after* measurement is trustworthy, not before.

---

## 10. Regression risks, rollback precautions and recommended tests

### 10.1 Cross-cutting regression risks

| Change | Risk | Precaution | Rollback |
| :--- | :--- | :--- | :--- |
| RAY-001 fix (whitening) | Any already-published image/claim may change; if the transform really was load-bearing, removing it "breaks" the model | Save the current default path behind `--legacy-unwhiten`; produce a 12-prompt comparison sheet before/after | Keep the old stats file in git history; the flag restores exact prior behaviour |
| RAY-002 refactor (arch selector) | `whiten_smoke` and `n5_smoke` step 7 load through `generate.load_models`; a signature change breaks them | Keep the legacy default path untouched; run both smokes plus `train_smoke` after the change | Revert to the previous `generate.py` (single file) |
| RAY-003 (sequence-length contract) | Changing cache padding invalidates the existing 100k cache → forces a re-build (hours of GPU time) | Do **not** change padding yet; first *record* the real length and use it everywhere (`train_dit_v3.py:186`) | Keep the old cache untouched on HF; switch the loader back |
| RAY-010 (metric recalibration) | Every historical accuracy number changes → old conclusions become incomparable | Land it together with a "superseded" banner in `runs/N5.1/REPORT.md` (do not edit the old report's numbers) and label the new metric `shape_accuracy_v2` | The old evaluator is one file; restore from git |
| RAY-006 (`private=True`) | Blocks a future *intended* public release | Add `--hf-public` opt-in in the same change | Flip the flag |
| Removing duplicates (§9.5.3) | Silent behavioural drift if two copies were subtly different (e.g. `preprocess_thumbnail` 0.50 similarity is a real divergence: one version validates the PIL type, the other does not) | Merge to the **stricter** variant and keep both call sites' tests | Cherry-pick-git-revert the single refactor commit |

### 10.2 Minimum test set to add (each maps to a finding)

1. `test_generate_arch_selection` — build tiny v1/v2/v3 checkpoints, assert the right class is constructed and a mismatch raises a descriptive error (**RAY-002**).
2. `test_whitening_contract` — assert `load_models` returns `whiten is None` for a checkpoint that does not declare whitening; assert identity stats ⇔ no-op (**RAY-001**).
3. `test_text_length_contract` — short caption ⇒ cache row length equals the recorded `seq_len`; trainer reads `L` from the payload; an `L ≠ 47` payload does not crash with a shape error (**RAY-003**).
4. `test_vae_resolution_guard` — `build_n7_from_n6` asserts `image_size == 256` and `latent_size == 32` before training (**RAY-004**).
5. `test_cache_chunk_args` — computed `skip == start_chunk * CHUNK_SIZE`; `create_repo` receives `args.hf_repo` and `private=True` (**RAY-005**, **RAY-006**; mock `HfApi`).
6. `test_checkpoint_window` — with a temp dir of fake checkpoints, only N remain and disk ≥ threshold refuses the write (**RAY-007**).
7. `test_upload_failure_does_not_raise` — mocked `upload_file` raising ⇒ `save_or_upload_checkpoint` returns without propagating (**RAY-008**).
8. `test_evaluator_exit_codes` — missing/empty dir ⇒ non-zero exit and no `*_accuracy=` line (**RAY-013**).
9. `test_evaluator_metric_parity` — a synthetic square with 4 % soft edge must be classified `square` (this test **fails today**, which is the point) (**RAY-010**).
10. `test_dit_v2_smoke_cpu` — wire `--device cpu` into CI (**RAY-009**, **RAY-030**).

---

## 11. Fix-tracking checklist

### Critical / High
- [ ] **RAY-001** Reconcile whitening between DiT training and inference *(High, Probable Bug)*
- [ ] **RAY-002** Explicit architecture selection in `generate.py`; support DiT v2 *(High, Confirmed)*
- [ ] **RAY-003** Data-driven text-context length contract caches ⇄ train ⇄ serve *(High, Potential Risk)*
- [ ] **RAY-004** Resolution guard for the 256 px VAE in the N7/N8 path *(High, Probable Bug)*
- [ ] **RAY-005** Parameterise cache chunk range/skip/repo *(High, Confirmed)*
- [ ] **RAY-006** `private=True` on HF repo creation by default *(High, Potential Risk)*

### Medium
- [ ] **RAY-007** Checkpoint window + disk guard in `train_dit_v2.py` *(Confirmed)*
- [ ] **RAY-008** Non-fatal, retried HF uploads *(Potential Risk)*
- [ ] **RAY-009** CI coverage for v2/v3/vae_v2 architectures *(Improvement)*
- [ ] **RAY-010** Recalibrate the shape metric; supersede old numbers *(Confirmed)*
- [ ] **RAY-011** Unify the cross-gate config key; validate at load *(Confirmed)*
- [ ] **RAY-012** Verify/declare the precision policy for T4 (bf16 vs fp16) *(Potential Risk)*
- [ ] **RAY-013** Non-zero exit for empty/missing evaluation directories *(Confirmed)*
- [ ] **RAY-014** Document the N8 pipeline; unpin stale branches in notebooks *(Potential Risk)*
- [ ] **RAY-035** Count and bound the stream skips in `next_batch`; split caption failures from image failures *(Probable Bug)*
- [ ] **RAY-036** Archive milestones on the fallback paths too; keep the token check out of the mislabelling `except` *(Confirmed)*
- [ ] **RAY-037** Durable default checkpoint directory (or require `--output-dir`) *(Potential Risk)*

### Low / Informational
- [ ] **RAY-015** `torch.empty` → validated assembly *(Confirmed)*
- [ ] **RAY-016** De-duplicate the 10 identical + 8 near-identical symbols *(Improvement)*
- [ ] **RAY-017** `probe_vae_latents`: architecture selector *(Confirmed)*
- [ ] **RAY-018** Remove dead code in `decision_boundaries()` *(Confirmed)*
- [ ] **RAY-019** Import (not copy) evaluator metrics; parity test *(Confirmed)*
- [ ] **RAY-020** Fix `--n` → `--samples` in the Kaggle notebook *(Confirmed)*
- [ ] **RAY-021** Document/limit `weights_only=False`; centralise loading *(Potential Risk)*
- [ ] **RAY-022** Add LICENSE (or amend README) *(Confirmed)*
- [ ] **RAY-023** Correct parameter counts in docs/docstrings *(Confirmed)*
- [ ] **RAY-024** Refresh `PROJECT_STATE.md` *(Confirmed)*
- [ ] **RAY-025** Record fp16-target decision; consider fp32 latents for small caches *(Potential Risk)*
- [ ] **RAY-026** Add a held-out monitor set + one real metric; record RNG state *(Potential Risk)*
- [ ] **RAY-027** Pin/constrain dependencies; add Dependabot; CI `permissions` *(Improvement)*
- [ ] **RAY-028** Deprecate or delete `train.py`, `evaluate.py` *(Improvement)*
- [ ] **RAY-029** Ablation: 2-D positions vs 1-D RoPE *(Informational)*
- [ ] **RAY-030** Use an in-range timestep in `dit_v2_smoke` *(Confirmed)*
- [ ] **RAY-031** Guard fully-masked text rows *(Potential Risk)*
- [ ] **RAY-032** Strip notebook outputs *(Potential Risk)*
- [ ] **RAY-033** Archive/kill stale branches; tidy the accidental backend remnants *(Improvement)*
- [ ] **RAY-034** Remove the always-on default for `--latent-stats` *(Informational)*
- [ ] **RAY-038** Milestone retention policy + `--milestone-interval 0` opt-out *(Potential Risk)*
- [ ] **RAY-039** Fix the duplicate final save/upload in `train_vae_v2.py` *(Confirmed)*

---

## 12. Final assessment

### 12.1 Preserve (do not rewrite)

* **The methodology.** `runs/N5.1/REPORT.md` is the best artefact in this repository: it measures, refuses to fabricate, separates fact from inference, and hands the reader an exact reproduction command. The probe-suite architecture (`probe_*` + `*_smoke` + CI) is the right shape for this kind of research; it needs extending to N8, not replacing.
* **`flow.py`, `whiten.py`, `dit_v3.py`'s internals, and the cache validator in `train_dit_v3.py`.** These are correct, small, and auditable. RAY-001 is a *wiring* bug around `whiten.py`; the class itself is fine.
* **The 6 CI smoke tests and the PR-based revert discipline.** Green in under three minutes and genuinely load-bearing for the toy pipeline.
* **The deterministic toy pipeline** (`toy.py` ↔ `make_toy_dataset.py` ↔ evaluator). Determinism is why the N5.1 defect could be *proven* rather than argued — a rare and valuable property.

### 12.2 Fix (in this order)

1. **RAY-001** (and Stage 0.1 first — it is a one-line question that decides the direction of the fix).
2. **RAY-002, RAY-004** — they make the current-generation pipeline unusable or silently wrong; both are small, localised changes.
3. **RAY-006, RAY-007** — quiet, cheap, and they protect against two failure modes that destroy work rather than annoy.
4. **RAY-010 + RAY-019** — until the measuring stick is fixed, every "improvement" claim in the project (including architecture experiments) is unfalsifiable.
5. **RAY-003, RAY-005** — contracts and reproducibility of the *data* layer.

### 12.3 Investigate (needs data or hardware I do not have)

* The **true latent-space convention** of the published DiT v2/v3 checkpoints (Ray-001 direction).
* The **real text sequence length** in the published cache, and whether `caption_florence-2-large` is uniformly >47 tokens (RAY-003).
* **bf16-on-T4 behaviour** and actual VRAM headroom for the 200 M-parameter DiT v2 (RAY-012) — the log claims bf16; the hardware does not natively support it.
* **Whether `--vae-checkpoint` generation was ever run against a *v2* checkpoint** (RAY-002) — if not, the bug has been latent since 2026-09-23.
* **RoPE vs 2-D positions** for the 16×16 patch grid (RAY-029) — an experiment, after measurement is repaired.

### 12.4 Improve (backlog, no urgency)

Dependency pinning + Dependabot, CI `permissions`/caching, de-duplication, dead-code retirement, branch cleanup, notebook output stripping, a README that says "N8 in progress" honestly, and a LICENSE file so the Apache-2.0 claim in the README becomes true.

### 12.5 One-paragraph verdict

RAY-IMAGE is an unusually self-critical research repository with a strong diagnostic culture and a working, CI-tested toy pipeline, wrapped around a current-generation (N6–N8) real-world pipeline that is running without a safety net. The evidence points to a small number of high-consequence wiring defects — most importantly an inference preprocessing step that training did not perform (RAY-001), an inference entry point that cannot load the family of model being trained (RAY-002), a resolution assumption that is never asserted (RAY-004), a hardcoded text-length contract (RAY-003), and a data-preparation script that cannot reproduce its own first half (RAY-005) — layered over a measurement tool that the project has already proven to be miscalibrated but never repaired (RAY-010). None of these require throwing anything away: each is a small, targeted change with a clear test, and each is far cheaper to fix now than to debug from a training log later. The repository's own N5.1 report shows the author already has the discipline to find and document exactly this class of problem — what is missing is applying it to the newest code path.

---

### Appendix A — Evidence ledger (commands whose output backs specific findings)

| Finding | Evidence artefact |
| :--- | :--- |
| RAY-001 | `grep -n "whiten\|LatentNormalizer\|normalize\|denormalize" ray_image/train_dit_v2.py train_dit_v3.py precompute_qwen_cache.py` → **no matches**; `gh api …/commits/3e24cf54` patch (quoted in §4.2) |
| RAY-002 | `ray_image/generate.py:36-87` read in full; `ray_image/text_encoder.py:92` (`encode_texts` exists only on `RAYTextEncoder`) |
| RAY-003 | `grep -n "47" ray_image/*.py` → `train_dit_v3.py:186,187`; `precompute_qwen_cache.py:31,108-116`; `text_encoder.py:84-90` (`padding=True`) |
| RAY-004 | `grep -n "image_size\|vae.encode\|z = mean" ray_image/train_n5.py` → `79, 118, 160-161`; `vae_v2.py:1-10` contract |
| RAY-005 | `grep -n "start_chunk\|end_chunk\|dataset.skip\|TOTAL_SAMPLES\|create_repo" ray_image/precompute_qwen_cache.py` → `214,215,226,227,228,232,280` |
| RAY-006 | `grep -n "create_repo" ray_image/*.py` → `train_dit_v2.py:260`, `train_dit_v3.py:320,336,407(private=True)`, `train_vae_v2.py:87`, `precompute_qwen_cache.py:214(private=True)` |
| RAY-007 | `grep -n "CHECKPOINT_EVERY\|unlink\|disk" ray_image/train_dit_v2.py` → `505-523` only; v3 has `48,50,281,384-388,400,429` |
| RAY-010 | `runs/N5.1/n5_1_report.json` → `bbox_fill_decision_bands_corners_zero = [{triangle,0.2,0.626},{circle,0.626,0.954},{square,0.954,1.001}]`; `runs/N5.1/REPORT.md:48-97` |
| RAY-011 | `grep -n "cross_gate" config.py utils.py generate.py train_dit_v3.py` → `config.py:16`, `utils.py:53`, `generate.py:64`, `train_dit_v3.py:306` |
| RAY-023 | Pure-Python constructor arithmetic (script output quoted in §5.8) |
| RAY-033 | `gh run list` shows a `Backend CI` run on 2026-10-09; `gh api …/git/trees/3d57f171?recursive=1` shows `backend/**` + `docs/openapi.yaml` (70 KB) on the food-backend branch; `main`'s tree has no `backend/` |
| CI health | `gh run view 38052442985` → all 13 steps `success`, 12:34:16Z → 12:36:57Z |
| Disclosed incident | `find . -name __pycache__ -exec ls -la --time-style=full-iso` (timestamps 08:14:29 vs checkout 08:12:38); `rm -rf`; `git status --porcelain --ignored` → empty; `git fsck` → clean |

### Appendix B — Repository facts for future reference

* Remote: `origin = https://github.com/Rishidev-20thcenturey/-rustes-generative-ai.git`; former name `anime-ai-companion` still redirects (verified via API).
* Branches at audit time: `main` (`7b73c865`), `arena/0d90ac7c-rustes-generative-ai` (`3d57f171`, 3 commits behind, contains the reverted backend), `arena/01a07cdc-anime-ai-companion` (`2dd79261`, **82 commits behind**, target of all Colab notebooks), `revert/food-backend-from-ray-image` (`5203b4b4`, 1 behind).
* PRs: #1 merged (anime-ai-companion → main), #3 merged (accidental food-delivery backend), #4 merged (revert), #5 closed unmerged (a second revert attempt). Issues enabled; 0 releases; 2 workflows active (`RAY-IMAGE tests`, GitHub's built-in Dependency Graph).
* Languages: Python ≈ 230 KB, Jupyter ≈ 382 KB. `diskUsage` reported by GitHub: 506 KB.
* Commit-message noise to be aware of when walking history: `temp`, `tmp`, `tmp2`, `tmp final`, `ignore`, plus several duplicate-message commits from 2026-09-11…15.

---

*End of report. This document is the sole artefact of the audit and was written outside the repository. No fix, refactor, cleanup, commit, branch, PR, workflow or repository setting was performed. All recommendations are deferred to the owner.*

*Superseded in part by the two addenda at the top of this file: the report now lives inside the repository, five supplementary findings (§4.3) have been added, and the owner's separate RB-01…RB-17 remediation work is implemented on the session branch (PR #6). The audit window itself (08:12–08:18 UTC, 2026-10-11) remains exactly as described above: read-only, nothing modified.*
