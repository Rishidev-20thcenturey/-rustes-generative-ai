"""Build a resumable MONET VAE + Qwen cache in 10k chunks.

Each chunk covers a fixed 10,000-sample source range and is uploaded immediately
as a Hugging Face Dataset file. Existing HF chunks are detected before work so a
rerun resumes from the first missing chunk without recomputing completed ranges.
"""

import argparse
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, hf_hub_download
from PIL import Image

from .text_encoder import RAYTextEncoder
from .vae_v2 import RAYVAE_v2

MONET_REPO = "jasperai/monet"
VAE_REPO = "Rishix500/ray-image-vae-n8"
VAE_CHECKPOINT = "ray_vae_v2_step_0047500.pt"
HF_CACHE_REPO = "Rishix500/monet-qwen-cache-50k"
DEFAULT_OUTPUT_DIR = "/kaggle/tmp"
IMAGE_SIZE = 256
LATENT_CHANNELS = 16
LATENT_SIZE = 32
TEXT_DIM = 2560
CACHE_TOKENS = 47
TOTAL_SAMPLES = 100_000
CHUNK_SIZE = 10_000
LOG_EVERY = 1_000


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chunk_filename(start: int, end: int) -> str:
    return f"monet_qwen_cache_{start // 1000}k_{end // 1000}k.pt"


def preprocess_thumbnail(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_vae(device: torch.device) -> RAYVAE_v2:
    checkpoint_path = hf_hub_download(
        repo_id=VAE_REPO,
        filename=VAE_CHECKPOINT,
        repo_type="model",
        token=os.getenv("HF_TOKEN"),
    )
    checkpoint_path = Path(checkpoint_path)
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vae = RAYVAE_v2(latent_channels=LATENT_CHANNELS, base_channels=64)
    vae.load_state_dict(checkpoint_data.get("vae", checkpoint_data), strict=True)
    vae.to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    print(f"[VAE] loaded {checkpoint_path.name}")
    return vae


def hf_repo_files(api: HfApi, repo_id: str) -> set[str]:
    try:
        return set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
    except Exception as exc:
        print(f"[HF] could not list {repo_id}: {exc}")
        return set()


def upload_chunk(path: Path, api: HfApi, repo_id: str) -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is required to upload cache chunks. "
            "Use --no-hf-upload only for a local build."
        )
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"feat(N8): upload cache chunk {path.name}",
    )
    print(f"[HF] uploaded {path.name} -> dataset/{repo_id}")


def build_chunk(ds_iter, start: int, end: int, batch_size: int, vae: RAYVAE_v2,
                text_encoder: RAYTextEncoder, device: torch.device,
                output_dir: Path) -> Path:
    latents = []
    text_embs = []
    masks = []
    source_seen = 0
    cached = 0
    # RB-17: skips are counted per reason so a bad chunk is diagnosable.
    skip_reasons = Counter()

    while source_seen < (end - start):
        images = []
        captions = []
        while len(images) < batch_size and source_seen < (end - start):
            sample = next(ds_iter)
            source_seen += 1
            image = sample.get("thumbnail")
            caption = sample.get("caption_florence-2-large")
            if image is None:
                skip_reasons["no_thumbnail"] += 1
                continue
            if not isinstance(caption, str) or not caption.strip():
                skip_reasons["empty_caption"] += 1
                continue
            try:
                images.append(preprocess_thumbnail(image))
                captions.append(caption.strip())
            except (TypeError, ValueError):
                skip_reasons["preprocess_error"] += 1

        if not images:
            continue

        image_batch = torch.stack(images).to(device, non_blocking=True)
        with torch.inference_mode():
            try:
                _, z, _, _ = vae(image_batch)
                # RB-09: pad to max_length so every cached row is exactly
                # (CACHE_TOKENS, TEXT_DIM) regardless of batch composition.
                text_hidden, mask = text_encoder.encode_texts(
                    captions, max_length=CACHE_TOKENS, pad_to_max=True
                )
            except Exception as exc:
                skip_reasons["encoder_failure"] += len(images)
                print(f"[skip] encoder batch failure: {exc}")
                continue

        z = z.detach().float().cpu().to(torch.float16)
        text_hidden = text_hidden.detach().cpu().to(torch.float16)
        mask = mask.detach().cpu().bool()

        for i in range(z.shape[0]):
            if not (
                torch.isfinite(z[i]).all().item()
                and torch.isfinite(text_hidden[i]).all().item()
            ):
                skip_reasons["non_finite"] += 1
                continue
            if not (
                tuple(z[i].shape) == (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE)
                and tuple(text_hidden[i].shape) == (CACHE_TOKENS, TEXT_DIM)
                and tuple(mask[i].shape) == (CACHE_TOKENS,)
            ):
                skip_reasons["shape_mismatch"] += 1
                continue
            latents.append(z[i])
            text_embs.append(text_hidden[i])
            masks.append(mask[i])
            cached += 1

        if source_seen % LOG_EVERY == 0 or source_seen >= (end - start):
            absolute = start + source_seen
            print(f"[progress] chunk={start//1000}k-{end//1000}k source={absolute}/{end} cached={cached} skipped={sum(skip_reasons.values())} {dict(skip_reasons)}")

    if not latents:
        raise RuntimeError(f"No valid samples produced for chunk {start}:{end}")

    filename = chunk_filename(start, end)
    output = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "ray-image-n8-qwen-cache-v1",
            "source": MONET_REPO,
            "source_start": start,
            "source_end": end,
            "samples_seen": source_seen,
            "samples_cached": cached,
            "samples_skipped": sum(skip_reasons.values()),
            "skip_reasons": dict(skip_reasons),
            "latent_dtype": "float16",
            "text_dtype": "float16",
            "latents": torch.stack(latents),
            "text_emb": torch.stack(text_embs),
            "mask": torch.stack(masks),
        },
        output,
    )
    print(
        f"[saved] {output} ({output.stat().st_size / (1024 ** 3):.2f} GB) cached={cached} "
        f"skipped={sum(skip_reasons.values())} skip_reasons={dict(skip_reasons)}"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the 100k MONET Qwen cache as resumable 10k chunks.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-repo", type=str, default=HF_CACHE_REPO)
    parser.add_argument("--no-hf-upload", action="store_true")
    # RB-06: the chunk window is a CLI argument. The stream skip is derived from
    # --start-chunk, so any window of the dataset can be built (including 0-5).
    parser.add_argument("--start-chunk", type=int, default=5,
                        help="first 10k chunk index to build (chunk N covers samples N*10000..N*10000+9999)")
    parser.add_argument("--end-chunk", type=int, default=10,
                        help="exclusive end chunk index to build")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.start_chunk < 0 or args.end_chunk <= args.start_chunk:
        raise ValueError("--start-chunk must be >= 0 and --end-chunk must be > --start-chunk")
    if args.end_chunk * CHUNK_SIZE > TOTAL_SAMPLES:
        raise ValueError(
            f"--end-chunk={args.end_chunk} exceeds the {TOTAL_SAMPLES:,}-sample cache budget "
            f"({TOTAL_SAMPLES // CHUNK_SIZE} chunks)"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen/VAE cache precompute requires CUDA")

    seed_everything(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_token = os.getenv("HF_TOKEN")
    if not args.no_hf_upload:
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN is required for resumable cache uploads. "
                "Use --no-hf-upload only for a local build."
            )
        # RB-07: honour --hf-repo instead of a hardcoded repository name.
        # private=True applies at creation only; existing repo visibility is
        # never modified.
        create_repo(
            repo_id=args.hf_repo,
            repo_type="dataset",
            exist_ok=True,
            private=True,
            token=hf_token,
        )

    api = HfApi(token=hf_token)
    existing_hf = set() if args.no_hf_upload else hf_repo_files(api, args.hf_repo)
    dataset = load_dataset(MONET_REPO, name="parquet", split="train", streaming=True)

    start_chunk = args.start_chunk
    end_chunk = args.end_chunk
    skip = start_chunk * CHUNK_SIZE
    print(f"[window] chunks {start_chunk}..{end_chunk - 1} (samples {skip:,}..{end_chunk * CHUNK_SIZE:,})")
    dataset = dataset.skip(skip)
    ds_iter = iter(dataset)

    completed = 0
    for chunk_index in range(start_chunk, end_chunk):
        start = chunk_index * CHUNK_SIZE
        end = start + CHUNK_SIZE
        filename = chunk_filename(start, end)
        local_path = output_dir / filename

        if filename in existing_hf:
            print(f"[resume] HF already has {filename}; skipping chunk {start}:{end}")
            for _ in range(CHUNK_SIZE):
                next(ds_iter)
            completed += 1
            continue

        if local_path.is_file():
            print(f"[resume] local chunk exists: {local_path}")
            for _ in range(CHUNK_SIZE):
                next(ds_iter)
            if not args.no_hf_upload:
                upload_chunk(local_path, api, args.hf_repo)
                existing_hf.add(filename)
            completed += 1
            continue

        print(f"[build] chunk {chunk_index + 1}: source {start}:{end}")
        vae = load_vae(device)
        text_encoder = RAYTextEncoder(
            model_name=RAYTextEncoder.MODEL_ID,
            max_tokens=CACHE_TOKENS,
            load_in_8bit=True,
        ).eval()

        try:
            built_path = build_chunk(
                ds_iter, start, end, args.batch_size,
                vae, text_encoder, device, output_dir,
            )
            if not args.no_hf_upload:
                upload_chunk(built_path, api, args.hf_repo)
                existing_hf.add(filename)
            completed += 1
        except Exception:
            print(f"[error] chunk {start}:{end} failed; resume will retry this chunk")
            raise
        finally:
            del text_encoder
            del vae
            torch.cuda.empty_cache()

    print(f"[done] completed {completed}/{end_chunk - start_chunk} cache chunks")


if __name__ == "__main__":
    main()
