"""Precompute the first 50k MONET VAE + Qwen embeddings for N8 DiT v3."""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image

from .text_encoder import RAYTextEncoder
from .vae_v2 import RAYVAE_v2

MONET_REPO = "jasperai/monet"
VAE_REPO = "Rishix500/ray-image-vae-n8"
VAE_CHECKPOINT = "ray_vae_v2_step_0047500.pt"
HF_CACHE_REPO = "Rishix500/monet-qwen-cache-50k"
DEFAULT_OUTPUT = "/kaggle/working/monet_qwen_cache_50k.pt"
IMAGE_SIZE = 256
LATENT_CHANNELS = 16
LATENT_SIZE = 32
TEXT_DIM = 2560
CACHE_TOKENS = 47
DEFAULT_SAMPLES = 50_000
LOG_EVERY = 1_000


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_thumbnail(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def load_vae(device: torch.device) -> RAYVAE_v2:
    from huggingface_hub import hf_hub_download

    checkpoint_path = hf_hub_download(
        repo_id=VAE_REPO,
        filename=VAE_CHECKPOINT,
        repo_type="model",
        token=os.getenv("HF_TOKEN"),
    )
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vae = RAYVAE_v2(latent_channels=LATENT_CHANNELS, base_channels=64)
    vae.load_state_dict(checkpoint_data.get("vae", checkpoint_data), strict=True)
    vae.to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    return vae


def upload_cache(path: Path, repo_id: str) -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        print("[HF] HF_TOKEN not set; keeping cache local")
        return
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=path.name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="feat(N8): upload MONET 50k Qwen embedding cache",
    )
    print(f"[HF] uploaded {path.name} -> dataset/{repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the MONET 50k Qwen cache.")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--hf-repo", type=str, default=HF_CACHE_REPO)
    parser.add_argument("--no-hf-upload", action="store_true")
    args = parser.parse_args()

    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("--samples and --batch-size must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen/VAE cache precompute requires CUDA")

    seed_everything(args.seed)
    device = torch.device("cuda")
    dataset = load_dataset(MONET_REPO, name="parquet", split="train", streaming=True)
    iterator = iter(dataset)
    vae = load_vae(device)
    text_encoder = RAYTextEncoder(
        model_name=RAYTextEncoder.MODEL_ID,
        max_tokens=CACHE_TOKENS,
        load_in_8bit=True,
    ).eval()

    latents = []
    text_embs = []
    masks = []
    seen = 0
    cached = 0
    skipped = 0

    # 50k x ([16,32,32] + [47,2560]) fp16 is about 15 GB before overhead.
    while seen < args.samples:
        images = []
        captions = []
        while len(images) < args.batch_size and seen < args.samples:
            try:
                sample = next(iterator)
            except StopIteration:
                break
            seen += 1
            image = sample.get("thumbnail")
            caption = sample.get("caption_florence-2-large")
            if image is None or not isinstance(caption, str) or not caption.strip():
                skipped += 1
                continue
            try:
                images.append(preprocess_thumbnail(image))
                captions.append(caption.strip())
            except (TypeError, ValueError):
                skipped += 1

        if not images:
            continue

        image_batch = torch.stack(images).to(device, non_blocking=True)
        with torch.inference_mode():
            try:
                _, z, _, _ = vae(image_batch)
                text_hidden, mask = text_encoder.encode_texts(
                    captions,
                    max_length=CACHE_TOKENS,
                )
            except Exception as exc:
                skipped += len(images)
                print(f"[skip] encoder batch failure: {exc}")
                continue

        z = z.detach().float().cpu().to(torch.float16)
        text_hidden = text_hidden.detach().cpu().to(torch.float16)
        mask = mask.detach().cpu().bool()

        for i in range(z.shape[0]):
            if not torch.isfinite(z[i]).all().item() or not torch.isfinite(text_hidden[i]).all().item():
                skipped += 1
                continue
            if tuple(z[i].shape) != (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE):
                skipped += 1
                continue
            if tuple(text_hidden[i].shape) != (CACHE_TOKENS, TEXT_DIM):
                skipped += 1
                continue
            latents.append(z[i])
            text_embs.append(text_hidden[i])
            masks.append(mask[i])
            cached += 1

        if seen % LOG_EVERY == 0 or seen >= args.samples:
            print(f"[progress] source={seen}/{args.samples} cached={cached} skipped={skipped}")

    if not latents:
        raise RuntimeError("No valid cache samples produced")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "ray-image-n8-qwen-cache-v1",
            "source": MONET_REPO,
            "samples_requested": args.samples,
            "samples_seen": seen,
            "samples_cached": cached,
            "samples_skipped": skipped,
            "latents": torch.stack(latents),
            "text_emb": torch.stack(text_embs),
            "mask": torch.stack(masks),
        },
        output,
    )
    print(f"[saved] {output} ({output.stat().st_size / (1024 ** 3):.2f} GB)")
    if not args.no_hf_upload:
        upload_cache(output, args.hf_repo)


if __name__ == "__main__":
    main()
