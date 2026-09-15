"""N8 DiT v3 flow-matching training on streamed MONET or a precomputed cache.

With --cache-path, frozen VAE/Qwen inference is bypassed and cached latent/text
embeddings are sampled directly from the local MONET 50k cache.
"""

import argparse
import io
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from .dit_v3 import RAYDiT_v3
from .text_encoder import RAYTextEncoder
from .vae_v2 import RAYVAE_v2

MONET_REPO = "jasperai/monet"
VAE_REPO = "Rishix500/ray-image-vae-n8"
VAE_CHECKPOINT = "ray_vae_v2_step_0047500.pt"
HF_DEFAULT_REPO = "Rishix500/ray-image-dit-n8-v3"
IMAGE_SIZE = 256
LATENT_CHANNELS = 16
LATENT_SIZE = 32
MODEL_DIM = 512
MODEL_DEPTH = 12
MODEL_HEADS = 8
MODEL_PATCH = 2
TEXT_DIM = 2560
TEXT_MAX_TOKENS = 256
CHECKPOINT_EVERY = 500
LOG_EVERY = 100
WARMUP_STEPS = 500
HF_CHECKPOINT_WINDOW = 4
LOCAL_CHECKPOINT_WINDOW = 2
LOCAL_CHECKPOINT_DIR = "/kaggle/tmp/checkpoints"
DISK_REFUSE_SAVE_PCT = 90.0
TRAIN_LR = 5e-5
GRAD_CLIP = 1.0
CACHE_CHUNK_RE = re.compile(r"^monet_qwen_cache_(\d+)k_(\d+)k\.pt$")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_thumbnail(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _pick_caption(sample) -> str:
    for key in (
        "caption_gemini-2.5-flash-lite",
        "caption_internvl-3-8b",
        "caption_sharegpt4v-7b",
        "caption_florence-2-large",
        "caption_original",
    ):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def next_batch(iterator, batch_size: int):
    images, captions = [], []
    while len(images) < batch_size:
        sample = next(iterator)
        image = sample.get("thumbnail")
        if image is None:
            continue
        try:
            images.append(preprocess_thumbnail(image))
            captions.append(_pick_caption(sample))
        except (TypeError, ValueError):
            continue
    return torch.stack(images), captions


def _load_cache_payload(path: Path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        payload = torch.load(path, map_location="cpu", weights_only=False)

    required = ("latents", "text_emb", "mask")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"cache missing keys in {path}: {missing}")
    latents = payload["latents"]
    text_emb = payload["text_emb"]
    mask = payload["mask"]
    if latents.ndim != 4 or tuple(latents.shape[1:]) != (LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE):
        raise ValueError(f"invalid cache latent shape in {path}: {tuple(latents.shape)}")
    if text_emb.ndim != 3 or text_emb.shape[0] != latents.shape[0] or text_emb.shape[2] != TEXT_DIM:
        raise ValueError(f"invalid cache text_emb shape in {path}: {tuple(text_emb.shape)}")
    if mask.shape != text_emb.shape[:2]:
        raise ValueError(f"invalid cache mask shape in {path}: {tuple(mask.shape)}")
    if latents.shape[0] == 0:
        raise ValueError(f"cache is empty: {path}")
    return payload


def _chunk_sort_key(path: Path):
    match = CACHE_CHUNK_RE.match(path.name)
    if not match:
        raise ValueError(f"invalid cache chunk filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def load_cache(path: Path):
    """Load one cache file or a directory of ordered MONET cache chunks."""
    path = Path(path)
    if path.is_file():
        payload = _load_cache_payload(path)
        latents = payload["latents"]
        print(f"[cache] loaded {path} samples={latents.shape[0]:,} latent={tuple(latents.shape[1:])} text={tuple(payload['text_emb'].shape[1:])}")
        return payload
    if not path.is_dir():
        raise FileNotFoundError(f"cache not found: {path}")

    chunk_paths = sorted(path.glob("monet_qwen_cache_*.pt"), key=_chunk_sort_key)
    if not chunk_paths:
        raise FileNotFoundError(f"no cache chunks found in directory: {path}")

    chunks = []
    previous_end = None
    total_samples = 0
    for chunk_path in chunk_paths:
        start_name, end_name = _chunk_sort_key(chunk_path)
        payload = _load_cache_payload(chunk_path)
        source_start = int(payload.get("source_start", start_name * 1000))
        source_end = int(payload.get("source_end", end_name * 1000))
        if source_start != start_name * 1000 or source_end != end_name * 1000:
            raise ValueError(f"cache chunk metadata mismatch: {chunk_path.name} filename={start_name * 1000}:{end_name * 1000} payload={source_start}:{source_end}")
        if source_end <= source_start:
            raise ValueError(f"invalid source range in {chunk_path}: {source_start}:{source_end}")
        if previous_end is not None and source_start < previous_end:
            raise ValueError(f"duplicate/overlapping cache source range: previous_end={previous_end}, current={source_start}:{source_end}")
        if previous_end is not None and source_start != previous_end:
            raise ValueError(f"cache chunks are not contiguous: previous_end={previous_end}, current_start={source_start}")
        total_samples += int(payload["latents"].shape[0])
        previous_end = source_end
        chunks.append(payload)

    offsets = [0]
    for payload in chunks:
        offsets.append(offsets[-1] + int(payload["latents"].shape[0]))
    cache = {"chunks": chunks, "offsets": torch.tensor(offsets, dtype=torch.long), "count": total_samples}
    print(f"[cache] loaded directory {path} chunks={len(chunks)} total_samples={total_samples:,}")
    print(f"[cache] source ranges verified: {chunks[0].get('source_start', 0)}:{chunks[-1].get('source_end', total_samples)}; no duplicate/overlapping ranges")
    return cache


def next_cached_batch(cache, batch_size: int, device: torch.device):
    if "chunks" not in cache:
        count = cache["latents"].shape[0]
        indices = torch.randint(0, count, (batch_size,))
        clean_latents = cache["latents"][indices].to(device=device, dtype=torch.float32, non_blocking=True)
        text_hidden = cache["text_emb"][indices].to(device=device, dtype=torch.float16, non_blocking=True)
        text_mask = cache["mask"][indices].to(device=device, dtype=torch.bool, non_blocking=True)
        return clean_latents, text_hidden, text_mask

    global_indices = torch.randint(0, int(cache["count"]), (batch_size,))
    offsets = cache["offsets"]
    chunk_ids = torch.bucketize(global_indices, offsets[1:-1], right=True)
    clean_latents = torch.empty((batch_size, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE), dtype=torch.float32)
    text_hidden = torch.empty((batch_size, 47, TEXT_DIM), dtype=torch.float16)
    text_mask = torch.empty((batch_size, 47), dtype=torch.bool)
    for chunk_id in torch.unique(chunk_ids, sorted=True).tolist():
        positions = torch.nonzero(chunk_ids == chunk_id, as_tuple=False).squeeze(1)
        local_indices = global_indices[positions] - offsets[chunk_id]
        chunk = cache["chunks"][chunk_id]
        clean_latents[positions] = chunk["latents"][local_indices].float()
        text_hidden[positions] = chunk["text_emb"][local_indices].half()
        text_mask[positions] = chunk["mask"][local_indices].bool()
    return clean_latents.to(device=device, non_blocking=True), text_hidden.to(device=device, non_blocking=True), text_mask.to(device=device, non_blocking=True)


def download_vae_checkpoint() -> Path:
    token = os.getenv("HF_TOKEN")
    try:
        path = hf_hub_download(repo_id=VAE_REPO, filename=VAE_CHECKPOINT, repo_type="model", token=token)
    except Exception as exc:
        raise RuntimeError(f"Could not download {VAE_CHECKPOINT} from {VAE_REPO}") from exc
    return Path(path)


def load_vae(device: torch.device) -> RAYVAE_v2:
    checkpoint_path = download_vae_checkpoint()
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    vae = RAYVAE_v2(latent_channels=LATENT_CHANNELS, base_channels=64)
    vae.load_state_dict(checkpoint_data.get("vae", checkpoint_data), strict=True)
    vae.to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    print(f"[VAE] loaded {checkpoint_path.name}")
    return vae


def build_dit() -> RAYDiT_v3:
    return RAYDiT_v3(latent_channels=LATENT_CHANNELS, latent_size=LATENT_SIZE, dim=MODEL_DIM, depth=MODEL_DEPTH, heads=MODEL_HEADS, patch=MODEL_PATCH, cond_dim=TEXT_DIM, cross_gate=True)


def checkpointed_forward(model, z_t, t, text, text_mask):
    b, c, h, w = z_t.shape
    if z_t.ndim != 4 or (c, h, w) != (model.latent_channels, model.latent_size, model.latent_size):
        raise ValueError(f"unexpected latent shape: {tuple(z_t.shape)}")
    x = model.in_proj(z_t)
    ph, pw = x.shape[-2:]
    x = x.flatten(2).transpose(1, 2)
    text = text.to(dtype=x.dtype)
    if text_mask is None:
        pooled = text.mean(dim=1)
    else:
        valid = (~text_mask).to(dtype=text.dtype).unsqueeze(-1)
        pooled = (text * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    cond = model.time(t).to(dtype=x.dtype) + model.text_proj(pooled)
    text_ctx = model.text_ctx_proj(text)

    def run_block(block_, x_, cond_, text_):
        return block_(x_, cond_, text_, text_mask=text_mask)

    for block in model.blocks:
        x = checkpoint(run_block, block, x, cond, text_ctx, use_reentrant=False)
    x = model.out(model.norm(x))
    x = x.view(b, ph, pw, model.patch, model.patch, model.latent_channels)
    return x.permute(0, 5, 1, 3, 2, 4).reshape(b, model.latent_channels, h, w)


def _checkpoint_sort_key(filename: str):
    try:
        return int(Path(filename).stem.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return -1


def cleanup_hf_checkpoints(api: HfApi, repo_id: str) -> None:
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        checkpoints = sorted({p for p in files if Path(p).name.startswith("ray_dit_v3_step_") and Path(p).suffix == ".pt"}, key=_checkpoint_sort_key)
        for path in checkpoints[:-HF_CHECKPOINT_WINDOW]:
            try:
                api.delete_file(path_in_repo=path, repo_id=repo_id, repo_type="model", commit_message=f"chore: remove old DiT v3 checkpoint {Path(path).name}")
                print(f"[HF cleanup] deleted {path}")
            except Exception as exc:
                print(f"[HF cleanup] failed to delete {path}: {exc}")
    except Exception as exc:
        print(f"[HF cleanup] listing failed for {repo_id}: {exc}")


def log_disk_usage(path: Path, prefix: str = "[disk]") -> float:
    usage = shutil.disk_usage(path)
    percent = (usage.used / usage.total) * 100.0 if usage.total else 100.0
    print(f"{prefix} used={usage.used / (1024 ** 3):.2f}GB free={usage.free / (1024 ** 3):.2f}GB total={usage.total / (1024 ** 3):.2f}GB ({percent:.1f}%)")
    return percent


def _local_checkpoint_paths(output_dir: Path):
    return sorted((p for p in output_dir.glob("ray_dit_v3_step_*.pt") if p.is_file()), key=lambda p: _checkpoint_sort_key(p.name))


def cleanup_local_checkpoints(output_dir: Path, keep: int = LOCAL_CHECKPOINT_WINDOW) -> None:
    checkpoints = _local_checkpoint_paths(output_dir)
    for path in checkpoints[:-keep] if keep > 0 else checkpoints:
        try:
            path.unlink()
            print(f"[local cleanup] deleted {path}")
        except OSError as exc:
            print(f"[local cleanup] failed to delete {path}: {exc}")


def _checkpoint_payload(model, optimizer, step, seed):
    return {
        "stage": "dit_v3_n8",
        "step": step,
        "seed": seed,
        "config": {
            "image_size": IMAGE_SIZE,
            "latent_channels": LATENT_CHANNELS,
            "latent_size": LATENT_SIZE,
            "dim": MODEL_DIM,
            "depth": MODEL_DEPTH,
            "heads": MODEL_HEADS,
            "patch": MODEL_PATCH,
            "text_dim": TEXT_DIM,
            "text_max_tokens": TEXT_MAX_TOKENS,
            "cross_gate": True,
            "flow_matching": "linear_gaussian",
        },
        "dit": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }


def upload_checkpoint(local_path: Path, repo_id: str) -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        print("[HF] HF_TOKEN not set; keeping checkpoint local only")
        return
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=str(local_path), path_in_repo=local_path.name, repo_id=repo_id, repo_type="model")
    print(f"[HF] uploaded {local_path.name} -> {repo_id}")
    cleanup_hf_checkpoints(api, repo_id)


def upload_checkpoint_direct(model, optimizer, step: int, seed: int, repo_id: str) -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for direct checkpoint upload when disk is near full")
    payload = _checkpoint_payload(model, optimizer, step, seed)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    buffer.seek(0)
    filename = f"ray_dit_v3_step_{step:07d}.pt"
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=buffer, path_in_repo=filename, repo_id=repo_id, repo_type="model", commit_message=f"upload DiT v3 checkpoint {filename}")
    print(f"[HF] uploaded {filename} directly from memory (local save skipped)")
    cleanup_hf_checkpoints(api, repo_id)
    buffer.close()


def checkpoint_has_corruption(model: RAYDiT_v3) -> bool:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all().item():
                print(f"[checkpoint guard] non-finite weight in {name}; refusing to save")
                return True
    return False


def save_checkpoint(path, model, optimizer, step, seed) -> bool:
    if checkpoint_has_corruption(model):
        print(f"[checkpoint guard] SKIP step={step}: corrupted model weights")
        return False
    torch.save(_checkpoint_payload(model, optimizer, step, seed), path)
    return True


def load_resume(path, model, optimizer, device) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    checkpoint_data = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data["dit"], strict=True)
    if "optimizer" in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data["optimizer"])
    step = int(checkpoint_data.get("step", 0))
    print(f"[resume] loaded {path} at step {step}")
    return step


def set_learning_rate(optimizer, base_lr, step) -> float:
    lr = base_lr * step / WARMUP_STEPS if step <= WARMUP_STEPS else base_lr
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def save_or_upload_checkpoint(output_dir: Path, dit, optimizer, step: int, seed: int, hf_repo: str, no_hf_upload: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    local_disk_pct = log_disk_usage(output_dir, prefix=f"[disk] checkpoint step={step}")
    checkpoint_path = output_dir / f"ray_dit_v3_step_{step:07d}.pt"

    if local_disk_pct >= DISK_REFUSE_SAVE_PCT:
        print(f"[checkpoint] local disk usage {local_disk_pct:.1f}% >= {DISK_REFUSE_SAVE_PCT:.1f}%; refusing local save")
        if no_hf_upload:
            print("[checkpoint] HF upload disabled; checkpoint skipped")
            return
        if checkpoint_has_corruption(dit):
            print(f"[checkpoint guard] SKIP step={step}: corrupted model weights")
            return
        upload_checkpoint_direct(dit, optimizer, step, seed, hf_repo)
        return

    try:
        if save_checkpoint(checkpoint_path, dit, optimizer, step, seed):
            print(f"[checkpoint] {checkpoint_path}")
            if not no_hf_upload:
                upload_checkpoint(checkpoint_path, hf_repo)
            cleanup_local_checkpoints(output_dir, keep=LOCAL_CHECKPOINT_WINDOW)
            log_disk_usage(output_dir, prefix=f"[disk] after checkpoint step={step}")
    except (RuntimeError, OSError) as exc:
        print(f"[checkpoint] local save failed at step={step}: {exc}")
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
            except OSError:
                pass
        if no_hf_upload:
            raise
        if checkpoint_has_corruption(dit):
            print(f"[checkpoint guard] SKIP step={step}: corrupted model weights")
            return
        upload_checkpoint_direct(dit, optimizer, step, seed, hf_repo)
        cleanup_local_checkpoints(output_dir, keep=LOCAL_CHECKPOINT_WINDOW)


def main():
    parser = argparse.ArgumentParser(description="Train RAY-IMAGE DiT v3 with flow matching.")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=TRAIN_LR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--cache-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=LOCAL_CHECKPOINT_DIR)
    parser.add_argument("--hf-repo", type=str, default=HF_DEFAULT_REPO)
    parser.add_argument("--no-hf-upload", action="store_true")
    args = parser.parse_args()
    if args.steps < 0 or args.batch_size <= 0 or args.lr <= 0 or args.start_step < 0:
        raise ValueError("invalid training arguments")
    if args.resume and args.start_step:
        raise ValueError("Use either --resume or --start-step, not both")
    if not torch.cuda.is_available():
        raise RuntimeError("N8 DiT v3 training requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = None
    dataset = None
    iterator = None
    vae = None
    text_encoder = None
    if args.cache_path:
        cache = load_cache(Path(args.cache_path))
        print("[data] using precomputed cache; live VAE/Qwen inference disabled")
    else:
        dataset = load_dataset(MONET_REPO, name="parquet", split="train", streaming=True)
        iterator = iter(dataset)
        vae = load_vae(device)
        text_encoder = RAYTextEncoder(model_name=RAYTextEncoder.MODEL_ID, max_tokens=TEXT_MAX_TOKENS, load_in_8bit=True).eval()
        print("[data] using live streamed MONET + frozen VAE + Qwen")
    dit = build_dit().to(device)
    optimizer = AdamW(dit.parameters(), lr=0.0, betas=(0.9, 0.99), weight_decay=0.01)
    last_step = args.start_step
    if args.resume:
        last_step = load_resume(Path(args.resume), dit, optimizer, device)
    if args.steps <= last_step:
        print(f"[done] target steps={args.steps} already reached at step={last_step}")
        return
    print(f"[DiT v3] trainable parameters: {sum(p.numel() for p in dit.parameters()):,}")
    print(f"[config] batch={args.batch_size} lr={args.lr:.2e} warmup={WARMUP_STEPS} target_steps={args.steps}")
    print("[precision] DiT forward fp16; cached Qwen embeddings are fp16")
    print("[stability] grad_clip=1.0; NaN guards=enabled; finite-checkpoints=enabled")
    print(f"[disk] checkpoint_dir={output_dir} local_window={LOCAL_CHECKPOINT_WINDOW} refuse_save_at={DISK_REFUSE_SAVE_PCT:.1f}%")
    nan_skip_count = 0
    running_loss = 0.0
    running_count = 0
    pbar = tqdm(total=args.steps - last_step, desc="RAY-DiT v3 N8")
    for step in range(last_step + 1, args.steps + 1):
        current_lr = set_learning_rate(optimizer, args.lr, step)
        if cache is not None:
            clean_latents, text_hidden, text_mask = next_cached_batch(cache, args.batch_size, device)
        else:
            try:
                images, captions = next_batch(iterator, args.batch_size)
            except StopIteration:
                iterator = iter(dataset)
                images, captions = next_batch(iterator, args.batch_size)
            images = images.to(device, non_blocking=True)
            with torch.inference_mode():
                _, clean_latents, _, _ = vae(images)
                text_hidden, text_mask = text_encoder.encode_texts(captions, max_length=TEXT_MAX_TOKENS)
            clean_latents = clean_latents.to(device=device, dtype=torch.float32)
            text_hidden = text_hidden.to(device=device, dtype=torch.float16)
            text_mask = text_mask.to(device=device, dtype=torch.bool)
        noise = torch.randn_like(clean_latents)
        t = torch.rand(clean_latents.shape[0], device=device)
        t_view = t[:, None, None, None]
        z_t = (1.0 - t_view) * clean_latents + t_view * noise
        target_velocity = noise - clean_latents
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            prediction = checkpointed_forward(dit, z_t, t, text_hidden, text_mask)
            loss = F.mse_loss(prediction.float(), target_velocity.float())
        if torch.isnan(loss).item() or torch.isinf(loss).item():
            nan_skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            pbar.update(1)
            pbar.set_postfix(loss="nan", lr=f"{current_lr:.1e}")
            if step % LOG_EVERY == 0:
                vram_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                print(f"step={step} loss=nan grad_norm=nan NaN_skip_count={nan_skip_count} vram_used={vram_gb:.2f}GB")
            continue
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), GRAD_CLIP)
        if not torch.isfinite(grad_norm).item():
            nan_skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            pbar.update(1)
            continue
        optimizer.step()
        running_loss += float(loss.item())
        running_count += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.1e}")
        if step % LOG_EVERY == 0:
            mean_loss = running_loss / max(running_count, 1)
            vram_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(f"step={step} loss={loss.item():.6f} loss_avg={mean_loss:.6f} grad_norm={float(grad_norm):.4f} NaN_skip_count={nan_skip_count} vram_used={vram_gb:.2f}GB")
            running_loss = 0.0
            running_count = 0
        if step % CHECKPOINT_EVERY == 0:
            save_or_upload_checkpoint(output_dir, dit, optimizer, step, args.seed, args.hf_repo, args.no_hf_upload)
    pbar.close()
    if args.steps % CHECKPOINT_EVERY != 0:
        save_or_upload_checkpoint(output_dir, dit, optimizer, args.steps, args.seed, args.hf_repo, args.no_hf_upload)


if __name__ == "__main__":
    main()
