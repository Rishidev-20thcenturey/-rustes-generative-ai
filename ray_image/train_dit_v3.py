"""N8 DiT v3 flow-matching training on streamed MONET image-text pairs.

DiT v3 is intentionally smaller than v2 for stable, fast T4 training:
  dim=512, depth=12, heads=8, patch=2 (~60M parameters)
  SwiGLU + RMSNorm + RoPE + gated text cross-attention.

Qwen3-4B text states are produced in bf16 by the frozen text encoder, then
cast to fp16 for the DiT forward pass under CUDA autocast.
"""

import argparse
import os
import random
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
TRAIN_LR = 5e-5
GRAD_CLIP = 1.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_thumbnail(image: Image.Image) -> torch.Tensor:
    """Convert MONET thumbnail to [3, 256, 256] float32 in [0, 1]."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"expected PIL image, got {type(image)!r}")
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _pick_caption(sample) -> str:
    """Prefer a detailed MONET caption with deterministic fallbacks."""
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
    """Read a full batch of valid MONET thumbnails and captions."""
    images = []
    captions = []
    while len(images) < batch_size:
        sample = next(iterator)
        image = sample.get("thumbnail")
        if image is None:
            continue
        try:
            image_tensor = preprocess_thumbnail(image)
        except (TypeError, ValueError):
            continue
        images.append(image_tensor)
        captions.append(_pick_caption(sample))
    return torch.stack(images), captions


def download_vae_checkpoint() -> Path:
    """Download the finalized N8 VAE checkpoint from the HF Hub."""
    token = os.getenv("HF_TOKEN")
    try:
        path = hf_hub_download(
            repo_id=VAE_REPO,
            filename=VAE_CHECKPOINT,
            repo_type="model",
            token=token,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {VAE_CHECKPOINT} from {VAE_REPO}. "
            "Set HF_TOKEN with access to the model repo."
        ) from exc
    return Path(path)


def load_vae(device: torch.device) -> RAYVAE_v2:
    checkpoint_path = download_vae_checkpoint()
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint_data.get("vae", checkpoint_data)

    vae = RAYVAE_v2(latent_channels=LATENT_CHANNELS, base_channels=64)
    vae.load_state_dict(state_dict, strict=True)
    vae.to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    print(f"[VAE] loaded {checkpoint_path.name} from {VAE_REPO}")
    return vae


def build_dit() -> RAYDiT_v3:
    """Construct the exact N8 DiT v3 architecture."""
    return RAYDiT_v3(
        latent_channels=LATENT_CHANNELS,
        latent_size=LATENT_SIZE,
        dim=MODEL_DIM,
        depth=MODEL_DEPTH,
        heads=MODEL_HEADS,
        patch=MODEL_PATCH,
        cond_dim=TEXT_DIM,
        cross_gate=True,
    )


def checkpointed_forward(
    model: RAYDiT_v3,
    z_t: torch.Tensor,
    t: torch.Tensor,
    text: torch.Tensor,
    text_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Forward v3 while gradient-checkpointing every transformer block."""
    if z_t.ndim != 4:
        raise ValueError(f"expected z_t [B,C,H,W], got {tuple(z_t.shape)}")

    b, c, h, w = z_t.shape
    if (c, h, w) != (model.latent_channels, model.latent_size, model.latent_size):
        raise ValueError(
            f"expected latent [{model.latent_channels},{model.latent_size},{model.latent_size}], "
            f"got [{c},{h},{w}]"
        )

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

    for block in model.blocks:
        x = checkpoint(
            lambda x_, cond_, text_: block(
                x_, cond_, text_, text_mask=text_mask
            ),
            x,
            cond,
            text_ctx,
            use_reentrant=False,
        )

    x = model.out(model.norm(x))
    x = x.view(b, ph, pw, model.patch, model.patch, model.latent_channels)
    return x.permute(0, 5, 1, 3, 2, 4).reshape(
        b, model.latent_channels, h, w
    )


def _checkpoint_sort_key(filename: str):
    stem = Path(filename).stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (ValueError, IndexError):
        return -1


def cleanup_hf_checkpoints(api: HfApi, repo_id: str) -> None:
    """Keep only the newest four HF v3 checkpoint files."""
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
        checkpoints = sorted(
            {
                path
                for path in files
                if Path(path).name.startswith("ray_dit_v3_step_")
                and Path(path).suffix == ".pt"
            },
            key=_checkpoint_sort_key,
        )
        for path in checkpoints[:-HF_CHECKPOINT_WINDOW]:
            try:
                api.delete_file(
                    path_in_repo=path,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"chore: remove old DiT v3 checkpoint {Path(path).name}",
                )
                print(f"[HF cleanup] deleted {path}")
            except Exception as exc:
                print(f"[HF cleanup] failed to delete {path}: {exc}")
    except Exception as exc:
        print(f"[HF cleanup] listing failed for {repo_id}: {exc}")


def upload_checkpoint(local_path: Path, repo_id: str) -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        print("[HF] HF_TOKEN not set; keeping checkpoint local only")
        return

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=local_path.name,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"[HF] uploaded {local_path.name} -> {repo_id}")
    cleanup_hf_checkpoints(api, repo_id)


def checkpoint_has_corruption(model: RAYDiT_v3) -> bool:
    """Return True when any trainable weight contains NaN or Inf."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all().item():
                print(f"[checkpoint guard] non-finite weight in {name}; refusing to save")
                return True
    return False


def save_checkpoint(
    path: Path,
    model: RAYDiT_v3,
    optimizer: AdamW,
    step: int,
    seed: int,
) -> bool:
    if checkpoint_has_corruption(model):
        print(f"[checkpoint guard] SKIP step={step}: corrupted model weights")
        return False

    torch.save(
        {
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
        },
        path,
    )
    return True


def load_resume(
    path: Path,
    model: RAYDiT_v3,
    optimizer: AdamW,
    device: torch.device,
) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")

    checkpoint_data = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint_data["dit"], strict=True)
    if "optimizer" in checkpoint_data:
        optimizer.load_state_dict(checkpoint_data["optimizer"])
    step = int(checkpoint_data.get("step", 0))
    print(f"[resume] loaded {path} at step {step}")
    return step


def set_learning_rate(optimizer: AdamW, base_lr: float, step: int) -> float:
    """500-step linear warmup from 0, then constant base LR."""
    if step <= WARMUP_STEPS:
        lr = base_lr * step / WARMUP_STEPS
    else:
        lr = base_lr
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def main():
    parser = argparse.ArgumentParser(description="Train RAY-IMAGE DiT v3 with flow matching.")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=TRAIN_LR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/ray_image_dit_n8_v3")
    parser.add_argument("--hf-repo", type=str, default=HF_DEFAULT_REPO)
    parser.add_argument("--no-hf-upload", action="store_true")
    args = parser.parse_args()

    if args.steps < 0:
        raise ValueError("--steps must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.start_step < 0:
        raise ValueError("--start-step must be >= 0")
    if args.resume and args.start_step:
        raise ValueError("Use either --resume or --start-step, not both")
    if not torch.cuda.is_available():
        raise RuntimeError("N8 DiT v3 training requires a CUDA GPU")

    seed_everything(args.seed)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        MONET_REPO,
        name="parquet",
        split="train",
        streaming=True,
    )
    iterator = iter(dataset)

    vae = load_vae(device)
    text_encoder = RAYTextEncoder(
        model_name=RAYTextEncoder.MODEL_ID,
        max_tokens=TEXT_MAX_TOKENS,
        load_in_8bit=True,
    )
    text_encoder.eval()

    dit = build_dit().to(device)
    optimizer = AdamW(
        dit.parameters(),
        lr=0.0,
        betas=(0.9, 0.99),
        weight_decay=0.01,
    )

    last_step = args.start_step
    if args.resume:
        last_step = load_resume(Path(args.resume), dit, optimizer, device)

    if args.steps <= last_step:
        print(f"[done] target steps={args.steps} already reached at step={last_step}")
        return

    parameter_count = sum(parameter.numel() for parameter in dit.parameters())
    print(f"[DiT v3] trainable parameters: {parameter_count:,}")
    print(f"[data] {MONET_REPO} / parquet / train (streaming)")
    print(f"[config] batch={args.batch_size} lr={args.lr:.2e} warmup={WARMUP_STEPS} target_steps={args.steps}")
    print("[precision] DiT forward autocast=fp16; Qwen text hidden cast bf16 -> fp16")
    print("[stability] grad_clip=1.0; NaN guards=enabled; finite-checkpoints=enabled")
    print("[architecture] SwiGLU + RMSNorm + RoPE + gated cross-attention")

    autocast_enabled = device.type == "cuda"
    autocast_dtype = torch.float16
    nan_skip_count = 0
    running_loss = 0.0
    running_count = 0

    pbar = tqdm(total=args.steps - last_step, desc="RAY-DiT v3 N8")

    for step in range(last_step + 1, args.steps + 1):
        current_lr = set_learning_rate(optimizer, args.lr, step)

        try:
            images, captions = next_batch(iterator, args.batch_size)
        except StopIteration:
            iterator = iter(dataset)
            images, captions = next_batch(iterator, args.batch_size)
        images = images.to(device, non_blocking=True)

        with torch.inference_mode():
            _, clean_latents, _, _ = vae(images)
            text_hidden, text_mask = text_encoder.encode_texts(
                captions,
                max_length=TEXT_MAX_TOKENS,
            )

        clean_latents = clean_latents.to(device=device, dtype=torch.float32)
        text_hidden = text_hidden.to(device=device, dtype=torch.float16)
        text_mask = text_mask.to(device=device)

        noise = torch.randn_like(clean_latents)
        t = torch.rand(clean_latents.shape[0], device=device)
        t_view = t[:, None, None, None]
        z_t = (1.0 - t_view) * clean_latents + t_view * noise
        target_velocity = noise - clean_latents

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            prediction = checkpointed_forward(
                dit,
                z_t,
                t,
                text_hidden,
                text_mask,
            )
            loss = F.mse_loss(prediction.float(), target_velocity.float())

        if torch.isnan(loss) or torch.isinf(loss):
            nan_skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            pbar.update(1)
            pbar.set_postfix(loss="nan", lr=f"{current_lr:.1e}")
            if step % LOG_EVERY == 0:
                vram_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                print(
                    f"step={step} loss=nan grad_norm=nan "
                    f"NaN_skip_count={nan_skip_count} vram_used={vram_gb:.2f}GB"
                )
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), GRAD_CLIP)

        if not torch.isfinite(grad_norm).item():
            nan_skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            pbar.update(1)
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.1e}")
            if step % LOG_EVERY == 0:
                vram_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
                print(
                    f"step={step} loss={loss.item():.6f} grad_norm=nan "
                    f"NaN_skip_count={nan_skip_count} vram_used={vram_gb:.2f}GB"
                )
            continue

        optimizer.step()

        running_loss += float(loss.item())
        running_count += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.1e}")

        if step % LOG_EVERY == 0:
            mean_loss = running_loss / max(running_count, 1)
            vram_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
            print(
                f"step={step} loss={loss.item():.6f} loss_avg={mean_loss:.6f} "
                f"grad_norm={float(grad_norm):.4f} NaN_skip_count={nan_skip_count} "
                f"vram_used={vram_gb:.2f}GB lr={current_lr:.2e}"
            )
            running_loss = 0.0
            running_count = 0

        if step % CHECKPOINT_EVERY == 0:
            checkpoint_path = output_dir / f"ray_dit_v3_step_{step:07d}.pt"
            if save_checkpoint(checkpoint_path, dit, optimizer, step, args.seed):
                print(f"[checkpoint] {checkpoint_path}")
                if not args.no_hf_upload:
                    upload_checkpoint(checkpoint_path, args.hf_repo)
            elif checkpoint_path.exists():
                checkpoint_path.unlink()

    pbar.close()

    if args.steps % CHECKPOINT_EVERY != 0:
        checkpoint_path = output_dir / f"ray_dit_v3_step_{args.steps:07d}.pt"
        if save_checkpoint(checkpoint_path, dit, optimizer, args.steps, args.seed):
            print(f"[checkpoint] {checkpoint_path}")
            if not args.no_hf_upload:
                upload_checkpoint(checkpoint_path, args.hf_repo)
        elif checkpoint_path.exists():
            checkpoint_path.unlink()


if __name__ == "__main__":
    main()
