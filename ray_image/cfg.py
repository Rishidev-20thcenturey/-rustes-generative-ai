"""Classifier-free guidance helpers (issue #7, RB-10).

Pure-tensor helpers shared by ``train_dit_v3.py`` (caption dropout, EMA) and
``generate.py`` (CFG sampling, EMA weights). This module never loads Qwen:
callers supply a ``build_fn`` that encodes the null prompt when no fetched
artifact is available.
"""

import os
import tempfile
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

NULL_PROMPT = "."
NULL_PROMPT_FILENAME = "aux/null_prompt_emb.pt"
NULL_PROMPT_REPO_DEFAULT = "Rishix500/ray-image-dit-n8-v3"
EMA_CHECKPOINT_PREFIX = "ray_dit_v3_ema_step_"


class NullPromptError(RuntimeError):
    """Raised when the null placeholder embedding is missing or invalid."""


def validate_null_payload(payload, text_dim: int) -> int:
    """Check a null-prompt payload and return its number of valid tokens.

    Mask convention matches ``RAYTextEncoder.encode_texts``: ``True`` marks
    padding, ``False`` marks a valid token. Valid tokens must form a contiguous
    prefix, because ``fit_null_to_length`` relies on that layout.
    """
    if not isinstance(payload, dict):
        raise NullPromptError("null prompt payload is not a dict")
    missing = [key for key in ("prompt", "emb", "mask") if key not in payload]
    if missing:
        raise NullPromptError(f"null prompt payload missing keys: {missing}")
    if payload["prompt"] != NULL_PROMPT:
        raise NullPromptError(f"null prompt mismatch: expected {NULL_PROMPT!r}, got {payload['prompt']!r}")
    emb = payload["emb"]
    mask = payload["mask"].to(torch.bool)
    if emb.ndim != 2 or emb.shape[1] != text_dim or tuple(mask.shape) != (emb.shape[0],):
        raise NullPromptError(f"invalid null prompt shapes: emb={tuple(emb.shape)} mask={tuple(mask.shape)} text_dim={text_dim}")
    if not torch.isfinite(emb.float()).all().item():
        raise NullPromptError("null prompt embedding contains NaN/Inf")
    valid = int((~mask).sum().item())
    if valid < 1:
        raise NullPromptError("null prompt has no valid tokens (mask.sum(False) == 0)")
    if mask[:valid].any().item() or (~mask[valid:]).any().item():
        raise NullPromptError("null prompt valid tokens are not a contiguous prefix")
    return valid


def encode_null_prompt(text_encoder, max_tokens: int) -> dict:
    """Encode the null placeholder with a loaded RAYTextEncoder (fixed length)."""
    hidden, mask = text_encoder.encode_texts([NULL_PROMPT], max_length=max_tokens, pad_to_max=True)
    return {
        "prompt": NULL_PROMPT,
        "emb": hidden[0].detach().to(device="cpu", dtype=torch.float16),
        "mask": mask[0].detach().to(device="cpu", dtype=torch.bool),
    }


def fit_null_to_length(emb: torch.Tensor, mask: torch.Tensor, length: int):
    """Trim or pad the null embedding to ``length`` tokens.

    Padding positions are masked, so the trimmed/padded result has the same
    valid tokens as the original and attention output is unchanged.
    """
    current = emb.shape[0]
    if current >= length:
        if (~mask[length:]).any().item():
            raise NullPromptError(f"null prompt has valid tokens beyond requested length {length}")
        return emb[:length], mask[:length]
    pad = length - current
    emb = torch.cat([emb, emb.new_zeros(pad, emb.shape[1])], dim=0)
    mask = torch.cat([mask, mask.new_ones(pad)], dim=0)
    return emb, mask


def apply_caption_dropout(text_hidden, text_mask, null_emb, null_mask, p: float, generator=None):
    """Replace each sample's text conditioning with the null prompt with prob ``p``.

    ``text_hidden`` is ``[B, T, D]``, ``text_mask`` is ``[B, T]`` (True = pad),
    and ``null_emb``/``null_mask`` must already be fitted to length ``T``.
    Returns ``(text_hidden, text_mask, dropped_count)``.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"caption dropout probability must be in [0, 1], got {p}")
    if null_emb.shape != text_hidden.shape[1:] or null_mask.shape != text_mask.shape[1:]:
        raise ValueError(
            f"null prompt shape {tuple(null_emb.shape)} does not match text {tuple(text_hidden.shape[1:])}"
        )
    batch = text_hidden.shape[0]
    drop = torch.rand(batch, generator=generator) < p
    dropped = int(drop.sum().item())
    if dropped == 0:
        return text_hidden, text_mask, 0
    drop = drop.to(text_hidden.device)
    null_e = null_emb.to(device=text_hidden.device, dtype=text_hidden.dtype)
    null_m = null_mask.to(device=text_mask.device, dtype=text_mask.dtype)
    hidden = torch.where(drop[:, None, None], null_e.unsqueeze(0).expand_as(text_hidden), text_hidden)
    mask = torch.where(drop[:, None], null_m.unsqueeze(0).expand_as(text_mask), text_mask)
    return hidden, mask, dropped


@torch.no_grad()
def update_ema(ema, model, decay: float) -> None:
    """ema = decay * ema + (1 - decay) * model, over parameters; buffers are copied."""
    ema_params = list(ema.parameters())
    model_params = list(model.parameters())
    if len(ema_params) != len(model_params):
        raise ValueError("EMA and model parameter lists differ")
    for e, m in zip(ema_params, model_params):
        e.mul_(decay).add_(m.detach(), alpha=1.0 - decay)
    for eb, mb in zip(ema.buffers(), model.buffers()):
        eb.copy_(mb)


@torch.no_grad()
def ema_drift(ema, model) -> float:
    """Mean absolute difference between EMA and model parameters."""
    total = 0.0
    count = 0
    for e, m in zip(ema.parameters(), model.parameters()):
        total += float((e - m.detach()).abs().sum().item())
        count += e.numel()
    return total / max(count, 1)


def ema_checkpoint_payload(ema_model, step: int, config: dict) -> dict:
    """EMA weights under key ``dit`` (loadable by generate.py --use-ema)."""
    return {"stage": "dit_v3_n8_ema", "step": step, "config": dict(config), "dit": ema_model.state_dict()}


def save_ema_checkpoint(path: Path, ema_model, step: int, config: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ema_checkpoint_payload(ema_model, step, config), path)
    return path


def _upload_null_payload(payload: dict, repo_id: str, token: str) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / "null_prompt_emb.pt"
        torch.save(payload, local)
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=NULL_PROMPT_FILENAME,
            repo_id=repo_id,
            repo_type="model",
            commit_message="add null placeholder embedding for CFG",
        )
    print(f"[null] uploaded {NULL_PROMPT_FILENAME} -> {repo_id}")


def load_or_build_null_prompt(repo_id: str, *, text_dim: int, build_fn, upload: bool) -> dict:
    """Fetch ``aux/null_prompt_emb.pt`` from HF; rebuild once if the fetch fails.

    * Fetch succeeds -> validate and return. A failed validation raises (the
      artifact is never silently overwritten).
    * Fetch fails -> call ``build_fn`` once, validate, optionally upload
      (``upload=True``, private repo), and return it.
    Only the exception type is printed, never the token or the raw message.
    """
    token = os.getenv("HF_TOKEN")
    try:
        path = hf_hub_download(repo_id=repo_id, filename=NULL_PROMPT_FILENAME, repo_type="model", token=token)
    except Exception as exc:  # network, missing file, auth
        print(f"[null] fetch failed for {repo_id}/{NULL_PROMPT_FILENAME} ({type(exc).__name__}); rebuilding once")
        if build_fn is None:
            raise NullPromptError("null prompt could not be fetched and no build_fn was supplied") from exc
        payload = build_fn()
        valid = validate_null_payload(payload, text_dim)
        print(f"[null] rebuilt prompt={NULL_PROMPT!r} valid_tokens={valid} tokens={payload['emb'].shape[0]}")
        if upload:
            if not token:
                raise NullPromptError("HF_TOKEN is required to upload the rebuilt null prompt")
            _upload_null_payload(payload, repo_id, token)
        return payload

    payload = torch.load(path, map_location="cpu", weights_only=False)
    valid = validate_null_payload(payload, text_dim)
    print(f"[null] fetched {repo_id}/{NULL_PROMPT_FILENAME} valid_tokens={valid} tokens={payload['emb'].shape[0]}")
    return payload
