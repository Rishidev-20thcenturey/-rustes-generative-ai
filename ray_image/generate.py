"""Generate an image from a trained RAY-IMAGE checkpoint."""
import argparse
import re
from pathlib import Path

import torch
from PIL import Image

from .cfg import (
    NULL_PROMPT_REPO_DEFAULT,
    encode_null_prompt,
    fit_null_to_length,
    load_or_build_null_prompt,
)
from .config import RAYConfig
from .flow import euler_sample
from .dit_v3 import RAYDiT_v3
from .text_encoder import RAYTextEncoder
from .utils import build_models, load_checkpoint, load_pretrained
from .whiten import LatentNormalizer

DEFAULT_LATENT_STATS = Path(__file__).with_name("latent_stats_corrected.json")

# Legacy generator stages that must never be silently served by this script.
UNSUPPORTED_STAGE_PREFIX = "generator_v0.3"

# Issue #7: EMA weight files written by train_dit_v3.py.
EMA_FILE_RE = re.compile(r"^ray_dit_v3_ema_step_\d+\.pt$")


def assert_supported_stage(checkpoint) -> None:
    """RB-04: refuse to generate from the legacy N5/N7/N8 generator stages.

    Those checkpoints are known-inconsistent with this loader (architecture
    selection, resolution and latent-space assumptions), so producing an image
    from them would be silently wrong. Fail loudly instead.
    """
    stage = checkpoint.get("stage")
    if isinstance(stage, str) and stage.startswith(UNSUPPORTED_STAGE_PREFIX):
        raise NotImplementedError(
            f"generation from stage={stage!r} is not supported: the legacy N5/N7/N8 "
            "generator path (train_n5.py) is inconsistent with this loader "
            "(architecture/flags/latent-space). Train with train_dit_v2.py or "
            "train_dit_v3.py, or extend generate.py before serving these weights."
        )


def resolve_whitening(checkpoint, latent_stats=None):
    """RB-01/RB-02: apply un-whitening ONLY when the checkpoint carries it.

    ``--latent-stats`` is no longer an implicit default; it is an optional
    *consistency check* against the checkpoint's own whitening state. Failing
    cases raise ``ValueError``:
      * stats were supplied but the checkpoint has no ``whiten`` state;
      * stats disagree with the checkpoint's ``whiten`` state;
      * the checkpoint stores ``whiten`` but declares ``latent_space == "raw"``.
    Returns a ``LatentNormalizer`` or ``None``.
    """
    state = checkpoint.get("whiten")
    config = checkpoint.get("config") or {}
    declared = config.get("latent_space") if isinstance(config, dict) else None

    stats = None
    if latent_stats is not None:
        stats = LatentNormalizer.from_stats_json(latent_stats)

    if state is None:
        if stats is not None:
            raise ValueError(
                f"--latent-stats {latent_stats} was supplied but the checkpoint has no "
                "'whiten' state; refusing to apply an un-verifiable transform"
            )
        if declared == "whitened":
            raise ValueError(
                "checkpoint config declares latent_space='whitened' but stores no "
                "'whiten' state; the checkpoint is inconsistent"
            )
        return None

    if declared == "raw":
        raise ValueError(
            "checkpoint stores 'whiten' state but its config declares "
            "latent_space='raw'; the checkpoint is inconsistent"
        )

    if isinstance(state, dict) and "mean" in state and "std" in state:
        normalizer = LatentNormalizer.from_state(state)
    else:
        normalizer = LatentNormalizer.from_stats_dict(state)

    if stats is not None:
        if normalizer.channels != stats.channels:
            raise ValueError(
                f"--latent-stats has {stats.channels} channels but the checkpoint's "
                f"whitening state has {normalizer.channels}"
            )
        import torch as _torch

        same_mean = _torch.allclose(normalizer.mean, stats.mean.to(normalizer.mean.dtype), rtol=1e-4, atol=1e-6)
        same_std = _torch.allclose(normalizer.std, stats.std.to(normalizer.std.dtype), rtol=1e-4, atol=1e-6)
        if not (same_mean and same_std):
            raise ValueError(
                f"--latent-stats {latent_stats} does not match the checkpoint's stored "
                "whitening state; refusing to guess which one is correct"
            )
    return normalizer



def load_models(checkpoint_path, device, vae_checkpoint=None, latent_stats=None):
    checkpoint = load_checkpoint(checkpoint_path, device)
    assert_supported_stage(checkpoint)  # RB-04
    raw_cfg = checkpoint.get("config", {})
    # Map checkpoint keys to RAYConfig field names
    mapped_cfg = {}
    for k, v in raw_cfg.items():
        if k in RAYConfig.__dataclass_fields__:
            mapped_cfg[k] = v
        elif k == "dim" and "model_dim" in RAYConfig.__dataclass_fields__:
            mapped_cfg["model_dim"] = v
        elif k == "patch" and "patch_size" in RAYConfig.__dataclass_fields__:
            mapped_cfg["patch_size"] = v
        elif k == "text_max_tokens" and "max_tokens" in RAYConfig.__dataclass_fields__:
            mapped_cfg["max_tokens"] = v
    cfg = RAYConfig(**mapped_cfg)

    # N7 path: use the separately trained N6 VAE and its matching 16-channel
    # latent statistics instead of the VAE embedded in the DiT checkpoint.
    if vae_checkpoint is not None:
        cfg.latent_channels = 16

        from ray_image.vae_v2 import RAYVAE_v2
        vae_checkpoint_data = load_checkpoint(vae_checkpoint, device)
        vae_cfg = vae_checkpoint_data.get("config", {})
        vae = RAYVAE_v2(
            latent_channels=vae_cfg.get("latent_channels", 16),
            base_channels=vae_cfg.get("base_channels", 64),
        )
        vae_state_dict = vae_checkpoint_data.get("vae", vae_checkpoint_data)
        load_pretrained({"vae": vae}, {"vae": vae_state_dict}, device)

        text_encoder = RAYTextEncoder(
            model_name=RAYTextEncoder.MODEL_ID,
            max_tokens=cfg.max_tokens,
            load_in_8bit=True,
        )
        dit = RAYDiT_v3(
            latent_channels=cfg.latent_channels,
            latent_size=cfg.latent_size,
            dim=cfg.model_dim,
            depth=cfg.depth,
            heads=cfg.heads,
            patch=cfg.patch_size,
            cond_dim=cfg.text_dim,
            cross_gate=getattr(cfg, "cross_gate", True),
        ).to(device)
        load_pretrained({"text_encoder": text_encoder, "dit": dit}, checkpoint, device)

        # RB-01/RB-02: un-whitening is driven by the checkpoint, never by an
        # implicit default. --latent-stats only verifies the checkpoint's state.
        whiten = resolve_whitening(checkpoint, latent_stats)
        if whiten is not None:
            whiten.validate_channels(16)
    else:
        # Legacy path: load the VAE, text encoder, DiT, and whitening state from
        # the generator checkpoint exactly as before (including 4-channel N5).
        modules = build_models(cfg, device)
        load_pretrained(modules, checkpoint, device)
        vae = modules["vae"]
        text_encoder = modules["text_encoder"]
        dit = modules["dit"]

        whiten = resolve_whitening(checkpoint, latent_stats)
        if whiten is not None:
            whiten.validate_channels(cfg.latent_channels)

    for module in (vae, text_encoder, dit):
        module.eval()

    return cfg, checkpoint.get("vocab", None), vae, text_encoder, dit, whiten


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vae-checkpoint", default=None,
                        help="Optional N6 VAE checkpoint. When provided, generation uses "
                             "the 16-channel N6 VAE instead of the embedded VAE.")
    parser.add_argument("--latent-stats", default=None,
                        help="Optional latent-stats JSON used ONLY as a consistency check "
                             "against the checkpoint's own whitening state. Un-whitening is "
                             f"applied only when the checkpoint stores it (suggested: "
                             f"{DEFAULT_LATENT_STATS}).")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="generated.png")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=1.0,
                        help="Issue #7: classifier-free guidance scale. Values > 1.0 run the DiT twice per "
                             "step (prompt and null prompt) and combine uncond + scale * (cond - uncond). "
                             "Requires --vae-checkpoint (v3 route).")
    parser.add_argument("--use-ema", action="store_true",
                        help="Issue #7: --checkpoint must be a ray_dit_v3_ema_step_*.pt file; its 'dit' weights are used. "
                             "Requires --vae-checkpoint (v3 route).")
    parser.add_argument("--null-prompt-repo", default=NULL_PROMPT_REPO_DEFAULT,
                        help="HF repo holding aux/null_prompt_emb.pt (used when --cfg-scale > 1).")
    args = parser.parse_args()
    if args.use_ema and not EMA_FILE_RE.match(Path(args.checkpoint).name):
        raise ValueError(f"--use-ema requires an EMA checkpoint named ray_dit_v3_ema_step_*.pt, got {args.checkpoint!r}")
    if (args.use_ema or args.cfg_scale > 1.0) and args.vae_checkpoint is None:
        raise ValueError("--use-ema and --cfg-scale > 1.0 are only supported on the v3 route (pass --vae-checkpoint)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    cfg, vocab, vae, text_encoder, dit, whiten = load_models(
        args.checkpoint,
        device,
        vae_checkpoint=args.vae_checkpoint,
        latent_stats=args.latent_stats,
    )

    with torch.no_grad():
        text, text_mask = text_encoder.encode_texts([args.prompt])
        text = text.to(device)
        text_mask = text_mask.to(device)
        null_text = null_mask = None
        if args.cfg_scale > 1.0:
            # Issue #7: null prompt for CFG. Fetched from HF; if that fails it is
            # rebuilt once with the already-loaded encoder (not uploaded here).
            null_payload = load_or_build_null_prompt(
                args.null_prompt_repo,
                text_dim=RAYTextEncoder.HIDDEN_SIZE,
                build_fn=lambda: encode_null_prompt(text_encoder, cfg.max_tokens),
                upload=False,
            )
            null_emb, null_valid = fit_null_to_length(
                null_payload["emb"].to(device=device, dtype=text.dtype),
                null_payload["mask"].to(device=device),
                text.shape[1],
            )
            null_text = null_emb.unsqueeze(0)
            null_mask = null_valid.unsqueeze(0)
            print(f"[cfg] scale={args.cfg_scale} null_tokens={null_payload['emb'].shape[0]}")
        latent = euler_sample(
            dit,
            text,
            (1, cfg.latent_channels, cfg.latent_size, cfg.latent_size),
            steps=args.steps,
            device=device,
            text_mask=text_mask,
            cfg_scale=args.cfg_scale,
            null_text=null_text,
            null_mask=null_mask,
        )
        # Un-whitening is applied only when the checkpoint stores whitening state
        # (see resolve_whitening); otherwise the sampler output is decoded as-is.
        if whiten is not None:
            latent = whiten.denormalize(latent)
        image = vae.decode(latent).clamp(0, 1)[0]

    array = (image.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output)
    print(f"saved image: {output}")


if __name__ == "__main__":
    main()
