"""Generate an image from a trained RAY-IMAGE checkpoint."""
import argparse
from pathlib import Path

import torch
from PIL import Image

from .config import RAYConfig
from .flow import euler_sample
from .dit_v3 import RAYDiT_v3
from .text_encoder import RAYTextEncoder
from .utils import build_models, load_checkpoint, load_pretrained
from .whiten import LatentNormalizer

DEFAULT_LATENT_STATS = Path(__file__).with_name("latent_stats_corrected.json")


def load_models(checkpoint_path, device, vae_checkpoint=None, latent_stats=None):
    checkpoint = load_checkpoint(checkpoint_path, device)
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
        if latent_stats is None:
            latent_stats = DEFAULT_LATENT_STATS

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

        whiten = LatentNormalizer.from_stats_json(latent_stats)
        whiten.validate_channels(16)
    else:
        # Legacy path: load the VAE, text encoder, DiT, and whitening state from
        # the generator checkpoint exactly as before (including 4-channel N5).
        modules = build_models(cfg, device)
        load_pretrained(modules, checkpoint, device)
        vae = modules["vae"]
        text_encoder = modules["text_encoder"]
        dit = modules["dit"]

        whiten = None
        if checkpoint.get("whiten") is not None:
            whiten = LatentNormalizer.from_state(checkpoint["whiten"])
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
    parser.add_argument("--latent-stats", default=str(DEFAULT_LATENT_STATS),
                        help="Optional N6 latent stats JSON. Required with --vae-checkpoint "
                             "for matching 16-channel whitening inversion.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="generated.png")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
        latent = euler_sample(
            dit,
            text,
            (1, cfg.latent_channels, cfg.latent_size, cfg.latent_size),
            steps=args.steps,
            device=device,
            text_mask=text_mask,
        )
        # The sampler operates in whitened space when the checkpoint was trained
        # with whitening; invert (z = z_norm*std + mean) before decoding.
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
