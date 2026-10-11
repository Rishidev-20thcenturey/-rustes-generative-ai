"""CPU smoke tests for checkpoint provenance, resume validation and whitening.

Runs without a GPU and without downloading any model. Covers:

  RB-01/RB-02  un-whitening is applied only when the checkpoint carries ``whiten``
               state, and ``--latent-stats`` is a consistency check that fails.
  RB-03        checkpoint config records latent_space / text_cap / vae_checkpoint
               / torch / run.
  RB-04        the legacy ``generator_v0.3`` stage is refused by generate.py.
  RB-11        resume raises on an architecture mismatch and only warns when keys
               are missing.

Run with:  python -m ray_image.resume_smoke
"""

import tempfile
from pathlib import Path

import torch

from . import train_dit_v3 as T
from .dit_v3 import RAYDiT_v3
from .generate import assert_supported_stage, resolve_whitening
from .whiten import LatentNormalizer

TINY = dict(latent_channels=16, latent_size=32, dim=64, depth=1, heads=4,
            patch=2, cond_dim=2560, cross_gate=True)


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    model = RAYDiT_v3(**TINY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return model, optimizer


def _tiny_expected():
    return {"image_size": 256, "latent_channels": 16, "latent_size": 32, "dim": 64,
            "depth": 1, "heads": 4, "patch": 2, "text_dim": 2560}


def main() -> None:
    device = torch.device("cpu")
    tmp = Path(tempfile.mkdtemp(prefix="ray_resume_smoke_"))

    # ---------------------------------------------------------------- RB-03
    model, optimizer = _tiny_model()
    payload = T._checkpoint_payload(model, optimizer, step=7, seed=3)
    config = payload["config"]
    for key in ("latent_space", "text_cap", "vae_checkpoint", "torch", "run"):
        assert key in config, f"RB-03: checkpoint config missing {key!r}"
    assert config["latent_space"] == "raw", config["latent_space"]
    assert config["text_cap"] == T.TEXT_MAX_TOKENS, config["text_cap"]
    assert config["vae_checkpoint"] == T.VAE_CHECKPOINT, config["vae_checkpoint"]
    assert isinstance(config["torch"], str) and config["torch"], config["torch"]
    print(f"[RB-03] provenance keys present (latent_space={config['latent_space']!r}, "
          f"text_cap={config['text_cap']}, torch={config['torch']}, run={config['run']!r})")

    # ---------------------------------------------------------------- RB-11
    ckpt_path = tmp / "tiny.pt"
    torch.save(payload, ckpt_path)

    step = T.load_resume(ckpt_path, model, optimizer, device, expected_config=_tiny_expected())
    assert step == 7, step
    print("[RB-11] matching config resumes cleanly (step=7)")

    mismatched = T._checkpoint_payload(model, optimizer, step=7, seed=3)
    mismatched["config"]["dim"] = 512  # pretend a different architecture
    bad_path = tmp / "mismatch.pt"
    torch.save(mismatched, bad_path)
    try:
        T.load_resume(bad_path, model, optimizer, device, expected_config=_tiny_expected())
    except RuntimeError as exc:
        assert "mismatch" in str(exc), exc
        print("[RB-11] architecture mismatch correctly raises RuntimeError")
    else:
        raise AssertionError("RB-11 regression: mismatched config was accepted")

    missing = T._checkpoint_payload(model, optimizer, step=7, seed=3)
    del missing["config"]["patch"]
    missing["config"].pop("latent_space", None)
    missing_path = tmp / "missing_keys.pt"
    torch.save(missing, missing_path)
    warned = T.load_resume(missing_path, model, optimizer, device,
                           expected_config=_tiny_expected())
    assert warned == ["patch"], warned
    print(f"[RB-11] missing keys warn only (warned={warned})")

    # ---------------------------------------------------------- RB-01/RB-02
    stats_path = tmp / "stats.json"
    mean = [0.5] * 16
    std = [2.0] * 16
    stats_path.write_text(
        '{"per_channel_mean": %s, "per_channel_std": %s}'
        % (mean, std)
    )

    # (a) no whiten state anywhere -> no transform, and stats must be rejected
    assert resolve_whitening({"config": {"latent_space": "raw"}}) is None
    try:
        resolve_whitening({"config": {"latent_space": "raw"}}, str(stats_path))
    except ValueError as exc:
        assert "no 'whiten' state" in str(exc), exc
        print("[RB-01] stats without checkpoint whiten state -> ValueError (as required)")
    else:
        raise AssertionError("RB-01 regression: un-verifiable stats were accepted")

    # (b) checkpoint carries whiten state -> transform applies, matching stats OK
    normalizer = LatentNormalizer(torch.tensor(mean), torch.tensor(std))
    ckpt_with_whiten = {"config": {"latent_space": "whitened"}, "whiten": normalizer.state()}
    recovered = resolve_whitening(ckpt_with_whiten, str(stats_path))
    assert recovered is not None and recovered.channels == 16
    print("[RB-02] checkpoint whiten state + matching stats -> un-whitening applied")

    # (c) disagreeing stats -> ValueError
    other = tmp / "stats_other.json"
    other.write_text('{"per_channel_mean": %s, "per_channel_std": %s}'
                     % ([0.1] * 16, [3.0] * 16))
    try:
        resolve_whitening(ckpt_with_whiten, str(other))
    except ValueError as exc:
        assert "does not match" in str(exc), exc
        print("[RB-02] disagreeing stats -> ValueError (as required)")
    else:
        raise AssertionError("RB-02 regression: disagreeing stats were accepted")

    # (d) contradictory declaration -> ValueError
    contradictory = {"config": {"latent_space": "raw"}, "whiten": normalizer.state()}
    try:
        resolve_whitening(contradictory)
    except ValueError as exc:
        assert "inconsistent" in str(exc), exc
        print("[RB-01] whitened state + latent_space='raw' -> ValueError")
    else:
        raise AssertionError("RB-01 regression: contradictory checkpoint accepted")

    # ---------------------------------------------------------------- RB-04
    try:
        assert_supported_stage({"stage": "generator_v0.3_n5"})
    except NotImplementedError as exc:
        assert "generator_v0.3" in str(exc), exc
        print("[RB-04] generator_v0.3 stage refused (NotImplementedError)")
    else:
        raise AssertionError("RB-04 regression: legacy stage was accepted")

    assert_supported_stage({"stage": "dit_v3_n8"})  # supported stage passes
    assert_supported_stage({})  # legacy checkpoints without a stage key pass
    print("[RB-04] supported / unlabelled stages still load")

    print(f"\nresume_smoke: PASS (artifacts under {tmp})")


if __name__ == "__main__":
    main()
