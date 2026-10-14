"""CPU smoke test for issue #7 (RB-10): caption dropout, null prompt, EMA.

(a) caption dropout replaces ~10% of 1,000 synthetic samples;
(b) the null-prompt payload has >= 1 valid token (and bad payloads are refused);
(c) EMA updates weights (drift > 0), follows the decay formula, and saves under key 'dit'.

No network access and no Qwen: the HF download is mocked for the fetch/rebuild check.
"""

import copy
import tempfile
from pathlib import Path

import torch

from . import cfg
from .dit_v3 import RAYDiT_v3


def check_caption_dropout() -> None:
    torch.manual_seed(0)
    samples, tokens, dim = 1000, 47, 8
    text_hidden = torch.randn(samples, tokens, dim)
    text_mask = torch.zeros(samples, tokens, dtype=torch.bool)
    text_mask[:, 10:] = True  # valid prefix of length 10
    null_emb = torch.full((tokens, dim), 7.0)
    null_mask = torch.ones(tokens, dtype=torch.bool)
    null_mask[0] = False  # exactly one valid token
    generator = torch.Generator().manual_seed(1234)

    hidden, mask, dropped = cfg.apply_caption_dropout(text_hidden, text_mask, null_emb, null_mask, 0.10, generator=generator)
    rate = dropped / samples
    assert 0.07 <= rate <= 0.13, f"caption dropout rate {rate:.3f} outside [0.07, 0.13] over {samples} samples"

    replaced = torch.isclose(hidden, null_emb.expand_as(hidden)).all(dim=(1, 2))
    assert int(replaced.sum()) == dropped, "dropped count does not match replaced rows"
    assert torch.equal(hidden[~replaced], text_hidden[~replaced]), "kept samples were modified"
    assert torch.equal(mask[replaced], null_mask.expand(int(replaced.sum()), tokens)), "dropped rows must take the null mask"
    print(f"[PASS] caption dropout: dropped={dropped}/{samples} rate={rate:.3f}")

    _, _, none_dropped = cfg.apply_caption_dropout(text_hidden, text_mask, null_emb, null_mask, 0.0)
    assert none_dropped == 0, "p=0 must drop nothing"


def check_null_payload() -> None:
    dim, length = 8, 256
    mask = torch.ones(length, dtype=torch.bool)
    mask[0] = False
    payload = {"prompt": ".", "emb": torch.randn(length, dim), "mask": mask}
    valid = cfg.validate_null_payload(payload, dim)
    assert valid == 1, valid

    fit_emb, fit_mask = cfg.fit_null_to_length(payload["emb"], payload["mask"], 47)
    assert fit_emb.shape == (47, dim) and fit_mask.shape == (47,)
    assert int((~fit_mask).sum()) == 1

    for label, bad in (
        ("wrong prompt", {**payload, "prompt": "a"}),
        ("no valid tokens", {**payload, "mask": torch.ones(length, dtype=torch.bool)}),
        ("non-prefix valid tokens", {**payload, "mask": torch.tensor([True, False] + [True] * (length - 2))}),
    ):
        try:
            cfg.validate_null_payload(bad, dim)
        except cfg.NullPromptError:
            continue
        raise AssertionError(f"null payload check did not refuse: {label}")
    print(f"[PASS] null prompt: valid_tokens={valid} (bad payloads refused)")


def check_fetch_or_rebuild() -> None:
    dim, length = 8, 16
    mask = torch.ones(length, dtype=torch.bool)
    mask[0] = False
    built = {"prompt": ".", "emb": torch.randn(length, dim), "mask": mask}
    calls = {"build": 0}

    def build_fn():
        calls["build"] += 1
        return built

    original = cfg.hf_hub_download

    def failing_download(**_kwargs):
        raise OSError("simulated fetch failure")

    try:
        cfg.hf_hub_download = failing_download
        payload = cfg.load_or_build_null_prompt("example/repo", text_dim=dim, build_fn=build_fn, upload=False)
        assert calls["build"] == 1, "fetch failure must rebuild exactly once"
        assert payload["prompt"] == "."

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "null_prompt_emb.pt"
            torch.save(built, path)
            cfg.hf_hub_download = lambda **_kwargs: str(path)
            fetched = cfg.load_or_build_null_prompt("example/repo", text_dim=dim, build_fn=build_fn, upload=False)
        assert calls["build"] == 1, "successful fetch must not rebuild"
        assert torch.equal(fetched["emb"], built["emb"])
    finally:
        cfg.hf_hub_download = original
    print("[PASS] null prompt fetch/rebuild: fetch failure rebuilds once; successful fetch does not rebuild")


def check_ema() -> None:
    torch.manual_seed(0)
    model = RAYDiT_v3(latent_channels=16, latent_size=32, dim=64, depth=2, heads=4, patch=2, cond_dim=2560, cross_gate=True)
    ema = copy.deepcopy(model)
    ema.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    before = [p.detach().clone() for p in ema.parameters()]
    x = torch.randn(2, 16, 32, 32)
    t = torch.rand(2)
    text = torch.randn(2, 5, 2560)
    loss = model(x, t, text).pow(2).mean()
    loss.backward()
    optimizer.step()

    decay = 0.999
    cfg.update_ema(ema, model, decay)
    drift = cfg.ema_drift(ema, model)
    assert drift > 0.0, "EMA drift must be > 0 after an update from a different starting point"
    for e0, e1, m in zip(before, ema.parameters(), model.parameters()):
        expected = decay * e0 + (1.0 - decay) * m.detach()
        assert torch.allclose(e1, expected, atol=1e-6), "EMA update does not match decay formula"

    with tempfile.TemporaryDirectory() as tmp:
        path = cfg.save_ema_checkpoint(Path(tmp) / "ray_dit_v3_ema_step_0000001.pt", ema, step=1, config={"dim": 64})
        payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "dit" in payload, "EMA checkpoint must store weights under key 'dit'"
    assert set(payload["dit"].keys()) == set(model.state_dict().keys()), "EMA state_dict keys mismatch"
    print(f"[PASS] EMA: drift={drift:.3e}; saved under key 'dit' ({len(payload['dit'])} tensors)")


def main() -> None:
    check_caption_dropout()
    check_null_payload()
    check_fetch_or_rebuild()
    check_ema()
    print("[PASS] cfg smoke test")


if __name__ == "__main__":
    main()
