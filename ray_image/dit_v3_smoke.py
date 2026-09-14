"""Smoke test for RAY-IMAGE DiT v3.

Checks a [1,16,32,32] latent + [1,47,2560] text context ->
[1,16,32,32] output and reports parameter count / peak VRAM.
"""

import torch

from .dit_v3 import RAYDiT_v3


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("DiT v3 smoke test requires a CUDA GPU")

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    model = RAYDiT_v3(
        latent_channels=16,
        latent_size=32,
        dim=512,
        depth=12,
        heads=8,
        patch=2,
        cond_dim=2560,
        cross_gate=True,
    ).to(device)
    model.eval()

    latent = torch.randn(1, 16, 32, 32, device=device, dtype=torch.float16)
    text = torch.randn(1, 47, 2560, device=device, dtype=torch.float16)
    timestep = torch.tensor([0.5], device=device, dtype=torch.float32)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(latent, timestep, text)

    expected = (1, 16, 32, 32)
    assert tuple(output.shape) == expected, (
        f"DiT v3 output shape mismatch: got {tuple(output.shape)}, expected {expected}"
    )
    assert torch.isfinite(output).all().item(), "DiT v3 smoke output contains NaN/Inf"

    params = sum(parameter.numel() for parameter in model.parameters())
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print(f"[PASS] output shape: {tuple(output.shape)}")
    print(f"[model] parameters: {params:,} ({params / 1e6:.2f}M)")
    print(f"[model] peak VRAM: {peak_vram_gb:.2f} GB")
    print("[model] precision: fp16 autocast")
    print("[model] cross_gate: True")


if __name__ == "__main__":
    main()
