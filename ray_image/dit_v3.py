"""RAY-IMAGE DiT v3.

A smaller, T4-friendly diffusion transformer for 16x32x32 latents:
  - dim=512, depth=12, heads=8, patch=2
  - RMSNorm + SwiGLU FFN
  - RoPE self-attention
  - gated text cross-attention enabled by default

Expected text context: Qwen3-4B hidden states with dim=2560.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(dtype=self.weight.dtype) * self.weight


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        if self.dim % 2 != 0:
            raise ValueError("timestep embedding dim must be even")
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        x = t.float()[:, None] * freq[None, :]
        emb = torch.cat([x.sin(), x.cos()], dim=-1)
        return self.mlp(emb)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 1D RoPE over token positions to self-attention q/k."""
    head_dim = q.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even attention head dimension")
    half = head_dim // 2
    device = q.device
    freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = positions[:, None] * freq[None, :]
    cos_half = angles.cos().to(dtype=q.dtype)[None, None, :, :]
    sin_half = angles.sin().to(dtype=q.dtype)[None, None, :, :]
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(round((4 * dim * 2 / 3) / 64) * 64)
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SelfAttentionRoPE(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, n)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Token-to-text cross-attention with optional learned output gate."""

    def __init__(self, dim: int, heads: int, context_dim: int, cross_gate: bool = True):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(context_dim, dim, bias=False)
        self.v = nn.Linear(context_dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.cross_gate = nn.Parameter(torch.ones(1)) if cross_gate else None

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, d = x.shape
        l = context.shape[1]
        q = self.q(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(context).view(b, l, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(context).view(b, l, self.heads, self.head_dim).transpose(1, 2)
        attn_mask = None
        if context_mask is not None:
            # v2's mask convention is True=padding/ignore. SDPA's bool mask is
            # True=keep, so invert the same mask before passing it to attention.
            attn_mask = (~context_mask).to(device=q.device)[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, n, d)
        out = self.proj(out)
        if self.cross_gate is not None:
            out = self.cross_gate * out
        return out


class RAYDiTBlock_v3(nn.Module):
    """Pre-norm DiT block with RMSNorm, RoPE self-attention and gated cross-attention."""

    def __init__(self, dim: int, heads: int, context_dim: int, cross_gate: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.self_attn = SelfAttentionRoPE(dim, heads)
        self.norm_cross = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, heads, context_dim, cross_gate=cross_gate)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim)
        self.cond = nn.Linear(dim, dim * 2)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        h = self.norm1(x)
        h = h * (1.0 + scale[:, None, :]) + shift[:, None, :]
        x = x + self.self_attn(h)
        x = x + self.cross_attn(
            self.norm_cross(x),
            text,
            context_mask=text_mask,
        )
        x = x + self.ffn(self.norm2(x))
        return x


class RAYDiT_v3(nn.Module):
    """~60M-parameter RAY-IMAGE DiT v3 for 16x32x32 latents."""

    def __init__(
        self,
        latent_channels: int = 16,
        latent_size: int = 32,
        dim: int = 512,
        depth: int = 12,
        heads: int = 8,
        patch: int = 2,
        cond_dim: int = 2560,
        cross_gate: bool = True,
    ):
        super().__init__()
        if latent_size % patch != 0:
            raise ValueError("latent_size must be divisible by patch")
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")

        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.patch = patch
        self.cond_dim = cond_dim

        self.in_proj = nn.Conv2d(latent_channels, dim, kernel_size=patch, stride=patch)
        self.time = TimestepEmbedding(dim)
        self.text_proj = nn.Linear(cond_dim, dim)
        self.text_ctx_proj = nn.Linear(cond_dim, dim) if cond_dim != dim else nn.Identity()
        self.blocks = nn.ModuleList(
            [RAYDiTBlock_v3(dim, heads, context_dim=dim, cross_gate=cross_gate) for _ in range(depth)]
        )
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, patch * patch * latent_channels)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"expected latent tensor [B,C,H,W], got {tuple(z.shape)}")
        b, c, h, w = z.shape
        if (c, h, w) != (self.latent_channels, self.latent_size, self.latent_size):
            raise ValueError(
                f"expected latent [{self.latent_channels},{self.latent_size},{self.latent_size}], got [{c},{h},{w}]"
            )
        if text.ndim != 3 or text.shape[0] != b or text.shape[2] != self.cond_dim:
            raise ValueError(f"expected text [B,L,{self.cond_dim}], got {tuple(text.shape)}")
        if t.ndim != 1 or t.shape[0] != b:
            raise ValueError(f"expected timestep [B], got {tuple(t.shape)}")
        if text_mask is not None and text_mask.shape != text.shape[:2]:
            raise ValueError(f"expected text_mask {tuple(text.shape[:2])}, got {tuple(text_mask.shape)}")

        x = self.in_proj(z)
        ph, pw = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        text = text.to(dtype=x.dtype)
        if text_mask is None:
            pooled = text.mean(dim=1)
        else:
            valid = (~text_mask).to(dtype=text.dtype).unsqueeze(-1)
            pooled = (text * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        cond = self.time(t).to(dtype=x.dtype) + self.text_proj(pooled)
        text_ctx = self.text_ctx_proj(text)
        for block in self.blocks:
            x = block(x, cond, text_ctx, text_mask=text_mask)

        x = self.out(self.norm(x))
        x = x.view(b, ph, pw, self.patch, self.patch, self.latent_channels)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(b, self.latent_channels, h, w)
