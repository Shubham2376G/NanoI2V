# vae/blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from vae.conv import CausalConv3d, CausalConv3d_1x1


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_norm(num_channels: int, num_groups: int = 32) -> nn.GroupNorm:
    """GroupNorm — works at any batch size, stable for video."""
    # If channels < 32, use fewer groups (must divide evenly)
    actual_groups = min(num_groups, num_channels)
    while num_channels % actual_groups != 0:
        actual_groups -= 1
    return nn.GroupNorm(actual_groups, num_channels, eps=1e-6, affine=True)


# ─────────────────────────────────────────────
# Core building block
# ─────────────────────────────────────────────

class ResBlock3D(nn.Module):
    """
    3D residual block for the VAE encoder/decoder.

    Structure:
        x → Norm → SiLU → Conv3D(3×3×3)
          → Norm → SiLU → Conv3D(3×3×3)
          → + skip(x)

    The skip connection uses a 1×1×1 conv if in/out channels differ,
    otherwise it's an identity (free).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
    ):
        super().__init__()

        self.norm1 = get_norm(in_channels, num_groups)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = get_norm(out_channels, num_groups)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)

        self.act = nn.SiLU()

        # Skip projection only needed when channels change
        if in_channels != out_channels:
            self.skip = CausalConv3d_1x1(in_channels, out_channels)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        residual = self.skip(x)

        x = self.act(self.norm1(x))
        x = self.conv1(x)

        x = self.act(self.norm2(x))
        x = self.conv2(x)

        return x + residual


# ─────────────────────────────────────────────
# Downsampling
# ─────────────────────────────────────────────

class SpatialDownsample(nn.Module):
    """
    2× spatial (H, W) downsampling via strided conv.
    Time dimension is untouched: stride = (1, 2, 2).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # stride (T=1, H=2, W=2) → H and W halved, T unchanged
        self.conv = CausalConv3d(
            in_channels, out_channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TemporalDownsample(nn.Module):
    """
    2× temporal downsampling via strided causal conv.
    Spatial dims are untouched: stride = (2, 1, 1).

    Note: we skip downsampling the first frame so the model can
    always condition on a clean frame-0 latent for I2V.
    The 'keep_first' flag handles this.
    """

    def __init__(self, in_channels: int, out_channels: int, keep_first: bool = True):
        super().__init__()
        self.keep_first = keep_first
        self.conv = CausalConv3d(
            in_channels, out_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        if self.keep_first:
            # Downsample all frames except the first
            first = x[:, :, :1, :, :]          # (B, C, 1, H, W) — kept
            rest  = x[:, :, 1:, :, :]          # (B, C, T-1, H, W)
            rest  = self.conv(rest)             # (B, C', (T-1)//2, H, W)
            return torch.cat([first, rest], dim=2)
        else:
            return self.conv(x)


# ─────────────────────────────────────────────
# Upsampling
# ─────────────────────────────────────────────

class SpatialUpsample(nn.Module):
    """
    2× spatial (H, W) upsampling.
    Uses interpolate (nearest) + conv to avoid checkerboard artifacts.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # After interpolation the spatial dims are doubled,
        # then a conv refines without changing resolution
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=(1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        # Reshape to (B*T, C, H, W) so we can use 2D interpolate
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = x.reshape(B, T, C, H * 2, W * 2).permute(0, 2, 1, 3, 4)
        return self.conv(x)


class TemporalUpsample(nn.Module):
    """
    2× temporal upsampling.
    Uses interpolate (nearest) along time + conv.
    Mirrors TemporalDownsample's 'keep_first' logic.
    """

    def __init__(self, in_channels: int, out_channels: int, keep_first: bool = True):
        super().__init__()
        self.keep_first = keep_first
        self.conv = CausalConv3d(in_channels, out_channels, kernel_size=(3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        if self.keep_first:
            first = x[:, :, :1, :, :]          # keep frame 0 as-is
            rest  = x[:, :, 1:, :, :]          # upsample the rest

            # Interpolate along T: nearest is fine here
            rest = F.interpolate(rest, scale_factor=(2.0, 1.0, 1.0), mode="nearest")
            rest = self.conv(rest)

            return torch.cat([first, rest], dim=2)
        else:
            x = F.interpolate(x, scale_factor=(2.0, 1.0, 1.0), mode="nearest")
            return self.conv(x)


# ─────────────────────────────────────────────
# Attention block (used in VAE bottleneck only)
# ─────────────────────────────────────────────

class SpatialAttention(nn.Module):
    """
    Single-head self-attention over spatial tokens at each timestep.
    Used only in the VAE bottleneck (middle block) — NOT the DiT.
    This gives the VAE global spatial context at the lowest resolution.

    For each frame independently:
        (B, C, T, H, W) → flatten H*W → attention → unflatten
    """

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.norm = get_norm(channels, num_groups)
        self.to_qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        residual = x

        x = self.norm(x)

        # Process each timestep independently as a spatial attention
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H * W)  # (BT, C, HW)

        qkv = self.to_qkv(x)                             # (BT, 3C, HW)
        q, k, v = qkv.chunk(3, dim=1)                    # each: (BT, C, HW)

        # Attention: (BT, HW, C) × (BT, C, HW) → (BT, HW, HW)
        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.bmm(attn, v.permute(0, 2, 1))        # (BT, HW, C)
        out = out.permute(0, 2, 1)                        # (BT, C, HW)
        out = self.proj_out(out)

        out = out.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return out + residual