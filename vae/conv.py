# vae/conv.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv3d(nn.Module):
    """
    Conv3d that is causal in the time dimension.
    Spatial dims (H, W) use standard symmetric padding.
    Time dim uses left-only padding so the kernel never sees future frames.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        dilation: int | tuple = 1,
        bias: bool = True,
    ):
        super().__init__()

        # Normalize to tuples: (T, H, W)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        # How much to pad on each side for each dim
        # For spatial: symmetric → pad = (k-1)//2 each side
        # For time:    causal   → pad = (k-1)*d on left, 0 on right
        kt, kh, kw = kernel_size
        dt, dh, dw = dilation

        self.time_pad = (kt - 1) * dt   # left-pad only

        pad_h = (kh - 1) * dh // 2
        pad_w = (kw - 1) * dw // 2

        # We handle time padding manually in forward();
        # pass spatial padding directly to Conv3d
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(0, pad_h, pad_w),   # time=0, we do it manually
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        if self.time_pad > 0:
            # F.pad pads last dims first, so pad = (W_left, W_right, H_left, H_right, T_left, T_right)
            x = F.pad(x, (0, 0, 0, 0, self.time_pad, 0))
        return self.conv(x)


class CausalConv3d_1x1(nn.Module):
    """Pointwise (1×1×1) conv — no causal padding needed, but keeps the API consistent."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)