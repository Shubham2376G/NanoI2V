# vae/encoder.py
import torch
import torch.nn as nn
from vae.conv import CausalConv3d, CausalConv3d_1x1
from vae.blocks import (
    ResBlock3D,
    SpatialDownsample,
    TemporalDownsample,
    SpatialAttention,
    get_norm,
)


class Encoder3D(nn.Module):
    """
    Causal 3D VAE encoder.

    Takes raw video:  (B, in_channels, T, H, W)
    Returns:          (B, 2*latent_channels, T', H', W')


    Compression schedule (default):
        Spatial:  2× down × 2 stages  →  H/4, W/4
        Temporal: 2× down × 2 stages  →  (T-1)/4 + 1  (first frame kept)

    Channel schedule with ch_mult=(1,2,4):
        base_channels=128
        stage 0: 128 channels
        stage 1: 256 channels
        stage 2: 512 channels  ← bottleneck enters here
    """

    def __init__(
        self,
        in_channels: int = 3,               # RGB input
        base_channels: int = 128,           # channels at first stage
        latent_channels: int = 16,          # VAE latent dim (C')
        ch_mult: tuple = (1, 2, 4),         # channel multiplier per stage
        num_res_blocks: int = 2,            # ResBlocks per stage
        spatial_ds: int = 2,               # how many spatial downsamples
        temporal_ds: int = 2,              # how many temporal downsamples
        num_groups: int = 32,
    ):
        super().__init__()

        self.ch_mult = ch_mult
        self.spatial_ds = spatial_ds
        self.temporal_ds = temporal_ds

        # ── Stage channel sizes ───────────────────────────────────────
        stage_channels = [base_channels * m for m in ch_mult]
        # e.g. [128, 256, 512] for ch_mult=(1,2,4)


        # ── Input projection ─────────────────────────────────────────
        # Map RGB (3) → base_channels
        self.in_conv = CausalConv3d(in_channels, stage_channels[0], kernel_size=3)

        # ── Downsampling stages ───────────────────────────────────────
        # Each stage: N ResBlocks + spatial/temporal downsample (optional)
        self.stages = nn.ModuleList()

        in_ch = stage_channels[0]
        for i, out_ch in enumerate(stage_channels):
            stage = nn.ModuleList()

            # ResBlocks (first one handles channel change)
            for j in range(num_res_blocks):
                block_in_ch = in_ch if j == 0 else out_ch
                stage.append(ResBlock3D(block_in_ch, out_ch, num_groups))
            in_ch = out_ch


            # We downsample after all stages except the last
            ds_block = nn.ModuleDict()

            if i < spatial_ds:
                ds_block["spatial"] = SpatialDownsample(out_ch, out_ch)

            if i < temporal_ds:
                ds_block["temporal"] = TemporalDownsample(out_ch, out_ch, keep_first=True)

            self.stages.append(nn.ModuleList([stage, nn.ModuleDict(ds_block)]))

        # ── Middle / bottleneck block ─────────────────────────────────
        # ResBlock → SpatialAttention → ResBlock
        # This is where the global spatial context is captured
        bottleneck_ch = stage_channels[-1]
        self.mid_block1   = ResBlock3D(bottleneck_ch, bottleneck_ch, num_groups)
        self.mid_attn     = SpatialAttention(bottleneck_ch, num_groups)
        self.mid_block2   = ResBlock3D(bottleneck_ch, bottleneck_ch, num_groups)

        # ── Output projection ─────────────────────────────────────────
        # Norm → SiLU → Conv → 2 * latent_channels (mean + logvar)
        self.out_norm = get_norm(bottleneck_ch, num_groups)
        self.out_act  = nn.SiLU()
        self.out_conv = CausalConv3d_1x1(bottleneck_ch, 2 * latent_channels)
        
        nn.init.zeros_(self.out_conv.conv.bias)
        nn.init.normal_(self.out_conv.conv.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, T, H, W)  raw video in [-1, 1]

        Returns:
            (B, 2*latent_channels, T', H', W')
            First  latent_channels → mean
            Second latent_channels → log_variance
        """
        # Input projection
        x = self.in_conv(x)

        # Downsampling stages
        for stage_resblocks, ds_ops in self.stages:

            # ResBlocks
            for resblock in stage_resblocks:
                x = resblock(x)

            # Spatial downsample first, then temporal
            # Order matters: spatial changes H,W; temporal changes T
            if "spatial" in ds_ops:
                x = ds_ops["spatial"](x)

            if "temporal" in ds_ops:
                x = ds_ops["temporal"](x)

        # Bottleneck
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        # Output projection → mean + logvar
        x = self.out_act(self.out_norm(x))
        x = self.out_conv(x)

        return x   # (B, 2*latent_channels, T', H', W')

    def get_latent_shape(self, video_shape: tuple) -> tuple:
        """
        Utility: compute latent shape without running a forward pass.
        video_shape = (B, C, T, H, W)
        """
        B, C, T, H, W = video_shape

        H_out = H // (2 ** self.spatial_ds)
        W_out = W // (2 ** self.spatial_ds)

        # Temporal: first frame kept, rest compressed
        # Each 2× downsample: T' = 1 + (T-1) // 2
        T_out = T
        for _ in range(self.temporal_ds):
            T_out = 1 + (T_out - 1) // 2

        return (B, self.out_conv.conv.out_channels, T_out, H_out, W_out)