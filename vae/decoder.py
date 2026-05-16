# vae/decoder.py
import torch
import torch.nn as nn

from vae.conv import CausalConv3d, CausalConv3d_1x1
from vae.blocks import (
    ResBlock3D,
    SpatialUpsample,
    TemporalUpsample,
    SpatialAttention,
    get_norm,
)


class Decoder3D(nn.Module):
    """
    Causal 3D VAE decoder.

    Takes sampled latent:  (B, latent_channels, T', H', W')
    Returns reconstructed: (B, out_channels,    T,  H,  W)

    Must be initialized with the same hyperparameters as Encoder3D
    so the channel schedule and upsample stages match exactly.
    """

    def __init__(
        self,
        out_channels: int = 3,              # RGB output
        base_channels: int = 128,           # channels at finest stage
        latent_channels: int = 16,          # must match encoder
        ch_mult: tuple = (1, 2, 4),         # same as encoder
        num_res_blocks: int = 2,            # same as encoder
        spatial_us: int = 2,               # spatial upsamples (= encoder spatial_ds)
        temporal_us: int = 2,              # temporal upsamples (= encoder temporal_ds)
        num_groups: int = 32,
    ):
        super().__init__()

        self.spatial_us = spatial_us
        self.temporal_us = temporal_us

        # Channel sizes — reversed compared to encoder
        # encoder: [128, 256, 512]  (low → high as we go deeper)
        # decoder: [512, 256, 128]  (high → low as we go up)
        stage_channels = [base_channels * m for m in ch_mult]  # [128, 256, 512]
        stage_channels_rev = list(reversed(stage_channels))     # [512, 256, 128]

        bottleneck_ch = stage_channels_rev[0]   # 512

        # ── Input projection ──────────────────────────────────────────
        # latent_channels → bottleneck channels
        # 1×1×1 conv — no spatial/temporal context needed here
        self.in_conv = CausalConv3d_1x1(latent_channels, bottleneck_ch)

        # ── Bottleneck ────────────────────────────────────────────────
        # Identical structure to encoder bottleneck
        # ResBlock → SpatialAttention → ResBlock
        self.mid_block1 = ResBlock3D(bottleneck_ch, bottleneck_ch, num_groups)
        self.mid_attn   = SpatialAttention(bottleneck_ch, num_groups)
        self.mid_block2 = ResBlock3D(bottleneck_ch, bottleneck_ch, num_groups)

        # ── Upsampling stages ─────────────────────────────────────────
        # Mirror of encoder stages but reversed
        # stage_channels_rev: [512, 256, 128]
        #   stage 0: 512 → 512  (no upsample — mirrors encoder stage 2)
        #   stage 1: 512 → 256  + TemporalUp + SpatialUp
        #   stage 2: 256 → 128  + TemporalUp + SpatialUp
        self.stages = nn.ModuleList()

        in_ch = bottleneck_ch
        num_stages = len(stage_channels_rev)

        for i, out_ch in enumerate(stage_channels_rev):
            stage = nn.ModuleList()

            # ResBlocks (first handles channel change)
            for j in range(num_res_blocks):
                block_in_ch = in_ch if j == 0 else out_ch
                stage.append(ResBlock3D(block_in_ch, out_ch, num_groups))
            in_ch = out_ch

            # Upsample after this stage?
            # We upsample after all stages except the first
            # (encoder had no downsample at the last stage = deepest)
            # so decoder has no upsample at the first stage = shallowest here
            us_ops = nn.ModuleDict()

            # stage index from the END tells us how many ups to apply
            # i=0 → deepest (no upsample), i=1 → 1st upsample, i=2 → 2nd
            ups_index = i  # how many upsamples have been applied so far

            if ups_index > 0 and (ups_index - 1) < temporal_us:
                # Temporal UP first (reverse of encoder: spatial first when encoding)
                us_ops["temporal"] = TemporalUpsample(out_ch, out_ch, keep_first=True)

            if ups_index > 0 and (ups_index - 1) < spatial_us:
                # Spatial UP after temporal
                us_ops["spatial"] = SpatialUpsample(out_ch, out_ch)

            self.stages.append(nn.ModuleList([stage, nn.ModuleDict(us_ops)]))

        # ── Output projection ─────────────────────────────────────────
        # Norm → SiLU → Conv → out_channels (3 for RGB)
        # tanh at the end maps output to [-1, 1] to match
        # the normalized video input range
        self.out_norm = get_norm(stage_channels_rev[-1], num_groups)
        self.out_act  = nn.SiLU()
        self.out_conv = CausalConv3d(stage_channels_rev[-1], out_channels, kernel_size=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_channels, T', H', W')  sampled latent

        Returns:
            (B, out_channels, T, H, W)  reconstructed video in [-1, 1]
        """
        # Input projection: latent → bottleneck channels
        x = self.in_conv(z)

        # Bottleneck
        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        # Upsampling stages
        for stage_resblocks, us_ops in self.stages:

            # ResBlocks first — learn features before upsampling
            for resblock in stage_resblocks:
                x = resblock(x)

            # Temporal UP first, then spatial
            # (reverse of encoder: spatial down first, then temporal)
            if "temporal" in us_ops:
                x = us_ops["temporal"](x)

            if "spatial" in us_ops:
                x = us_ops["spatial"](x)

        # Output projection
        x = self.out_act(self.out_norm(x))
        x = self.out_conv(x)

        # Clamp to [-1, 1] — videos are normalized to this range
        return torch.tanh(x)