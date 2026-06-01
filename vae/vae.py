# vae/vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from vae.encoder import Encoder3D
from vae.decoder import Decoder3D


# ─────────────────────────────────────────────
# Output container — cleaner than returning tuples
# ─────────────────────────────────────────────

@dataclass
class VAEOutput:
    z: torch.Tensor            # sampled latent  (B, C', T', H', W')
    mean: torch.Tensor         # encoder mean    (B, C', T', H', W')
    logvar: torch.Tensor       # encoder logvar  (B, C', T', H', W')
    recon: torch.Tensor        # decoded video   (B, 3,  T,  H,  W)
    loss_recon: torch.Tensor   # scalar
    loss_kl: torch.Tensor      # scalar
    loss_total: torch.Tensor   # scalar


# ─────────────────────────────────────────────
# Diagonal Gaussian — wraps mean + logvar
# ─────────────────────────────────────────────

class DiagonalGaussian(nn.Module):
    """
    Represents q(z|x) = N(mean, diag(std²)).
    'Diagonal' means each latent dim is independent —
    the covariance matrix is diagonal (no cross-correlations).
    This is the standard VAE assumption.
    """

    def __init__(self, deterministic: bool = False):
        super().__init__()
        # deterministic=True → always return mean, no sampling
        # useful at inference when you want a stable encode
        self.deterministic = deterministic

    def sample(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        if self.deterministic:
            return mean

        # Clamp logvar for numerical stability:
        # exp(logvar) blows up if logvar is large
        logvar = logvar.clamp(-10.0, 10.0)
        std    = torch.exp(0.5 * logvar)

        # Reparameterization trick
        eps = torch.randn_like(mean)
        return mean + std * eps

    def kl_loss(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL(q(z|x) || p(z)) where p(z) = N(0, I).
        Closed-form solution for diagonal Gaussians:
            -0.5 * sum(1 + logvar - mean² - exp(logvar))
        Averaged over all dimensions and batch.
        """
        # 1. Upcast to float32 to survive the massive sum and pow(2)
        mean_f32 = mean.float()
        logvar_f32 = logvar.float().clamp(-10.0, 10.0)
        
        # 2. Calculate KL in float32 (safe from 65,504 limit)
        kl = -0.5 * (1.0 + logvar_f32 - mean_f32.pow(2) - logvar_f32.exp())
        
        # 3. Summing in float32 is now completely safe
        return kl.flatten(1).sum(dim=1).mean()


# ─────────────────────────────────────────────
# Full VAE
# ─────────────────────────────────────────────

class VAE3D(nn.Module):
    """
    Full 3D causal VAE.
    Encoder → DiagonalGaussian → Decoder.

    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        latent_channels: int = 16,
        ch_mult: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        spatial_ds: int = 2,
        temporal_ds: int = 2,
        num_groups: int = 32,
        kl_weight: float = 1e-6,        # β in L = L_recon + β * L_KL
        recon_loss: str = "l1",         # "l1" or "l2"
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.kl_weight = kl_weight
        self.recon_loss = recon_loss

        # Shared config for encoder and decoder
        shared = dict(
            base_channels=base_channels,
            latent_channels=latent_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            num_groups=num_groups,
        )

        self.encoder = Encoder3D(
            in_channels=in_channels,
            spatial_ds=spatial_ds,
            temporal_ds=temporal_ds,
            **shared,
        )

        self.decoder = Decoder3D(
            out_channels=in_channels,
            spatial_us=spatial_ds,
            temporal_us=temporal_ds,
            **shared,
        )

        self.posterior = DiagonalGaussian(deterministic=False)

    # ── Core ops ─────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent, sampling from q(z|x).
        At inference: use encode_mean() for deterministic latent.

        Args:
            x: (B, 3, T, H, W)  normalized to [-1, 1]
        Returns:
            z: (B, latent_channels, T', H', W')
        """
        enc_out        = self.encoder(x)
        mean, logvar   = enc_out.chunk(2, dim=1)
        z              = self.posterior.sample(mean, logvar)
        return z


    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent back to video.

        Args:
            z: (B, latent_channels, T', H', W')
        Returns:
            (B, 3, T, H, W)  in [-1, 1]
        """
        return self.decoder(z)

    # ── Training forward ─────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> VAEOutput:
        """
        Full forward pass for training.
        Encodes, samples z, decodes, computes losses.

        Args:
            x: (B, 3, T, H, W)  normalized to [-1, 1]
        Returns:
            VAEOutput with z, mean, logvar, recon, and losses
        """
        # 1. Encode → mean + logvar
        enc_out      = self.encoder(x)
        mean, logvar = enc_out.chunk(2, dim=1)

        # 2. Sample z via reparameterization
        z = self.posterior.sample(mean, logvar)

        # 3. Decode
        recon = self.decoder(z)

        # 4. Reconstruction loss
        if self.recon_loss == "l1":
            # L1 is more robust to outliers, preferred for video
            loss_recon = F.l1_loss(recon, x)
        else:
            loss_recon = F.mse_loss(recon, x)

        # 5. KL loss
        loss_kl = self.posterior.kl_loss(mean, logvar)

        # 6. Total loss
        loss_total = loss_recon + self.kl_weight * loss_kl

        return VAEOutput(
            z          = z,
            mean       = mean,
            logvar     = logvar,
            recon      = recon,
            loss_recon = loss_recon,
            loss_kl    = loss_kl,
            loss_total = loss_total,
        )

