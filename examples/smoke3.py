import torch
from vae.encoder import Encoder3D
from vae.decoder import Decoder3D

torch.manual_seed(42)

# ------------------------------------------------------------------
# Shared configuration
# ------------------------------------------------------------------
cfg = dict(
    base_channels=64,
    latent_channels=16,
    ch_mult=(1, 2, 4),
    num_res_blocks=2,
    num_groups=32,
)

encoder = Encoder3D(
    in_channels=3,
    spatial_ds=2,
    temporal_ds=2,
    **cfg,
)

decoder = Decoder3D(
    out_channels=3,
    spatial_us=2,
    temporal_us=2,
    **cfg,
)

# ------------------------------------------------------------------
# Dummy video: (B, C, T, H, W)
# ------------------------------------------------------------------
x = torch.randn(1, 3, 17, 128, 128)

print("=== Encoder & Decoder Smoke Test ===")
print(f"Input video: {tuple(x.shape)}")

# ------------------------------------------------------------------
# Encode
# ------------------------------------------------------------------
enc_out = encoder(x)

assert enc_out.shape == (1, 32, 5, 32, 32)

print("\n✓ Encoder")
print(f"  Input video          : {tuple(x.shape)}")
print("      ↓")
print("  2× SpatialDownsample : 128×128 → 32×32")
print("  2× TemporalDownsample: 17 → 9 → 5")
print("      ↓")
print("  Bottleneck")
print("    • ResBlock3D")
print("    • SpatialAttention")
print("    • ResBlock3D")
print("      ↓")
print(f"  Latent distribution  : {tuple(enc_out.shape)}")

# ------------------------------------------------------------------
# Split latent distribution
# ------------------------------------------------------------------
mean, logvar = enc_out.chunk(2, dim=1)

assert mean.shape == (1, 16, 5, 32, 32)
assert logvar.shape == (1, 16, 5, 32, 32)

print("\n✓ Latent Distribution")
print(f"  Mean                : {tuple(mean.shape)}")
print(f"  Log Variance        : {tuple(logvar.shape)}")

# ------------------------------------------------------------------
# Sample latent (reparameterization trick)
# ------------------------------------------------------------------
std = torch.exp(0.5 * logvar)
z = mean + std * torch.randn_like(std)

assert z.shape == mean.shape

print("\n✓ Reparameterization")
print("  z = μ + σ · ε")
print(f"  Sampled latent      : {tuple(z.shape)}")

# ------------------------------------------------------------------
# Decode
# ------------------------------------------------------------------
recon = decoder(z)

assert recon.shape == x.shape

print("\n✓ Decoder")
print(f"  Input latent        : {tuple(z.shape)}")
print("      ↓")
print("  Bottleneck")
print("    • ResBlock3D")
print("    • SpatialAttention")
print("    • ResBlock3D")
print("      ↓")
print("  2× TemporalUpsample : 5 → 9 → 17")
print("  2× SpatialUpsample  : 32×32 → 128×128")
print("      ↓")
print(f"  Reconstructed video : {tuple(recon.shape)}")

# ------------------------------------------------------------------
# Output checks
# ------------------------------------------------------------------
assert recon.min() >= -1.0 and recon.max() <= 1.0

print("\n✓ Output")
print(f"  Range               : [{recon.min().item():.3f}, {recon.max().item():.3f}]")
print("  Matches input shape : ✓")


print("\n✅ Encoder and Decoder smoke test passed.")