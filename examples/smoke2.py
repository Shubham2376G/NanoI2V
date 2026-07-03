import torch
from vae.blocks import (
    ResBlock3D,
    SpatialDownsample,
    TemporalDownsample,
    SpatialUpsample,
    TemporalUpsample,
)

torch.manual_seed(42)

# ------------------------------------------------------------------
# Dummy input: (B, C, T, H, W)
# ------------------------------------------------------------------
x = torch.randn(2, 64, 8, 32, 32)

print("=== VAE Blocks Smoke Test ===")
print(f"Input shape: {tuple(x.shape)}")

# ------------------------------------------------------------------
# ResBlock (same channels)
# ------------------------------------------------------------------
block = ResBlock3D(64, 64)
y = block(x)

assert y.shape == (2, 64, 8, 32, 32)

print("\n✓ ResBlock3D (64 → 64)")
print(f"  {tuple(x.shape)} → {tuple(y.shape)}")

# ------------------------------------------------------------------
# ResBlock (channel expansion)
# ------------------------------------------------------------------
block = ResBlock3D(64, 128)
y = block(x)

assert y.shape == (2, 128, 8, 32, 32)

print("\n✓ ResBlock3D (64 → 128)")
print(f"  {tuple(x.shape)} → {tuple(y.shape)}")

# ------------------------------------------------------------------
# Spatial downsampling
# ------------------------------------------------------------------
sd = SpatialDownsample(64, 64)
y = sd(x)

assert y.shape == (2, 64, 8, 16, 16)

print("\n✓ SpatialDownsample")
print(f"  {tuple(x.shape)} → {tuple(y.shape)}")

# ------------------------------------------------------------------
# Temporal downsampling
# ------------------------------------------------------------------
td = TemporalDownsample(64, 64)
y = td(x)

assert y.shape == (2, 64, 5, 32, 32)

print("\n✓ TemporalDownsample")
print(f"  Input               : {tuple(x.shape)}")
print("      ↓")
print("  Keep frame 0")
print("      ↓")
print("  Downsample frames 1...T")
print("      ↓")
print(f"  Output              : {tuple(y.shape)}")

# ------------------------------------------------------------------
# Spatial upsampling
# ------------------------------------------------------------------
su = SpatialUpsample(64, 64)
y = su(sd(x))

assert y.shape == (2, 64, 8, 32, 32)

print("\n✓ SpatialUpsample")
print(f"  {tuple(sd(x).shape)} → {tuple(y.shape)}")

# ------------------------------------------------------------------
# Temporal upsampling
# ------------------------------------------------------------------
tu = TemporalUpsample(64, 64)
y = tu(td(x))

assert y.shape == (2, 64, 9, 32, 32)

print("\n✓ TemporalUpsample")
print(f"  Input               : {tuple(td(x).shape)}")
print("      ↓")
print("  Keep frame 0")
print("      ↓")
print("  Upsample frames 1...T")
print("      ↓")
print(f"  Output              : {tuple(y.shape)}")

print("\n✅ All VAE block smoke tests passed.")