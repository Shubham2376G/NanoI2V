# quick smoke test — paste into a notebook or test script

import torch
from vae.conv import CausalConv3d

torch.manual_seed(42)

# ------------------------------------------------------------------
# Create module
# ------------------------------------------------------------------
conv = CausalConv3d(
    in_channels=4,
    out_channels=8,
    kernel_size=(3, 3, 3),
)

# ------------------------------------------------------------------
# Dummy input: (B, C, T, H, W)
# ------------------------------------------------------------------
x = torch.randn(2, 4, 8, 32, 32)

print("=== CausalConv3d Smoke Test ===")
print(f"Module:       {conv.__class__.__name__}")
print(f"Input shape:  {tuple(x.shape)}")

# ------------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------------
y = conv(x)

print(f"Output shape: {tuple(y.shape)}")

# ------------------------------------------------------------------
# Shape checks
# ------------------------------------------------------------------
assert y.shape[0] == x.shape[0], "Batch size changed!"
assert y.shape[1] == 8, "Unexpected number of output channels!"
assert y.shape[2] == x.shape[2], "Temporal dimension should be preserved!"
assert y.shape[3] == x.shape[3], "Height should be preserved!"
assert y.shape[4] == x.shape[4], "Width should be preserved!"

print("\n✓ Forward pass successful.")
print("✓ Output dimensions are as expected.")
print(f"  Batch    : {x.shape[0]} → {y.shape[0]}")
print(f"  Channels : {x.shape[1]} → {y.shape[1]}")
print(f"  Time     : {x.shape[2]} → {y.shape[2]} (preserved)")
print(f"  Height   : {x.shape[3]} → {y.shape[3]} (preserved)")
print(f"  Width    : {x.shape[4]} → {y.shape[4]} (preserved)")

print("\n✅ Smoke test passed.")