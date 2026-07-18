import torch
from vae.vae import VAE3D

torch.manual_seed(42)

# ------------------------------------------------------------------
# Create VAE
# ------------------------------------------------------------------
vae = VAE3D(
    base_channels=64,
    latent_channels=16,
    ch_mult=(1, 2, 4),
    num_res_blocks=2,
    spatial_ds=2,
    temporal_ds=2,
    kl_weight=1e-6,
    recon_loss="l1",
)

num_params = sum(p.numel() for p in vae.parameters())

print("=== VAE3D Smoke Test ===")
print(f"Trainable parameters: {num_params / 1e6:.2f}M")

# ------------------------------------------------------------------
# Dummy input video
# ------------------------------------------------------------------
x = torch.randn(1, 3, 17, 128, 128).clamp(-1, 1)

print(f"\nInput video: {tuple(x.shape)}")

# ------------------------------------------------------------------
# Training forward pass
# ------------------------------------------------------------------
out = vae(x)

assert out.z.shape == (1, 16, 5, 32, 32)
assert out.recon.shape == x.shape

print("\n✓ Training Forward Pass")
print(f"  Input video          : {tuple(x.shape)}")
print("      ↓")
print("  Encoder")
print("      ↓")
print("  μ, logσ²")
print("      ↓")
print("  Reparameterization")
print("      ↓")
print(f"  Sampled latent (z)   : {tuple(out.z.shape)}")
print("      ↓")
print("  Decoder")
print("      ↓")
print(f"  Reconstruction       : {tuple(out.recon.shape)}")

# ------------------------------------------------------------------
# Losses
# ------------------------------------------------------------------
print("\n✓ Training Losses")
print(f"  Reconstruction Loss : {out.loss_recon.item():.6f}")
print(f"  KL Divergence       : {out.loss_kl.item():.6f}")
print(f"  Total Loss          : {out.loss_total.item():.6f}")

# ------------------------------------------------------------------
# Backward pass
# ------------------------------------------------------------------
out.loss_total.backward()

print("\n✓ Backward Pass")
print("  Gradients computed successfully.")

# ------------------------------------------------------------------
# Inference API
# ------------------------------------------------------------------
vae.set_inference_mode()

with torch.no_grad():
    z = vae.encode_mean(x)
    recon = vae.decode(z)

assert z.shape == (1, 16, 5, 32, 32)
assert recon.shape == x.shape
assert recon.min() >= -1.0 and recon.max() <= 1.0

print("\n✓ Inference API")
print(f"  Deterministic latent : {tuple(z.shape)}")
print(f"  Decoded video        : {tuple(recon.shape)}")
print(f"  Output range         : [{recon.min().item():.3f}, {recon.max().item():.3f}]")

print("\n✅ VAE3D smoke test passed.")