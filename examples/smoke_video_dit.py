import torch
from dit.dit import VideoDiT

print("=== VideoDiT Smoke Test ===")

# ------------------------------------------------------------
# Small configuration for a fast smoke test
# ------------------------------------------------------------

dit = VideoDiT(
    latent_channels=16,
    latent_t=5,
    latent_h=32,
    latent_w=32,
    patch_t=1,
    patch_h=2,
    patch_w=2,
    hidden_dim=512,
    num_heads=8,
    num_layers=4,
    cond_dim=512,
    text_dim=1024,
    clip_dim=1024,
)

params = dit.count_params()

print(f"\n✓ Model Parameters")
print(f"  Total      : {params['total_M']:.2f}M")
print(f"  Trainable  : {params['trainable_M']:.2f}M")

# ------------------------------------------------------------
# Dummy inputs
# ------------------------------------------------------------

B = 2

z_t = torch.randn(B, 16, 5, 32, 32)          # noisy latent
z_img = torch.randn(B, 16, 1, 32, 32)        # first-frame latent

t = torch.rand(B)

txt_tokens = torch.randn(B, 77, 1024)
img_tokens = torch.randn(B, 257, 1024)

print("\n✓ Inputs")
print(f"  Noisy latent        : {tuple(z_t.shape)}")
print(f"  Image latent        : {tuple(z_img.shape)}")
print(f"  Timesteps           : {tuple(t.shape)}")
print(f"  Text tokens         : {tuple(txt_tokens.shape)}")
print(f"  Image tokens        : {tuple(img_tokens.shape)}")

# ------------------------------------------------------------
# Image latent tiling (I2V conditioning)
# ------------------------------------------------------------

z_img_tiled = z_img.expand_as(z_t)

print("\n✓ Image Conditioning")
print(f"  Original image latent : {tuple(z_img.shape)}")
print(f"  Tiled across time     : {tuple(z_img_tiled.shape)}")
print("  The first-frame latent is repeated across all latent frames")
print("  and concatenated with the noisy video latent before patchifying.")

# ------------------------------------------------------------
# Patchify
# ------------------------------------------------------------

z_in = torch.cat([z_t, z_img_tiled], dim=1)
tokens = dit.patchify(z_in)

print("\n✓ Patchify")
print(f"  Concatenated latent : {tuple(z_in.shape)}")
print(f"  Latent tokens       : {tuple(tokens.shape)}")
print(f"  Total tokens        : {tokens.shape[1]} = 5 × 16 × 16")

# ------------------------------------------------------------
# Conditioning
# ------------------------------------------------------------

c, cond_tokens = dit.build_conditioning(
    t,
    txt_tokens,
    img_tokens,
)

print("\n✓ Conditioning")
print(f"  Global conditioning vector : {tuple(c.shape)}")
print("    • timestep embedding")
print("    • pooled text embedding")
print("    • pooled image embedding")
print("    • used by adaLN-Zero in every DiT block")

print(f"\n  Cross-attention tokens     : {tuple(cond_tokens.shape)}")
print("    • 77 T5 text tokens")
print("    • 257 CLIP image tokens")
print("    • attended by every transformer block")

# ------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------

v_pred = dit(
    z_t,
    t,
    txt_tokens,
    img_tokens,
    z_img,
)

print("\n✓ DiT Forward Pass")
print("  Noisy video latent")
print("          +")
print("  Image latent (first frame)")
print("          ↓")
print("  Tile image latent across time")
print("          ↓")
print("  Concatenate channels")
print("          ↓")
print("  Patchify into latent tokens")
print("          ↓")
print("  Add 3D RoPE position information")
print("          ↓")
print("  4 × DiT Blocks")
print("      • adaLN-Zero modulation")
print("      • RoPE self-attention")
print("      • Cross-attention")
print("      • Feed-forward network")
print("          ↓")
print("  Final layer")
print("          ↓")
print("  Unpatchify")
print("          ↓")
print("  Predicted velocity")

print(f"\n  Output velocity : {tuple(v_pred.shape)}")

assert v_pred.shape == z_t.shape



# ------------------------------------------------------------
# CFG null conditioning
# ------------------------------------------------------------

null_txt, null_img = dit.get_null_conditioning(
    B,
    77,
    257,
    z_t.device,
)

v_uncond = dit(
    z_t,
    t,
    null_txt,
    null_img,
    z_img,
)

assert v_uncond.shape == z_t.shape

print("\n✓ Classifier-Free Guidance")
print(f"  Null text tokens  : {tuple(null_txt.shape)}")
print(f"  Null image tokens : {tuple(null_img.shape)}")
print(f"  Unconditional prediction : {tuple(v_uncond.shape)}")

# ------------------------------------------------------------
# CFG combination
# ------------------------------------------------------------

cfg_scale = 7.0
v_guided = v_uncond + cfg_scale * (v_pred - v_uncond)

assert v_guided.shape == z_t.shape

print(f"\n✓ CFG Combination (scale = {cfg_scale})")
print(f"  Guided velocity : {tuple(v_guided.shape)}")

# ------------------------------------------------------------
# Backward
# ------------------------------------------------------------

loss = v_pred.mean()
loss.backward()

print("\n✓ Backward Pass")
print("  Gradients propagated successfully.")

print("\n✅ All VideoDiT smoke tests passed.")