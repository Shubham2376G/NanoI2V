import torch
from dit.rope import RoPE3D, make_3d_pos_ids
from dit.blocks import (
    DiTBlock,
    TimestepEmbedder,
    FinalLayer,
)

torch.manual_seed(42)

# ------------------------------------------------------------------
# Model configuration
# ------------------------------------------------------------------
B = 2

# Latent volume after VAE:
# (B, 16, 5, 32, 32)
#
# Using latent patches of size (1, 2, 2):
#   Temporal patches : 5 / 1  = 5
#   Height patches   : 32 / 2 = 16
#   Width patches    : 32 / 2 = 16
#
# Total latent tokens:
#   5 × 16 × 16 = 1280
#
T, H, W = 5, 16, 16
N = T * H * W

D = 512          # transformer hidden dimension
D_txt = 1024     # text/image conditioning dimension
D_cond = 512
heads = 8

rope = RoPE3D(head_dim=D // heads)

t_embed = TimestepEmbedder(hidden_dim=D_cond)

block = DiTBlock(
    hidden_dim=D,
    num_heads=heads,
    cond_dim=D_cond,
    text_dim=D_txt,
    rope=rope,
)

final = FinalLayer(
    hidden_dim=D,
    patch_dim=16 * 2 * 2 * 2,
    cond_dim=D_cond,
)

# ------------------------------------------------------------------
# Dummy inputs
# ------------------------------------------------------------------
x = torch.randn(B, N, D)

t = torch.rand(B)

# 77 T5 tokens + 257 image tokens
cond_tokens = torch.randn(B, 334, D_txt)

# Each latent token gets one (time, row, column) coordinate.
# Shape:
#   (1280, 3) → (B, 1280, 3)
#
# Example:
#   (0,0,0)
#   (0,0,1)
#   ...
#   (0,15,15)
#   (1,0,0)
#   ...
#   (4,15,15)
#
pos_ids = make_3d_pos_ids(T, H, W).expand(B, -1, -1)

print("=== DiT Block Smoke Test ===")

print(f"Latent tokens      : {tuple(x.shape)}")
print(f"  • Batch size = {B}")
print(f"  • {N} = {T} × {H} × {W} latent patches")
print(f"  • Each token is a {D}-dim embedding")

print(f"\nCondition tokens   : {tuple(cond_tokens.shape)}")
print("  • 77 text tokens (T5)")
print("  • 257 image conditioning tokens")
print("  • 334 conditioning tokens total")

print(f"\nPosition IDs       : {tuple(pos_ids.shape)}")
print("  • One (time, row, column) coordinate per latent token")
print("  • Used by 3D RoPE for spatio-temporal attention")

# ------------------------------------------------------------------
# Timestep embedding
# ------------------------------------------------------------------
c = t_embed(t)

assert c.shape == (B, D_cond)

print("\n✓ Timestep Embedding")
print(f"  Timesteps         : {tuple(t.shape)}")
print(f"  Conditioning      : {tuple(c.shape)}")

# ------------------------------------------------------------------
# DiT block
# ------------------------------------------------------------------
out = block(x, c, cond_tokens, pos_ids)

assert out.shape == x.shape

print("\n✓ DiT Block")
print("  Latent tokens")
print("      ↓")
print("  adaLN-Zero modulation")
print("      ↓")
print("  RoPE Self-Attention")
print("      ↓")
print("  Cross-Attention (Text + Image)")
print("      ↓")
print("  Feed-Forward (MLP)")
print("      ↓")
print(f"  Output tokens     : {tuple(out.shape)}")

# ------------------------------------------------------------------
# Final layer
# ------------------------------------------------------------------
out_final = final(out, c)

assert out_final.shape == (B, N, 128)

print("\n✓ Final Layer")
print("  adaLN-Zero modulation")
print("      ↓")
print("  Linear projection")
print("      ↓")
print(f"  Predicted patches : {tuple(out_final.shape)}")

# ------------------------------------------------------------------
# adaLN-Zero initialization
# ------------------------------------------------------------------
with torch.no_grad():

    x_test = torch.randn(1, N, D)

    c_test = torch.zeros(1, D_cond)

    cond_test = torch.randn(1, 334, D_txt)

    pos_test = make_3d_pos_ids(T, H, W)

    out_test = block(
        x_test,
        c_test,
        cond_test,
        pos_test,
    )

identity = torch.allclose(out_test, x_test, atol=1e-5)

assert identity

print("\n✓ adaLN-Zero Initialization")
print(f"  Block behaves as identity at initialization : {identity}")

print("\n✅ DiT block smoke test passed.")