import torch
from flow.scheduler import FlowMatchingScheduler

torch.manual_seed(42)

# ------------------------------------------------------------------
# Dummy DiT
# ------------------------------------------------------------------
# Returns random velocity predictions.
# We're only verifying the scheduler logic here.
class DummyDiT(torch.nn.Module):
    def forward(self, x_t, t, c_txt, c_img):
        return torch.randn_like(x_t)


model = DummyDiT()

scheduler = FlowMatchingScheduler(
    cfg_scale=7.0,
    p_uncond=0.1,
)

# ------------------------------------------------------------------
# Dummy latent video + conditioning
# ------------------------------------------------------------------
B, C, T, H, W = 2, 16, 5, 32, 32

x_0 = torch.randn(B, C, T, H, W)

c_txt = torch.randn(B, 77, 1024)
c_img = torch.randn(B, 16, 1024)

null_txt = torch.zeros(1, 77, 1024)
null_img = torch.zeros(1, 16, 1024)

print("=== Flow Matching Scheduler Smoke Test ===")

print(f"Latent video      : {tuple(x_0.shape)}")
print(f"Text conditioning : {tuple(c_txt.shape)}")
print(f"Image conditioning: {tuple(c_img.shape)}")

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
out = scheduler.training_loss(
    model,
    x_0,
    c_txt,
    c_img,
    null_txt,
    null_img,
)

assert out.loss.ndim == 0
assert out.x_t.shape == x_0.shape
assert out.t.shape == (B,)

print("\n✓ Training Step")
print("  Clean latent (x₀)")
print("      ↓")
print("  Sample timestep t")
print("      ↓")
print("  Add flow noise")
print("      ↓")
print("  Noisy latent (xₜ)")
print("      ↓")
print("  Dummy DiT predicts velocity")
print("      ↓")
print("  Flow Matching loss")

print(f"\n  Loss            : {out.loss.item():.6f}")
print(f"  Noisy latent    : {tuple(out.x_t.shape)}")
print(f"  Timesteps       : {tuple(out.t.shape)}")

# ------------------------------------------------------------------
# Inference (Euler)
# ------------------------------------------------------------------
print("\n✓ Euler Sampling")

x_gen = scheduler.euler_sample(
    model,
    shape=(B, C, T, H, W),
    c_txt=c_txt,
    c_img=c_img,
    null_c_txt=null_txt,
    null_c_img=null_img,
    num_steps=5,
    device=torch.device("cpu"),
    verbose=True,
)

assert x_gen.shape == (B, C, T, H, W)

print("\n  Initial Gaussian noise")
print("      ↓")
print("  Euler integration (5 steps)")
print("      ↓")
print("  Classifier-Free Guidance (CFG)")
print("      ↓")
print(f"  Generated latent : {tuple(x_gen.shape)}")

print("\n✅ Flow Matching scheduler smoke test passed.")