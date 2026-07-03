# smoke_train_vae.py

import torch
import numpy as np
from types import SimpleNamespace

from vae.vae import VAE3D
from vae.encoder import Encoder3D  # just for sanity import


# ------------------------------------------------------------
# 1. Fake Smoke Dataset (VERY SMALL)
# ------------------------------------------------------------

class SmokeVideoDataset(torch.utils.data.Dataset):
    """
    Generates random video clips instead of loading from disk.
    Shape: (T, H, W, 3) uint8 → same as real pipeline
    """

    def __init__(self, n=20, T=17, H=128, W=128):
        self.n = n
        self.T = T
        self.H = H
        self.W = W

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        video = np.random.randint(
            0, 255, (self.T, self.H, self.W, 3), dtype=np.uint8
        )

        video = torch.from_numpy(video).float() / 127.5 - 1.0
        return video.permute(3, 0, 1, 2)  # (3, T, H, W)


# ------------------------------------------------------------
# 2. Minimal Config (override your full config)
# ------------------------------------------------------------

cfg = SimpleNamespace(
    base_channels=64,
    latent_channels=16,
    ch_mult=(1, 2, 4),
    num_res_blocks=2,
    spatial_ds=2,
    temporal_ds=2,
    kl_weight=1e-6,
    perceptual_weight=0.0,   # disable LPIPS for smoke
    batch_size=2,
    lr=1e-4,
    kl_warmup_steps=10,
    max_steps=2,
    grad_accum=1,
    use_amp=False,
    grad_clip=1.0,
)


# ------------------------------------------------------------
# 3. Build VAE
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE3D(
    in_channels=3,
    base_channels=cfg.base_channels,
    latent_channels=cfg.latent_channels,
    ch_mult=cfg.ch_mult,
    num_res_blocks=cfg.num_res_blocks,
    spatial_ds=cfg.spatial_ds,
    temporal_ds=cfg.temporal_ds,
    kl_weight=cfg.kl_weight,
).to(device)

optimizer = torch.optim.AdamW(vae.parameters(), lr=cfg.lr)


# ------------------------------------------------------------
# 4. Smoke loader
# ------------------------------------------------------------

dataset = SmokeVideoDataset()
loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size)


# ------------------------------------------------------------
# 5. Smoke training loop
# ------------------------------------------------------------

print("\n=== VAE SMOKE TRAIN START ===\n")

vae.train()

step = 0

for video in loader:
    video = video.to(device)

    # LR + KL warmup
    lr = cfg.lr
    kl_weight = cfg.kl_weight * min(1.0, step / cfg.kl_warmup_steps)

    for g in optimizer.param_groups:
        g["lr"] = lr

    # -----------------------------
    # Forward
    # -----------------------------
    out = vae(video)

    loss = (
        out.loss_recon
        + kl_weight * out.loss_kl
    )

    # -----------------------------
    # Backward
    # -----------------------------
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # -----------------------------
    # Logging
    # -----------------------------
    print(f"Step {step}")
    print(f"  recon  : {out.loss_recon.item():.4f}")
    print(f"  kl     : {out.loss_kl.item():.4f}")
    print(f"  total  : {loss.item():.4f}")
    print(f"  z      : {tuple(out.z.shape)}")
    print(f"  recon  : {tuple(out.recon.shape)}")

    # shape sanity checks
    assert out.recon.shape == video.shape

    step += 1
    if step >= cfg.max_steps:
        break


# ------------------------------------------------------------
# 6. Encode / Decode smoke test
# ------------------------------------------------------------

vae.eval()
with torch.no_grad():
    x = next(iter(loader)).to(device)
    z = vae.encode_mean(x)
    recon = vae.decode(z)

    print("\n=== INFERENCE SMOKE ===")
    print("z     :", tuple(z.shape))
    print("recon :", tuple(recon.shape))

    assert recon.shape == x.shape

print("\n✅ VAE SMOKE RUN PASSED")