# train_vae.py
# Run with: torchrun --nproc_per_node=8 train_vae.py



#   LPIPS  — perceptual loss using deep features instead of raw pixels
#   EMA    — exponential moving average of weights for stable checkpoints

import os
import time
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler

# Optional — install with: pip install lpips
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("[warn] lpips not installed — perceptual loss disabled. pip install lpips")

from vae.vae import VAE3D


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

@dataclass
class VAETrainConfig:

    # ── Paths ──────────────────────────────────────────────────────
    data_dir:   str = "data/videos"
    output_dir: str = "runs/vae"
    resume:     str = None          # path to checkpoint to resume from

    # ── Video ──────────────────────────────────────────────────────
    num_frames: int = 17
    frame_h:    int = 128
    frame_w:    int = 128

    # ── VAE arch ───────────────────────────────────────────────────
    base_channels:   int   = 128
    latent_channels: int   = 16
    ch_mult:         tuple = (1, 2, 4)
    num_res_blocks:  int   = 2
    spatial_ds:      int   = 2
    temporal_ds:     int   = 2
    num_groups:      int   = 32

    # ── Loss weights ───────────────────────────────────────────────
    kl_weight:          float = 1e-6    # β: KL regularization
    perceptual_weight:  float = 0.1     # LPIPS weight (0 to disable)
    # kl_weight starts very small and warms up — prevents posterior collapse
    kl_warmup_steps:    int   = 5_000   # steps to ramp kl_weight to full

    # ── Training ───────────────────────────────────────────────────
    batch_size:       int   = 4         # per GPU
    lr:               float = 1e-4
    weight_decay:     float = 1e-2
    max_steps:        int   = 200_000
    warmup_steps:     int   = 1_000
    grad_clip:        float = 1.0
    grad_accum:       int   = 1         # gradient accumulation steps
    use_amp:          bool  = True

    # ── Logging ────────────────────────────────────────────────────
    log_every:        int   = 50
    save_every:       int   = 2_000
    recon_vis_every:  int   = 500       # save reconstruction samples
    num_vis_samples:  int   = 4         # how many clips to visualize

    # ── System ─────────────────────────────────────────────────────
    num_workers:      int   = 4
    seed:             int   = 42




def setup_ddp():
    """Initialize the distributed process group."""
    dist.init_process_group(backend="nccl")   # nccl = NVIDIA's fast GPU comms
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

class VideoDataset(Dataset):
    """
    Loads pre-extracted video clips as .npy files.
    Each file: (T, H, W, 3) float32 in [-1, 1].

    No captions needed for VAE training —
    the VAE learns video compression independently of text.
    """

    def __init__(self, data_dir: str, num_frames: int, frame_h: int, frame_w: int):
        self.paths      = sorted(Path(data_dir).glob("*.npy"))
        self.num_frames = num_frames
        self.frame_h    = frame_h
        self.frame_w    = frame_w
        assert len(self.paths) > 0, f"No .npy files in {data_dir}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        video = np.load(self.paths[idx])         # (T, H, W, 3)
        video = torch.from_numpy(video).float()

        # Random temporal crop
        T = video.shape[0]
        if T > self.num_frames:
            start = torch.randint(0, T - self.num_frames + 1, (1,)).item()
            video = video[start: start + self.num_frames]
        elif T < self.num_frames:
            pad   = video[-1:].repeat(self.num_frames - T, 1, 1, 1)
            video = torch.cat([video, pad], dim=0)

        # (T, H, W, 3) → (3, T, H, W)
        return video.permute(3, 0, 1, 2)


# ─────────────────────────────────────────────────────────────────────
# LPIPS Perceptual Loss
# ─────────────────────────────────────────────────────────────────────
# LPIPS (Learned Perceptual Image Patch Similarity):
#   Instead of comparing pixels directly (L1/L2), LPIPS compares
#   deep features from a pretrained VGG/AlexNet network.
#   This penalizes perceptual differences rather than pixel differences.
#
#   Why this matters for video VAE:
#       L1 loss minimizes pixel error → blurry reconstructions
#       (the model hedges by averaging plausible textures)
#       LPIPS penalizes blurriness because blurry features
#       differ significantly from sharp features in VGG space.
#
#   Applied per-frame: we flatten (B, 3, T, H, W) → (B*T, 3, H, W).

class PerceptualLoss(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        if not LPIPS_AVAILABLE:
            self.fn = None
            return
        # net='vgg' gives slightly better perceptual quality than 'alex'
        self.fn = lpips.LPIPS(net="vgg").to(device)
        self.fn.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, 3, T, H, W) in [-1, 1]
        Returns: scalar perceptual loss
        """
        if self.fn is None:
            return torch.tensor(0.0, device=pred.device)

        B, C, T, H, W = pred.shape
        # Flatten time into batch: (B*T, 3, H, W)
        pred_flat   = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        target_flat = target.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        return self.fn(pred_flat, target_flat).mean()


# ─────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: VAETrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    # Cosine decay → 10% of peak LR (don't decay fully to 0)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress)))


def get_kl_weight(step: int, cfg: VAETrainConfig) -> float:
    """
    Warm up the KL weight from 0 → cfg.kl_weight over kl_warmup_steps.

    Why: at init the encoder outputs mean≈0, logvar≈0 (near N(0,1)).
    Immediately applying full KL weight causes the encoder to collapse
    to the prior before it has learned to encode anything useful.
    Warming it up gives the encoder time to develop first.
    """
    if step >= cfg.kl_warmup_steps:
        return cfg.kl_weight
    return cfg.kl_weight * (step / cfg.kl_warmup_steps)


# ─────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────

class VAETrainer:

    def __init__(self, cfg: VAETrainConfig, rank: int, world_size: int, local_rank: int):
        self.cfg        = cfg
        self.rank       = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device     = torch.device(f"cuda:{local_rank}")
        self.main       = is_main(rank)

        torch.manual_seed(cfg.seed + rank)

        if self.main:
            os.makedirs(cfg.output_dir, exist_ok=True)
            os.makedirs(f"{cfg.output_dir}/samples", exist_ok=True)
            # TensorBoard writer — only rank 0 writes logs
            self.writer = SummaryWriter(log_dir=f"{cfg.output_dir}/tb_logs")
            print(f"\n{'='*60}")
            print(f"VAE Training | GPUs: {world_size} | Device: {self.device}")
            print(f"Output: {cfg.output_dir}")
            print(f"{'='*60}\n")

        self._build_model()
        self._build_data()
        self._build_optimizer()
        self.step = 0

        # Resume from checkpoint if provided
        if cfg.resume is not None:
            self._load_checkpoint(cfg.resume)

    def _build_model(self):
        cfg = self.cfg

        self.vae = VAE3D(
            in_channels     = 3,
            base_channels   = cfg.base_channels,
            latent_channels = cfg.latent_channels,
            ch_mult         = cfg.ch_mult,
            num_res_blocks  = cfg.num_res_blocks,
            spatial_ds      = cfg.spatial_ds,
            temporal_ds     = cfg.temporal_ds,
            num_groups      = cfg.num_groups,
            kl_weight       = cfg.kl_weight,
            recon_loss      = "l1",
        ).to(self.device)

        # Wrap in DDP
        # find_unused_parameters=False → faster, fine since all params are used
        self.vae_ddp = DDP(
            self.vae,
            device_ids         = [self.local_rank],
            find_unused_parameters = False,
        )

        # Perceptual loss (not wrapped in DDP — no grad through it)
        self.perceptual = PerceptualLoss(self.device)

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=cfg.use_amp)

        if self.main:
            n = sum(p.numel() for p in self.vae.parameters()) / 1e6
            print(f"VAE parameters: {n:.1f}M")

    def _build_data(self):
        cfg = self.cfg

        dataset = VideoDataset(
            cfg.data_dir, cfg.num_frames, cfg.frame_h, cfg.frame_w
        )

        # DistributedSampler ensures each GPU sees a non-overlapping
        # subset of the dataset. Without it every GPU would train on
        # the same data → no speedup, just redundancy.
        sampler = DistributedSampler(
            dataset,
            num_replicas = self.world_size,
            rank         = self.rank,
            shuffle      = True,
            drop_last    = True,
        )

        self.loader = DataLoader(
            dataset,
            batch_size  = cfg.batch_size,
            sampler     = sampler,
            num_workers = cfg.num_workers,
            pin_memory  = True,
            drop_last   = True,
        )
        self.sampler = sampler

    def _build_optimizer(self):
        cfg = self.cfg

        decay, no_decay = [], []
        for name, p in self.vae.named_parameters():
            if "bias" in name or "norm" in name:
                no_decay.append(p)
            else:
                decay.append(p)

        self.optimizer = torch.optim.AdamW([
            {"params": decay,    "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ], lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)

    # ── Checkpoint ───────────────────────────────────────────────────

    def _save_checkpoint(self):
        if not self.main:
            return
        path = f"{self.cfg.output_dir}/ckpt_step{self.step:07d}.pt"
        torch.save({
            "step":      self.step,
            "vae":       self.vae.state_dict(),         # save unwrapped model
            "optimizer": self.optimizer.state_dict(),
            "scaler":    self.scaler.state_dict(),
            "cfg":       asdict(self.cfg),
        }, path)
        # Also save as latest for easy resuming
        torch.save({
            "step":      self.step,
            "vae":       self.vae.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler":    self.scaler.state_dict(),
            "cfg":       asdict(self.cfg),
        }, f"{self.cfg.output_dir}/latest.pt")
        print(f"[step {self.step}] Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(ckpt["vae"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]
        if self.main:
            print(f"Resumed from {path} at step {self.step}")

    # ── Visualization ────────────────────────────────────────────────

    @torch.no_grad()
    def _save_recon_samples(self, real: torch.Tensor, recon: torch.Tensor):
        """
        Save side-by-side real vs reconstructed first frames to TensorBoard.
        real, recon: (B, 3, T, H, W) in [-1, 1]
        """
        if not self.main:
            return

        n = min(self.cfg.num_vis_samples, real.shape[0])

        # Take first frame of each clip: (n, 3, H, W)
        real_frames  = real[:n, :, 0].cpu()
        recon_frames = recon[:n, :, 0].cpu()

        # Denormalize [-1,1] → [0,1]
        real_frames  = (real_frames  + 1) / 2
        recon_frames = (recon_frames + 1) / 2

        # Stack side by side: (n, 3, H, 2W)
        comparison = torch.cat([real_frames, recon_frames], dim=3)

        # Log as image grid to TensorBoard
        self.writer.add_images(
            "vae/reconstruction",
            comparison,
            global_step = self.step,
        )

    # ── Training loop ────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg

        if self.main:
            print("Starting VAE training...\n")

        self.vae_ddp.train()
        t0 = time.time()

        while self.step < cfg.max_steps:
            # Tell sampler which epoch we're on so shuffling varies
            self.sampler.set_epoch(self.step // len(self.loader))

            for video in self.loader:
                if self.step >= cfg.max_steps:
                    break

                video = video.to(self.device)    # (B, 3, T, H, W)

                # ── LR + KL weight update ─────────────────────────────
                lr        = get_lr(self.step, cfg)
                kl_weight = get_kl_weight(self.step, cfg)

                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                # ── Forward ───────────────────────────────────────────
                with torch.autocast(device_type="cuda", enabled=cfg.use_amp):
                    out = self.vae_ddp(video)

                    # Perceptual loss on reconstruction
                    loss_percep = self.perceptual(out.recon, video)

                    # Total loss with current kl_weight (overrides fixed weight)
                    loss = (
                        out.loss_recon
                        + kl_weight * out.loss_kl
                        + cfg.perceptual_weight * loss_percep
                    )

                    # Gradient accumulation: scale loss
                    loss = loss / cfg.grad_accum

                # ── Backward ──────────────────────────────────────────
                self.scaler.scale(loss).backward()

                if (self.step + 1) % cfg.grad_accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.vae.parameters(), cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # ── Logging (rank 0 only) ─────────────────────────────
                if self.main and self.step % cfg.log_every == 0:
                    elapsed = time.time() - t0
                    steps_per_sec = cfg.log_every / max(elapsed, 1e-6)
                    t0 = time.time()

                    # Console
                    print(
                        f"step {self.step:7d}/{cfg.max_steps} | "
                        f"loss={loss.item() * cfg.grad_accum:.4f} | "
                        f"recon={out.loss_recon.item():.4f} | "
                        f"kl={out.loss_kl.item():.5f} | "
                        f"percep={loss_percep.item():.4f} | "
                        f"lr={lr:.2e} | "
                        f"kl_w={kl_weight:.2e} | "
                        f"{steps_per_sec:.1f} s/s"
                    )

                    # TensorBoard
                    self.writer.add_scalar("loss/total",       loss.item() * cfg.grad_accum, self.step)
                    self.writer.add_scalar("loss/recon",       out.loss_recon.item(),          self.step)
                    self.writer.add_scalar("loss/kl",          out.loss_kl.item(),             self.step)
                    self.writer.add_scalar("loss/perceptual",  loss_percep.item(),             self.step)
                    self.writer.add_scalar("train/lr",         lr,                             self.step)
                    self.writer.add_scalar("train/kl_weight",  kl_weight,                      self.step)
                    self.writer.add_scalar("train/steps_per_s", steps_per_sec,                 self.step)

                    # Latent statistics — useful to monitor posterior collapse
                    # If mean stays near 0 and std stays near 1 → healthy
                    # If std → 0 → posterior collapse (encoder ignores input)
                    with torch.no_grad():
                        mean_abs  = out.mean.abs().mean().item()
                        std_mean  = out.logvar.mul(0.5).exp().mean().item()
                    self.writer.add_scalar("latent/mean_abs",  mean_abs, self.step)
                    self.writer.add_scalar("latent/std_mean",  std_mean, self.step)

                # ── Reconstruction visualization ───────────────────────
                if self.main and self.step % cfg.recon_vis_every == 0:
                    with torch.no_grad():
                        self._save_recon_samples(
                            video.detach(),
                            out.recon.detach(),
                        )

                # ── Checkpoint ────────────────────────────────────────
                if self.step % cfg.save_every == 0 and self.step > 0:
                    self._save_checkpoint()

                self.step += 1

        # Final checkpoint
        self._save_checkpoint()
        if self.main:
            self.writer.close()
            print("\nVAE training complete.")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str, default="data/videos")
    parser.add_argument("--output_dir",  type=str, default="runs/vae")
    parser.add_argument("--resume",      type=str, default=None)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--max_steps",   type=int, default=200_000)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--base_channels", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_ddp()

    cfg = VAETrainConfig(
        data_dir    = args.data_dir,
        output_dir  = args.output_dir,
        resume      = args.resume,
        batch_size  = args.batch_size,
        max_steps   = args.max_steps,
        lr          = args.lr,
        base_channels = args.base_channels,
    )

    trainer = VAETrainer(cfg, rank, world_size, local_rank)
    trainer.train()
    cleanup_ddp()


if __name__ == "__main__":
    main()