# train_dit.py
# Run with: torchrun --nproc_per_node=8 train_dit.py
#
# VAE loading priority:
#   1. --vae_ckpt path      → our trained VAE checkpoint
#   2. --pretrained_vae     → HuggingFace model ID (e.g. THUDM/CogVideoX-2b)
#   3. Neither provided     → error (VAE must come from somewhere)
#
# EMA (Exponential Moving Average):
#   Shadow copy of weights updated as:
#       ema_w = decay * ema_w + (1 - decay) * current_w
#   Train with raw weights, sample with EMA weights.

import os
import time
import copy
import argparse
import contextlib
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

from vae.vae import VAE3D
from dit.dit import VideoDiT
from flow.scheduler import FlowMatchingScheduler
from conditioning.encoders import ConditioningPipeline


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DiTTrainConfig:

    # ── Paths ──────────────────────────────────────────────────────
    data_dir:        str  = "data/data/videos"
    output_dir:      str  = "runs/dit"
    resume:          str  = None       # DiT checkpoint to resume from
    vae_ckpt:        str  = None       # our trained VAE checkpoint
    pretrained_vae:  str  = None       # HF model ID fallback (e.g. "THUDM/CogVideoX-2b")

    # ── Video ──────────────────────────────────────────────────────
    num_frames:      int  = 17
    frame_h:         int  = 128
    frame_w:         int  = 128

    # ── Data format ────────────────────────────────────────────────
    # Must match how the VAE was trained. The VAE dataset asserted uint8
    # and normalized with video / 127.5 - 1.0. Keep this identical or the
    # frozen VAE sees out-of-distribution input and produces garbage latents.
    expect_uint8:    bool = True

    # ── VAE arch (must match checkpoint if using our own) ──────────
    vae_base_ch:     int   = 128
    latent_channels: int   = 16
    vae_ch_mult:     tuple = (1, 2, 4)
    vae_res_blocks:  int   = 2
    spatial_ds:      int   = 2
    temporal_ds:     int   = 2

    # ── DiT arch ───────────────────────────────────────────────────
    dit_hidden:  int   = 1024
    dit_heads:   int   = 16
    dit_layers:  int   = 24
    dit_cond:    int   = 1024
    patch_t:     int   = 1
    patch_h:     int   = 2
    patch_w:     int   = 2
    ffn_mult:    float = 8/3
    dropout:     float = 0.0

    # ── Conditioning ───────────────────────────────────────────────
    t5_model:    str = "google/t5-v1_1-base"     # swap to base for testing
    clip_model:  str = "openai/clip-vit-large-patch14"
    text_seq_len: int = 226

    # ── Flow matching ──────────────────────────────────────────────
    cfg_scale:          float = 7.0
    p_uncond:           float = 0.1
    timestep_strategy:  str   = "logit"

    # ── Training ───────────────────────────────────────────────────
    batch_size:    int   = 2        # per GPU
    lr:            float = 1e-4
    weight_decay:  float = 1e-2
    max_steps:     int   = 450_000
    warmup_steps:  int   = 2_000
    grad_clip:     float = 1.0
    grad_accum:    int   = 4        # effective batch = batch_size * world_size * grad_accum
    use_amp:       bool  = True

    # ── EMA ────────────────────────────────────────────────────────
    use_ema:       bool  = True
    ema_decay:     float = 0.9999
    ema_start:     int   = 1_000   # step at which to start EMA updates

    # ── Logging ────────────────────────────────────────────────────
    log_every:     int   = 50
    save_every:    int   = 15_000
    sample_every:  int   = 5_000   # generate sample video and log
    num_vis:       int   = 2       # number of sample videos to generate

    # ── System ─────────────────────────────────────────────────────
    num_workers:   int   = 4
    seed:          int   = 42


# ─────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────

def setup_ddp():
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


def barrier():
    """Safe barrier — no-op if dist not initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# ─────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────

class EMA:
    """
    Exponential Moving Average of model weights.

        ema_w = decay * ema_w + (1 - decay) * current_w

    The shadow is (re)synced from the live model via `copy_from` — call this
    after loading a resume checkpoint that has no EMA state, so the shadow
    starts from the resumed weights rather than a stale random init.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(self._unwrap(model))
        self.shadow.requires_grad_(False)
        self.shadow.eval()

    @staticmethod
    def _unwrap(model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, DDP) else model

    @torch.no_grad()
    def copy_from(self, model: nn.Module):
        """Hard-copy live weights into the shadow (used on cold resume)."""
        m = self._unwrap(model)
        for ema_p, model_p in zip(self.shadow.parameters(), m.parameters()):
            ema_p.data.copy_(model_p.data)
        for ema_b, model_b in zip(self.shadow.buffers(), m.buffers()):
            ema_b.copy_(model_b)

    @torch.no_grad()
    def update(self, model: nn.Module):
        m = self._unwrap(model)
        for ema_p, model_p in zip(self.shadow.parameters(), m.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        # Buffers (e.g. RoPE tables) are copied, not averaged
        for ema_b, model_b in zip(self.shadow.buffers(), m.buffers()):
            ema_b.copy_(model_b)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state):
        self.shadow.load_state_dict(state)


# ─────────────────────────────────────────────────────────────────────
# VAE loading — our checkpoint OR pretrained fallback
# ─────────────────────────────────────────────────────────────────────

class PretrainedVAEWrapper(nn.Module):
    """
    Wraps a HuggingFace pretrained 3D VAE (e.g. CogVideoX) to expose the same
    encode_mean / decode interface as our VAE3D.
    """

    def __init__(self, model_id: str, device: torch.device):
        super().__init__()
        try:
            from diffusers import AutoencoderKLCogVideoX
        except ImportError:
            raise ImportError(
                "diffusers not installed. pip install diffusers\n"
                "Or provide --vae_ckpt to use your own trained VAE."
            )

        print(f"Loading pretrained VAE from HuggingFace: {model_id}")
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_id,
            subfolder   = "vae",
            torch_dtype = torch.float16,
        ).to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        # The pretrained VAE runs in fp16, but training tensors arrive in
        # fp32. Cast inputs to the VAE's dtype on the way in, and cast the
        # latent/recon back to fp32 on the way out, so the rest of the
        # pipeline (DiT, loss) stays in its expected dtype.
        self.vae_dtype = next(self.vae.parameters()).dtype

        # CogVideoX VAE compression factors (verify against the model card)
        self.spatial_compression  = 8
        self.temporal_compression = 4
        self.latent_channels = self.vae.config.latent_channels

        # Latent scaling. CogVideoX normalizes latents by a scalar
        # scaling_factor (2b: 1.15258426) so the diffusion model sees
        # ~unit-variance inputs. We multiply on encode and divide on decode.
        # latents_mean/std are null for 2b, so the scalar is the whole story;
        # if a future model sets them, this would need the vector path.
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0) or 1.0
        assert getattr(self.vae.config, "latents_mean", None) is None, (
            "This VAE uses per-channel latents_mean/std; scalar scaling is "
            "insufficient. Add the vector normalization path."
        )

    @torch.no_grad()
    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        x = x.to(self.vae_dtype)
        dist_ = self.vae.encode(x).latent_dist
        return (dist_.mean * self.scaling_factor).to(out_dtype)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        out_dtype = z.dtype
        z = (z / self.scaling_factor).to(self.vae_dtype)
        return self.vae.decode(z).sample.to(out_dtype)

    def get_latent_shape(self, video_shape: tuple) -> tuple:
        B, C, T, H, W = video_shape
        lat_h = H // self.spatial_compression
        lat_w = W // self.spatial_compression
        lat_t = 1 + (T - 1) // self.temporal_compression
        return (B, self.latent_channels, lat_t, lat_h, lat_w)

    def set_inference_mode(self): pass
    def set_training_mode(self): pass


def load_vae(cfg: DiTTrainConfig, device: torch.device):
    if cfg.vae_ckpt is not None:
        print(f"Loading our VAE from checkpoint: {cfg.vae_ckpt}")
        vae = VAE3D(
            in_channels     = 3,
            base_channels   = cfg.vae_base_ch,
            latent_channels = cfg.latent_channels,
            ch_mult         = cfg.vae_ch_mult,
            num_res_blocks  = cfg.vae_res_blocks,
            spatial_ds      = cfg.spatial_ds,
            temporal_ds     = cfg.temporal_ds,
        ).to(device)

        ckpt = torch.load(cfg.vae_ckpt, map_location=device)
        state = ckpt.get("vae", ckpt)
        vae.load_state_dict(state)
        vae.requires_grad_(False)
        vae.eval()
        vae.set_inference_mode()

        n = sum(p.numel() for p in vae.parameters()) / 1e6
        print(f"Our VAE loaded ({n:.1f}M params)")
        return vae

    elif cfg.pretrained_vae is not None:
        vae = PretrainedVAEWrapper(cfg.pretrained_vae, device)
        print(f"Pretrained VAE loaded ({cfg.pretrained_vae})")
        return vae

    else:
        raise ValueError(
            "Must provide either --vae_ckpt (our trained VAE) "
            "or --pretrained_vae (HF model ID like THUDM/CogVideoX-2b)"
        )


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────

class VideoTextDataset(Dataset):
    """
    Loads .npy video files and matching .txt caption files.

    IMPORTANT: normalization must match the VAE's training preprocessing.
    The VAE was trained on uint8 .npy clips normalized with:
        video = video / 127.5 - 1.0      → [-1, 1]
    We replicate that exactly here so the frozen VAE sees in-distribution input.
    """

    def __init__(self, data_dir: str, num_frames: int,
                 frame_h: int, frame_w: int, expect_uint8: bool = True):
        self.paths        = sorted(Path(data_dir).glob("*.npy"))
        self.num_frames   = num_frames
        self.frame_h      = frame_h
        self.frame_w      = frame_w
        self.expect_uint8 = expect_uint8
        assert len(self.paths) > 0, f"No .npy files in {data_dir}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]

        video = np.load(p)   # (T, H, W, 3)

        if self.expect_uint8:
            # Match VAE preprocessing exactly: uint8 [0,255] → float [-1,1]
            assert video.dtype == np.uint8, (
                f"{p.name}: expected uint8 (to match VAE preprocessing), "
                f"got {video.dtype}. Set expect_uint8=False only if your "
                f"clips are already float in [-1, 1]."
            )
            video = torch.from_numpy(video.copy()).float()
            video = video / 127.5 - 1.0
        else:
            # Clips already float in [-1, 1]
            video = torch.from_numpy(video.copy()).float()

        # Temporal crop / pad
        T = video.shape[0]
        if T > self.num_frames:
            s     = torch.randint(0, T - self.num_frames + 1, (1,)).item()
            video = video[s: s + self.num_frames]
        elif T < self.num_frames:
            pad   = video[-1:].repeat(self.num_frames - T, 1, 1, 1)
            video = torch.cat([video, pad], dim=0)

        # (T, H, W, 3) → (3, T, H, W)
        video = video.permute(3, 0, 1, 2)

        # First frame as PIL for CLIP (denormalize [-1,1] → [0,255])
        ff_np  = ((video[:, 0].permute(1, 2, 0).numpy() + 1) * 127.5)
        ff_pil = Image.fromarray(ff_np.clip(0, 255).astype(np.uint8))

        caption_path = p.with_suffix(".txt")
        caption = caption_path.read_text().strip() if caption_path.exists() else ""

        return {"video": video, "first_frame": ff_pil, "caption": caption}


def collate_fn(batch):
    return {
        "video":       torch.stack([b["video"] for b in batch]),
        "first_frame": [b["first_frame"] for b in batch],
        "caption":     [b["caption"] for b in batch],
    }


# ─────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: DiTTrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress)))


# ─────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────

class DiTTrainer:

    def __init__(self, cfg: DiTTrainConfig, rank: int, world_size: int, local_rank: int):
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
            self.writer = SummaryWriter(log_dir=f"{cfg.output_dir}/tb_logs")
            print(f"\n{'='*60}")
            print(f"DiT Training | GPUs: {world_size} | Device: {self.device}")
            print(f"Effective batch: {cfg.batch_size * world_size * cfg.grad_accum}")
            print(f"Output: {cfg.output_dir}")
            print(f"{'='*60}\n")

        self._build_models()
        self._build_data()
        self._build_optimizer()
        self.step = 0

        if cfg.resume is not None:
            self._load_checkpoint(cfg.resume)

    def _build_models(self):
        cfg = self.cfg

        # ── VAE (frozen) ──────────────────────────────────────────────
        print(f"[rank {self.rank}] Loading VAE...")
        self.vae = load_vae(cfg, self.device)

        # Derive latent shape EMPIRICALLY by probing the real encoder.
        # This reads the post-chunk (mean-only) shape and cannot disagree
        # with what the encoder actually produces — unlike analytical shape
        # math, which is easy to get wrong (e.g. mean‖logvar channel count,
        # causal temporal compression).
        with torch.no_grad():
            probe = torch.zeros(
                1, 3, cfg.num_frames, cfg.frame_h, cfg.frame_w,
                device=self.device,
            )
            lat = self.vae.encode_mean(probe)
        _, lat_c, lat_t, lat_h, lat_w = lat.shape
        self.lat_c, self.lat_t, self.lat_h, self.lat_w = lat_c, lat_t, lat_h, lat_w
        del probe, lat

        if self.main:
            print(f"Latent shape (probed): C={lat_c}, T={lat_t}, H={lat_h}, W={lat_w}")

        # ── Conditioning encoders (frozen) ────────────────────────────
        if self.main:
            print("Loading conditioning encoders...")
        self.cond = ConditioningPipeline(
            t5_model   = cfg.t5_model,
            clip_model = cfg.clip_model,
            device     = self.device,
        )

        # ── DiT ───────────────────────────────────────────────────────
        if self.main:
            print("Building DiT...")
        self.dit = VideoDiT(
            latent_channels = lat_c,
            latent_t        = lat_t,
            latent_h        = lat_h,
            latent_w        = lat_w,
            patch_t         = cfg.patch_t,
            patch_h         = cfg.patch_h,
            patch_w         = cfg.patch_w,
            hidden_dim      = cfg.dit_hidden,
            num_heads       = cfg.dit_heads,
            num_layers      = cfg.dit_layers,
            cond_dim        = cfg.dit_cond,
            text_dim        = self.cond.text_dim,
            clip_dim        = self.cond.clip_dim,
            ffn_mult        = cfg.ffn_mult,
            dropout         = cfg.dropout,
        ).to(self.device)

        # DDP wrap.
        # find_unused_parameters=True: with CFG conditioning dropout, on steps
        # where drop_mask is all-False the learnable null embeddings receive no
        # gradient. With find_unused_parameters=False that triggers a DDP
        # "expected to have finished reduction" error. True is the safe default
        # here; if your null conditioning is non-learnable buffers, you can set
        # this back to False for a small speedup.
        self.dit_ddp = DDP(
            self.dit,
            device_ids             = [self.local_rank],
            find_unused_parameters = True,
        )

        # ── EMA (rank 0 only, to save memory) ─────────────────────────
        if cfg.use_ema and self.main:
            self.ema = EMA(self.dit, decay=cfg.ema_decay)
        else:
            self.ema = None

        # ── Scheduler ────────────────────────────────────────────────
        self.scheduler = FlowMatchingScheduler(
            timestep_strategy = cfg.timestep_strategy,
            cfg_scale         = cfg.cfg_scale,
            p_uncond          = cfg.p_uncond,
        )

        # ── Mixed precision ───────────────────────────────────────────
        self.scaler = GradScaler(enabled=cfg.use_amp)

        if self.main:
            params = self.dit.count_params()
            print(f"DiT params: {params['total_M']:.1f}M trainable")

    def _build_data(self):
        cfg = self.cfg

        dataset = VideoTextDataset(
            cfg.data_dir, cfg.num_frames, cfg.frame_h, cfg.frame_w,
            expect_uint8=cfg.expect_uint8,
        )
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
            collate_fn  = collate_fn,
            pin_memory  = True,
            drop_last   = True,
        )
        self.sampler = sampler

    def _build_optimizer(self):
        cfg = self.cfg
        decay, no_decay = [], []
        for name, p in self.dit.named_parameters():
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
        ckpt = {
            "step":      self.step,
            "dit":       self.dit.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler":    self.scaler.state_dict(),
            "cfg":       asdict(self.cfg),
        }
        if self.ema is not None:
            ckpt["ema"] = self.ema.state_dict()

        torch.save(ckpt, path)
        torch.save(ckpt, f"{self.cfg.output_dir}/latest.pt")
        print(f"[step {self.step}] Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.dit.load_state_dict(ckpt["dit"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = ckpt["step"]

        if self.ema is not None:
            if "ema" in ckpt:
                self.ema.load_state_dict(ckpt["ema"])
            else:
                # Cold resume from a checkpoint without EMA state: sync the
                # shadow to the resumed weights so it doesn't drift from a
                # stale random init.
                self.ema.copy_from(self.dit)

        if self.main:
            print(f"Resumed from {path} at step {self.step}")

    # ── Encode batch (frozen encoders) ───────────────────────────────

    @torch.no_grad()
    def _encode_batch(self, batch: dict) -> dict:
        video = batch["video"].to(self.device)   # (B, 3, T, H, W)

        z0    = self.vae.encode_mean(video)       # (B, C, T', H', W')

        first = video[:, :, :1, :, :]
        z_img = self.vae.encode_mean(first)       # (B, C, 1, H', W')

        txt_cond = self.cond.encode_text(batch["caption"])
        img_cond = self.cond.encode_image(batch["first_frame"])

        return {
            "z0":         z0,
            "z_img":      z_img,
            "txt_tokens": txt_cond.tokens,
            "img_tokens": img_cond.tokens,
        }

    # ── Training step ────────────────────────────────────────────────

    def _step(self, encoded: dict) -> dict:
        cfg = self.cfg
        B   = encoded["z0"].shape[0]

        z0         = encoded["z0"]
        z_img      = encoded["z_img"]
        txt_tokens = encoded["txt_tokens"]
        img_tokens = encoded["img_tokens"]

        t     = self.scheduler.t_sampler.sample(B, self.device)
        noise = torch.randn_like(z0)
        x_t, noise = self.scheduler.add_noise(z0, t, noise)
        v_target   = self.scheduler.get_velocity_target(z0, noise)

        # CFG conditioning dropout
        drop_mask = torch.rand(B, device=self.device) < cfg.p_uncond

        null_txt, null_img = self.dit.get_null_conditioning(
            B, txt_tokens.shape[1], img_tokens.shape[1], self.device
        )

        txt_in = torch.where(drop_mask.view(B, 1, 1), null_txt, txt_tokens)
        img_in = torch.where(drop_mask.view(B, 1, 1), null_img, img_tokens)

        with torch.autocast(device_type="cuda", enabled=cfg.use_amp):
            v_pred = self.dit_ddp(x_t, t, txt_in, img_in, z_img)
            loss   = F.mse_loss(v_pred, v_target)
            loss   = loss / cfg.grad_accum

        return {
            "loss":     loss,
            "v_pred":   v_pred.detach(),
            "v_target": v_target.detach(),
            "t":        t.detach(),
        }

    # ── Sample generation for visualization ─────────────────────────

    @torch.no_grad()
    def _generate_samples(self, batch: dict):
        """
        Rank-0-only sample generation using EMA weights (or current weights).
        Runs under no_grad with no DDP forward, so it does not participate in
        gradient allreduce. Callers must barrier afterwards so the other ranks
        wait rather than racing ahead into the next DDP step (which can trip
        NCCL timeouts when sampling is slow).
        """
        if not self.main:
            return

        # Pick the sampling model. Prefer EMA shadow if present.
        if self.ema is not None:
            sample_model = self.ema.shadow
        else:
            sample_model = self.dit

        was_training = sample_model.training
        sample_model.eval()

        cfg = self.cfg
        B   = min(cfg.num_vis, len(batch["caption"]))

        try:
            txt_cond = self.cond.encode_text(batch["caption"][:B])
            img_cond = self.cond.encode_image(batch["first_frame"][:B])

            first_vid = batch["video"][:B].to(self.device)
            first_fr  = first_vid[:, :, :1, :, :]
            z_img     = self.vae.encode_mean(first_fr)

            L = txt_cond.tokens.shape[1]
            M = img_cond.tokens.shape[1]
            null_txt, null_img = self.dit.get_null_conditioning(B, L, M, self.device)

            z0_gen = self.scheduler.euler_sample(
                model      = lambda z, t, txt, img: sample_model(z, t, txt, img, z_img),
                shape      = (B, self.lat_c, self.lat_t, self.lat_h, self.lat_w),
                c_txt      = txt_cond.tokens,
                c_img      = img_cond.tokens,
                null_c_txt = null_txt,
                null_c_img = null_img,
                num_steps  = 20,
                cfg_scale  = cfg.cfg_scale,
                device     = self.device,
            )

            gen_video = self.vae.decode(z0_gen)     # (B, 3, T, H, W)

            gen_ff  = (gen_video[:, :, 0] + 1) / 2          # (B, 3, H, W)
            real_ff = (first_vid[:, :, 0].cpu() + 1) / 2

            comparison = torch.cat([real_ff, gen_ff.cpu()], dim=3)  # (B, 3, H, 2W)
            self.writer.add_images(
                "samples/real_vs_generated",
                comparison,
                global_step = self.step,
            )

            gen_np = gen_video.cpu().permute(0, 2, 3, 4, 1).numpy()  # (B, T, H, W, 3)
            for i, vid in enumerate(gen_np):
                np.save(
                    f"{cfg.output_dir}/samples/step{self.step:07d}_sample{i}.npy",
                    vid
                )
        finally:
            # Restore the sampling model's previous mode. The EMA shadow should
            # stay in eval; the live DiT (if used) goes back to whatever it was.
            if was_training:
                sample_model.train()

    # ── Training loop ────────────────────────────────────────────────

    def train(self):
        cfg = self.cfg

        if self.main:
            print("Starting DiT training...\n")

        self.dit_ddp.train()
        self.optimizer.zero_grad()

        running_loss = 0.0
        last_grad_norm = 0.0
        t0 = time.time()

        vis_batch = None

        while self.step < cfg.max_steps:
            self.sampler.set_epoch(self.step // max(1, len(self.loader)))

            for batch in self.loader:
                if self.step >= cfg.max_steps:
                    break

                if vis_batch is None:
                    vis_batch = batch

                # ── LR ────────────────────────────────────────────────
                lr = get_lr(self.step, cfg)
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                # ── Encode (no grad, frozen encoders) ─────────────────
                encoded = self._encode_batch(batch)

                # ── DiT forward + backward ────────────────────────────
                is_accum_step = (self.step + 1) % cfg.grad_accum != 0

                # Skip allreduce on accumulation steps for efficiency.
                sync_ctx = (
                    self.dit_ddp.no_sync()
                    if is_accum_step else contextlib.nullcontext()
                )
                with sync_ctx:
                    out = self._step(encoded)
                    self.scaler.scale(out["loss"]).backward()

                if not is_accum_step:
                    self.scaler.unscale_(self.optimizer)
                    last_grad_norm = nn.utils.clip_grad_norm_(
                        self.dit.parameters(), cfg.grad_clip
                    ).item()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    if self.ema is not None and self.step >= cfg.ema_start:
                        self.ema.update(self.dit_ddp)

                running_loss += out["loss"].item() * cfg.grad_accum

                # ── Logging ───────────────────────────────────────────
                if self.main and self.step % cfg.log_every == 0 and self.step > 0:
                    elapsed = time.time() - t0
                    sps     = cfg.log_every / max(elapsed, 1e-6)
                    t0      = time.time()
                    avg_loss = running_loss / cfg.log_every
                    running_loss = 0.0

                    t_mean = out["t"].mean().item()
                    t_std  = out["t"].std().item()
                    v_err = (out["v_pred"] - out["v_target"]).abs().mean().item()

                    print(
                        f"step {self.step:7d}/{cfg.max_steps} | "
                        f"loss={avg_loss:.4f} | "
                        f"v_err={v_err:.4f} | "
                        f"grad={last_grad_norm:.3f} | "
                        f"t_mean={t_mean:.3f} | "
                        f"lr={lr:.2e} | "
                        f"{sps:.1f} s/s"
                    )

                    self.writer.add_scalar("loss/train",        avg_loss,       self.step)
                    self.writer.add_scalar("train/lr",          lr,             self.step)
                    self.writer.add_scalar("train/grad_norm",   last_grad_norm, self.step)
                    self.writer.add_scalar("train/v_error",     v_err,          self.step)
                    self.writer.add_scalar("train/steps_per_s", sps,            self.step)

                    self.writer.add_histogram(
                        "train/timestep_dist", out["t"], self.step
                    )

                    t_vals = out["t"].cpu()
                    v_pred = out["v_pred"].cpu()
                    v_tgt  = out["v_target"].cpu()
                    for lo, hi in [(0, 0.33), (0.33, 0.66), (0.66, 1.0)]:
                        mask = (t_vals >= lo) & (t_vals < hi)
                        if mask.any():
                            bucket_loss = F.mse_loss(v_pred[mask], v_tgt[mask]).item()
                            self.writer.add_scalar(
                                f"loss/bucket_t{lo:.2f}_{hi:.2f}",
                                bucket_loss, self.step
                            )

                # ── Sample visualization ───────────────────────────────
                # Rank 0 generates; all ranks barrier so nobody races into the
                # next DDP step while rank 0 is still sampling.
                if self.step % cfg.sample_every == 0 and self.step > 0:
                    if self.main:
                        print(f"[step {self.step}] Generating samples...")
                        self._generate_samples(vis_batch)
                    barrier()

                # ── Checkpoint ────────────────────────────────────────
                if self.step % cfg.save_every == 0 and self.step > 0:
                    self._save_checkpoint()
                    barrier()

                self.step += 1

        self._save_checkpoint()
        barrier()
        if self.main:
            self.writer.close()
            print("\nDiT training complete.")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       type=str, default="data/data/videos")
    p.add_argument("--output_dir",     type=str, default="runs/dit")
    p.add_argument("--resume",         type=str, default=None,
                   help="DiT checkpoint to resume from")
    p.add_argument("--vae_ckpt",       type=str, default=None,
                   help="Our trained VAE checkpoint (runs/vae/latest.pt)")
    p.add_argument("--pretrained_vae", type=str, default="zai-org/CogVideoX-2b",
                   help="HuggingFace model ID e.g. THUDM/CogVideoX-2b")
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--max_steps",      type=int,   default=450_000)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--grad_accum",     type=int,   default=4)
    p.add_argument("--dit_hidden",     type=int,   default=1024)
    p.add_argument("--dit_layers",     type=int,   default=24)
    p.add_argument("--t5_model",       type=str,   default="google/t5-v1_1-base")
    p.add_argument("--no_uint8",       action="store_true",
                   help="Set if .npy clips are already float in [-1, 1] "
                        "instead of uint8 [0, 255]")
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_ddp()

    cfg = DiTTrainConfig(
        data_dir        = args.data_dir,
        output_dir      = args.output_dir,
        resume          = args.resume,
        vae_ckpt        = args.vae_ckpt,
        pretrained_vae  = args.pretrained_vae,
        batch_size      = args.batch_size,
        max_steps       = args.max_steps,
        lr              = args.lr,
        grad_accum      = args.grad_accum,
        dit_hidden      = args.dit_hidden,
        dit_layers      = args.dit_layers,
        t5_model        = args.t5_model,
        expect_uint8    = not args.no_uint8,
    )

    trainer = DiTTrainer(cfg, rank, world_size, local_rank)
    trainer.train()
    cleanup_ddp()


if __name__ == "__main__":
    main()