#!/usr/bin/env python3
"""
inference_dit.py — text + reference-image → video, from a trained DiT checkpoint.

Single-GPU inference. Mirrors the sampling path used during training
(_generate_samples in train_dit.py) so behavior matches what you saw in
the TensorBoard samples.

Example:
    python inference_dit.py \
        --ckpt        runs/dit/latest.pt \
        --vae_ckpt    runs/vae/latest.pt \
        --prompt      "a man is driving a car down a country road" \
        --ref_image   examples/car.jpg \
        --out         outputs/car.mp4 \
        --num_steps   50 \
        --cfg_scale   7.0

VAE selection mirrors training:
    --vae_ckpt path      → your trained VAE3D
    --pretrained_vae id  → HF model id (e.g. zai-org/CogVideoX-2b)
The VAE MUST match whatever the DiT was trained against — the latent shape
(channels / spatial / temporal) is baked into the DiT weights. Using a
different VAE will mismatch and either crash or produce garbage.

The reference image can be any resolution / aspect ratio; it is
center-cropped to square and resized to --size (default 128) so it matches
the VAE and CLIP input the model trained on.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch

from vae.vae import VAE3D
from dit.dit import VideoDiT
from flow.scheduler import FlowMatchingScheduler
from conditioning.encoders import ConditioningPipeline


# ─────────────────────────────────────────────────────────────────────
# VAE loading — same priority as training
# ─────────────────────────────────────────────────────────────────────

class PretrainedVAEWrapper(torch.nn.Module):
    """Mirror of the training wrapper: HF CogVideoX VAE with our interface."""

    def __init__(self, model_id: str, device: torch.device):
        super().__init__()
        from diffusers import AutoencoderKLCogVideoX

        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float16,
        ).to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.vae_dtype = next(self.vae.parameters()).dtype
        self.spatial_compression  = 8
        self.temporal_compression = 4
        self.latent_channels = self.vae.config.latent_channels
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0) or 1.0

    @torch.no_grad()
    def encode_mean(self, x):
        out_dtype = x.dtype
        x = x.to(self.vae_dtype)
        dist_ = self.vae.encode(x).latent_dist
        return (dist_.mean * self.scaling_factor).to(out_dtype)

    @torch.no_grad()
    def decode(self, z):
        out_dtype = z.dtype
        z = (z / self.scaling_factor).to(self.vae_dtype)
        return self.vae.decode(z).sample.to(out_dtype)


def load_vae(args, cfg: dict, device: torch.device):
    """
    Load the VAE that the DiT was trained with. cfg is the checkpoint's saved
    config dict (so VAE arch matches what produced the latents).
    """
    if args.vae_ckpt is not None:
        vae = VAE3D(
            in_channels     = 3,
            base_channels   = cfg.get("vae_base_ch", 128),
            latent_channels = cfg.get("latent_channels", 16),
            ch_mult         = tuple(cfg.get("vae_ch_mult", (1, 2, 4))),
            num_res_blocks  = cfg.get("vae_res_blocks", 2),
            spatial_ds      = cfg.get("spatial_ds", 2),
            temporal_ds     = cfg.get("temporal_ds", 2),
        ).to(device)
        ckpt = torch.load(args.vae_ckpt, map_location=device)
        state = ckpt.get("vae", ckpt)
        vae.load_state_dict(state)
        vae.requires_grad_(False)
        vae.eval()
        if hasattr(vae, "set_inference_mode"):
            vae.set_inference_mode()
        print(f"Loaded VAE3D from {args.vae_ckpt}")
        return vae

    elif args.pretrained_vae is not None:
        vae = PretrainedVAEWrapper(args.pretrained_vae, device)
        print(f"Loaded pretrained VAE {args.pretrained_vae}")
        return vae

    else:
        raise ValueError("Provide --vae_ckpt or --pretrained_vae (must match training).")


# ─────────────────────────────────────────────────────────────────────
# Reference image → model input
# ─────────────────────────────────────────────────────────────────────

def load_ref_image(path: str, size: int):
    """
    Load an arbitrary-resolution image, center-crop to square, resize to
    size x size. Returns:
        pil_resized : PIL.Image at (size, size)  — for CLIP
        tensor      : (1, 3, 1, size, size) float in [-1, 1] on CPU — for VAE
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    img  = img.crop((left, top, left + side, top + side))
    img  = img.resize((size, size), Image.BICUBIC)

    arr = np.asarray(img, dtype=np.uint8)            # (size, size, 3)
    t   = torch.from_numpy(arr).float() / 127.5 - 1.0  # [-1, 1]
    t   = t.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)  # (1, 3, 1, H, W)
    return img, t


# ─────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────

def save_video(video: np.ndarray, out_path: str, fps: int = 8):
    """
    video: (T, H, W, 3) float in [-1, 1] OR uint8 in [0, 255].
    Saves .mp4 if out_path ends in .mp4 (needs imageio[ffmpeg]) and always
    a .npy alongside.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if video.dtype != np.uint8:
        video = ((video + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    np.save(out.with_suffix(".npy"), video)

    if out.suffix.lower() == ".mp4":
        try:
            import imageio
            imageio.mimwrite(str(out), list(video), fps=fps, quality=8)
            print(f"Saved video: {out}")
        except Exception as e:
            print(f"[warn] mp4 write failed ({e}); .npy saved at "
                  f"{out.with_suffix('.npy')}")
    print(f"Saved frames: {out.with_suffix('.npy')}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",          required=True, help="DiT checkpoint (.pt)")
    ap.add_argument("--vae_ckpt",      default=None, help="Trained VAE3D checkpoint")
    ap.add_argument("--pretrained_vae", default=None, help="HF VAE id (must match training)")
    ap.add_argument("--prompt",        required=True, help="Text prompt")
    ap.add_argument("--ref_image",     required=True, help="Reference image (any resolution)")
    ap.add_argument("--out",           default="outputs/sample.mp4")
    ap.add_argument("--num_steps",     type=int,   default=50)
    ap.add_argument("--cfg_scale",     type=float, default=None,
                    help="Override CFG scale (default: value from checkpoint cfg)")
    ap.add_argument("--seed",          type=int,   default=0)
    ap.add_argument("--fps",           type=int,   default=8)
    ap.add_argument("--use_raw",       action="store_true",
                    help="Use raw (non-EMA) weights even if EMA is present")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # ── Load checkpoint + its saved config ───────────────────────────
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg  = ckpt.get("cfg", {})
    if not cfg:
        print("[warn] checkpoint has no saved cfg; falling back to defaults. "
              "If arch differs from defaults, pass matching values manually.")

    num_frames = cfg.get("num_frames", 17)
    frame_h    = cfg.get("frame_h", 128)
    frame_w    = cfg.get("frame_w", 128)
    size       = frame_h  # square training crop
    cfg_scale  = args.cfg_scale if args.cfg_scale is not None else cfg.get("cfg_scale", 7.0)

    # ── VAE ───────────────────────────────────────────────────────────
    vae = load_vae(args, cfg, device)

    # Probe latent shape through the real encoder (same as training)
    probe = torch.zeros(1, 3, num_frames, frame_h, frame_w, device=device)
    lat   = vae.encode_mean(probe)
    _, lat_c, lat_t, lat_h, lat_w = lat.shape
    del probe, lat
    print(f"Latent shape: C={lat_c}, T={lat_t}, H={lat_h}, W={lat_w}")

    # ── Conditioning encoders (must match training ids) ──────────────
    cond = ConditioningPipeline(
        t5_model   = cfg.get("t5_model", "google/t5-v1_1-base"),
        clip_model = cfg.get("clip_model", "openai/clip-vit-large-patch14"),
        device     = device,
    )

    # ── DiT ───────────────────────────────────────────────────────────
    dit = VideoDiT(
        latent_channels = lat_c,
        latent_t        = lat_t,
        latent_h        = lat_h,
        latent_w        = lat_w,
        patch_t         = cfg.get("patch_t", 1),
        patch_h         = cfg.get("patch_h", 2),
        patch_w         = cfg.get("patch_w", 2),
        hidden_dim      = cfg.get("dit_hidden", 1024),
        num_heads       = cfg.get("dit_heads", 16),
        num_layers      = cfg.get("dit_layers", 24),
        cond_dim        = cfg.get("dit_cond", 1024),
        text_dim        = cond.text_dim,
        clip_dim        = cond.clip_dim,
        ffn_mult        = cfg.get("ffn_mult", 8/3),
        dropout         = 0.0,
    ).to(device)

    # Prefer EMA weights (what training sampled with) unless told otherwise
    if "ema" in ckpt and not args.use_raw:
        dit.load_state_dict(ckpt["ema"])
        print("Loaded EMA weights")
    else:
        dit.load_state_dict(ckpt["dit"])
        print("Loaded raw (non-EMA) weights")
    dit.eval()

    # ── Scheduler ──────────────────────────────────────────────────────
    scheduler = FlowMatchingScheduler(
        timestep_strategy = cfg.get("timestep_strategy", "logit"),
        cfg_scale         = cfg_scale,
        p_uncond          = cfg.get("p_uncond", 0.1),
    )

    # ── Encode conditioning ────────────────────────────────────────────
    ref_pil, ref_tensor = load_ref_image(args.ref_image, size)
    ref_tensor = ref_tensor.to(device)

    txt_cond = cond.encode_text([args.prompt])
    img_cond = cond.encode_image([ref_pil])

    z_img = vae.encode_mean(ref_tensor)   # (1, C, 1, H', W') — I2V conditioning

    # Diagnostic stats — compare these against infer_from_trainclip.py output.
    # If z_img / txt / img differ a lot from the working training-clip run,
    # this image's preprocessing is the problem.
    print("\n=== CONDITIONING STATS (compare to infer_from_trainclip) ===")
    print(f"z_img : shape={tuple(z_img.shape)} dtype={z_img.dtype} "
          f"mean={z_img.mean():.4f} std={z_img.std():.4f}")
    print(f"txt   : shape={tuple(txt_cond.tokens.shape)} "
          f"mean={txt_cond.tokens.mean():.4f} std={txt_cond.tokens.std():.4f}")
    print(f"img   : shape={tuple(img_cond.tokens.shape)} "
          f"mean={img_cond.tokens.mean():.4f} std={img_cond.tokens.std():.4f}")
    print("=" * 52 + "\n")

    L = txt_cond.tokens.shape[1]
    M = img_cond.tokens.shape[1]
    null_txt, null_img = dit.get_null_conditioning(1, L, M, device)

    # ── Sample ──────────────────────────────────────────────────────────
    print(f"Sampling: steps={args.num_steps}, cfg={cfg_scale}, seed={args.seed}")
    z0_gen = scheduler.euler_sample(
        model      = lambda z, t, txt, img: dit(z, t, txt, img, z_img),
        shape      = (1, lat_c, lat_t, lat_h, lat_w),
        c_txt      = txt_cond.tokens,
        c_img      = img_cond.tokens,
        null_c_txt = null_txt,
        null_c_img = null_img,
        num_steps  = args.num_steps,
        cfg_scale  = cfg_scale,
        device     = device,
        verbose    = True,
    )

    # ── Decode ──────────────────────────────────────────────────────────
    video = vae.decode(z0_gen)                       # (1, 3, T, H, W) in [-1,1]
    video = video[0].permute(1, 2, 3, 0).cpu().float().numpy()  # (T, H, W, 3)

    save_video(video, args.out, fps=args.fps)
    print("Done.")


if __name__ == "__main__":
    main()