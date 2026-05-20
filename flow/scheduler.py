# flow/scheduler.py
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Callable


# ─────────────────────────────────────────────
# Output container
# ─────────────────────────────────────────────

@dataclass
class FlowMatchingOutput:
    loss: torch.Tensor          # training loss (scalar)
    v_pred: torch.Tensor        # predicted velocity
    v_target: torch.Tensor      # target velocity (ε - x_0)
    x_t: torch.Tensor           # noisy latent at timestep t
    t: torch.Tensor             # sampled timesteps (B,)


# ─────────────────────────────────────────────
# Timestep sampling strategies
# ─────────────────────────────────────────────

class TimestepSampler:
    """
    Samples timesteps t ∈ [0, 1] during training.

    Uniform sampling works but recent work (SD3, Wan2.1) shows that
    sampling more timesteps near t=0.5 (the "middle" of the trajectory)
    improves training efficiency — those timesteps are hardest for the
    model and carry the most gradient signal.

    Strategies:
        uniform  → t ~ Uniform(0, 1)
        logit    → t ~ logit-normal (more mass near 0.5)
    """

    def __init__(self, strategy: str = "logit", logit_mean: float = 0.0, logit_std: float = 1.0):
        assert strategy in ("uniform", "logit")
        self.strategy   = strategy
        self.logit_mean = logit_mean
        self.logit_std  = logit_std

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.strategy == "uniform":
            return torch.rand(batch_size, device=device)

        elif self.strategy == "logit":
            # Sample from logit-normal:
            # u ~ N(mean, std), then t = sigmoid(u)
            # This concentrates mass near t=0.5 (sigmoid(0) = 0.5)
            u = torch.randn(batch_size, device=device)
            u = self.logit_mean + self.logit_std * u
            return torch.sigmoid(u)


# ─────────────────────────────────────────────
# Flow Matching Scheduler
# ─────────────────────────────────────────────

class FlowMatchingScheduler(nn.Module):
    """
    Implements flow matching for video latent diffusion.

    The flow interpolates linearly between data (x_0) and noise (ε):
        x_t = (1 - t) * x_0 + t * ε

    The target velocity is constant:
        v* = ε - x_0

    The model learns: v_θ(x_t, t, c) ≈ v*

    At inference, we integrate the ODE from t=1 (noise) to t=0 (data):
        x_{t - dt} = x_t - dt * v_θ(x_t, t, c)
    """

    def __init__(
        self,
        timestep_strategy: str = "logit",    # how to sample t during training
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        cfg_scale: float = 7.0,              # classifier-free guidance scale
        p_uncond: float = 0.1,               # prob of dropping conditioning
    ):
        super().__init__()
        self.cfg_scale = cfg_scale
        self.p_uncond  = p_uncond
        self.t_sampler = TimestepSampler(timestep_strategy, logit_mean, logit_std)

    # ── Forward noising ──────────────────────────────────────────────

    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate between data and noise at timestep t.

        Args:
            x_0:   (B, C, T, H, W)  clean latent from VAE
            t:     (B,)              timesteps in [0, 1]
            noise: optional pre-sampled noise (for reproducibility)

        Returns:
            x_t:   noisy latent at timestep t
            noise: the noise that was added (needed to compute target)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Reshape t: (B,) → (B, 1, 1, 1, 1) to broadcast over (C, T, H, W)
        t_broadcast = t.view(-1, 1, 1, 1, 1)

        # Linear interpolation: x_t = (1-t)*x_0 + t*ε
        x_t = (1.0 - t_broadcast) * x_0 + t_broadcast * noise

        return x_t, noise

    def get_velocity_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Target velocity the model should predict.
        For linear flow: v* = ε - x_0  (constant along trajectory)

        Args:
            x_0:   (B, C, T, H, W)  clean latent
            noise: (B, C, T, H, W)  sampled noise

        Returns:
            v_target: (B, C, T, H, W)
        """
        return noise - x_0

    # ── Training step ────────────────────────────────────────────────

    def training_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        c_txt: torch.Tensor,                 # text conditioning (B, L, D)
        c_img: torch.Tensor,                 # image conditioning (B, N, D)
        null_c_txt: torch.Tensor,            # null text embedding for CFG dropout
        null_c_img: torch.Tensor,            # null image embedding for CFG dropout
    ) -> FlowMatchingOutput:
        """
        Single training step.

        1. Sample t ~ TimestepSampler
        2. Sample noise ε ~ N(0, I)
        3. Compute x_t = (1-t)*x_0 + t*ε
        4. Drop conditioning with prob p_uncond  (CFG training)
        5. Predict velocity v_θ(x_t, t, c)
        6. Loss = MSE(v_pred, v_target)
        """
        B      = x_0.shape[0]
        device = x_0.device

        # 1. Sample timesteps
        t = self.t_sampler.sample(B, device)

        # 2. Sample noise and create noisy latent
        noise = torch.randn_like(x_0)
        x_t, noise = self.add_noise(x_0, t, noise)

        # 3. Compute target velocity
        v_target = self.get_velocity_target(x_0, noise)

        # 4. CFG conditioning dropout
        # For each sample independently, decide whether to drop conditioning
        drop_mask = torch.rand(B, device=device) < self.p_uncond  # (B,) bool

        # Replace conditioning with null embeddings where drop_mask is True
        c_txt_in = torch.where(
            drop_mask.view(B, 1, 1),        # (B, 1, 1) broadcasts over (L, D)
            null_c_txt.expand_as(c_txt),
            c_txt,
        )
        c_img_in = torch.where(
            drop_mask.view(B, 1, 1),
            null_c_img.expand_as(c_img),
            c_img,
        )

        # 5. Predict velocity
        # Model takes: noisy latent, timestep, text cond, image cond
        v_pred = model(x_t, t, c_txt_in, c_img_in)

        # 6. MSE loss
        loss = torch.nn.functional.mse_loss(v_pred, v_target)

        return FlowMatchingOutput(
            loss     = loss,
            v_pred   = v_pred,
            v_target = v_target,
            x_t      = x_t,
            t        = t,
        )

    # ── Inference: ODE solvers ───────────────────────────────────────

    @torch.no_grad()
    def euler_sample(
        self,
        model: nn.Module,
        shape: tuple,
        c_txt: torch.Tensor,
        c_img: torch.Tensor,
        null_c_txt: torch.Tensor,
        null_c_img: torch.Tensor,
        num_steps: int = 20,
        cfg_scale: Optional[float] = None,
        device: torch.device = torch.device("cuda"),
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Euler ODE solver — simplest inference method.
        Integrates from t=1 (noise) to t=0 (data) in fixed steps.

        Args:
            shape:     latent shape (B, C, T', H', W')
            c_txt:     text conditioning
            c_img:     image conditioning
            num_steps: number of Euler steps (more = better quality, slower)
            cfg_scale: override self.cfg_scale if provided

        Returns:
            x_0: (B, C, T', H', W')  denoised latent
        """
        cfg = cfg_scale if cfg_scale is not None else self.cfg_scale

        # Start from pure noise at t=1
        x_t = torch.randn(shape, device=device)

        # Timestep schedule: linearly from 1 → 0
        # We use num_steps+1 points: [1.0, ..., 0.0]
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        dt_const  = 1.0 / num_steps   # step size (constant for Euler)

        for i in range(num_steps):
            t_cur = timesteps[i]
            t_batch = t_cur.expand(shape[0])   # (B,)

            # CFG: run model twice
            # 1. Conditioned
            v_cond = model(x_t, t_batch, c_txt, c_img)

            # 2. Unconditioned
            v_uncond = model(
                x_t, t_batch,
                null_c_txt.expand_as(c_txt),
                null_c_img.expand_as(c_img),
            )

            # 3. Guided velocity
            v_guided = v_uncond + cfg * (v_cond - v_uncond)

            # 4. Euler step: x_{t-dt} = x_t - dt * v
            x_t = x_t - dt_const * v_guided

            if verbose:
                print(f"Step {i+1}/{num_steps} | t={t_cur:.3f} | "
                      f"|v|={v_guided.abs().mean():.4f}")

        return x_t   # x_0: denoised latent

    @torch.no_grad()
    def dpm_sample(
        self,
        model: nn.Module,
        shape: tuple,
        c_txt: torch.Tensor,
        c_img: torch.Tensor,
        null_c_txt: torch.Tensor,
        null_c_img: torch.Tensor,
        num_steps: int = 20,
        cfg_scale: Optional[float] = None,
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        """
        Heun's method (2nd order Runge-Kutta).
        Better quality than Euler at the same step count.
        Costs 2× model evaluations per step.

        Euler:  x_{t-dt} = x_t - dt * v(x_t)            1 eval/step
        Heun:   x̃  = x_t - dt * v(x_t)                  1st eval
                x_{t-dt} = x_t - dt/2 * (v(x_t) + v(x̃)) 2nd eval (corrector)
        """
        cfg = cfg_scale if cfg_scale is not None else self.cfg_scale

        x_t       = torch.randn(shape, device=device)
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        dt_const  = 1.0 / num_steps

        for i in range(num_steps):
            t_cur  = timesteps[i]
            t_next = timesteps[i + 1]
            t_batch = t_cur.expand(shape[0])

            # ── Predictor (Euler step)
            v_cond   = model(x_t, t_batch, c_txt, c_img)
            v_uncond = model(
                x_t, t_batch,
                null_c_txt.expand_as(c_txt),
                null_c_img.expand_as(c_img),
            )
            v1 = v_uncond + cfg * (v_cond - v_uncond)
            x_pred = x_t - dt_const * v1

            # ── Corrector (evaluate at predicted next point)
            t_batch_next = t_next.expand(shape[0])
            v_cond2   = model(x_pred, t_batch_next, c_txt, c_img)
            v_uncond2 = model(
                x_pred, t_batch_next,
                null_c_txt.expand_as(c_txt),
                null_c_img.expand_as(c_img),
            )
            v2 = v_uncond2 + cfg * (v_cond2 - v_uncond2)

            # Average the two velocity estimates
            x_t = x_t - dt_const * 0.5 * (v1 + v2)

        return x_t