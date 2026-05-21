# dit/rope.py
import torch
import torch.nn as nn
from typing import Tuple


# ─────────────────────────────────────────────
# 1D RoPE frequencies
# ─────────────────────────────────────────────

def get_1d_freqs(
    dim: int,
    max_len: int,
    base: float = 10000.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute cosine/sine frequency table for 1D RoPE.

    Args:
        dim:     number of dims to use for this axis (must be even)
        max_len: maximum sequence length for this axis
        base:    RoPE base (controls frequency range)

    Returns:
        freqs: (max_len, dim/2)  — the angles m*θ_i for each position m
    """
    assert dim % 2 == 0, "RoPE dim must be even"

    # θ_i = 1 / base^(2i/dim)   for i = 0, 1, ..., dim/2 - 1
    i = torch.arange(0, dim, 2, device=device).float()       # (dim/2,)
    theta = 1.0 / (base ** (i / dim))                        # (dim/2,)

    # positions m = 0, 1, ..., max_len-1
    positions = torch.arange(max_len, device=device).float() # (max_len,)

    # outer product: freqs[m, i] = m * θ_i
    freqs = torch.outer(positions, theta)                    # (max_len, dim/2)

    return freqs


def freqs_to_cos_sin(freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert frequency angles to cos and sin tables.
    Returns cos, sin each of shape (max_len, dim/2).
    """
    return freqs.cos(), freqs.sin()


# ─────────────────────────────────────────────
# Apply rotation to a single tensor
# ─────────────────────────────────────────────

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rearranges pairs for rotation:
    [x_0, x_1, x_2, x_3, ...] → [-x_1, x_0, -x_3, x_2, ...]

    This implements the 2D rotation without explicit matrix multiply:
    [cos  -sin] [x_0]   [x_0*cos - x_1*sin]
    [sin   cos] [x_1] = [x_0*sin + x_1*cos]
    Which equals: x*cos + rotate_half(x)*sin
    """
    # x: (..., dim)
    # Split last dim into pairs
    x1 = x[..., 0::2]   # even indices: x_0, x_2, x_4, ...
    x2 = x[..., 1::2]   # odd  indices: x_1, x_3, x_5, ...

    # Interleave: [-x_1, x_0, -x_3, x_2, ...]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope_1d(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply 1D RoPE rotation to tensor x.

    Args:
        x:   (..., seq_len, dim)
        cos: (seq_len, dim/2)
        sin: (seq_len, dim/2)

    Returns:
        rotated x, same shape
    """
    # Repeat cos/sin to match full dim: (seq_len, dim/2) → (seq_len, dim)
    cos = cos.repeat_interleave(2, dim=-1)   # (seq_len, dim)
    sin = sin.repeat_interleave(2, dim=-1)   # (seq_len, dim)

    # Broadcast over batch and head dims
    # x: (B, heads, seq_len, dim)
    # cos/sin need to broadcast: (1, 1, seq_len, dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Apply rotation: x*cos + rotate_half(x)*sin
    return x * cos + rotate_half(x) * sin


# ─────────────────────────────────────────────
# 3D RoPE
# ─────────────────────────────────────────────

class RoPE3D(nn.Module):
    """
    3D Rotary Position Embedding for video tokens.

    Each token has position (t, h, w).
    The head dimension D_head is split into three chunks:
        [0 : D_t]           → temporal RoPE
        [D_t : D_t+D_h]     → height RoPE
        [D_t+D_h : D_head]  → width RoPE

    Q and K are rotated independently in each chunk.
    """

    def __init__(
        self,
        head_dim: int,
        max_t: int = 64,      # max temporal positions (latent frames)
        max_h: int = 64,      # max height positions (latent H)
        max_w: int = 64,      # max width positions (latent W)
        base_t: float = 10000.0,
        base_h: float = 10000.0,
        base_w: float = 10000.0,
        theta_t_ratio: float = 0.25,   # fraction of head_dim for time
        theta_h_ratio: float = 0.375,  # fraction for height
        # width gets the remaining fraction automatically
    ):
        super().__init__()

        self.head_dim = head_dim

        # Compute per-axis dims (must all be even, must sum to head_dim)
        self.dim_t = int(head_dim * theta_t_ratio) // 2 * 2
        self.dim_h = int(head_dim * theta_h_ratio) // 2 * 2
        self.dim_w = head_dim - self.dim_t - self.dim_h

        assert self.dim_w % 2 == 0, \
            f"Width dim {self.dim_w} must be even. Adjust ratios."
        assert self.dim_t + self.dim_h + self.dim_w == head_dim, \
            "Dim split must sum to head_dim"

        # Pre-compute frequency tables and register as buffers
        # (buffers move with .to(device) but aren't trained parameters)
        freqs_t = get_1d_freqs(self.dim_t, max_t, base_t)
        freqs_h = get_1d_freqs(self.dim_h, max_h, base_h)
        freqs_w = get_1d_freqs(self.dim_w, max_w, base_w)

        cos_t, sin_t = freqs_to_cos_sin(freqs_t)
        cos_h, sin_h = freqs_to_cos_sin(freqs_h)
        cos_w, sin_w = freqs_to_cos_sin(freqs_w)

        self.register_buffer("cos_t", cos_t)  # (max_t, dim_t/2)
        self.register_buffer("sin_t", sin_t)
        self.register_buffer("cos_h", cos_h)  # (max_h, dim_h/2)
        self.register_buffer("sin_h", sin_h)
        self.register_buffer("cos_w", cos_w)  # (max_w, dim_w/2)
        self.register_buffer("sin_w", sin_w)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 3D RoPE to Q and K.

        Args:
            q:       (B, heads, N, head_dim)   query
            k:       (B, heads, N, head_dim)   key
            pos_ids: (B, N, 3)                 position ids [t, h, w] per token

        Returns:
            q_rot, k_rot: same shape as q, k
        """
        # Extract per-axis position ids
        # pos_ids: (B, N, 3) → each: (B, N)
        t_ids = pos_ids[..., 0]   # (B, N)
        h_ids = pos_ids[..., 1]   # (B, N)
        w_ids = pos_ids[..., 2]   # (B, N)

        # Look up cos/sin for each token's position
        # cos_t: (max_t, dim_t/2) → index with t_ids: (B, N, dim_t/2)
        cos_t = self.cos_t[t_ids]   # (B, N, dim_t/2)
        sin_t = self.sin_t[t_ids]
        cos_h = self.cos_h[h_ids]   # (B, N, dim_h/2)
        sin_h = self.sin_h[h_ids]
        cos_w = self.cos_w[w_ids]   # (B, N, dim_w/2)
        sin_w = self.sin_w[w_ids]

        # Add head dimension for broadcasting
        # (B, N, dim/2) → (B, 1, N, dim/2)
        cos_t = cos_t.unsqueeze(1)
        sin_t = sin_t.unsqueeze(1)
        cos_h = cos_h.unsqueeze(1)
        sin_h = sin_h.unsqueeze(1)
        cos_w = cos_w.unsqueeze(1)
        sin_w = sin_w.unsqueeze(1)

        # Split Q and K along head_dim into (T, H, W) chunks
        # q: (B, heads, N, head_dim)
        q_t = q[..., :self.dim_t]
        q_h = q[..., self.dim_t : self.dim_t + self.dim_h]
        q_w = q[..., self.dim_t + self.dim_h :]

        k_t = k[..., :self.dim_t]
        k_h = k[..., self.dim_t : self.dim_t + self.dim_h]
        k_w = k[..., self.dim_t + self.dim_h :]

        # Apply 1D RoPE independently to each axis chunk
        q_t = apply_rope_1d(q_t, cos_t, sin_t)
        q_h = apply_rope_1d(q_h, cos_h, sin_h)
        q_w = apply_rope_1d(q_w, cos_w, sin_w)

        k_t = apply_rope_1d(k_t, cos_t, sin_t)
        k_h = apply_rope_1d(k_h, cos_h, sin_h)
        k_w = apply_rope_1d(k_w, cos_w, sin_w)

        # Concatenate back along head_dim
        q_rot = torch.cat([q_t, q_h, q_w], dim=-1)
        k_rot = torch.cat([k_t, k_h, k_w], dim=-1)

        return q_rot, k_rot


# ─────────────────────────────────────────────
# Position ID generator
# ─────────────────────────────────────────────

def make_3d_pos_ids(
    t: int,
    h: int,
    w: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate (t*h*w, 3) position id tensor for a video latent of shape (T, H, W).
    Tokens are ordered as: T-major → H-major → W  (row-major flattening).

    Returns:
        pos_ids: (1, T*H*W, 3)   with [t_id, h_id, w_id] per token
    """
    # Build grids for each axis
    t_grid = torch.arange(t, device=device)  # (T,)
    h_grid = torch.arange(h, device=device)  # (H,)
    w_grid = torch.arange(w, device=device)  # (W,)

    # Meshgrid → each has shape (T, H, W)
    grid_t, grid_h, grid_w = torch.meshgrid(t_grid, h_grid, w_grid, indexing="ij")

    # Stack and flatten: (T, H, W, 3) → (T*H*W, 3)
    pos_ids = torch.stack([grid_t, grid_h, grid_w], dim=-1)  # (T, H, W, 3)
    pos_ids = pos_ids.reshape(-1, 3)                          # (T*H*W, 3)

    return pos_ids.unsqueeze(0)   # (1, N, 3) — batch dim for broadcasting