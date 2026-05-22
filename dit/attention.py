# dit/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dit.rope import RoPE3D, make_3d_pos_ids


# ─────────────────────────────────────────────
# Multi-head Self-Attention with 3D RoPE
# ─────────────────────────────────────────────

class SelfAttention3D(nn.Module):
    """
    Full 3D self-attention over video tokens.
    Every token attends to every other token (spatial + temporal together).

    Uses:
        - RoPE3D for position encoding (applied to Q, K only)
        - Flash Attention via F.scaled_dot_product_attention
        - RMSNorm on Q, K before RoPE (QK-norm) for training stability
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        rope: RoPE3D = None,        # shared RoPE instance across all blocks
        qk_norm: bool = True,       # normalize Q,K before attention
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}"

        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.dropout    = dropout
        self.rope       = rope

        # Q, K, V projections — fused into one matrix for efficiency
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # QK-norm: normalize Q and K separately before attention
        # Prevents attention logit explosion at large scales
        # Used in Wan2.1, SD3, FLUX
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        pos_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, N, hidden_dim)   video tokens
            pos_ids: (B, N, 3)            [t, h, w] position per token

        Returns:
            (B, N, hidden_dim)
        """
        B, N, D = x.shape

        # 1. QKV projection
        qkv = self.qkv_proj(x)                        # (B, N, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)                # each: (B, N, D)

        # 2. Reshape to multi-head: (B, N, D) → (B, heads, N, head_dim)
        def reshape_heads(t):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)    # (B, heads, N, head_dim)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # 3. QK-norm (before RoPE — norm the raw projections)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. Apply 3D RoPE to Q and K
        if self.rope is not None:
            q, k = self.rope(q, k, pos_ids)

        # 5. Flash Attention
        # F.scaled_dot_product_attention handles scaling (1/√head_dim)
        # and dispatches to Flash Attention kernel automatically
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )   # (B, heads, N, head_dim)

        # 6. Merge heads: (B, heads, N, head_dim) → (B, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        # 7. Output projection
        return self.out_proj(out)


# ─────────────────────────────────────────────
# Cross-Attention (video queries, text/image keys+values)
# ─────────────────────────────────────────────

class CrossAttention(nn.Module):
    """
    Cross-attention where:
        Query  → video tokens        (B, N_video, D)
        Key/V  → conditioning tokens (B, N_cond,  D_cond)

    No RoPE here — conditioning tokens (text, image) don't
    have meaningful 3D spatial positions.
    QK-norm still applied for stability.
    """

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,          # dimension of conditioning tokens (text/image encoder output)
        num_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = True,
    ):
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.dropout    = dropout

        # Query comes from video tokens (dim = hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Key and Value come from conditioning (dim = cond_dim → hidden_dim)
        self.k_proj = nn.Linear(cond_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(cond_dim, hidden_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,         # video tokens  (B, N_video, hidden_dim)
        c: torch.Tensor,         # cond tokens   (B, N_cond,  cond_dim)
    ) -> torch.Tensor:
        """
        Returns:
            (B, N_video, hidden_dim)
        """
        B, N_video, _ = x.shape
        N_cond = c.shape[1]

        # 1. Project Q from video, K/V from conditioning
        q = self.q_proj(x)    # (B, N_video, D)
        k = self.k_proj(c)    # (B, N_cond,  D)
        v = self.v_proj(c)    # (B, N_cond,  D)

        # 2. Reshape to multi-head
        def reshape_heads(t, N):
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q, N_video)   # (B, heads, N_video, head_dim)
        k = reshape_heads(k, N_cond)    # (B, heads, N_cond,  head_dim)
        v = reshape_heads(v, N_cond)    # (B, heads, N_cond,  head_dim)

        # 3. QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. Flash Attention
        # Q attends to K/V from conditioning
        # output shape: (B, heads, N_video, head_dim)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        # 5. Merge heads
        out = out.transpose(1, 2).contiguous().view(B, N_video, self.head_dim * self.num_heads)

        return self.out_proj(out)