# dit/blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dit.attention import SelfAttention3D, CrossAttention
from dit.rope import RoPE3D


# ─────────────────────────────────────────────
# Timestep embedding
# ─────────────────────────────────────────────

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timestep t ∈ [0,1] into a vector.

    Uses sinusoidal embedding (like transformer positional encoding)
    followed by a small MLP to get a rich conditioning signal.

    Why sinusoidal: the model needs to distinguish t=0.01 from t=0.02
    very precisely near t=0, but t=0.5 from t=0.51 less so.
    Sinusoids at multiple frequencies naturally provide this resolution.
    """

    def __init__(self, hidden_dim: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim

        # MLP: freq_dim → hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """
        t: (B,) timesteps in [0, 1]
        Returns: (B, dim) sinusoidal embeddings
        """
        assert dim % 2 == 0
        device = t.device

        # Frequencies: ω_i = 1 / 10000^(2i/dim)
        half = dim // 2
        freqs = torch.arange(half, device=device).float()
        freqs = torch.exp(-freqs * (torch.log(torch.tensor(10000.0)) / (half - 1)))

        # Outer product: (B, half)
        # Scale t to [0, 1000] so the embedding has enough resolution
        args = t[:, None] * freqs[None, :] * 1000.0

        embedding = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) in [0, 1]
        Returns: (B, hidden_dim)
        """
        sincos = self.sinusoidal_embedding(t, self.freq_dim)
        return self.mlp(sincos)


# ─────────────────────────────────────────────
# adaLN modulation
# ─────────────────────────────────────────────

class AdaLNModulation(nn.Module):
    """
    Predicts (γ, β, α) for one sub-block from the conditioning signal.

    Output: 3 * hidden_dim values split into:
        γ — feature scale  (multiplies normalized x)
        β — feature shift  (added after scale)
        α — residual gate  (scales the sub-block output before adding to residual)

    Initialized to zero → identity block at init (adaLN-Zero trick).
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()

        self.norm = nn.RMSNorm(cond_dim)

        # Single linear predicts all 3 modulation params at once
        # Initialize to zero → adaLN-Zero
        self.proj = nn.Linear(cond_dim, 3 * hidden_dim, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, c: torch.Tensor) -> tuple:
        """
        c: (B, cond_dim)  pooled conditioning signal
        Returns: γ, β, α each of shape (B, 1, hidden_dim)
                 (middle dim=1 broadcasts over sequence length N)
        """
        c = self.norm(c)
        out = self.proj(c)                        # (B, 3*hidden_dim)
        γ, β, α = out.chunk(3, dim=-1)            # each: (B, hidden_dim)

        # Unsqueeze for broadcasting over sequence: (B, 1, hidden_dim)
        return γ.unsqueeze(1), β.unsqueeze(1), α.unsqueeze(1)


def modulate(x: torch.Tensor, γ: torch.Tensor, β: torch.Tensor) -> torch.Tensor:
    """
    Apply adaLN scale and shift to normalized x.
    x: (B, N, D)
    γ, β: (B, 1, D)  — broadcast over N
    """
    return x * (1.0 + γ) + β


# ─────────────────────────────────────────────
# Feed-Forward Network (SwiGLU)
# ─────────────────────────────────────────────

class FeedForward(nn.Module):
    """
    FFN with SwiGLU activation.

    Standard FFN:   Linear → ReLU/GELU → Linear
    SwiGLU FFN:     (Linear_gate * SiLU(Linear_gate)) → Linear_out

    SwiGLU splits the hidden projection into two halves —
    one acts as a gate, one as the value. Their elementwise product
    gives a learned gating mechanism. Used in LLaMA, Wan2.1, SD3.

    hidden_mult: multiplier for inner dim relative to hidden_dim
                 typically 4 for standard FFN, but SwiGLU uses 8/3
                 to keep parameter count the same after the gating split
    """

    def __init__(self, hidden_dim: int, hidden_mult: float = 8/3):
        super().__init__()

        # Inner dim — round to nearest multiple of 256 for efficiency
        inner_dim = int(hidden_dim * hidden_mult)
        inner_dim = (inner_dim + 255) // 256 * 256

        # gate and value projections (fused)
        self.w_gate  = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.w_val   = nn.Linear(hidden_dim, inner_dim, bias=False)
        self.w_out   = nn.Linear(inner_dim,  hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate(x) * SiLU(val(x))
        gate = self.w_gate(x)
        val  = self.w_val(x)
        return self.w_out(F.silu(gate) * val)


# ─────────────────────────────────────────────
# Full DiT Block
# ─────────────────────────────────────────────

class DiTBlock(nn.Module):
    """
    Single DiT transformer block.

    Sub-block order:
        1. adaLN → 3D Self-Attention  → gated residual
        2. adaLN → Cross-Attention    → gated residual
        3. adaLN → FFN (SwiGLU)      → gated residual

    Each sub-block gets its own (γ, β, α) from adaLN modulation.
    The conditioning signal c = timestep_emb + pooled_text_emb + pooled_img_emb.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        cond_dim: int,          # dim of conditioning signal c
        text_dim: int,          # dim of text encoder output (for cross-attn)
        rope: RoPE3D = None,
        dropout: float = 0.0,
        ffn_mult: float = 8/3,
    ):
        super().__init__()

        # ── Norms (pre-norm architecture) ────────────────────────────
        self.norm1 = nn.RMSNorm(hidden_dim)   # before self-attn
        self.norm2 = nn.RMSNorm(hidden_dim)   # before cross-attn
        self.norm3 = nn.RMSNorm(hidden_dim)   # before FFN

        # ── Attention ────────────────────────────────────────────────
        self.self_attn  = SelfAttention3D(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            rope=rope,
            qk_norm=True,
        )
        self.cross_attn = CrossAttention(
            hidden_dim=hidden_dim,
            cond_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout,
            qk_norm=True,
        )

        # ── FFN ──────────────────────────────────────────────────────
        self.ffn = FeedForward(hidden_dim, ffn_mult)

        # ── adaLN modulation (one per sub-block) ─────────────────────
        # Three separate modulators: self-attn, cross-attn, FFN
        self.ada1 = AdaLNModulation(hidden_dim, cond_dim)
        self.ada2 = AdaLNModulation(hidden_dim, cond_dim)
        self.ada3 = AdaLNModulation(hidden_dim, cond_dim)

    def forward(
        self,
        x: torch.Tensor,         # (B, N, hidden_dim)  video tokens
        c: torch.Tensor,         # (B, cond_dim)        pooled conditioning
        cond_tokens: torch.Tensor,  # (B, N_cond, text_dim)  text+image tokens
        pos_ids: torch.Tensor,   # (B, N, 3)            3D position ids
    ) -> torch.Tensor:

        # ── 1. Self-attention with adaLN ─────────────────────────────
        γ1, β1, α1 = self.ada1(c)
        x = x + α1 * self.self_attn(
            modulate(self.norm1(x), γ1, β1),
            pos_ids,
        )

        # ── 2. Cross-attention with adaLN ────────────────────────────
        γ2, β2, α2 = self.ada2(c)
        x = x + α2 * self.cross_attn(
            modulate(self.norm2(x), γ2, β2),
            cond_tokens,
        )

        # ── 3. FFN with adaLN ────────────────────────────────────────
        γ3, β3, α3 = self.ada3(c)
        x = x + α3 * self.ffn(
            modulate(self.norm3(x), γ3, β3),
        )

        return x


# ─────────────────────────────────────────────
# Final output layer
# ─────────────────────────────────────────────

class FinalLayer(nn.Module):
    """
    Last layer of the DiT.
    adaLN → Linear → unpatch-ready output.

    Projects hidden_dim → patch_dim (C * patch_t * patch_h * patch_w)
    so the output can be reshaped back into a video latent.
    """

    def __init__(self, hidden_dim: int, patch_dim: int, cond_dim: int):
        super().__init__()

        self.norm     = nn.RMSNorm(hidden_dim)
        self.ada_out  = AdaLNModulation(hidden_dim, cond_dim)
        self.proj_out = nn.Linear(hidden_dim, patch_dim, bias=True)

        # Initialize output projection to zero
        # → model outputs zero velocity at init → stable early training
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, hidden_dim)
        c: (B, cond_dim)
        Returns: (B, N, patch_dim)
        """
        γ, β, _ = self.ada_out(c)   # α not used in final layer
        x = modulate(self.norm(x), γ, β)
        return self.proj_out(x)



        