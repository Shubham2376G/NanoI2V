# dit/dit.py
import torch
import torch.nn as nn
from dit.rope import RoPE3D, make_3d_pos_ids
from dit.blocks import DiTBlock, TimestepEmbedder, FinalLayer


class VideoDiT(nn.Module):
    """
    Full Video Diffusion Transformer for I2V generation.

    Takes noisy video latent + image conditioning latent,
    predicts the flow matching velocity field.

    Architecture:
        Patchify → N × DiTBlock → FinalLayer → Unpatchify

    I2V conditioning:
        Image latent is tiled across time and concatenated with
        the noisy latent on the channel axis before patchifying.
        This doubles the input channels — the model learns to
        "complete" the video given the first frame.
    """

    def __init__(
        self,
        # Latent dims (from VAE)
        latent_channels: int = 16,
        latent_t: int = 5,
        latent_h: int = 32,
        latent_w: int = 32,

        # Patch size (in latent space)
        patch_t: int = 1,           # temporal patch size
        patch_h: int = 2,           # height patch size
        patch_w: int = 2,           # width patch size

        # Transformer
        hidden_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 24,
        dropout: float = 0.0,
        ffn_mult: float = 8/3,

        # Conditioning dims
        cond_dim: int = 1024,       # adaLN conditioning dim
        text_dim: int = 4096,       # T5-XXL output dim
        clip_dim: int = 1024,       # CLIP ViT-L output dim

        # RoPE
        rope_base_t: float = 10000.0,
        rope_base_h: float = 10000.0,
        rope_base_w: float = 10000.0,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.patch_t = patch_t
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.hidden_dim = hidden_dim

        # ── Derived dims ──────────────────────────────────────────────
        # After patchifying
        self.num_t = latent_t // patch_t
        self.num_h = latent_h // patch_h
        self.num_w = latent_w // patch_w
        self.num_tokens = self.num_t * self.num_h * self.num_w

        # Each patch has (2 * latent_channels) channels after I2V concat
        # × (patch_t * patch_h * patch_w) spatial elements
        self.in_channels  = 2 * latent_channels   # I2V: noisy + image latent
        self.patch_dim    = self.in_channels * patch_t * patch_h * patch_w
        self.out_patch_dim = latent_channels * patch_t * patch_h * patch_w

        # ── Patchify: Conv3d with stride = patch_size ─────────────────
        # Equivalent to splitting into non-overlapping patches and
        # projecting each to hidden_dim.
        # kernel_size = stride = patch_size → no overlap, no gaps
        self.patch_embed = nn.Conv3d(
            in_channels  = self.in_channels,
            out_channels = hidden_dim,
            kernel_size  = (patch_t, patch_h, patch_w),
            stride       = (patch_t, patch_h, patch_w),
            bias         = True,
        )

        # ── Conditioning ─────────────────────────────────────────────

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_dim=cond_dim)

        # Pooled text → cond_dim (for adaLN signal)
        # T5 tokens are mean-pooled then projected
        self.txt_pool_proj = nn.Sequential(
            nn.Linear(text_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Pooled image → cond_dim (for adaLN signal)
        # CLIP CLS token projected
        self.img_pool_proj = nn.Sequential(
            nn.Linear(clip_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Image patch tokens → text_dim (for cross-attention)
        # We project CLIP patch tokens to text_dim so we can
        # concatenate them with T5 tokens for a single cross-attention
        self.img_token_proj = nn.Linear(clip_dim, text_dim, bias=True)

        # Learned null embeddings for CFG dropout
        # These replace conditioning when it's dropped during training
        self.null_txt_token = nn.Parameter(torch.zeros(1, 1, text_dim))
        self.null_img_token = nn.Parameter(torch.zeros(1, 1, clip_dim))
        self.null_txt_pool  = nn.Parameter(torch.zeros(1, text_dim))
        self.null_img_pool  = nn.Parameter(torch.zeros(1, clip_dim))

        # ── RoPE (shared across all blocks) ──────────────────────────
        # All DiT blocks share one RoPE instance —
        # same frequency tables, same rotation applied everywhere
        self.rope = RoPE3D(
            head_dim = hidden_dim // num_heads,
            max_t    = self.num_t + 8,     # small buffer for flexibility
            max_h    = self.num_h + 8,
            max_w    = self.num_w + 8,
            base_t   = rope_base_t,
            base_h   = rope_base_h,
            base_w   = rope_base_w,
        )

        # ── DiT blocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim = hidden_dim,
                num_heads  = num_heads,
                cond_dim   = cond_dim,
                text_dim   = text_dim,
                rope       = self.rope,
                dropout    = dropout,
                ffn_mult   = ffn_mult,
            )
            for _ in range(num_layers)
        ])

        # ── Final layer ───────────────────────────────────────────────
        self.final_layer = FinalLayer(
            hidden_dim = hidden_dim,
            patch_dim  = self.out_patch_dim,
            cond_dim   = cond_dim,
        )

        # ── Weight init ───────────────────────────────────────────────
        self._init_weights()

    # ── Weight initialization ────────────────────────────────────────

    def _init_weights(self):
        """
        Standard transformer init:
            Linear layers: Xavier uniform
            Embeddings: Normal(0, 0.02)
            adaLN projections: already zeroed in AdaLNModulation
            Final layer: already zeroed in FinalLayer
        """
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv3d):
                nn.init.xavier_uniform_(module.weight.view(module.weight.size(0), -1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

        # Patch embed: slightly smaller init — it sees raw latents
        nn.init.normal_(self.patch_embed.weight, std=0.02)

        # Timestep MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    # ── Patchify / Unpatchify ────────────────────────────────────────

    def patchify(self, z: torch.Tensor) -> torch.Tensor:
        """
        Convert video latent into token sequence.

        z: (B, C_in, T', H', W')
        Returns: (B, N, hidden_dim)
            N = num_t * num_h * num_w
        """
        # Conv3d with stride=patch_size does the patchification:
        # Each (patch_t × patch_h × patch_w) block → one hidden_dim vector
        x = self.patch_embed(z)      # (B, hidden_dim, num_t, num_h, num_w)

        # Flatten spatial dims into sequence
        # (B, D, T, H, W) → (B, D, N) → (B, N, D)
        B, D, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, N, D)
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert token sequence back into video latent.

        x: (B, N, out_patch_dim)
            out_patch_dim = latent_channels * patch_t * patch_h * patch_w
        Returns: (B, latent_channels, T', H', W')
        """
        B  = x.shape[0]
        pt, ph, pw = self.patch_t, self.patch_h, self.patch_w
        C  = self.latent_channels

        # Reshape: (B, N, out_patch_dim) → (B, num_t, num_h, num_w, C, pt, ph, pw)
        x = x.view(B, self.num_t, self.num_h, self.num_w, C, pt, ph, pw)

        # Rearrange to (B, C, T', H', W'):
        # T' = num_t * pt, H' = num_h * ph, W' = num_w * pw
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        # Now: (B, C, num_t, pt, num_h, ph, num_w, pw)
        x = x.contiguous().view(
            B, C,
            self.num_t * pt,
            self.num_h * ph,
            self.num_w * pw,
        )
        return x

    # ── Conditioning builders ────────────────────────────────────────

    def build_conditioning(
        self,
        t: torch.Tensor,           # (B,)
        txt_tokens: torch.Tensor,  # (B, L, text_dim)
        img_tokens: torch.Tensor,  # (B, M, clip_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build:
            c           — pooled signal for adaLN  (B, cond_dim)
            cond_tokens — sequence for cross-attn  (B, L+M, text_dim)
        """
        # ── Pooled adaLN signal ───────────────────────────────────────
        t_emb     = self.t_embedder(t)                          # (B, cond_dim)

        txt_pool  = txt_tokens.mean(dim=1)                      # (B, text_dim)
        txt_pool  = self.txt_pool_proj(txt_pool)                # (B, cond_dim)

        img_pool  = img_tokens[:, 0]                            # (B, clip_dim) ← CLS token
        img_pool  = self.img_pool_proj(img_pool)                # (B, cond_dim)

        # Sum all three signals — each contributes to adaLN modulation
        c = t_emb + txt_pool + img_pool                         # (B, cond_dim)

        # ── Cross-attention token sequence ───────────────────────────
        # Project CLIP tokens → text_dim so they can be concatenated
        img_tokens_proj = self.img_token_proj(img_tokens)       # (B, M, text_dim)

        # Concatenate: text tokens first, then image tokens
        cond_tokens = torch.cat([txt_tokens, img_tokens_proj], dim=1)  # (B, L+M, text_dim)

        return c, cond_tokens

    # ── Main forward ─────────────────────────────────────────────────

    def forward(
        self,
        z_t: torch.Tensor,         # (B, C, T', H', W')  noisy video latent
        t: torch.Tensor,           # (B,)                timesteps in [0,1]
        txt_tokens: torch.Tensor,  # (B, L, text_dim)    T5 sequence
        img_tokens: torch.Tensor,  # (B, M, clip_dim)    CLIP tokens
        z_img: torch.Tensor,       # (B, C, T', H', W')  image latent (I2V)
    ) -> torch.Tensor:
        """
        Predict velocity field for flow matching.

        Returns:
            v_pred: (B, C, T', H', W')  predicted velocity
        """
        B = z_t.shape[0]
        device = z_t.device

        # ── Step 1: I2V conditioning ──────────────────────────────────
        # Tile image latent across time to match z_t shape
        # z_img is the VAE-encoded first frame, shape (B, C, 1, H', W')
        # or already tiled (B, C, T', H', W') — handle both
        if z_img.shape[2] == 1:
            z_img = z_img.expand_as(z_t)       # tile across T'

        # Concatenate on channel dim: (B, 2C, T', H', W')
        z_in = torch.cat([z_t, z_img], dim=1)

        # ── Step 2: Patchify ─────────────────────────────────────────
        x = self.patchify(z_in)                # (B, N, hidden_dim)

        # ── Step 3: Build conditioning ───────────────────────────────
        c, cond_tokens = self.build_conditioning(t, txt_tokens, img_tokens)
        # c:           (B, cond_dim)
        # cond_tokens: (B, L+M, text_dim)

        # ── Step 4: 3D position ids for RoPE ─────────────────────────
        pos_ids = make_3d_pos_ids(
            self.num_t, self.num_h, self.num_w, device=device
        ).expand(B, -1, -1)                    # (B, N, 3)

        # ── Step 5: DiT blocks ────────────────────────────────────────
        for block in self.blocks:
            x = block(x, c, cond_tokens, pos_ids)

        # ── Step 6: Final layer → unpatchify ─────────────────────────
        x = self.final_layer(x, c)             # (B, N, out_patch_dim)
        v = self.unpatchify(x)                 # (B, C, T', H', W')

        return v

    # ── Utilities ────────────────────────────────────────────────────

    def count_params(self) -> dict:
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_M":     total / 1e6,
            "trainable_M": trained / 1e6,
        }

    def get_null_conditioning(
        self, B: int, L: int, M: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns null text and image tokens for CFG unconditioned pass.
        Expands learned null embeddings to batch size.
        """
        null_txt = self.null_txt_token.expand(B, L, -1)   # (B, L, text_dim)
        null_img = self.null_img_token.expand(B, M, -1)   # (B, M, clip_dim)
        return null_txt.to(device), null_img.to(device)