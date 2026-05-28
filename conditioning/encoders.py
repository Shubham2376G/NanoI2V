# conditioning/encoders.py
import torch
import torch.nn as nn
from transformers import T5EncoderModel, T5Tokenizer
from transformers import CLIPVisionModel, CLIPImageProcessor
from typing import Optional
from dataclasses import dataclass


# ─────────────────────────────────────────────
# Output containers
# ─────────────────────────────────────────────

@dataclass
class TextCondition:
    tokens: torch.Tensor      # (B, L, text_dim)  sequence for cross-attn
    pool:   torch.Tensor      # (B, text_dim)      mean-pooled


@dataclass
class ImageCondition:
    tokens: torch.Tensor      # (B, M, clip_dim)  patch tokens for cross-attn
    pool:   torch.Tensor      # (B, clip_dim)      CLS token


# ─────────────────────────────────────────────
# T5 Text Encoder
# ─────────────────────────────────────────────

class T5TextEncoder(nn.Module):
    """
    Frozen T5 encoder for text conditioning.

    Extracts last-layer hidden states as conditioning sequence.
    Used as cross-attention keys/values in the DiT.

    """

    def __init__(
        self,
        model_name: str = "google/t5-v1_1-base",   # swap to xxl for real training
        max_length: int = 226,                       # Wan2.2 uses 226 token limit
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.max_length = max_length
        self.device     = device

        # Load tokenizer and model
        print(f"Loading T5 tokenizer: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

        print(f"Loading T5 encoder: {model_name}")
        self.model = T5EncoderModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,   # load in fp16 to save memory
        )

        # Freeze all parameters — T5 is never trained
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(device)

        # Store output dim for downstream use
        self.output_dim = self.model.config.d_model

    @torch.no_grad()
    def forward(self, prompts: list[str]) -> TextCondition:
        """
        Encode a batch of text prompts.

        Args:
            prompts: list of B strings

        Returns:
            TextCondition with tokens (B, L, D) and pool (B, D)
        """
        # Tokenize
        encoding = self.tokenizer(
            prompts,
            max_length      = self.max_length,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )

        input_ids      = encoding.input_ids.to(self.device)
        attention_mask = encoding.attention_mask.to(self.device)

        # Encode — extract last hidden states
        outputs = self.model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )

        # (B, L, D) — full sequence for cross-attention
        tokens = outputs.last_hidden_state.float()

        # Mask padding tokens before pooling
        # attention_mask: 1 for real tokens, 0 for padding
        mask   = attention_mask.unsqueeze(-1).float()    # (B, L, 1)
        pool   = (tokens * mask).sum(dim=1) / mask.sum(dim=1)  # (B, D)

        return TextCondition(tokens=tokens, pool=pool)

    def get_null_condition(self, device: torch.device) -> TextCondition:
        """
        Encode an empty string → null conditioning for CFG.
        Returns single sample — caller should expand to batch size.
        """
        return self.forward([""])


# ─────────────────────────────────────────────
# CLIP Image Encoder
# ─────────────────────────────────────────────

class CLIPImageEncoder(nn.Module):
    """
    Frozen CLIP ViT encoder for image conditioning.

    Extracts:
        - Patch tokens (B, num_patches, clip_dim) → cross-attention
        - CLS token    (B, clip_dim)              → pooled signal

    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()

        self.device = device

        print(f"Loading CLIP processor: {model_name}")
        self.processor = CLIPImageProcessor.from_pretrained(model_name)

        print(f"Loading CLIP encoder: {model_name}")
        self.model = CLIPVisionModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

        # Freeze — CLIP is never trained
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(device)

        self.output_dim = self.model.config.hidden_size   # 1024 for ViT-L

    @torch.no_grad()
    def forward(self, images: list) -> ImageCondition:
        """
        Encode a batch of PIL images or numpy arrays.

        Args:
            images: list of B PIL.Image objects

        Returns:
            ImageCondition with:
                tokens: (B, num_patches+1, clip_dim)  ← includes CLS
                pool:   (B, clip_dim)                  ← CLS token
        """
        # Preprocess: resize, normalize to CLIP's expected range
        inputs = self.processor(
            images         = images,
            return_tensors = "pt",
        )
        pixel_values = inputs.pixel_values.to(self.device)

        # Encode
        outputs = self.model(pixel_values=pixel_values)

        # last_hidden_state: (B, num_patches+1, D)
        # position 0 is the CLS token, rest are patch tokens
        tokens = outputs.last_hidden_state.float()   # (B, 257, 1024) for ViT-L
        pool   = tokens[:, 0]                         # (B, 1024) — CLS token

        return ImageCondition(tokens=tokens, pool=pool)

    def get_null_condition(
        self, B: int, device: torch.device
    ) -> ImageCondition:
        """
        Zero tokens for CFG null image conditioning.
        Shape matches real image output so CFG math works cleanly.
        """
        M   = self.model.config.num_patches + 1   # 257 for ViT-L
        D   = self.output_dim

        tokens = torch.zeros(B, M, D, device=device)
        pool   = torch.zeros(B, D, device=device)
        return ImageCondition(tokens=tokens, pool=pool)


# ─────────────────────────────────────────────
# Combined conditioning module
# ─────────────────────────────────────────────

class ConditioningPipeline(nn.Module):
    """
    Wraps T5 + CLIP into a single module.
    Called once per batch to produce all conditioning tensors
    before the DiT training step.

    Keeps encoders on a separate device from the DiT if needed
    (e.g. CPU offload for T5-XXL to save GPU memory).
    """

    def __init__(
        self,
        t5_model:   str = "google/t5-v1_1-base",
        clip_model: str = "openai/clip-vit-large-patch14",
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__()

        self.text_encoder  = T5TextEncoder(t5_model,   device=device)
        self.image_encoder = CLIPImageEncoder(clip_model, device=device)

        # Store output dims for DiT constructor
        self.text_dim = self.text_encoder.output_dim
        self.clip_dim = self.image_encoder.output_dim

    @torch.no_grad()
    def encode_text(self, prompts: list[str]) -> TextCondition:
        return self.text_encoder(prompts)

    @torch.no_grad()
    def encode_image(self, images: list) -> ImageCondition:
        return self.image_encoder(images)

    def get_null_conditions(
        self, B: int, L: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns null txt_tokens and img_tokens for CFG.
        L = sequence length of text tokens.
        """
        null_txt = torch.zeros(B, L, self.text_dim, device=device)
        null_img = self.image_encoder.get_null_condition(B, device)
        return null_txt, null_img.tokens