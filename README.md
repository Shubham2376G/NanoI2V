# NanoI2V: Building an Image-to-Video Model from Scratch

Over the past few months, I moved from coding LLMs to working on diffusion-based video generation models, inspired by architectures like WAN, CogVideoX, and modern open-source video systems.

These models generate coherent video from a single image - producing motion, temporal consistency, camera dynamics, and cinematic output. But how do they actually work under the hood?

This repository is a step-by-step series where we build every core component of a modern I2V pipeline from scratch.

---

> The series is published as a dedicated website with structured lessons, explanations, and code walkthroughs:
> **[shubham2376g.github.io/NanoI2V](https://shubham2376g.github.io/NanoI2V)**
>
> Each topic is a self-contained lesson - read in order or jump to what you need.

---

## What This Series Covers

This series explores the core building blocks behind modern image-to-video (I2V) models, including topics such as:

| Area | Topics |
|---|---|
| VAE | Causal 3D convolutions, residual blocks, video encoders & decoders |
| DiT | Rotary positional embeddings (RoPE), attention mechanisms, adaptive LayerNorm |
| Flow & Diffusion | Flow matching, schedulers, denoising concepts |
| Conditioning | Text conditioning, image conditioning, multimodal embeddings |
| Training | End-to-end training pipeline, optimization, inference |

Additional topics and modules will be added as the series evolves.

---

## Repository Structure

```text
NanoI2V/
├── vae/
│   ├── conv.py          # CausalConv3d
│   ├── blocks.py        # 3D ResBlocks and SpatialAttention
│   ├── encoder.py
│   ├── decoder.py
│   └── vae.py
│
├── dit/
│   ├── rope.py          # 3D RoPE
│   ├── attention.py     # Self & Cross Attention
│   ├── blocks.py        # DiT blocks with adaLN
│   └── dit.py
│
├── flow/
│   └── scheduler.py     # Flow matching scheduler
│
├── conditioning/
│   └── encoders.py      # Text & image encoders
│
├── docs/
│   └── index.html       # Series website (GitHub Pages)
│
└── train.py
```

---

## Prerequisites

A basic understanding of the following will help:

- PyTorch basics
- Transformer architecture and attention mechanisms
- LLM fundamentals
- Diffusion model intuition (helpful but not required)

If you've worked with LLMs before, many concepts here will feel familiar.

---

## ⭐ Support the Project

If you find this useful:
- Star the repository ⭐
- Share it with others interested in diffusion or video models

<p>
  <a href="https://www.linkedin.com/in/shubham-aggarwal-a63b40276">
    <img src="https://img.shields.io/badge/Follow%20on-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
</p>