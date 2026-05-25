# NanoI2V: Building an Image-to-Video Model from Scratch

<p align="center">
  <img src="docs/images/i2v.png" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue">
  <img src="https://img.shields.io/badge/PyTorch-2.0-red">
  <img src="https://img.shields.io/badge/Transformers-DiT-purple">
  <img src="https://img.shields.io/badge/License-MIT-green">
  <img src="https://img.shields.io/github/stars/Shubham2376G/NanoI2V">
</p>

# Overview

NanoI2V is a **from-scratch implementation** of an **Image-to-Video (I2V)** generation pipeline.

The project focuses on understanding and implementing the core concepts and building blocks behind modern video generation systems such as:

- Variational Autoencoders (VAE)
- Latent Video Modeling
- Diffusion / Flow Matching
- DiT Transformers
- Cross-Attention Conditioning
- Classifier-Free-Guidance


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