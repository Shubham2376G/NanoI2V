# 🚀 Building an Image-to-Video (I2V) Model from Scratch

Over the past few months, I moved from coding LLMs to working on diffusion-based video generation models inspired by architectures like WAN, DiT, and modern open-source video models 🎥✨

These systems are capable of generating coherent videos from a single image; producing motion, temporal consistency, camera dynamics, and cinematic outputs 🎬🔥

But how do these models actually work under the hood? 🤔

This repository is a step-by-step learning series where we build and understand the core components behind modern Image-to-Video (I2V) models from scratch.

Instead of treating video generation like magic, we’ll break everything down into small understandable modules and implement them one piece at a time

---

# 📚 What This Series Covers

We’ll gradually build the components used in modern diffusion video pipelines, including:

- ✅ Video VAEs (3D latent compression)
- ✅ Causal 3D convolutions
- ✅ Diffusion Transformers (DiT)
- ✅ Rotary Positional Embeddings (RoPE)
- ✅ Self & Cross Attention
- ✅ Adaptive LayerNorm (adaLN)
- ✅ Flow Matching
- ✅ Text/Image Conditioning
- ✅ Schedulers
- ✅ Training Pipeline
- ✅ Full Image-to-Video generation pipeline

By the end of the series, we’ll connect all the pieces together into a simplified but functional I2V architecture 🚀


---

# ⭐ Support the Project

If you find this series useful:

- Star the repository ⭐
- Follow the updates on LinkedIn 🚀
- Share it with others interested in diffusion/video models

Your support helps the project reach more builders and researchers.



---

# 🧠 Prerequisites

A basic understanding of the following topics will help:

- Transformers
- Attention mechanisms
- LLM fundamentals
- PyTorch basics
- Diffusion model intuition (helpful but not mandatory)

If you've previously worked with LLMs, many concepts here will feel surprisingly familiar 👀

---

# 📂 Repository Structure

```text
wan_i2v/
├── vae/
│   ├── conv.py
│   ├── blocks.py
│   ├── encoder.py
│   ├── decoder.py
│   └── vae.py
│
├── dit/
│   ├── rope.py
│   ├── attention.py
│   ├── blocks.py
│   └── dit.py
│
├── flow/
│   └── scheduler.py
│
├── conditioning/
│   └── encoders.py
│
├── notebooks/
│   ├── 01_causal_conv3d.ipynb
│   ├── 02_resblock3d.ipynb
│   ├── 03_video_vae.ipynb
│   └── ...
│
└── train.py
```

---

# 📓 Notebooks

The `notebooks/` folder will contain detailed deep dives for every concept covered in the series.

Each notebook will include:

- Intuition behind the concept
- Mathematical explanation
- PyTorch implementation
- Visualizations
- Experiments
- Debugging insights
- Links to related papers

The goal is to make this repository both:
- a learning resource 📘
- and a practical implementation guide 🔧

---

# 🎯 Learning Philosophy

This series focuses on:

✅ Small incremental concepts  
✅ Practical implementations  
✅ Clear intuition over heavy theory  
✅ Building everything piece by piece  
✅ Understanding *why* components exist

Instead of jumping directly into massive codebases, we’ll recreate the important ideas ourselves.

---

# 🔗 Follow the Series

I’ll be posting:

- daily breakdowns
- architecture deep dives
- implementation walkthroughs
- visual explanations
- diffusion model intuition

throughout this series 🚀

<br>

<p align="left">
  <a href="https://www.linkedin.com/in/shubham-aggarwal-a63b40276">
    <img src="https://img.shields.io/badge/Follow%20the%20Series%20on-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" />
  </a>
</p>



---

