# Spectral-Grouping-and-Drift-Correction-for-Diffusion-Model-Quantization
DL CS437 Project

This repository contains our project implementation for post-training quantization (PTQ) of diffusion models, with a focus on timestep-aware calibration using spectral grouping. The project studies how activation distributions and quantization errors evolve across the denoising trajectory, and evaluates several methods for improving low-bit diffusion model quantization.

## Project Overview

Diffusion models generate images through an iterative denoising process. Unlike standard feed-forward models, the same denoising network is evaluated repeatedly across many timesteps. This makes PTQ challenging because activation distributions change significantly over time, and small quantization errors can accumulate across the sampling trajectory.

Our main idea is to group timesteps according to spectral properties of intermediate activations rather than using uniform timestep intervals. We compute spectral descriptors from activation feature maps, identify regions along the denoising trajectory, and assign quantization groups based on those regions.

We evaluate this approach on CIFAR-10 diffusion generation under W8A8 and W4A8 quantization settings.

## Main Contributions

- **Spectral timestep grouping** for diffusion PTQ.
- **Layer-wise weight clipping calibration** for improving W4A8 quantization.
- **Sensitivity-aware mixed precision**, where the most sensitive W4 layers are kept at W8.
- **Spectral drift correction**, which estimates denoiser-output mismatch between FP32 and quantized models.
- **Timestep-dependent Householder rotations** as an ablation for activation outlier suppression.
- Comparison against uniform timestep grouping and contextual Q-Diffusion baselines.

## Repository Structure

```text
.
├── README.md
├── PTQ Diffusion Survey.pdf
├── Spectral Grouping and Drift Correction for Diffusion Models.pdf
├── ptq_diffusion_dl_final_notebook.ipynb
└── Q_diffusion_cifar10.ipynb
