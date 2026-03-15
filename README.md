# Lib-SRDM: Library-guided Spectral-prior Diffusion Model for HSI Reconstruction

This repository implements a library-guided, spectral-prior diffusion model (Lib-SRDM) for hyperspectral image (HSI) spectral super-resolution and relighting. The framework leverages pre-trained diffusion models in both the spatial (HSI) and spectral domains, guided by a scene-adaptive spectral library built with open-vocabulary segmentation.

This project is built on top of [guided-diffusion](https://github.com/openai/guided-diffusion).

---

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Datasets](#datasets)
- [Pre-trained Models](#pre-trained-models)
- [Spectral Library Generation](#spectral-library-generation)
- [Training the HSI Diffusion Model](#training-the-hsi-diffusion-model)
- [Inference](#inference)
- [Spectral Super-resolution & Relighting](#spectral-super-resolution--relighting)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

---

## Overview

Lib-SRDM solves HSI restoration (spectral super-resolution, relighting) by combining:

1. **Spatial HSI Diffusion Model** – a U-Net-based DDPM trained on hyperspectral data.
2. **Spectral Prior Model** – a lightweight spectral model that acts as a scene-adaptive spectral library prior, built per-image using open-vocabulary segmentation (RAM++ + GroundingDINO + SAM-HQ).

At inference, the posterior gradient is computed analytically (see `measurement.py`) and injected into each diffusion step, steering the sample towards measurements while preserving the learned prior.

---

## Environment Setup

Install all pip dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

The following packages must be installed separately from source:

```bash
# CLIP
pip install git+https://github.com/openai/CLIP.git
# RAM++ — https://github.com/xinyu1205/recognize-anything
# GroundingDINO — https://github.com/IDEA-Research/GroundingDINO
# SAM-HQ — https://github.com/SysCV/sam-hq
```

---

## Datasets

The following datasets are used in the paper. Download them and place under your dataset root (e.g. `/your/dataset/`).

| Dataset | Description | Bands | Link |
|---------|-------------|-------|------|
| **CAVE** | 32 hyperspectral images, 512×512 | 31 | [CAVE](https://www.cs.columbia.edu/CAVE/databases/multispectral/) |
| **ICVL** | 201 hyperspectral images, 1392×1300 | 31 | [ICVL](http://icvl.cs.bgu.ac.il/hyperspectral/) |
| **NTIRE 2022** | Spectral super-resolution challenge data | 31 | [NTIRE 2022](https://data.vision.ee.ethz.ch/cvl/ntire22/) |

Update the `data_dir` / `base_samples` paths in the relevant shell scripts under `run/` or directly in the Python configuration dictionaries at the bottom of each script.

---

## Pre-trained Models

### HSI Diffusion Model

Train the HSI DM yourself (see [Training the HSI Diffusion Model](#training-the-hsi-diffusion-model)) or use a checkpoint trained on ICVL / CAVE / NTIRE 2022. Place the checkpoint at a path of your choice and update `model_path` in `config/model_config.yaml`:

```yaml
# config/model_config.yaml
in_channels: 31
image_size: 256
num_channels: 128
num_res_blocks: 1
attention_resolutions: "16"
...
```

### Open-vocabulary Segmentation Checkpoints (for Lib-SRDM)

Download the following and place in `Grounded-Segment-Anything/`:

- **RAM++** (`ram_plus_swin_large_14m.pth`): https://github.com/xinyu1205/recognize-anything
- **GroundingDINO** (`groundingdino_swint_ogc.pth`): https://github.com/IDEA-Research/GroundingDINO
- **SAM-HQ** (`sam_hq_vit_h.pth`): https://github.com/SysCV/sam-hq

---

## Spectral Library Generation

Before running Lib-SRDM inference, build a per-scene spectral library offline from the training split of your dataset.

```bash
# Edit paths inside the script first
python scripts/gen_spectral_library.py
```

The script:
1. Converts HSI training images to RGB via the camera spectral response function (SRF).
2. Tags each image with RAM++ open-vocabulary labels.
3. Detects objects with GroundingDINO and creates segmentation masks via SAM-HQ.
4. Samples spectra per object class into a per-class library saved as `library/<name>/lib.pkl`.

Configuration is inside the script (`cfg` dict at the top):

| Key | Description |
|-----|-------------|
| `data_root` | List of paths to HSI training data |
| `save_dir` | Output directory for the library |
| `sample_per_tag` | Maximum spectra sampled per object class |
| `gamma` | Gamma correction applied before segmentation |
| `pretrained` | Path to RAM++ checkpoint |

---

## Training the HSI Diffusion Model

```bash
cd run
bash train_hsi.sh
```

or directly:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/hsi_train.py \
    --data_dir "/path/to/CAVE/" \
    --batch_size 8 \
    --save_interval 20000 \
    --model_config config/model_config.yaml \
    --lr_anneal_steps 100000 \
    --save_dir models/icvl/
```

Key arguments:

| Argument | Description |
|----------|-------------|
| `--data_dir` | Root directory of the training dataset |
| `--batch_size` | Training batch size |
| `--model_config` | Path to model YAML configuration |
| `--lr_anneal_steps` | Total training steps with learning rate annealing |
| `--save_dir` | Where to save model checkpoints |
| `--resume_checkpoint` | Path to a checkpoint to resume from (optional) |

---

## Inference

Run Lib-SRDM inference using a pre-built spectral library (see [Spectral Library Generation](#spectral-library-generation)):

```bash
python scripts/hsi_spr.py
```

---

## Spectral Super-resolution & Relighting

Additional tasks are provided as standalone scripts:

| Script | Task |
|--------|------|
| `scripts/hsi_spr.py` | RGB → HSI spectral super-resolution using the Lib-SRDM pipeline |
| `scripts/hsi_relighting.py` | HSI relighting (change illuminant from source to target) |
| `scripts/ablative.py` | Ablation study runner |

Each script contains a `cfg` dictionary at the top that controls all paths and hyperparameters. Edit the relevant fields before running.

---

## Project Structure

```
config/
  model_config.yaml         # U-Net HSI diffusion model configuration
  spectral_dm.yaml          # Spectral diffusion model configuration
  spectral_prior.yaml       # Spectral prior model configuration

guided_diffusion/
  gaussian_diffusion.py     # Core DDPM forward/reverse process and loss functions
  image_datasets.py         # Dataloaders for CAVE, ICVL, NTIRE 2022
  cave.py                   # CAVE dataset loader
  icvl.py                   # ICVL dataset loader
  ntire_gen.py              # NTIRE 2022 dataset loader
  script_util.py            # Model factory and default configurations
  train_util.py             # Training loop
  unet.py                   # U-Net spatial model
  specmodel.py              # Spectral diffusion model
  prior_model.py            # Scene-adaptive spectral prior (PriorModel / PriorModels)
  low_rank_model.py         # Low-rank spectral gradient approximation
  losses.py                 # Loss functions
  fp16_util.py              # Mixed-precision utilities
  dist_util.py              # Distributed training utilities
  logger.py                 # Logging utilities
  resample.py               # Diffusion timestep resampling
  respace.py                # Timestep respacing

run/
  spec+hsi_denoise.sh       # Spectral-library-guided inference script
  train_hsi.sh              # HSI DM training script

scripts/
  hsi_train.py              # Train the HSI diffusion model
  hsi_spr.py                # Spectral super-resolution (RGB → HSI)
  hsi_relighting.py         # HSI relighting
  gen_spectral_library.py   # Build the per-class spectral library
  image_train.py            # Train the RGB diffusion model
  ablative.py               # Ablation experiments
  rgb_modified.py           # Modified RGB processing utilities

measurement.py              # Analytical posterior gradient (log-likelihood gradient)
utils.py                    # Metric computation (PSNR, SSIM, SAM), noise estimation,
                            #   covariance estimation, spectral plotting, and I/O helpers
srf_tools.py                # Spectral response function (SRF) loading and RGB synthesis
illuminant.py               # CIE illuminant loading and spectral relighting utilities
awb.py                      # Automatic white balance (Gray World and White Block methods)
lib_and_mask.py             # Per-image spectral library and segmentation mask generation
requirement.txt             # Legacy Conda environment specification (YAML)
requirements.txt            # pip requirements file (all pip-installable dependencies)
```

---

## Acknowledgements

This project builds on the following open-source work:

- [guided-diffusion](https://github.com/openai/guided-diffusion) (OpenAI) – base diffusion model framework.
- [Recognize Anything (RAM++)](https://github.com/xinyu1205/recognize-anything) – open-vocabulary image tagging.
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) – open-set object detection.
- [SAM-HQ](https://github.com/SysCV/sam-hq) – high-quality segment-anything model.
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) – integration of GroundingDINO and SAM.
