<h1 align="center">PQGAN: Product-Quantised Image Representation<br>for High-Resolution Image Synthesis</h1>

<h3 align="center">Denis Zavadski &nbsp;·&nbsp; Nikita Philip Tatsch &nbsp;·&nbsp; Carsten Rother</h3>

<div align="center">

[![ICLR 2026](https://img.shields.io/badge/ICLR_2026-Paper-blue)](https://openreview.net/forum?id=D8oqcochgq)
[![pytest](https://github.com/MrWhatZitToYaa/VQ-VAE-Praktikum-SS24-HD/actions/workflows/pytest.yml/badge.svg)](https://github.com/MrWhatZitToYaa/VQ-VAE-Praktikum-SS24-HD/actions/workflows/pytest.yml)
[![quality checks](https://github.com/MrWhatZitToYaa/VQ-VAE-Praktikum-SS24-HD/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/MrWhatZitToYaa/VQ-VAE-Praktikum-SS24-HD/actions/workflows/pre-commit.yml)

</div>

---

## Abstract

Product quantisation (PQ) is a classical method for scalable vector encoding, yet has seen limited adoption in high-fidelity generative modelling. In this work, we introduce **PQGAN**, a quantised image autoencoder that integrates PQ into the VQGAN framework, leveraging its full potential for the first time in the context of latent representation learning. PQGAN achieves state-of-the-art reconstruction performance, improving PSNR from 27 dB to 37 dB and reducing FID, LPIPS, and CMMD by up to 96% compared to prior quantised methods — surpassing also continuous counterparts — despite using codebooks as small as 128 to 512 entries per subspace. We analyse the interaction between codebook size, embedding dimensionality, and subspace factorisation, showing that PQ generalises both vector and scalar quantisation as special cases. Our codebook diagnostics reveal efficient and compositional latent structures, enabling systematic control over compression and fidelity. Finally, we demonstrate that PQ latents integrate seamlessly into pre-trained diffusion models, enabling either significantly faster and more compute-efficient generation, or a doubling of output resolution at no additional cost — positioning PQ as a strong, drop-in alternative to both discrete and continuous latent representations in generative modelling.

---

## Method

PQGAN replaces the single-codebook vector quantiser of VQGAN with **Product Quantisation**: the latent vector is factorised into *S* subspaces, each independently quantised with its own codebook of size *K*. The effective codebook is the Cartesian product of all subspaces, yielding *K^S* representable codes without the storage cost of a single large codebook.

```
Encoder  →  [z₁ | z₂ | … | zₛ]  →  VQ(z₁) ⊕ VQ(z₂) ⊕ … ⊕ VQ(zₛ)  →  Decoder
```

This factorisation generalises both standard VQ (*S*=1) and scalar quantisation (*S*=*d*), giving fine-grained control over the compression–fidelity trade-off.

---

## Results

All models are trained on ImageNet (256×256) and evaluated on the ImageNet validation set.

### Reconstruction quality (F=8 downsampling factor)

| Model | Codebook | PSNR ↑ | FID ↓ | LPIPS ↓ | CMMD ↓ |
|-------|----------|--------|-------|---------|--------|
| VQGAN | *K*=16384, *S*=1 | 27.0 | 0.93 | 0.18 | — |
| RQ-VAE | *K*=512 ×4 | 28.1 | 0.69 | 0.16 | — |
| **PQGAN (ours)** | *K*=512, *S*=64 | **37.4** | **0.036** | **0.038** | **–96%** |

### Diffusion model integration (Stable Diffusion 2.1)

By swapping the SD2.1 VAE with a PQGAN encoder, we obtain three operating points:

| Variant | Description |
|---------|-------------|
| **PQSD-HR** | Double output resolution (512→1024) at no additional compute |
| **PQSD-Precise** | Same resolution, improved fidelity |
| **PQSD-Quick** | Same resolution, 2× faster sampling |

---

## Installation

**Requirements:** Python 3.11, CUDA-enabled GPU (≥40 GB VRAM recommended)

### 1. Create a virtual environment

```sh
# conda
conda create -n pqgan python=3.11
conda activate pqgan

# or venv
python3.11 -m venv pqgan
source pqgan/bin/activate
```

### 2. Install the package

```sh
pip install --upgrade pip
pip install -e .
```

### 3. (Optional) Development setup

```sh
pip install -e .[dev]
pre-commit install
```

---

## Usage

### Training

Config files for all experiments are in `config/latent_diffusion/`. The main model from the paper (F=8, *d*=128, *S*=64, *K*=512):

```sh
vv fit --config config/latent_diffusion/F16_K4_Z128_S64/config.yaml
```

Set `data_dir` in the config to point to your ImageNet directory. Multi-GPU training and cluster scripts are in `scripts/`.

### Evaluation

```sh
vv eval \
  --source_path /path/to/checkpoints \
  --destination_path /path/to/results \
  --eval_config config/eval/eval_config.yaml
```

Metrics computed: FID, KID, IS, LPIPS, PSNR, CMMD.

### Inference speed evaluation

```sh
vv eval --eval_config config/eval/eval_config_speed.yaml \
        --source_path /path/to/checkpoints \
        --destination_path /path/to/results
```

### Visualisation

```sh
vv plot --source_path /path/to/results
```

---

## Repository Structure

```
├── config/               # Training and evaluation configs
│   ├── latent_diffusion/ # Per-experiment YAML configs
│   └── eval/             # Evaluation configs
├── scripts/              # Shell scripts for training and evaluation
├── src/vv/
│   ├── data/             # Dataset loaders (ImageNet, FFHQ, LSUN, …)
│   ├── models/           # Model definitions (PQGAN, baselines)
│   ├── evaluation/       # Evaluation framework
│   ├── logger/           # Training-time metric loggers
│   └── special_layers/   # Quantisation and other custom layers
└── tests/                # Unit tests
```

---

## Testing

```sh
pytest tests --cov src
```

---

## Citation

```bibtex
@inproceedings{zavadski2026pqgan,
  title     = {Product-Quantised Image Representation for High-Resolution Image Synthesis},
  author    = {Zavadski, Denis and Tatsch, Nikita Philip and Rother, Carsten},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=D8oqcochgq}
}
```
