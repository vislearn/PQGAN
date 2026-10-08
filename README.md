<h1 align="center">PQGAN: Product-Quantised Image Representation<br>for High-Quality Image Synthesis</h1>

<h3 align="center">Denis Zavadski &nbsp;·&nbsp; Nikita Philip Tatsch &nbsp;·&nbsp; Carsten Rother</h3>

<div align="center">

[![ICLR 2026](https://img.shields.io/badge/ICLR_2026-Paper-blue)](https://openreview.net/forum?id=D8oqcochgq)
[![arXiv](https://img.shields.io/badge/arXiv-2510.03191-b31b1b)](https://arxiv.org/abs/2510.03191)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-yellow)](https://huggingface.co/CVL-Heidelberg/PQGAN)
[![pytest](https://github.com/vislearn/PQGAN/actions/workflows/pytest.yml/badge.svg)](https://github.com/vislearn/PQGAN/actions/workflows/pytest.yml)
[![quality checks](https://github.com/vislearn/PQGAN/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/vislearn/PQGAN/actions/workflows/pre-commit.yml)

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

### Reconstruction quality

Selected rows from Table 1 of the paper (ImageNet 256×256 validation set, 50k images). *F*: downsampling factor, *d*: latent channels, *K*: codebook size (per subspace for PQ).

| Model | Latent | *F* | Latent resolution | *d* | *K* | PSNR ↑ | rFID ↓ | CMMD ↓ | LPIPS ↓ |
|-------|--------|-----|-------------------|-----|-----|--------|--------|--------|---------|
| VQGAN (LDM) | VQ | 8 | 32×32 | 4 | 16 384 | 23.1 | 1.29 | 0.258 | 0.0815 |
| VQGAN-LC | VQ | 8 | 32×32 | 4 | 100 000 | 27.0 | 1.29 | 0.080 | 0.0712 |
| SDv2.1 VAE | KL | 8 | 32×32 | 4 | – | 25.3 | 0.75 | 0.133 | 0.0610 |
| SDXL VAE | KL | 8 | 32×32 | 4 | – | 25.3 | 0.74 | 0.148 | 0.0573 |
| **PQGAN (ours)** | PQ | 16 | 16×16 | 128 | 128 | 28.3 | 0.41 | 0.094 | 0.0304 |
| **PQGAN (ours)** | PQ | 8 | 32×32 | 128 | 512 | **37.4** | **0.036** | **0.011** | **0.0024** |

Both PQGAN models use *S*=64 subspaces and are available as [pretrained weights](#pretrained-models).

### Diffusion model integration (Stable Diffusion 2.1)

By adapting SD2.1 to PQGAN latents, we obtain three operating points:

| Variant | Description |
|---------|-------------|
| **PQSD-HR** | Double output resolution (768→1536) at the same sampling cost |
| **PQSD-Precise** | Same resolution and cost, higher-fidelity latent space |
| **PQSD-Quick** | Same resolution, ~4× faster sampling |

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

## Pretrained Models

Weights for the two PQGAN models from Table 1 are available on Hugging Face at [CVL-Heidelberg/PQGAN](https://huggingface.co/CVL-Heidelberg/PQGAN). Both were trained on ImageNet at 256×256.

| Model | *F* | *d* | *K* | *S* | Latent (256×256 input) | PSNR ↑ | rFID ↓ | CMMD ↓ | LPIPS ↓ |
|-------|-----|-----|-----|-----|------------------------|--------|--------|--------|---------|
| `PQGAN_F8_K512_Z128_S64` | 8 | 128 | 512 | 64 | 128×32×32 | 37.4 | 0.036 | 0.011 | 0.0024 |
| `PQGAN_F16_K128_Z128_S64` | 16 | 128 | 128 | 64 | 128×16×16 | 28.3 | 0.41 | 0.094 | 0.0304 |

Each model comes as `checkpoints/<name>.safetensors` (weights) and `checkpoints/<name>.yaml` (model config). The weights contain only the autoencoder and quantiser. Discriminator and optimizer states are not included.

```python
import torch
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from vv.models.latent_diffusion_copy import VQModel

name = "PQGAN_F8_K512_Z128_S64"  # or "PQGAN_F16_K128_Z128_S64"
config = yaml.safe_load(open(hf_hub_download("CVL-Heidelberg/PQGAN", f"checkpoints/{name}.yaml")))
model = VQModel(**config["model"]["init_args"])
model.load_state_dict(load_file(hf_hub_download("CVL-Heidelberg/PQGAN", f"checkpoints/{name}.safetensors")))
model = model.eval().cuda()

# Models expect 256x256 inputs with ImageNet normalisation
mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(mean, std)])
x = preprocess(Image.open("image.jpg").convert("RGB")).unsqueeze(0).cuda()

with torch.no_grad():
    quant, _, (_, _, indices) = model.encode(x)  # quant: (1, 128, 32, 32) for F=8, (1, 128, 16, 16) for F=16
    rec = model.decode(quant)

rec = (rec.cpu() * std[:, None, None] + mean[:, None, None]).clamp(0, 1)  # back to [0, 1]
```

`indices` is a list with one tensor of codebook indices per subspace (64 entries).

The metrics above were reproduced from these weights with `vv eval` on the full ImageNet validation set. The training configs are in [`config/latent_diffusion/F8_K512_Z128_S64`](config/latent_diffusion/F8_K512_Z128_S64/config.yaml) and [`config/latent_diffusion/F16_K128_Z128_S64`](config/latent_diffusion/F16_K128_Z128_S64/config.yaml).

---

## Usage

### Training

Config files for all experiments are in `config/latent_diffusion/`. The main model from the paper (F=8, *d*=128, *S*=64, *K*=512):

```sh
vv fit --config config/latent_diffusion/F8_K512_Z128_S64/config.yaml
```

The F=16 model (*d*=128, *S*=64, *K*=128) uses `config/latent_diffusion/F16_K128_Z128_S64/config.yaml`.

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
  title     = {{PQGAN}: Product-Quantised Image Representation for High-Quality Image Synthesis},
  author    = {Zavadski, Denis and Tatsch, Nikita Philip and Rother, Carsten},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=D8oqcochgq}
}
```
