<div align="center">

# NS-VLA: Towards Neuro-Symbolic Vision-Language-Action Models

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://zuzuzzy.github.io/NS-VLA/)
[![Demo](https://img.shields.io/badge/Demo-Online-green)](https://zuzuzzy.github.io/NS-VLA/demo/)
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/zuzuzzy/NS-VLA)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/zuzuzzy/NS-VLA-Dataset)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

[**Homepage**](https://zuzuzzy.github.io/NS-VLA/) | [**Demo**](https://zuzuzzy.github.io/NS-VLA/demo/) | [**Paper**](https://arxiv.org/abs/XXXX.XXXXX) | [**Model**](https://huggingface.co/zuzuzzy/NS-VLA) | [**Dataset**](https://huggingface.co/datasets/zuzuzzy/NS-VLA-Dataset)

</div>

## Overview

**NS-VLA** is a novel **Neuro-Symbolic Vision-Language-Action** framework trained via online reinforcement learning. It introduces:

- 🧩 **Symbolic Encoder**: Embeds vision and language features and extracts structured primitives
- ⚡ **Symbolic Solver**: Lightweight solver for data-efficient action sequencing via visual token sparsification
- 🔄 **Online RL**: GRPO-based optimization with primitive-segmented rewards for expansive exploration

<p align="center">
  <img src="assets/pipeline.png" width="90%" alt="NS-VLA Framework"/>
</p>

## Performance

| Method | Params | LIBERO (Full) | LIBERO (1-shot) | LIBERO-Plus | CALVIN 5-Task |
|:---|:---:|:---:|:---:|:---:|:---:|
| OpenVLA | 7B | 76.5 | 35.7 | 15.6 | 43.5 |
| OpenVLA-OFT | 7B | 97.1 | 48.9 | 69.6 | 66.5 |
| π₀ | 3B | 94.2 | 37.4 | 53.6 | — |
| UniVLA | 7B | 95.2 | 55.1 | 42.9 | 56.5 |
| VLA-Adapter | 0.5B | 97.3 | 65.3 | 58.9 | 80.0 |
| **NS-VLA (Ours)** | **2B** | **98.6** | **69.1** | **79.4** | **91.2** |

## Installation

```bash
# Clone the repository
git clone https://github.com/Zuzuzzy/NS-VLA.git
cd NS-VLA

# Create conda environment
conda create -n nsvla python=3.10 -y
conda activate nsvla

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
NS-VLA/
├── configs/                  # Training and evaluation configurations
│   ├── train/
│   └── eval/
├── nsvla/                    # Core NS-VLA framework
│   ├── encoder/              # Neuro-symbolic encoder
│   ├── solver/               # Symbolic solver and action generator
│   ├── rl/                   # Online RL optimization (GRPO)
│   ├── primitives/           # Primitive definitions and classifier
│   └── utils/                # Utility functions
├── scripts/                  # Training and evaluation scripts
│   ├── train.sh
│   ├── eval.sh
│   └── demo.sh
├── data/                     # Data processing and primitives
│   ├── libero/
│   ├── calvin/
│   └── primitive_annotations/
├── assets/                   # Figures and media
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

## Quick Start

### Training

```bash
# Stage I: Supervised pretraining
bash scripts/train.sh --stage pretrain --config configs/train/pretrain.yaml

# Stage II: Online RL optimization
bash scripts/train.sh --stage rl --config configs/train/rl_grpo.yaml
```

### Evaluation

```bash
# Evaluate on LIBERO
bash scripts/eval.sh --benchmark libero --checkpoint path/to/checkpoint

# Evaluate on CALVIN ABC→D
bash scripts/eval.sh --benchmark calvin --checkpoint path/to/checkpoint
```

## Citation

If you find our work useful, please consider citing:

```bibtex
@article{zhu2026nsvla,
  title={NS-VLA: Towards Neuro-Symbolic Vision-Language-Action Models},
  author={Zhu, Ziyue and Wu, Shangyang and Zhao, Shuai and Zhao, Zhiqiu and Li, Shengjie and Wang, Yi and Li, Fang and Luo, Haoran},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## Acknowledgement

We thank the developers of [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [CALVIN](https://github.com/mees/calvin), [OpenVLA](https://github.com/openvla/openvla), and [Qwen-VL](https://github.com/QwenLM/Qwen-VL) for their open-source contributions.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
