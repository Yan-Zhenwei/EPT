# EPT: Expert Pyramid Tuning

> **Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Task Allocation Based on Professional Specialization**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

## 📋 Overview

<p align="center">
  <img src="framework.png" width="800"/>
</p>

**EPT** is a novel parameter-efficient fine-tuning method that integrates the multi-scale feature pyramid concept into MoE-LoRA architectures.

### Key Features

- 🔹 **Shared Meta-knowledge Subspace**: Encodes universal linguistic patterns in low dimensions
- 🔹 **Pyramid Projection Mechanism**: Reconstructs high-dimensional features at varying scales
- 🔹 **Task-aware Router**: Dynamically selects optimal multi-scale feature combinations
- 🔹 **Zero Additional Latency**: Re-parameterization enables efficient inference

## 🔧 Environment Setup

### Prerequisites

- Python 3.10+
- CUDA 11.8+
- NVIDIA GPU (A100 80GB recommended)

### Installation

```bash
# 1. Create conda environment
conda create -n ept python=3.10 -y
conda activate ept

# 2. Install PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
pip install -r requirements.txt
```

### Requirements

```
accelerate>=1.0.0
datasets>=4.0.0
deepspeed>=0.12.0
transformers>=4.36.0
peft>=0.7.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
pandas>=2.0.0
sentencepiece>=0.1.99
evaluate>=0.4.0
```

## 📁 Project Structure

```
EPT/
├── finetune.py          # Main training script
├── trainer.py           # Custom trainer with task routing
├── metrics.py           # Evaluation metrics (accuracy, F1, etc.)
├── rank.py              # Task ID management
├── requirements.txt     # Python dependencies
├── t5-base/             # Pre-trained T5-base model
├── data/                # Dataset processing utilities
├── peft/                # Modified PEFT library with EPT
├── transformers/        # Modified Transformers library
└── script/
    └── train_ept.sh     # Training script
```

## 📊 Datasets

- **GLUE**: CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2, STS-B
- **Commonsense**: BoolQ, PIQA, OpenBookQA, ARC-Easy, ARC-Challenge

## 🚀 Training

### Quick Start

```bash
cd EPT
bash script/train_ept.sh
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Expert Kernel Sizes | 2, 2, 4, 4, 6, 6, 8, 8 |
| Top-k Experts | 2 |
| LoRA Rank | 8 |
| LoRA Alpha | 32 |
| Learning Rate | 3e-4 |
| Batch Size | 128 |
| Warmup Steps | 500 |

### Multi-GPU Training

```bash
# Dual GPU training
deepspeed --include localhost:0,1 --master_port 17621 finetune.py \
  --model_name_or_path ./t5-base \
  --expert_kernel_sizes 2 2 4 4 6 6 8 8 \
  --moe_top_k 2 \
  --per_device_train_batch_size 256 \
  ...
```

