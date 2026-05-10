#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip

python -m pip install \
  accelerate \
  beautifulsoup4 \
  datasets \
  deepspeed \
  gradio \
  lmdeploy \
  matplotlib \
  omegaconf \
  openai \
  peft \
  qwen_vl_utils \
  tensorboardx \
  torchvision \
  tqdm \
  wandb

# Match the Visual-RFT environment. Adjust this wheel if your CUDA/PyTorch
# versions differ.
python -m pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Keep these versions pinned for the released training scripts.
python -m pip install vllm==0.7.2
python -m pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef
python -m pip install trl==0.14.0
