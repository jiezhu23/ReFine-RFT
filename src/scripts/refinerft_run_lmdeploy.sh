#!/bin/bash

export WANDB_INIT_ON_PRIMARY_PROCESS_ONLY=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_GPUS=4


# Qwen/Qwen2-VL-7B-Instruct
lmdeploy serve api_server Qwen/Qwen2-VL-7B-Instruct \
  --server-name 0.0.0.0 \
  --server-port 8001 \
  --download-dir ./share_models \
  --dtype bfloat16 \
  --session-len 1024 \
  --max-batch-size 32 \
  --vision-max-batch-size 32 \
  --tp $NUM_GPUS \
  --backend turbomind \


# OpenGVLab/InternVL3-38B
# lmdeploy serve api_server OpenGVLab/InternVL3-9B \
#   --server-name 0.0.0.0 \
#   --server-port 8001 \
#   --download-dir ./share_models \
#   --quant-policy 8 \
#   --session-len 65536 \
#   --max-batch-size 96 \
#   --vision-max-batch-size 96 \
#   --tp $NUM_GPUS \
#   --backend turbomind \
#   --chat-template internvl2_5

# Qwen/Qwen2.5-VL-32B-Instruct
# lmdeploy serve api_server Qwen/Qwen2.5-VL-32B-Instruct \
#   --server-name 0.0.0.0 \
#   --server-port 8001 \
#   --download-dir ./share_models \
#   --quant-policy 8 \
#   --session-len 65536 \
#   --max-batch-size 96 \
#   --vision-max-batch-size 96 \
#   --tp $NUM_GPUS \
#   --backend turbomind \