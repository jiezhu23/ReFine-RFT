#!/bin/bash

export WANDB_INIT_ON_PRIMARY_PROCESS_ONLY=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Must have GPUs >= 2
export CUDA_VISIBLE_DEVICES=1,2,3,5
NUM_GPUS=4
PYTHON_PATH=~/anaconda3/envs/refinerft/bin/python


cd ./classification

# Zero-shot model evaluation
# $PYTHON_PATH Zeroshot_classification_infere.py \
#     --model_name qwen2-vl-2b \
#     --cache_dir /research/cvlshare/cvl-zhujie4/Refine-RFT/share_models \
#     --dataset_name pets \
#     --prompt_type cot

# Fine-tuned model
$PYTHON_PATH Qwen2_VL_classification_infere.py \
    --model_path /research/cvlshare/cvl-zhujie4/Refine-RFT/share_models/pets_4_shot-sft-lora-r64a128-ao_1105_0658 \
    --model_base Qwen/Qwen2-VL-2B-Instruct \
    --cache_dir /research/cvlshare/cvl-zhujie4/Refine-RFT/share_models/share_models \
    --dataset_name pets \
    --prompt_type ao

