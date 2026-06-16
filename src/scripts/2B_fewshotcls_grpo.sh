# Set wandb to only initialize on main process
export WANDB_INIT_ON_PRIMARY_PROCESS_ONLY=true
# Optimize CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


ACCELERATE_PATH=~/anaconda3/envs/refinerft/bin/accelerate
GPU_IDS=0,1,2,3
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# Run training with accelerate
$ACCELERATE_PATH launch \
  --gpu_ids ${GPU_IDS} \
  --num_processes=${NUM_GPUS} \
  --main_process_port 29520 \
  src/refinerft/grpo_classification.py \
  --configs src/refinerft/configs/train_configs_grpo_lora_r64a128_aircrafts.yaml \
  --use_accelerate \

# $ACCELERATE_PATH launch \
#   --gpu_ids ${GPU_IDS} \
#   --num_processes=${NUM_GPUS} \
#   --main_process_port 29532 \
#   src/refinerft/grpo_classification.py \
#   --configs src/refinerft/configs/train_configs_grpo_lora_r64a128_cars.yaml \
#   --use_accelerate \

# $ACCELERATE_PATH launch \
#   --gpu_ids ${GPU_IDS} \
#   --num_processes=${NUM_GPUS} \
#   --main_process_port 29543 \
#   src/refinerft/grpo_classification.py \
#   --configs src/refinerft/configs/train_configs_grpo_lora_r64a128_pets.yaml \
#   --use_accelerate \

# $ACCELERATE_PATH launch \
#   --gpu_ids ${GPU_IDS} \
#   --num_processes=${NUM_GPUS} \
#   --main_process_port 29558 \
#   src/refinerft/grpo_classification.py \
#   --configs src/refinerft/configs/train_configs_grpo_lora_r64a128_flower.yaml \
#   --use_accelerate \




# cd ./classification
# python Qwen2_VL_classification_infere.py \
#     --model_path ${SAVE_PATH} \