# Set wandb to only initialize on main process
export WANDB_INIT_ON_PRIMARY_PROCESS_ONLY=true
# Optimize CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


ACCELERATE_PATH=~/anaconda3/envs/refinerft/bin/accelerate
GPU_IDS=0,1
NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=${GPU_IDS}


# Run training with accelerate

$ACCELERATE_PATH launch \
  --gpu_ids ${GPU_IDS} \
  --num_processes=${NUM_GPUS} \
  --main_process_port 29534 \
  src/refinerft/grpo_classification.py \
  --configs src/refinerft/configs/train_configs_sft_cars.yaml \
  --use_accelerate \

$ACCELERATE_PATH launch \
  --gpu_ids ${GPU_IDS} \
  --num_processes=${NUM_GPUS} \
  --main_process_port 29534 \
  src/refinerft/grpo_classification.py \
  --configs src/refinerft/configs/train_configs_sft_flower102.yaml \
  --use_accelerate \

$ACCELERATE_PATH launch \
  --gpu_ids ${GPU_IDS} \
  --num_processes=${NUM_GPUS} \
  --main_process_port 29534 \
  src/refinerft/grpo_classification.py \
  --configs src/refinerft/configs/train_configs_sft_pets.yaml \
  --use_accelerate \


# $ACCELERATE_PATH launch \
#   --gpu_ids ${GPU_IDS} \
#   --num_processes=${NUM_GPUS} \
#   --main_process_port 29543 \
#   src/refinerft/grpo_classification.py \
#   --configs src/refinerft/configs/train_configs_sft_cot.yaml \
#   --use_accelerate \

# $ACCELERATE_PATH launch \
#   --gpu_ids ${GPU_IDS} \
#   --num_processes=${NUM_GPUS} \
#   --main_process_port 29510 \
#   src/refinerft/grpo_classification.py \
#   --configs src/refinerft/configs/train_configs_aircraft_mrpo_dynamic_lora_r64a128_aptlr5e-6.yaml \
#   --use_accelerate \

# $ACCELERATE_PATH launch \
#   --gpu_ids ${GPU_IDS} \
#   --num_processes=${NUM_GPUS} \
#   --main_process_port 29510 \
#   src/refinerft/grpo_classification.py \
#   --configs src/refinerft/configs/train_configs_car_grpo.yaml \
#   --use_accelerate \

# $ACCELERATE_PATH launch \
#   --gpu_ids ${GPU_IDS} \
#   --num_processes=${NUM_GPUS} \
#   --main_process_port 29510 \
#   src/refinerft/grpo_classification.py \
#   --configs src/refinerft/configs/train_configs_car_mrpo.yaml \
#   --use_accelerate \

# cd ./classification
# python Qwen2_VL_classification_infere.py \
#     --model_path ${SAVE_PATH} \