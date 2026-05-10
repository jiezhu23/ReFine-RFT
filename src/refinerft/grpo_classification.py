# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import json
import re
import torch
import wandb
import argparse
import time
from omegaconf import OmegaConf
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import logging
from datasets import Dataset,load_dataset, load_from_disk
from transformers import (
set_seed,
TrainingArguments,
Trainer,
Qwen2VLForConditionalGeneration, 
AutoProcessor,
BitsAndBytesConfig
)
from accelerate import Accelerator
from peft import LoraConfig, TaskType
from math_verify import parse, verify
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from refinerft.trainer.grpo_trainer import Qwen2VLGRPOTrainer # third-party trainer from open-R1
from refinerft.trainer.mrpo_trainer import Qwen2VLGRPOTrainer as Qwen2VLMRPOTrainer
from refinerft.trainer.sft_trainer import Qwen2VLSFTTrainer
from refinerft.trainer.ppo_trainer import Qwen2VLPPOTrainer, PPOConfig
from refinerft.utils.prompt_utils import SYSTEM_PROMPT, get_user_prompt_cot, get_user_prompt_ao
from refinerft.reward_func import accuracy_reward, format_reward, think_length_reward, reasoning_reward_openai, accuracy_reward_openai, think_punishment_reward, embedding_similarity_reward

try:
    # Add classification evaluation path
    sys.path.append("./classification")
    # from Qwen2_VL_classification_utils import run as evaluate
    from Qwen2_VL_classification_infere import load_test_data
    CLASSIFICATION_AVAILABLE = True
except Exception as e:
    print(e)
    CLASSIFICATION_AVAILABLE = False
    print("Warning: classification evaluation modules not found. Classification evaluation will be disabled.")

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--configs', type=str, default='/research/cvlshare/cvl-zhujie4/Refine-RFT/src/refinerft/configs/debug.yaml')
    parser.add_argument('--configs', type=str, default='/research/cvlshare/cvl-zhujie4/Refine-RFT/src/refinerft/configs/train_configs_grpo_lora_r64a128_think10_vqarad.yaml')

    parser.add_argument('--image_root', type=str, default='')
    parser.add_argument('--load_quantized', action='store_true')
    parser.add_argument('--use_accelerate', action='store_true', help='Whether to use Accelerate for distributed training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--test_only', action='store_true', help='Only run evaluation on the test set')

    args = parser.parse_args()
    configs = OmegaConf.load(args.configs)
    configs = OmegaConf.to_container(configs, resolve=True)

    # add configs to args
    for k, v in configs.items():
        setattr(args, k, v)

    return args


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "reasoning": reasoning_reward_openai,
    "think_length": think_length_reward,
    "think_punishment": think_punishment_reward,
    "accuracy_mllm": accuracy_reward_openai,
    "embedding_similarity": embedding_similarity_reward,
    # "overlong": soft_overlong_punishment,
}

dataset_name_mapping = {
    "laolao77/ViRFT_CLS_fgvc_aircraft_4_shot": "fgvc_aircraft",
    "laolao77/ViRFT_CLS_flower_4_shot": "oxford_flowers",
    "laolao77/ViRFT_CLS_car196_4shot": "stanford_cars",
    "laolao77/ViRFT_CLS_pets37_4shot": "pets",
    "flaviagiammarino/vqa-rad": "vqa-rad",

}


def main(args, accelerator):
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # # Fix random seeds for reproducibility
    # set_seed(args.seed)
    # print(f"✅ Random seed set to {args.seed} for reproducibility")

    run_name = args.wandb_run_name
    os.makedirs(os.path.join(args.output_dir, run_name), exist_ok=True)    
    
    # Get reward functions
    # args.reward_funcs = ['accuracy','format','overlong']
    if args.train_method != "sft":
        reward_funcs = [reward_funcs_registry[func] for func in args.reward_funcs]

    # Load the dataset
    if args.use_cot:
        local_save_path = os.path.join(args.dataset_cache_dir, f"{args.dataset_name.split('/')[-1]}_cot")
        dataset = load_from_disk(local_save_path)
    else:
        dataset = load_dataset(args.dataset_name,
                            cache_dir=args.dataset_cache_dir)
        
    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": example["problem"]},
                    ],
                },
            ],
        }

    # Load the test data
    if "vqa-rad" in args.dataset_name:
        # Add specified prompt to each question
        _prompt = f"Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows:\n<think> ... </think> <answer>your answer</answer>\nPlease strictly follow the format and limit your response in {args.max_len} words."
        dataset['train'] = dataset['train'].map(lambda example: {"problem": example["question"] + " " + _prompt, "solution": example["answer"]})
        dataset['test'] = dataset['test'].map(lambda example: {"problem": example["question"] + " " + _prompt, "category": example["answer"], "eval_dataset_name": args.dataset_name})
        test_dataset = dataset['test'].map(make_conversation_image)
    elif "RobustAD" in args.dataset_name:
        # Add specified prompt to each question
        _question = "You are a robust anomaly detection model. Given an image, output 0 if normal, 1 if anomalous. return 0 if normal, 1 if anomalous."
        _prompt = f"Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows:\n<think> ... </think> <answer>your answer</answer>\nPlease strictly follow the format and limit your response in {args.max_len} words."
        dataset['train'] = dataset['train'].map(lambda example: {"problem": _question + " " + _prompt, "solution": str(example["label"])})
        dataset['test'] = dataset['test'].map(lambda example: {"problem": _question + " " + _prompt, "category": str(example["label"]), "eval_dataset_name": args.dataset_name})
        # delete the label column which will cause error
        dataset['test'] = dataset['test'].remove_columns("label")
        dataset['train'] = dataset['train'].remove_columns("label")
        test_dataset = dataset['test'].map(make_conversation_image)
    else:    
        test_dataset = Dataset.from_list(load_test_data(dataset_name=dataset_name_mapping[args.dataset_name], prompt_type=args.prompt_type))
    
    ### lzy modified
    # from datasets import DatasetDict
    # dataset = DatasetDict.load_from_disk(script_args.dataset_name)


    if "image" in dataset["train"].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")
    
    # Log dataset statistics
    if accelerator.is_main_process:
        logger.info(msg=f"Train Dataset size: {len(dataset['train'])} examples")
        print(dataset['train'][-1])
    # Configure model loading for better GPU utilization
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()  # Clear GPU cache before loading model
    # Optimize CUDA operations
    torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes

    # # Configure quantization for efficient memory usage
    # bnb_config = BitsAndBytesConfig(
    #             load_in_4bit=args.load_quantized == "4bit",
    #             load_in_8bit=args.load_quantized == "8bit",
    #             bnb_4bit_compute_dtype=torch.bfloat16,
    #             llm_int8_has_fp16_weight=True,  # For better performance with 8-bit quantization
    #             llm_int8_threshold=6.0,
    # ) if args.load_quantized else None
    
    # # Load tokenizer first to avoid GPU memory fragmentation
    # tokenizer = AutoProcessor.from_pretrained(args.model,
    #                                         trust_remote_code=True,
    #                                         max_pixels=args.max_pixels,
    #                                         cache_dir=args.model_cache_dir)

    # Configure LoRA with optimized parameters
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",  # To save memory
        fan_in_fan_out=False,  # Set to True for specific modules that require it
    ) if args.use_lora else None    
        
    if args.train_method == "grpo":
        trainer_cls = Qwen2VLGRPOTrainer
    elif args.train_method == "sft":
        trainer_cls = Qwen2VLSFTTrainer
    elif args.train_method == "ppo":
        trainer_cls = Qwen2VLPPOTrainer
    else: # mrpo
        trainer_cls = Qwen2VLMRPOTrainer
        
    print("using: ", trainer_cls)
    if args.use_lora:
        print("Detected using lora, deepspeed will be disabled right now")
        args.deepspeed = None
    
    if args.train_method == 'sft':
        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, run_name),
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size =args.per_device_eval_batch_size ,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps if args.max_steps is not None else -1,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            fp16=args.mixed_precision,
            bf16=args.bf16,
            deepspeed=args.deepspeed if args.deepspeed is not None and not args.use_lora else None,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.eval_steps if hasattr(args, 'eval_steps') else args.save_steps,
            save_total_limit=1,
            load_best_model_at_end=True if args.use_lora else False,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to="wandb" if args.use_wandb and accelerator.is_main_process else 'none',
            remove_unused_columns=False,
            dataloader_num_workers=args.workers,
            dataloader_persistent_workers=True,
            dataloader_prefetch_factor=args.prefetch_factor,
            optim="adamw_torch", # More efficient implementation
            max_grad_norm=1.0,  # Helps with training stability
            ddp_find_unused_parameters=False,  # Important for DDP
            gradient_checkpointing=args.gradient_checkpointing,
            save_on_each_node=False,
            eval_strategy="steps" if args.use_lora else "no", # do not evaluate during training for fully finetuning
            eval_steps=args.eval_steps if hasattr(args, 'eval_steps') else 10,
            eval_delay=args.eval_delay,
            seed=args.seed,
        )
    
    elif args.train_method == "ppo":
        training_args = PPOConfig(
            output_dir=os.path.join(args.output_dir, run_name),
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # GRPO: num_generations -> PPO: num_sample_generations
            num_sample_generations=args.num_generations,
            learning_rate=args.learning_rate,
            # GRPO: beta -> PPO: kl_coef
            kl_coef=args.beta if hasattr(args, 'beta') else 0.05,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps if args.max_steps is not None else -1,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            fp16=args.mixed_precision,
            bf16=args.bf16,
            deepspeed=args.deepspeed if args.deepspeed is not None and not args.use_lora else None,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.eval_steps if hasattr(args, 'eval_steps') else args.save_steps,
            save_total_limit=1,
            load_best_model_at_end=True if args.use_lora else False,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to="wandb" if args.use_wandb and accelerator.is_main_process else 'none',
            remove_unused_columns=False,
            dataloader_num_workers=args.workers,
            dataloader_persistent_workers=True, 
            dataloader_prefetch_factor=args.prefetch_factor,
            max_prompt_length=args.max_prompt_length,
            # GRPO: max_completion_length -> PPO: response_length
            response_length=args.max_completion_length,
            optim="adamw_torch", 
            max_grad_norm=1.0,  # Helps with training stability
            ddp_find_unused_parameters=False,  # Important for DDP
            gradient_checkpointing=args.gradient_checkpointing,
            save_on_each_node=False,
            eval_strategy="steps" if args.use_lora else "no",  # do not evaluate during training for fully finetuning
            eval_steps=args.eval_steps if hasattr(args, 'eval_steps') else 10,
            eval_delay=args.eval_delay,
            seed=args.seed,
            # PPO Hyperparams
            cliprange=0.2,  # PPO 的 clip range，默认 0.2
            # vf_coef=0.1,  # value function coefficient
            # gamma=1.0,  # discount factor
            # lam=0.95,  # GAE lambda
            # whiten_rewards=False,  # whether to whiten rewards
        )
        
    else:
        training_args = GRPOConfig(
            output_dir=os.path.join(args.output_dir, run_name),
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size =args.per_device_eval_batch_size ,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_generations=args.num_generations,
            learning_rate=args.learning_rate,
            beta=args.beta,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps if args.max_steps is not None else -1,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            fp16=args.mixed_precision,
            bf16=args.bf16,
            deepspeed=args.deepspeed if args.deepspeed is not None and not args.use_lora else None,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            save_steps=args.eval_steps if hasattr(args, 'eval_steps') else args.save_steps,
            save_total_limit=1,
            load_best_model_at_end=True if args.use_lora else False,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to="wandb" if args.use_wandb and accelerator.is_main_process else 'none',
            remove_unused_columns=False,
            dataloader_num_workers=args.workers,
            dataloader_persistent_workers=True, 
            dataloader_prefetch_factor=args.prefetch_factor,
            max_prompt_length=args.max_prompt_length,  
            max_completion_length=args.max_completion_length,  
            optim="adamw_torch", # More efficient implementation
            max_grad_norm=1.0,  # Helps with training stability
            ddp_find_unused_parameters=False,  # Important for DDP
            gradient_checkpointing=args.gradient_checkpointing,
            save_on_each_node=False,
            eval_strategy="steps" if args.use_lora else "no", # do not evaluate during training for fully finetuning
            eval_steps=args.eval_steps if hasattr(args, 'eval_steps') else 10,
            eval_delay=args.eval_delay,
            seed=args.seed,
        )

    # Initialize the GRPO trainer
    if args.train_method == 'sft':
        trainer = trainer_cls(
            model=args.model,
            model_cache_dir=args.model_cache_dir,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
            peft_config=peft_config,
            max_pixels=args.max_pixels,
            custom_args=args,
        )
    else:
        trainer = trainer_cls(
            model=args.model,
            model_cache_dir=args.model_cache_dir,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=test_dataset if training_args.eval_strategy != "no" else None, # use test_dataset for evaluation
            peft_config=peft_config,
            # attn_implementation=model_args.attn_implementation,
            max_pixels=args.max_pixels,
            # min_pixels=args.min_pixels,
            reward_weights=args.reward_weights,
            custom_args = args,
        )

    if args.test_only:
        logger.info("Running evaluation only...")
        eval_metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {eval_metrics}")
    else:
        # Train and push the model to the Hub
        trainer.train()

        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=args.dataset_name)
        
    # clean trainer and dataset
    # del trainer
    # del dataset
    # import gc
    # gc.collect()
    torch.cuda.empty_cache()
    # Self-evaluation
    if CLASSIFICATION_AVAILABLE:
        
        try:
            # NOTE: problem with fully finetuned model with deepspeed
            raise Exception("Not implemented")
            accelerator.wait_for_everyone()
            time.sleep(5)
            eval_results = evaluate_multi_gpu(
                accelerator=accelerator,
                model_path=training_args.output_dir,
                model_base=args.model,
                cache_dir=args.model_cache_dir,
                aircraft_data_path="/research/cvl-zhujie4/data/fgvc-aircraft-2013b/data/images/",
                val_data_path="./classification/val_data/"
            )
        except Exception as e:
            import traceback, subprocess, re
            # print(f"Direct evaluation failed: {e}")
            traceback.print_exc()
            if accelerator.is_main_process:
                print("Falling back to subprocess evaluation...")
                try:
                    start_time = time.time()
                    ckpt_path = os.path.abspath(training_args.output_dir)
                    os.chdir("./classification")
                    result = subprocess.run([
                        "python", "Qwen2_VL_classification_infere.py", 
                        "--model_path", ckpt_path,
                        "--model_base", args.model,
                        # "--cache_dir", args.model_cache_dir,
                        "--dataset_name", dataset_name_mapping[args.dataset_name],
                        "--prompt_type", args.prompt_type,
                    ], capture_output=True, text=True, cwd=".")
                    print(f"Subprocess evaluation time: {(time.time() - start_time)/60:.2f} minutes")
                    if result.stderr:
                        print(result.stderr)
                        eval_results_str = result.stderr.split('Accuracy')[1].strip()
                        eval_results = {}
                        nums = re.findall(r"\d+(?:\.\d+)?", eval_results_str)
                        eval_results['accuracy'] = round(float(nums[0]) / 100, 3)
                        eval_results['right_count'] = int(nums[1])
                        eval_results['total_count'] = int(nums[2])
                        print(f"Accuracy: {eval_results}")
                except Exception as subprocess_error:
                    # print(f"Subprocess evaluation also failed: {subprocess_error}")
                    traceback.print_exc()
                    eval_results = None
        
        # Log to wandb if on main process and results are available
        if accelerator.is_main_process and eval_results is not None and args.use_wandb:
            wandb.log({
                f"final/{dataset_name_mapping[args.dataset_name]}_accuracy": eval_results['accuracy'],
            })
    
    # Close wandb on main process only
    if args.use_wandb and  accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":

    args = parse_args()
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if hasattr(args, 'api_url'):
        API_URL = args.api_url    
    # Initialize accelerator early to check main process
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if args.mixed_precision else ("bf16" if args.bf16 else "no"),
    )
    args.wandb_run_name = args.wandb_run_name + "_" + datetime.now().strftime("%m%d_%H%M")
    # Initialize wandb only on the main process
    if args.use_wandb and accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            save_code=True,
            config=args,
            settings=wandb.Settings(code_dir=os.path.dirname(os.path.abspath(__file__)))
        )
    main(args, accelerator)
