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

import re
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import json
import logging
import argparse
import time
import torch
import wandb
import math
from accelerate import Accelerator
from omegaconf import OmegaConf
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset,load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
from peft import LoraConfig, TaskType
from math_verify import parse, verify
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from refinerft.trainer.grpo_trainer import Qwen2VLGRPOTrainer # third-party trainer from open-R1
from refinerft.trainer.mrpo_trainer import Qwen2VLGRPOTrainer as Qwen2VLMRPOTrainer, dataset_name_mapping
# from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from refinerft.reward_func import  format_reward, think_length_reward, accuracy_reward_openai, think_punishment_reward, embedding_similarity_reward

try:
    # Add evaluation path
    sys.path.append("./coco_evaluation")
    # from Qwen2_VL_classification_utils import run as evaluate
    from Qwen2_VL_coco_infere import run as evaluate
    COCO_EVALUATION_AVAILABLE = True
except Exception as e:
    print(e)
    COCO_EVALUATION_AVAILABLE = False
    print("Warning: coco evaluation modules not found. Coco evaluation will be disabled.")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='/research/cvlshare/cvl-zhujie4/Refine-RFT/src/refinerft/configs/coco_fewshot_mrpo_lora_r64a128.yaml')
    parser.add_argument('--image_root', type=str, default='')
    parser.add_argument('--load_quantized', action='store_true')
    parser.add_argument('--use_accelerate', action='store_true', help='Whether to use Accelerate for distributed training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    configs = OmegaConf.load(args.configs)
    configs = OmegaConf.to_container(configs, resolve=True)

    # add configs to args
    for k, v in configs.items():
        setattr(args, k, v)

    return args

def extract_bbox(response):
    start_tag = "<answer>"
    end_tag = "</answer>"
    input_str = response
    # Check if the start tag is in the string
    if start_tag in input_str:
        # Extract the content between the start tag and end tag
        start_idx = input_str.find(start_tag) + len(start_tag)
        end_idx = input_str.find(end_tag)
        
        # If end_tag is not found (i.e., the string is truncated), assume it should be at the end
        if end_idx == -1:
            end_idx = len(input_str)
    
        content_str = input_str[start_idx:end_idx]
    
        # Check if it ends with a closing bracket, if not, fix it
        if not content_str.endswith("]"):
            # If the string is truncated, remove the incomplete part
            content_str = content_str.rsplit("},", 1)[0] + "}]"
    
        # Replace single quotes with double quotes for valid JSON
        content_str_corrected = content_str.replace("'", '"')
    
        # Convert the corrected string to a list of dictionaries (JSON format)
        try:
            bbox_list = json.loads(content_str_corrected)
        except json.JSONDecodeError as e:
            bbox_list = None
    else:
        bbox_list = None
    return bbox_list

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area
    return iou

def sort_and_calculate_iou(list1, list2, iou_threshold=0.5):
    # to avoid NaN error
    for i in range(len(list2)):
        try:
            list2[i]['Confidence'] = float(list2[i]['Confidence'])
            if math.isnan(list2[i]['Confidence']):
                raise ValueError("Confidence is NaN")
        except:
            print("Confidence is not a float, setting it to 0.0. Original value:", list2[i]['Confidence'])
            list2[i]['Confidence'] = 0.0    

    list2_sorted = sorted(list2, key=lambda x: x['Confidence'], reverse=True)
    
    iou_results = []
    
    matched_list1_indices = set()

    for bbox2 in list2_sorted:
        max_iou = 0
        matched_bbox1 = -1
        best_iou = 0
        for i, bbox1 in enumerate(list1):
            if i not in matched_list1_indices:
                iou = calculate_iou(bbox1['Position'], bbox2['Position'])
                if iou > best_iou:
                    best_iou = iou
                    matched_bbox1 = i

        if best_iou > iou_threshold:
            iou_results.append((best_iou, bbox2['Confidence']))
            matched_list1_indices.add(matched_bbox1)
        else:
            iou_results.append((0, bbox2['Confidence']))
    
    ### [(0.7192676547515258, 1.0), (0, 0.7)]
    return iou_results

def remove_duplicates(bbox_list):
    seen = set()
    unique_bboxes = []
    
    for bbox in bbox_list:
        # Convert the position tuple to a tuple for set hashing
        position_tuple = tuple(bbox['Position'])
        
        if position_tuple not in seen:
            seen.add(position_tuple)
            unique_bboxes.append(bbox)
    
    return unique_bboxes

# V1
def compute_reward_iou(iou_results):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    iou_reward = iou_reward/len(iou_results)
    confidence_reward = confidence_reward/len(iou_results)
    return iou_reward

# V2
def compute_reward_iou_v2(iou_results, len_gt):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    if len_gt>=len(iou_results):
        iou_reward = iou_reward/len_gt
    else:
        iou_reward = iou_reward/len(iou_results)
    return iou_reward

def compute_reward_confidence(iou_results):
    iou_reward = 0.0
    confidence_reward = 0.0
    for i in range(len(iou_results)):
        temp_iou = iou_results[i][0]
        temp_confidence = iou_results[i][1]

        temp_iou_reward = temp_iou
        if temp_iou == 0:
            temp_confidence_reward = (1-temp_iou)*(1-temp_confidence)
        else:
            temp_confidence_reward = temp_confidence

        iou_reward += temp_iou_reward
        confidence_reward += temp_confidence_reward
        
    iou_reward = iou_reward/len(iou_results)
    confidence_reward = confidence_reward/len(iou_results)
    return confidence_reward

def accuracy_reward_iou(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        student_answer_bbox = []
        ground_truth_bbox = []
        iou_results = []
        show_flage = 0

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                show_flage = 1
                # Extract answer from solution if it has think/answer tags
                ground_truth = sol.strip()
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = '<answer>'+student_answer+'</answer>'

                # fix format error
                student_answer = student_answer.replace("[[",'[')  
                student_answer = student_answer.replace("]]",']')  
                student_answer = student_answer.replace("\n",'')  
                # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
                ground_truth_bbox = extract_bbox(ground_truth)
                student_answer_bbox = extract_bbox(student_answer)
                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:
                    reward = 0.0
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
                    ### new iou reward
                    reward = compute_reward_iou_v2(iou_results, len(ground_truth_bbox))
                    if reward>1:
                        reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward of IoU: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if show_flage==1:
                    f.write(f"student_answer_bbox: {student_answer_bbox}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
                    if student_answer_bbox!=None:
                        f.write(f"iou_results: {iou_results}\n")
        show_flage = 0 
    return rewards

def accuracy_reward_confidence(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        student_answer_bbox = []
        ground_truth_bbox = []
        iou_results = []
        show_flage = 0

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                show_flage = 1
                # Extract answer from solution if it has think/answer tags
                ground_truth = sol.strip()
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                student_answer = '<answer>'+student_answer+'</answer>'

                # fix format error
                student_answer = student_answer.replace("[[",'[')
                student_answer = student_answer.replace("]]",']')
                student_answer = student_answer.replace("\n",'')
                # [{'Position': [254, 303, 291, 365], 'Confidence': 0.9}, {'Position': [100, 100, 200, 200], 'Confidence': 0.8}]
                ground_truth_bbox = extract_bbox(ground_truth)
                student_answer_bbox = extract_bbox(student_answer)
                # pdb.set_trace()
                if student_answer_bbox==None or type(student_answer_bbox[0])!=dict:  # wrong bbox
                    reward = 0.0
                else:
                    student_answer_bbox = remove_duplicates(student_answer_bbox)   # remove duplicates
                    iou_results = sort_and_calculate_iou(ground_truth_bbox, student_answer_bbox)
                    reward = compute_reward_confidence(iou_results)
                    if reward>1:
                        reward = 1.0
                    if reward<0:
                        reward = 0.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward of Confidence: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if show_flage==1:
                    f.write(f"student_answer_bbox: {student_answer_bbox}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
                    if student_answer_bbox!=None:
                        f.write(f"iou_results: {iou_results}\n")
        show_flage = 0 
    return rewards


###  reward registry three parts
reward_funcs_registry = {
    "accuracy_iou": accuracy_reward_iou,
    "accuracy_confidence": accuracy_reward_confidence,
    "format": format_reward,
    "think_length": think_length_reward,
    "think_punishment": think_punishment_reward,
    "accuracy_mllm": accuracy_reward_openai,
    "embedding_similarity": embedding_similarity_reward,    
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


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
    # args.reward_funcs = ['accuracy_iou','accuracy_confidence','format']
    if args.train_method != "sft":
        reward_funcs = [reward_funcs_registry[func] for func in args.reward_funcs]

    # Load the dataset from huggingface
    dataset = load_dataset(args.dataset_name,
                        cache_dir=args.dataset_cache_dir)
    # Load the test data
    # For COCO: minimal eval_dataset to trigger evaluation_loop (COCO eval runs via subprocess inside)
    coco_eval_dataset = dataset["train"].select(range(1))
    # test_dataset = Dataset.from_list(load_test_data(dataset_name_mapping[args.dataset_name]))
    
    # Load the dataset from local disk
    # from datasets import DatasetDict
    # dataset = DatasetDict.load_from_disk(script_args.dataset_name)


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
    
    # trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    trainer_cls = Qwen2VLGRPOTrainer if args.train_method == "grpo" else Qwen2VLMRPOTrainer
    print("using: ", trainer_cls)
    if args.use_lora:
        print("Detected using lora, deepspeed will be disabled right now")
        args.deepspeed = None
        
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
        save_total_limit=2,
        load_best_model_at_end=False if args.use_lora else False,
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
        eval_strategy="no",
        # eval_strategy="steps" if coco_eval_dataset is not None else ("no" if args.use_lora else "no"),
        eval_steps=args.eval_steps if hasattr(args, 'eval_steps') else 10,
        eval_delay=args.eval_delay,
        seed=args.seed,
    )

    # eval_dataset: COCO uses coco_eval_dataset to trigger evaluation_loop (runs subprocess); others use test_dataset
    eval_dataset = coco_eval_dataset if coco_eval_dataset is not None else None

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=args.model,
        model_cache_dir=args.model_cache_dir,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        # attn_implementation=model_args.attn_implementation,
        max_pixels=args.max_pixels,
        # min_pixels=args.min_pixels,
        reward_weights=args.reward_weights,
        custom_args = args,
    )

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
    if COCO_EVALUATION_AVAILABLE:
        
        try:
            # NOTE: problem with fully finetuned model with deepspeed
            raise Exception("Not implemented")
        except Exception as e:
            import traceback, subprocess, re
            if accelerator.is_main_process:
                print("Falling back to subprocess evaluation...")
                try:
                    start_time = time.time()
                    ckpt_path = os.path.abspath(training_args.output_dir)
                    os.chdir("./coco_evaluation")
                    result = subprocess.run([
                        "python", "Qwen2_VL_coco_infere.py", 
                        "--model_path", ckpt_path,
                        "--model_base", args.model,
                        "--cache_dir", args.model_cache_dir,
                        "--coco_data_path", args.coco_data_path,
                        "--val_type", "few",
                    ], capture_output=True, text=True, cwd=".")
                    print(f"Subprocess evaluation time: {(time.time() - start_time)/60:.2f} minutes")
                    if result.stderr:
                        eval_results = {}
                        # Parse key-value pairs for COCO evaluation
                        key_value_pairs = re.findall(r"Key: (.*), Value: ([\d.]+)", result.stderr)
                        for key, value in key_value_pairs:
                            eval_results[key.strip()] = float(value)

                        # Parse accuracy for classification evaluation
                        if 'mAP for selected categories:' in result.stderr:
                            try:
                                eval_results_str = result.stderr.split('selected categories:')[1].strip()
                                nums = re.findall(r"\d+(?:\.\d+)?", eval_results_str)
                                eval_results['mAP'] = round(float(nums[0]) / 100, 3)
                            except (IndexError, ValueError) as e:
                                print(f"Could not parse mAP from stderr: {e}")
                        # Parse avg completion length (tokens)
                        if 'Avg completion length (tokens):' in result.stderr:
                            try:
                                avg_len_match = re.search(r"Avg completion length \(tokens\): ([\d.]+)", result.stderr)
                                if avg_len_match:
                                    eval_results['avg_completion_length'] = round(float(avg_len_match.group(1)), 2)
                            except (IndexError, ValueError) as e:
                                print(f"Could not parse avg completion length from stderr: {e}")
                        if eval_results:
                            print(f"Evaluation results: {eval_results}")
                except Exception as subprocess_error:
                    # print(f"Subprocess evaluation also failed: {subprocess_error}")
                    traceback.print_exc()
                    eval_results = None
        
        # Log to wandb if on main process and results are available
        if accelerator.is_main_process and eval_results is not None and args.use_wandb:
            wandb_log_dict = {f"final/{k.replace(' ', '_')}": v for k, v in eval_results.items()}
            wandb.log(wandb_log_dict)
    
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
