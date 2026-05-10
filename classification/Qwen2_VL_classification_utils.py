import io
import os
import re
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM, PeftModel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
torch.manual_seed(1234)

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色


def plot_images(image_paths):
    num_images = len(image_paths)
    
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))
    
    for i, image_path in enumerate(image_paths):
        img = mpimg.imread(image_path)
        if num_images == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(img)
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def run(accelerator,
        model_path, 
        model_base, 
        cache_dir, 
        aircraft_data_path, 
        val_data_path="./val_data/", 
        dataset_name="fgvc_aircraft"):
    """
    Modified run function that initializes model inside and uses accelerator for device management.
    
    Args:
        accelerator: Accelerator instance for distributed training
        rank: Process rank
        world_size: Total number of processes
        model_path: Path to the model checkpoint
        model_base: Base model name
        cache_dir: Cache directory for models
        aircraft_data_path: Path to aircraft images
        val_data_path: Path to validation data
        dataset_name: Name of the dataset
    
    Returns:
        List[int]: [error_count, right_count, total_count]
    """
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    # Check if trained with lora and load model accordingly
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_base, 
            cache_dir=cache_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        lora_model = PeftModel.from_pretrained(model, model_path)
        model = lora_model.merge_and_unload()
        print(f"Load lora ckpt from {model_path}")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print(f"Load fully finetuned ckpt from {model_path}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path)
    
    model = model.eval()

    ### get categories name
    with open(os.path.join(val_data_path, f'{dataset_name}.txt'), 'r') as file:
        lines = file.readlines()
    categories = []
    for line in lines:
        categories.append(line.strip())
    print(len(categories))
    print(categories)   ### 对应 0-101

    ### get validation data
    pth_file_path = os.path.join(val_data_path, f'{dataset_name}.pth')
    predictions = torch.load(pth_file_path)
    
    val_set = []
    for item in predictions:
        for k,v in item.items():
            val_set.append({k:int(v['label'])})
    print(f"Number of test samples: {len(val_set)}")

    import math
    split_length = math.ceil(len(val_set)/world_size)
    print("Split Chunk Length:" + str(split_length))
    split_images = val_set[int(rank*split_length) : int((rank+1)*split_length)]
    print(len(split_images))

    ### 遍历 val 中的所有图片
    error_count = 0
    right_count = 0
    total_count = 0
    for image in tqdm(split_images): 
        ### 获取图片信息
        for k,v in image.items():
            image_path = k
            image_label = v
        image_cate = categories[image_label]   
        # plot_images([image_path])
    
        question = (
        "This is an image containing an aircraft. Please identify the model of the aircraft based on the image.\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "The output answer format should be as follows:\n"
        "<think> ... </think> <answer>species name</answer>\n"
        "Please strictly follow the format."
        )
    
        image_path = os.path.join(aircraft_data_path, os.path.basename(image_path))
        query = "<image>\n"+question
        # print(RED+query+RESET)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path}
                ] + [{"type": "text", "text": query}],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # Use accelerator to handle device placement
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Inference: Generation of the output
        with torch.no_grad():  # Add no_grad for memory efficiency during evaluation
            generated_ids = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = response[0]
        # print("\033[92m" + response + "\033[0m")
        total_count += 1
        try:
            match = re.search(r"<answer>(.*?)</answer>", response)
            if match:
                answer_content = match.group(1)
                # print(image_cate, answer_content)
                image_cate = image_cate.replace(' ','').replace('_','').lower()
                answer_content = answer_content.replace(' ','').replace('_','').lower()
                # judgement
                if image_cate in answer_content or answer_content in image_cate:
                    right_count += 1
                else:
                    pass
        except Exception as e:
            error_count+=1
            
    return [error_count, right_count, total_count]

def aircraft_classification_evaluation(model_path,
                                       model_base,
                                       cache_dir, 
                                       device, 
                                       aircraft_data_path="/research/cvl-zhujie4/data/fgvc-aircraft-2013b/data/images/", 
                                       categories_file="./classification/val_data/fgvc_aircraft.txt",
                                       val_data_file="./classification/val_data/fgvc_aircraft.pth",
                                       local_rank=0,
                                       world_size=1):
    """
    Aircraft classification evaluation function for integration with trainer.
    
    Args:
        device: Device to run evaluation on
        aircraft_data_path: Path to aircraft images
        categories_file: Path to categories file
        val_data_file: Path to validation data file
        local_rank: Local rank of current GPU (default: 0)
        world_size: Total number of GPUs (default: 1)
    
    Returns:
        dict: Evaluation metrics including accuracy, error_count, right_count, total_count
    """
    # Ensure model is on the correct device and in eval mode
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_base, 
                                                                cache_dir=cache_dir,
                                                                torch_dtype=torch.bfloat16,
                                                                attn_implementation="flash_attention_2",
                                                                device_map="cpu")
        lora_model = PeftModel.from_pretrained(model, model_path)
        model = lora_model.merge_and_unload()
        print(f"Load lora ckpt from {model_path}")
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
        print(f"Load fully finetuned ckpt from {model_path}")
    # processor = AutoProcessor.from_pretrained(model_base, cache_dir=cache_dir) 
    processor = AutoProcessor.from_pretrained(model_path) 

    model = model.to(torch.device(local_rank))
    model = model.eval()    
    
    ### get categories name
    with open(categories_file, 'r') as file:
        lines = file.readlines()
    categories = []
    for line in lines:
        categories.append(line.strip())
    
    ### get validation data
    predictions = torch.load(val_data_file)
    
    val_set = []
    for item in predictions:
        for k,v in item.items():
            val_set.append({k:int(v['label'])})
    
    
    import math
    split_length = math.ceil(len(val_set)/world_size)
    print("Split Chunk Length:" + str(split_length))
    split_images = val_set[int(local_rank*split_length) : int((local_rank+1)*split_length)]
    print(len(split_images))

    ### 遍历当前GPU分配的图片
    error_count = 0
    right_count = 0
    total_count = 0
        
    for image in tqdm(split_images, desc=f"Aircraft Classification Evaluation (GPU {local_rank})"): 
        ### 获取图片信息
        for k,v in image.items():
            image_path = k
            image_label = v
        image_cate = categories[image_label]   
    
        question = (
        "This is an image containing an aircraft. Please identify the model of the aircraft based on the image.\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "The output answer format should be as follows:\n"
        "<think> ... </think> <answer>species name</answer>\n"
        "Please strictly follow the format."
        )
    
        full_image_path = os.path.join(aircraft_data_path, os.path.basename(image_path))
        query = "<image>\n"+question
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": full_image_path}
                ] + [{"type": "text", "text": query}],
            }
        ]
        
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        
        # Inference: Generation of the output
        with torch.no_grad():  # Add no_grad for memory efficiency
            generated_ids = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = response[0]
        
        total_count += 1
        try:
            match = re.search(r"<answer>(.*?)</answer>", response)
            if match:
                answer_content = match.group(1)
                image_cate_clean = image_cate.replace(' ','').replace('_','').lower()
                answer_content_clean = answer_content.replace(' ','').replace('_','').lower()
                # judgement
                if image_cate_clean in answer_content_clean or answer_content_clean in image_cate_clean:
                    right_count += 1
        except Exception as e:
            error_count += 1
    
    accuracy = right_count / total_count if total_count > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "error_count": error_count,
        "right_count": right_count,
        "total_count": total_count,
    }