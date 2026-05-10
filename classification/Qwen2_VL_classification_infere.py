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

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'


import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import functools
import itertools
import argparse
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.refinerft.utils.prompt_utils import get_user_prompt_cot, get_user_prompt_ao

DATASET_MAPPING = {
    "fgvc_aircraft": "aircraft",
    "oxford_flowers": "plant",
    "stanford_cars": "car",
    "pets": "pet",
}

DATASET_IMG_PATH = {
    "fgvc_aircraft": "~/data/fgvc-aircraft-2013b/data/images/",
    "oxford_flowers": "~/data/flowers-102/jpg/",
    "stanford_cars": "~/data/stanford-cars/cars_test/",
    "pets": "~/data/oxford-pet/images",
}

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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./share_models/aircraft-4-shot_new_advantage_0731_1220")
    parser.add_argument("--model_base", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--model_name", type=str, default=None, choices=["llava-1.5-7b", "llava-next-7b", "qwen2-vl-7b", "internvl-1.5-8b"])
    parser.add_argument("--cache_dir", type=str, default="./share_models")
    parser.add_argument("--aircraft_data_path", type=str, default="~/data/fgvc-aircraft-2013b/data/images/")
    parser.add_argument("--dataset_name", type=str, default="oxford_flowers", choices=["fgvc_aircraft", "oxford_flowers", "stanford_cars", "pets"])
    parser.add_argument("--prompt_type", type=str, default="cot", choices=["cot", "ao"])
    return parser.parse_args()


def load_test_data(dataset_name, 
                   val_data_path="./classification/val_data/", prompt_type="cot"):

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
    
    # Format data for HuggingFace dataset
    test_examples = []
    for item in predictions:
        for image_path, label_info in item.items():
            label = int(label_info['label'])
            category = categories[label]
        # question = (
        # f"This is an image containing an {DATASET_MAPPING[dataset_name]}. Please identify the model of the {DATASET_MAPPING[dataset_name]} based on the image.\n"
        # "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        # "The output answer format should be as follows:\n"
        # "<think> ... </think> <answer>species name</answer>\n"
        # "Please strictly follow the format."
        # )
        if prompt_type == "cot":
            question = get_user_prompt_cot(dataset_name)
        elif prompt_type == "ao":
            question = get_user_prompt_ao(dataset_name)
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")

        image_path = os.path.join(DATASET_IMG_PATH[dataset_name], os.path.basename(image_path))
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
        test_examples.append({
            "prompt": messages,
            "image": image_path,
            "label": label,
            "category": category,
            "eval_dataset_name": dataset_name
        })        
    return test_examples


def run(rank, 
        world_size, 
        model_path, 
        model_base, 
        cache_dir, 
        aircraft_data_path, 
        val_data_path="./val_data/", 
        dataset_name="oxford_flowers",
        prompt_type="cot"):
    # check if trained with lora
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

    model = model.to(torch.device(rank))
    model = model.eval()

    ### get categories name
    with open(os.path.join(val_data_path, f'{dataset_name}.txt'), 'r') as file:
        lines = file.readlines()
    categories = []
    for line in lines:
        categories.append(line.strip())
    # print(len(categories))
    # print(categories)   ### 对应 0-101
    print(f"Using Prompt Type: {prompt_type}")
    
    ### get validation data
    pth_file_path = os.path.join(val_data_path, f'{dataset_name}.pth')
    predictions = torch.load(pth_file_path)
    
    val_set = []
    for i, item in enumerate(predictions):
        for k, v in item.items():
            val_set.append({'image_path': k, 'label': int(v['label']), 'index': i})

    # print(len(val_set))
    # print(val_set[0])

    import math
    split_length = math.ceil(len(val_set)/world_size)
    logger.info("Split Chunk Length:" + str(split_length))
    split_images = val_set[int(rank*split_length) : int((rank+1)*split_length)]
    logger.info(len(split_images))

    ### 遍历 val 中的所有图片
    error_count = 0
    right_count = 0
    total_count = 0
    local_conversations = []
    for image in tqdm(split_images): 
        ### 获取图片信息
        image_path = image['image_path']
        image_label = image['label']
        image_index = image['index']
        image_cate = categories[image_label]   
        # plot_images([image_path])
    
        # question = (
        # f"This is an image containing an {DATASET_MAPPING[dataset_name]}. Please identify the model of the {DATASET_MAPPING[dataset_name]} based on the image.\n"
        # "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        # "The output answer format should be as follows:\n"
        # "<think> ... </think> <answer>species name</answer>\n"
        # "Please strictly follow the format."
        # )
        if prompt_type == "cot":
            question = get_user_prompt_cot(dataset_name)
        elif prompt_type == "ao":
            question = get_user_prompt_ao(dataset_name)
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
    
        image_path = os.path.join(DATASET_IMG_PATH[dataset_name], os.path.basename(image_path))
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
        inputs = inputs.to(model.device)
        
        # Inference: Generation of the output
        # NOTE: must set use_cache=True to avoid nonsense output
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
        is_correct = False
        if prompt_type == "cot":
            try:
                match = re.search(r"<answer>(.*?)</answer>", response)
                answer_content = match.group(1)
                # print(image_cate, answer_content)
                image_cate = image_cate.replace(' ','').replace('_','').lower()
                answer_content = answer_content.replace(' ','').replace('_','').lower()
                # judgement
                if image_cate in answer_content or answer_content in image_cate:
                    print('yes')
                    right_count += 1
                    is_correct = True
                    logger.info('Local Right Number: ' + str(right_count))
                else:
                    print('no')
            except Exception as e:
                error_count+=1
        elif prompt_type == "ao":
            true_category_norm = image_cate.replace(' ', '').replace('_', '').lower()
            answer_content_norm = response.replace(' ', '').replace('_', '').lower()
            if true_category_norm in answer_content_norm or answer_content_norm in true_category_norm:
                right_count += 1
                is_correct = True
            else:
                error_count += 1

        conversation_data = {
            "index": image_index,
            "prompt": messages,
            "response": response,
            "label": categories[image_label],
            "is_correct": is_correct,
        }
        local_conversations.append(conversation_data)
            
    return [error_count, right_count, total_count, local_conversations]


def main(model_path, 
         model_base="Qwen/Qwen2-VL-2B-Instruct", 
         cache_dir="./share_models", 
         aircraft_data_path="~/data/fgvc-aircraft-2013b/data/images/",
         dataset_name="fgvc_aircraft",
         prompt_type="cot"):
    multiprocess = torch.cuda.device_count() >= 2
    os.makedirs(model_path, exist_ok=True)
    mp.set_start_method('spawn')
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, 
                                     world_size=world_size,
                                     model_path=model_path,
                                     model_base=model_base,
                                     cache_dir=cache_dir,
                                     aircraft_data_path=aircraft_data_path,
                                     dataset_name=dataset_name,
                                     prompt_type=prompt_type)
            result_lists = pool.map(func, range(world_size))

        global_count_error = 0
        global_count_right = 0
        global_total_count = 0
        all_conversations = []
        for i in range(world_size):
            global_count_error += int(result_lists[i][0])
            global_count_right = global_count_right + result_lists[i][1]
            global_total_count = global_total_count + result_lists[i][2]
            all_conversations.extend(result_lists[i][3])
        
        accuracy = (global_count_right / global_total_count * 100) if global_total_count > 0 else 0
        if all_conversations:
            model_name = os.path.basename(model_path)
            output_filename = f"conversations_{model_name}_{dataset_name}_{prompt_type}_acc{accuracy:.1f}.jsonl"
            output_path = os.path.join(model_path, output_filename)
            if all("index" in conv for conv in all_conversations):
                all_conversations = sorted(all_conversations, key=lambda x: x['index'])
            
            with open(output_path, 'w') as f:
                for item in all_conversations:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Saved conversations to {output_path}")

        logger.info('Model checkpoint:' + model_path)    
        logger.info('Error number: ' + str(global_count_error))  
        logger.info('Total Right Number: ' + str(global_count_right))
        logger.info(f'Accuracy: {global_count_right / global_total_count:.1%} ({global_count_right}/{global_total_count})')
    else:
        logger.info("Not enough GPUs for multiprocessing, running on single GPU")
        error_count, right_count, total_count, conversations = run(
            0, 1, model_path, model_base, cache_dir, aircraft_data_path,
            dataset_name=dataset_name, prompt_type=prompt_type
        )
        accuracy = (right_count / total_count * 100) if total_count > 0 else 0
        if conversations:
            model_name = os.path.basename(model_path)
            output_filename = f"conversations_{model_name}_{dataset_name}_{prompt_type}_acc{accuracy:.1f}.jsonl"
            output_path = os.path.join(model_path, output_filename)
            if all("index" in conv for conv in conversations):
                conversations = sorted(conversations, key=lambda x: x['index'])
            with open(output_path, 'w') as f:
                for item in conversations:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Saved conversations to {output_path}")
        logger.info('Model checkpoint:' + model_path)
        logger.info('Error number: ' + str(error_count))
        logger.info('Total Right Number: ' + str(right_count))
        logger.info(f'Accuracy: {right_count / total_count:.1%} ({right_count}/{total_count})')
 
 
if __name__ == "__main__":
    args = parse_args()
    # model path and model base
    model_path = args.model_path
    model_base = args.model_base
    cache_dir = args.cache_dir
    aircraft_data_path = args.aircraft_data_path
    dataset_name = args.dataset_name
    prompt_type = args.prompt_type
    main(model_path, model_base, cache_dir, aircraft_data_path, dataset_name, prompt_type)
