import io
import os
import re
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
from transformers.generation import GenerationConfig
from peft import AutoPeftModelForCausalLM, PeftModel
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
torch.manual_seed(1234)

from qwen_vl_utils import process_vision_info

# 定义颜色的ANSI代码
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色


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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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


DATASET_MAPPING = {
    "fgvc_aircraft": "aircraft",
    "oxford_flowers": "plant",
    "stanford_cars": "car",
    "pets": "pet",
}

DATASET_IMG_PATH = {
    "fgvc_aircraft": "/research/cvlshare/cvl-zhujie4/data/fgvc-aircraft-2013b/data/images/",
    "oxford_flowers": "/research/cvlshare/cvl-zhujie4/data/flowers-102/jpg/",
    "stanford_cars": "/research/cvlshare/cvl-zhujie4/data/stanford-cars/cars_test/",
    "pets": "/research/cvlshare/cvl-zhujie4/data/oxford-pet/images",
}

    
ZERO_SHOT_MODEL_MAPPING = {
    "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
    "llava-next-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "r1-onevision-7b": "Fancy-MLLM/R1-Onevision-7B",
    "internvl3-8b": "OpenGVLab/InternVL3-8B",
    "internvl2.5-8b": "OpenGVLab/InternVL2_5-8B",
    
}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=["llava-1.5-7b", "llava-next-7b", "qwen2-vl-2b", "qwen2-vl-7b", "qwen2.5-vl-7b", "r1-onevision-7b", "internvl3-8b", "internvl2.5-8b"])
    parser.add_argument("--cache_dir", type=str, default="/research/cvl-zhujie4/Visual-RFT/share_models")
    parser.add_argument("--dataset_name", type=str, default="oxford_flowers", choices=["fgvc_aircraft", "oxford_flowers", "stanford_cars", "pets"])
    parser.add_argument("--prompt_type", type=str, default="cot", choices=["cot", "ao"])
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Directory to save conversation logs.")
    return parser.parse_args()


def load_model(model_name, cache_dir, device):
    """
    Load model and processor based on the model name.
    """

    
    model_hf_path = ZERO_SHOT_MODEL_MAPPING[model_name]
    logger.info(f"Loading zero-shot model: {model_hf_path}")
    
    if "llava-1.5" in model_name:
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        model = LlavaForConditionalGeneration.from_pretrained(
            model_hf_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        ).eval()
        processor = AutoProcessor.from_pretrained(model_hf_path, cache_dir=cache_dir)
    elif "llava-next" in model_name:
        from transformers import LlavaNextForConditionalGeneration, AutoProcessor
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_hf_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            cache_dir=cache_dir
        ).eval()
        processor = AutoProcessor.from_pretrained(model_hf_path, cache_dir=cache_dir)
    elif "qwen" in model_name:
        if "qwen2.5" in model_name:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_hf_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="cpu",
                cache_dir=cache_dir
            ).eval()
        else:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_hf_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map="cpu",
                cache_dir=cache_dir
            ).eval()
        processor = AutoProcessor.from_pretrained(model_hf_path, cache_dir=cache_dir)
    elif "r1-onevision" in model_name:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_hf_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).eval()
        processor = AutoProcessor.from_pretrained(model_hf_path, cache_dir=cache_dir)
    elif "internvl" in model_name:
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(
            model_hf_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            cache_dir=cache_dir
        ).eval()
        processor = AutoTokenizer.from_pretrained(model_hf_path, cache_dir=cache_dir, trust_remote_code=True)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    return model, processor


def run(rank, 
        world_size, 
        model_name,
        cache_dir, 
        val_data_path="./val_data/", 
        dataset_name="oxford_flowers",
        prompt_type="cot"):
    
    model, processor = load_model(model_name, cache_dir, rank)

    model = model.to(torch.device(rank))
    model = model.eval()

    ### get categories name
    with open(os.path.join(val_data_path, f'{dataset_name}.txt'), 'r') as file:
        lines = file.readlines()
    categories = []
    for line in lines:
        categories.append(line.strip())
    # print(len(categories))
    # print(categories)

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
        
        if prompt_type == "cot":
            question = get_user_prompt_cot(dataset_name)
        elif prompt_type == "ao":
            question = get_user_prompt_ao(dataset_name)
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
    
        image_path = os.path.join(DATASET_IMG_PATH[dataset_name], os.path.basename(image_path))

        pil_image = Image.open(image_path).convert("RGB")

        prompt_for_logging = None
        if "llava" in model_name:
            query = f"USER: <image>\n{question}\nASSISTANT:"
            prompt_for_logging = query
            inputs = processor(text=query, images=pil_image, return_tensors="pt")
        elif "internvl" in model_name:
            query = f"<image>\n{question}"
            prompt_for_logging = query
            pixel_values = load_image(image_path, max_num=12).to(model.device, dtype=torch.bfloat16)
            generation_config = dict(max_new_tokens=1024, do_sample=False)
            response = model.chat(processor, pixel_values, query, generation_config)
            inputs = {}
        elif "qwen" in model_name or "r1-" in model_name:
            query = "<image>\n"+question
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path}
                    ] + [{"type": "text", "text": query}],
                }
            ]
            prompt_for_logging = messages
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
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Inference: Generation of the output
        if "internvl" in model_name:
             # The response is already generated by the `chat` method.
             pass
        else:
            generated_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)
            if "llava" in model_name:
                 generated_ids_trimmed = generated_ids
            else: # Qwen
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
                ]

        if "internvl" not in model_name:
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = response[0]
        
        total_count += 1
        is_correct = False
        if prompt_type == "cot":
            try:
                match = re.search(r"<answer>(.*?)</answer>", response)
                answer_content = match.group(1)
                image_cate = image_cate.replace(' ','').replace('_','').lower()
                answer_content = answer_content.replace(' ','').replace('_','').lower()

                if image_cate in answer_content or answer_content in image_cate:
                    right_count += 1
                    is_correct = True
                    logger.info('Local Right Number: ' + str(right_count))

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
            "prompt": prompt_for_logging,
            "response": response,
            "label": categories[image_label],
            "is_correct": is_correct,
        }
        local_conversations.append(conversation_data)
            
    return [error_count, right_count, total_count, local_conversations]


def main(model_name,
         cache_dir,
         dataset_name,
         output_dir,
         prompt_type):
    multiprocess = torch.cuda.device_count() >= 2
    os.makedirs(output_dir, exist_ok=True)
    mp.set_start_method('spawn')
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, 
                                     world_size=world_size,
                                     model_name=model_name,
                                     cache_dir=cache_dir,
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
            output_filename = f"conversations_{model_name}_{dataset_name}_{prompt_type}_acc{accuracy:.1f}.jsonl"
            output_path = os.path.join(output_dir, output_filename)
            if all("index" in conv for conv in all_conversations):
                all_conversations = sorted(all_conversations, key=lambda x: x['index'])
            
            with open(output_path, 'w') as f:
                for item in all_conversations:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Saved conversations to {output_path}")

        logger.info('Model:' + model_name)    
        logger.info('Error number: ' + str(global_count_error))  
        logger.info('Total Right Number: ' + str(global_count_right))
        logger.info(f'Accuracy: {global_count_right / global_total_count:.1%} ({global_count_right}/{global_total_count})')
    else:
        logger.info("Not enough GPUs for multiprocessing, running on single GPU")
        error_count, right_count, total_count, conversations = run(
            0, 1, model_name, cache_dir, dataset_name=dataset_name, prompt_type=prompt_type
        )
        accuracy = (right_count / total_count * 100) if total_count > 0 else 0
        if conversations:
            output_filename = f"conversations_{model_name}_{dataset_name}_{prompt_type}_acc{accuracy:.1f}.jsonl"
            output_path = os.path.join(output_dir, output_filename)
            if all("index" in conv for conv in conversations):
                conversations = sorted(conversations, key=lambda x: x['index'])
            with open(output_path, 'w') as f:
                for item in conversations:
                    f.write(json.dumps(item) + '\n')
            logger.info(f"Saved conversations to {output_path}")
        logger.info('Model:' + model_name)
        logger.info('Error number: ' + str(error_count))
        logger.info('Total Right Number: ' + str(right_count))
        logger.info(f'Accuracy: {right_count / total_count:.1%} ({right_count}/{total_count})')


if __name__ == "__main__":
    args = parse_args()
    main(args.model_name, args.cache_dir, args.dataset_name, args.output_dir, args.prompt_type)
