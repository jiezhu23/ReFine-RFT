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
from peft import AutoPeftModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os

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
import multiprocessing as mp
from argparse import ArgumentParser
from multiprocessing import Pool

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

def xyxy2xywh(bbox):
    """
    change bbox to coco format
    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]

import matplotlib.patches as patches
def plot_bbox(image_path, bbox_list):
    image = Image.open(image_path)
    image_width, image_height = image.size

    fig, ax = plt.subplots(1)

    ax.imshow(image)
    
    for bbox in bbox_list:
        position = bbox["Position"]
        x1, y1, x2, y2 = position
    
        x1 /= 1000.0
        y1 /= 1000.0
        x2 /= 1000.0
        y2 /= 1000.0
    
        x1 = x1 * image_width
        y1 = y1 * image_height
        x2 = x2 * image_width
        y2 = y2 * image_height

        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def trans_bbox(image_height, image_width, position):
        x1, y1, x2, y2 = position

        x1 /= 1000.0
        y1 /= 1000.0
        x2 /= 1000.0
        y2 /= 1000.0
    
        x1 = x1 * image_width
        y1 = y1 * image_height
        x2 = x2 * image_width
        y2 = y2 * image_height

        return [
                x1,
                y1,
                x2 - x1,
                y2 - y1,
            ]
    

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

### model path and model base
# model_path = "./share_models/Qwen2-VL-2B-Instruct/"
# ori_processor_path = "./share_models/Qwen2-VL-2B-Instruct/"

model_path = './share_models/Qwen2-VL-2B-Instruct_GRPO_model/checkpoint-200'
ori_processor_path = "./share_models/Qwen2-VL-2B-Instruct/"

def run(rank, world_size):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    processor = AutoProcessor.from_pretrained(ori_processor_path) 

    model = model.to(torch.device(rank))
    model = model.eval()

    with open('./data/lvis/annotations/lvis_v1_val.json', 'r') as json_file:
        instances_val = json.load(json_file)
    print(type(instances_val))
    
    print(instances_val.keys())
    print("############# Images #############")
    print(len(instances_val['images']))
    print(instances_val['images'][0])
    print("############# Annotations #############")
    print(len(instances_val['annotations']))
    # print(instances_val['annotations'][0])
    print("############# Categories #############")
    print(len(instances_val['categories']))
    print(instances_val['categories'][1000])
    category_ids_2_categoty = {item['id']:item['name'] for item in instances_val['categories']}
    category_2_categoty_ids = {item['name']:item['id'] for item in instances_val['categories']}
    print(category_ids_2_categoty[1], category_ids_2_categoty[90])
    print(category_2_categoty_ids['person'], category_2_categoty_ids['toothbrush'])


    rank = rank
    world_size = world_size
    import math
    instances_val['images'] = instances_val['images']
    split_length = math.ceil(len(instances_val['images'])/world_size)
    logger.info("Split Chunk Length:" + str(split_length))
    split_images = instances_val['images'][int(rank*split_length) : int((rank+1)*split_length)]
    logger.info(len(split_images))

    ### Traverse all images in val.
    error_count = 0
    bbox_count = 0
    pred_results = []
    if '2B' in model_path:
        exist_cat = json.load(open("./exist_map_lvis_Qwen2_vl_2B_baseline.json", 'r'))
    elif '7B' in model_path:
        exist_cat = json.load(open("./exist_map_lvis_Qwen2_vl_7B_baseline.json", 'r'))
        
    for image in tqdm(split_images): 
        image_id = image['id']
        image_height = image['height']
        image_width = image['width']
        image_coco_url = image['coco_url']
        parts = image_coco_url.split("/")
        image_path = './data/lvis/' + "/".join(parts[-2:])   ### Modify according to your own image path.

        ### Traverse all class in image.
        for cate in exist_cat[str(image_id)]:
            category = cate

            ### few-shot experiment: 6 classes
            # selected_cate = ['horse_buggy', 'die', 'kitchen_table', 'omelet', 'papaya', 'stepladder']
            # if category not in selected_cate:
            #       continue
            
            ### open vocabulary experiment:  13 classes
            selected_cate = ['casserole', 'die', 'egg_roll', 'futon', 'garbage', 'handsaw', 'hippopotamus', 'kitchen_table', 'mallet', 'omelet', 'shot_glass', 'stepladder', 'sugar_bowl']
            if category not in selected_cate:
                continue

            category_id = category_2_categoty_ids[category]

            question = (
                f"Detect all objects belonging to the category '{category}' in the image, and provide the bounding boxes (between 0 and 1000, integer) and confidence (between 0 and 1, with two decimal places).\n"
                f"If no object belonging to the category '{category}' in the image, return 'No Objects'.\n"
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                "The output answer format should be as follows:\n"
                "<think> ... </think> <answer>[{'Position': [x1, y1, x2, y2], 'Confidence': number}, ...]</answer>\n"
                "Please strictly follow the format."
            )

            image_path = image_path
            query = '<image>\n' + question
            # logger.info(RED+query+RESET)
            
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
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = response[0]
            logger.info(response)
            # Fix possible formatting errors in the response.
            response = response.replace("[[",'[')
            response = response.replace("]]",']')
            response = response.replace("\n",'')
            response = response.replace(", ...",'')
            # logger.info("\033[92m" + response + "\033[0m")

            # extract answer
            content_match = re.search(r'<answer>(.*?)</answer>', response)
            response = content_match.group(1).strip() if content_match else response.strip()
            response = '<answer>'+response+'</answer>'

            # extract bbox
            try:
                bbox_list = extract_bbox(response)
                if bbox_list==None or type(bbox_list[0])==str:
                    pass
                else:
                    bbox_list = remove_duplicates(bbox_list)
                    for bbox in bbox_list:
                        temp_bbox = trans_bbox(image_height, image_width, bbox['Position'])
                        temp_confidence = bbox['Confidence']
                        new_pred_dict = {
                            'image_id': image_id,
                            'category_id': category_id,
                            'bbox': temp_bbox,
                            'score': temp_confidence
                        }
                        pred_results.append(new_pred_dict)
                        bbox_count += 1
                        logger.info('bbox_count: '+str(bbox_count))
            except Exception as e:
                error_count+=1  
    return [error_count, pred_results]

def main():
    multiprocess = torch.cuda.device_count() >= 2
    mp.set_start_method('spawn')
    if multiprocess:
        logger.info('started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus
        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size)
            result_lists = pool.map(func, range(world_size))

        global_count_error = 0
        global_results = []
        for i in range(world_size):
            global_count_error += int(result_lists[i][0])
            global_results = global_results + result_lists[i][1]
            
        logger.info('Error number: ' + str(global_count_error))  
        ### save path
        with open('prediction_results.json', 'w') as json_file:
            json.dump(global_results, json_file)
        logger.info("Done")
        logger.info('finished running')
    else:
        logger.info("Not enough GPUs")

if __name__ == "__main__":
    main()
