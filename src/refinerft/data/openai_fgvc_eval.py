import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from typing import List, Dict, Any
from collections import defaultdict
import argparse
import glob
from tqdm import tqdm
from datasets import Dataset, load_dataset
from openai import OpenAI

from utils.openai_utils import submit_batch, estimate_text_tokens, estimate_image_tokens
from utils.img_utils import encode_image, get_image_dimensions_from_base64, pil_image_to_base64
from utils.prompt_utils import get_user_prompt_cot, get_user_prompt_ao

dataset_name_mapping = {
    "laolao77/ViRFT_CLS_fgvc_aircraft_4_shot": "fgvc_aircraft",
    "laolao77/ViRFT_CLS_flower_4_shot": "oxford_flowers",
    "laolao77/ViRFT_CLS_car196_4shot": "stanford_cars",
    "laolao77/ViRFT_CLS_pets37_4shot": "pets",
}

def parse_args():
    """
    Parse command line arguments for RSC evaluation.
    """
    parser = argparse.ArgumentParser(description='CoT Dataset Generation Script')
    
    # Input/Output parameters
    parser.add_argument('--dataset_name', type=str, default='laolao77/ViRFT_CLS_flower_4_shot')
    parser.add_argument('--dataset_cache_dir', type=str, default='./share_data')
    parser.add_argument('--output_dir', type=str, default="./src/refinerft/data/")
    parser.add_argument('--prompt_type', type=str, default="cot", choices=["cot", "ao"], help='Prompt type')
    # API parameters
    parser.add_argument('--url', type=str, default="/v1/chat/completions",
                      help='API endpoint URL')
    parser.add_argument('--model', type=str, default="gpt-4o",
                      help='Model name to use in OpenAI API')
    parser.add_argument('--max_tokens', type=int, default=512,
                      help='Maximum number of tokens in response')

    # Batch processing parameters
    parser.add_argument('--max_sample', type=int, default=None,
                      help='Maximum number of samples to process')
    parser.add_argument('--num_slice', type=int, default=1,
                      help='Number of slices to split the data into')
    parser.add_argument('--check_interval', type=int, default=60,
                      help='Time interval in seconds between batch status checks')
    parser.add_argument('--start_from_slice_idx', type=int, default=0,
                      help='Start processing from this slice index (0-based)')
    args = parser.parse_args()
    return args

def generate_query(args):
    dataset_name = dataset_name_mapping[args.dataset_name]
    if args.prompt_type == "cot":
        prompt = get_user_prompt_cot(dataset_name)
    elif args.prompt_type == "ao":
        prompt = get_user_prompt_ao(dataset_name)
    else:
        raise ValueError(f"Invalid prompt type: {args.prompt_type}")
    return prompt

if __name__ == "__main__":
    args = parse_args()

    #--------------------------------
    # 1. generate query slices for prompt gpt
    #--------------------------------
    dataset = load_dataset(args.dataset_name,
                           cache_dir=args.dataset_cache_dir)
    # generate query to prompt gpt
    requests = []
    total_image_tokens = 0
    total_query_tokens = 0
    for idx, sample in enumerate(tqdm(dataset['train'], total=len(dataset['train']), desc="Generating query")):
        base64_image = pil_image_to_base64(sample['image'])
        msg = {
            'role': 'user',
            'content': [{
                'type': 'text',
                'text': generate_query(args),
            }, {
                'type': 'image_url',
                'image_url': {
                    'url': f"data:image/jpeg;base64,{base64_image}",
                },
            }],
            }    
        image_width, image_height = get_image_dimensions_from_base64(base64_image)
        image_tokens = estimate_image_tokens(image_width, image_height)
        query_tokens = estimate_text_tokens(msg['content'][0]['text'])
        total_image_tokens += image_tokens
        total_query_tokens += query_tokens
        request = {
            "custom_id": f"{args.dataset_name}-{idx}",
            "method": "POST",
            "url": args.url,
            "body": {
                "model": args.model,
                "messages": [msg],
                "max_tokens": args.max_tokens,
                "seed": 42, # NOTE: for reproducibility
                "temperature": 0.0, # NOTE: for reproducibility
            }
        }        
        requests.append(request)
        
    print(f"Total image tokens: {total_image_tokens / 1_000_000:.3f}M. Total query tokens: {total_query_tokens / 1_000_000:.3f}M")
    print(f"Total tokens: {(total_image_tokens + total_query_tokens) / 1_000_000:.3f}M")
    
    # Divide requests into slices and save each slice to a JSONL file
    # NOTE: each file should be < 200MB for tier 3 API
    slice_size, remainder = divmod(len(requests), args.num_slice)
    start_idx = 0
    if '/' in args.dataset_name:
        save_file_name = args.dataset_name.split('/')[-1]
    for i in range(args.num_slice):
        end_idx = start_idx + slice_size + (1 if i < remainder else 0)
        slice_data = requests[start_idx:end_idx]
        start_idx = end_idx
        with open(os.path.join(args.output_dir, f"{save_file_name}_{args.prompt_type}_{args.model}_eval_slice_{i}.jsonl"), "w") as f:
            for data in slice_data:
                json.dump(data, f)
                f.write('\n')

    #--------------------------------
    # 2. generate cot dataset and get response from gpt
    #--------------------------------
    
    # Submit batch requests
    
    # Find all response files in the output directory
    response_files = glob.glob(os.path.join(args.output_dir, f"{save_file_name}_{args.prompt_type}_{args.model}_eval_slice_*.jsonl"))
    response_files.sort()  # Ensure consistent ordering
    client = OpenAI()
    submit_batch(client, response_files,
            args.output_dir, 
            check_interval=args.check_interval,
            start_from_slice_idx=args.start_from_slice_idx)