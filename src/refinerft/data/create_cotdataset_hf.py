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


MAX_COMPLETION_LENGTH = 100
# EXTRA_PROMPT = "The ground truth answer is '{solution}'. Limit your response to 100 words."
EXTRA_PROMPT = "The ground truth answer is '{solution}'. Limit your response to {max_completion_length} words."

def parse_args():
    """
    Parse command line arguments for RSC evaluation.
    """
    parser = argparse.ArgumentParser(description='CoT Dataset Generation Script')
    
    # Input/Output parameters
    parser.add_argument('--dataset_name', type=str, default='laolao77/ViRFT_CLS_flower_4_shot')
    parser.add_argument('--dataset_cache_dir', type=str, default='./share_data')
    parser.add_argument('--output_dir', type=str, default="./src/refinerft/data/")
    parser.add_argument('--push_to_hub', action='store_true', help='Whether to push the merged dataset to Hugging Face Hub')
    parser.add_argument('--new_dataset_name', type=str, default=None, help='Name for the new merged dataset (if push_to_hub is True)')
    parser.add_argument('--merge_only', action='store_true',
                      help='Only merge existing *_cot_dataset_query_result_*.jsonl files with the source HF dataset')
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


def merge_dataset_with_api_responses(args):
    """
    Merge the original HF dataset with API responses to create a new dataset.
    
    Args:
        args: Parsed command line arguments containing dataset configuration
    
    Returns:
        None: Saves the merged dataset to dataset_cache_dir and optionally pushes to HF Hub
    """
    print("Loading original dataset...")
    # Load the original HF dataset
    original_dataset = load_dataset(args.dataset_name, cache_dir=args.dataset_cache_dir)
    
    # Get the dataset name for file naming
    if '/' in args.dataset_name:
        dataset_file_name = args.dataset_name.split('/')[-1]
    else:
        dataset_file_name = args.dataset_name
    
    print("Reading API response files...")
    # Read all API response files
    api_responses = {}
    
    # Find all response files in the output directory
    response_files = glob.glob(os.path.join(args.output_dir, f"{dataset_file_name}_cot_dataset_query_result_*.jsonl"))
    response_files.sort()  # Ensure consistent ordering
    
    if not response_files:
        raise FileNotFoundError(f"No response files found matching pattern: {dataset_file_name}_cot_dataset_result_slice_*.jsonl")
    
    print(f"Found {len(response_files)} response files")
    
    # Read all responses and organize by custom_id
    for response_file in response_files:
        print(f"Reading {response_file}...")
        with open(response_file, 'r') as f:
            for line in f:
                response_data = json.loads(line.strip())
                custom_id = response_data['custom_id']
                # Extract the index from custom_id (format: dataset_name-index)
                idx = int(custom_id.split('-')[-1])
                api_content = response_data['response']['body']['choices'][0]['message']['content']
                api_responses[idx] = api_content
    
    print(f"Loaded {len(api_responses)} API responses")
    
    # Create new dataset by adding API responses as a new column
    def add_api_response(example, idx):
        """Add CoT reasoning as a new column to each example"""
        if idx in api_responses:
            example['reference'] = api_responses[idx]
        else:
            example['reference'] = ""  # Empty string for missing responses
            print(f"Warning: No API response found for index {idx}")
        return example
    
    print("Merging dataset with API responses...")
    # Apply the function to add CoT reasoning column
    merged_dataset = original_dataset.map(
        add_api_response, 
        with_indices=True,
        desc="Adding CoT reference column"
    )
    
    # Save the merged dataset locally
    local_save_path = os.path.join(args.dataset_cache_dir, f"{dataset_file_name}_cot_{MAX_COMPLETION_LENGTH}words")
    print(f"Saving merged dataset to {local_save_path}...")
    merged_dataset.save_to_disk(local_save_path)
    
    print(f"Dataset successfully saved to {local_save_path}")
    print(f"Dataset info: {merged_dataset}")
    
    # Push to Hugging Face Hub if requested
    if args.push_to_hub:
        if args.new_dataset_name is None:
            new_dataset_name = f"{args.dataset_name}_cot"
        else:
            new_dataset_name = args.new_dataset_name
            
        print(f"Pushing dataset to Hugging Face Hub as '{new_dataset_name}'...")
        try:
            merged_dataset.push_to_hub(new_dataset_name)
            print(f"Dataset successfully pushed to Hugging Face Hub: {new_dataset_name}")
        except Exception as e:
            print(f"Error pushing to Hugging Face Hub: {str(e)}")
            print("Please make sure you are logged in to Hugging Face (huggingface-cli login)")
    
    return merged_dataset


if __name__ == "__main__":
    args = parse_args()

    if args.merge_only:
        print("Merge-only mode: using existing API response JSONL files.")
        merge_dataset_with_api_responses(args)
        sys.exit(0)

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
                'text': sample['problem'] + " " + EXTRA_PROMPT.format(solution=sample['solution'], max_completion_length=MAX_COMPLETION_LENGTH),
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
        with open(os.path.join(args.output_dir, f"{save_file_name}_cot_dataset_query_slice_{i}_{MAX_COMPLETION_LENGTH}words.jsonl"), "w") as f:
            for data in slice_data:
                json.dump(data, f)
                f.write('\n')

    #--------------------------------
    # 2. generate cot dataset and get response from gpt
    #--------------------------------
    
    # Submit batch requests
    
    # Find all response files in the output directory
    response_files = glob.glob(os.path.join(args.output_dir, f"{save_file_name}_cot_dataset_query_slice_*.jsonl"))
    response_files.sort()  # Ensure consistent ordering
    client = OpenAI()
    submit_batch(client, response_files,
            args.output_dir, 
            check_interval=args.check_interval,
            start_from_slice_idx=args.start_from_slice_idx)
    
    #--------------------------------
    # 3. merge original dataset with API responses
    #--------------------------------
    print("Starting dataset merging process...")
    
    merged_dataset = merge_dataset_with_api_responses(args)
    
    print("Dataset merging completed successfully!")
    
    
    ##########################
    # split jsonl file if exceed the max token limit for OpenAI API
    ##########################
    # def split_jsonl_file(input_file, output_file1, output_file2):
    #     """
    #     Splits a JSONL file into two separate files.

    #     Args:
    #         input_file (str): Path to the input JSONL file.
    #         output_file1 (str): Path to the first output JSONL file.
    #         output_file2 (str): Path to the second output JSONL file.
    #     """
    #     assert os.path.exists(input_file)
    #     assert not os.path.exists(output_file1)
    #     assert not os.path.exists(output_file2)
        
        
    #     with open(input_file, 'r') as infile:
    #         lines = infile.readlines()
        
    #     # Calculate the split point
    #     split_point = len(lines) // 2
        

    #     # Write the first half to the first output file
    #     with open(output_file1, 'w') as outfile1:
    #         for line in lines[:split_point]:
    #             outfile1.write(line)
        
    #     # Write the second half to the second output file
    #     with open(output_file2, 'w') as outfile2:
    #         for line in lines[split_point:]:
    #             outfile2.write(line)

    # split_jsonl_file("./data/5shot_sft_cot_dataset_query_slice_5.jsonl", 
    #                  "./data/5shot_sft_cot_dataset_query_slice_5_part1.jsonl", 
    #                  "./data/5shot_sft_cot_dataset_query_slice_5_part2.jsonl")
    
