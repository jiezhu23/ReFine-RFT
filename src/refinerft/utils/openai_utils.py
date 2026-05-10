import json
import sys
import re
from typing import List, Dict, Any, Optional
import os
from math import ceil
from openai import OpenAI
# from utils.img_utils import pil_image_to_base64, encode_image, get_image_dimensions_from_base64
# from data.datasets_utils import PROMPT_QUERY_DICT, SYSTEM_PROMPT, PROMPT_END, SYSTEM_PROMPT_SHORT, DATASET_NAME_MAPPING
import yaml
import logging
import time
from datetime import datetime


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def estimate_text_tokens(text: str, method: str = "max") -> int:
    """
    Estimate the number of tokens in a text using different methods.
    
    Args:
        text (str): The input text to estimate tokens for
        method (str): The method to use for estimation. Can be:
            - "average": Average of word and char based estimates
            - "words": Word count divided by 0.75
            - "chars": Character count divided by 4
            - "max": Maximum of word and char based estimates
            - "min": Minimum of word and char based estimates
            Defaults to "max"
            
    Returns:
        int: Estimated number of tokens
    """
    # Calculate word and character based estimates
    word_count = len(text.split())
    char_count = len(text)
    tokens_count_word_est = word_count / 0.75
    tokens_count_char_est = char_count / 4.0
    
    # Calculate final estimate based on method
    if method == "average":
        output = (tokens_count_word_est + tokens_count_char_est) / 2
    elif method == "words":
        output = tokens_count_word_est
    elif method == "chars":
        output = tokens_count_char_est
    elif method == "max":
        output = max(tokens_count_word_est, tokens_count_char_est)
    elif method == "min":
        output = min(tokens_count_word_est, tokens_count_char_est)
    else:
        raise ValueError("Invalid method. Use 'average', 'words', 'chars', 'max', or 'min'.")
    
    return int(output)

def estimate_image_tokens(width: int, height: int):
    if width > 2048 or height > 2048:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            width, height = 2048, int(2048 / aspect_ratio)
        else:
            width, height = int(2048 * aspect_ratio), 2048
            
    if width >= height and height > 768:
        width, height = int((768 / height) * width), 768
    elif height > width and width > 768:
        width, height = 768, int((768 / width) * height)

    tiles_width = ceil(width / 512)
    tiles_height = ceil(height / 512)
    total_tokens = 85 + 170 * (tiles_width * tiles_height)
    
    return total_tokens

def check_batch_status(client: OpenAI, batch_id: str, check_interval: int = 60) -> bool:
    """
    Check the status of a batch job and wait until it's completed.
    
    Args:
        client (OpenAI): The OpenAI client instance.
        batch_id (str): The ID of the batch job to check
        check_interval (int): Time interval in seconds between status checks
        
    Returns:
        bool: True if batch completed successfully, False if failed
    """
    while True:
        batch_status = client.batches.retrieve(batch_id)
        logger.info(f"Batch {batch_id} status: {batch_status.status}")
        
        if batch_status.status == "completed":
            return True
        elif batch_status.status in ["failed", "expired"]:
            logger.error(f"Batch {batch_id} error: {batch_status.status}")
            return False
        
        time.sleep(check_interval)

def submit_file(client: OpenAI, file_list: List[str]) -> List[str]:
    batch_input_files = []
    for json_file in file_list:
        # create batch input file
        with open(json_file, "rb") as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )
        logger.info(batch_input_file)
        batch_input_files.append(batch_input_file)
    return batch_input_files

def download_results(client: OpenAI, batch_id: str, output_dir: str, file_name: str = None):
    """
    Download results from a completed batch job.
    
    Args:
        client (OpenAI): The OpenAI client instance.
        batch_id (str): The ID of the batch job
        output_dir (str): Directory to save the results
    """
    batch_status = client.batches.retrieve(batch_id)
    logger.info(f"Checking batch {batch_id} status: {batch_status.status}")
    
    if batch_status.status != "completed":
        logger.error(f"Batch {batch_id} is not completed. Current status: {batch_status.status}")
        return None
    file_name = file_name if file_name is not None else batch_id
    file_response = client.files.content(batch_status.output_file_id)
    output_file = os.path.join(output_dir, f"{file_name}")
    with open(output_file, "w") as f:
        f.write(file_response.content.decode('utf-8'))
    logger.info(f"Downloaded results to {output_file}")
    return output_file

def merge_json_files(json_path_list: List[str], output_path: str) -> None:
    """
    Merge multiple JSON files into a single JSON file in order.
    
    Args:
        json_path_list (List[str]): List of paths to input JSON files
        output_path (str): Path to save the merged JSON file
    """
    merged_data = []
    
    for json_path in json_path_list:
        logger.info(f"Merging results from {json_path}")
        with open(json_path, 'r') as f:
            for line in f:
                merged_data.append(json.loads(line))
    
    with open(output_path, 'w') as f:
        for item in merged_data:
            f.write(json.dumps(item) + '\n')
    
    logger.info(f"Successfully merged {len(merged_data)} records into {output_path}")

def submit_batch(client: OpenAI,
                 file_list: List[str], 
                 output_dir: str, 
                 check_interval: int = 60,
                 start_from_slice_idx: int = 0,
                 max_retries: int = 3,
                 retry_delay: int = 600):
    """
    Submit batch requests to OpenAI with status checking and retry functionality.
    If a batch fails (check_batch_status returns False), it will retry after retry_delay seconds for max_retries times.
    
    Args:
        client (OpenAI): The OpenAI client instance.
        file_list (List[str]): List of JSONL files to submit
        output_dir (str): Directory to save the results
        check_interval (int): Time interval in seconds between status checks
        start_from_slice_idx (int): Start processing from this slice index
        max_retries (int): Maximum number of retry attempts for failed batches
        retry_delay (int): Delay in seconds before retrying a failed batch
    """
    
    for json_file in file_list[start_from_slice_idx:]:
        retry_count = 0
        while retry_count <= max_retries:
            batch_input_file = submit_file(client, [json_file])[0]
            logger.info(f"Submitting batch for file: {json_file} (Attempt {retry_count + 1}/{max_retries + 1})")
            batch_input_file_id = batch_input_file.id
            batch = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            logger.info(f"Created batch with ID: {batch.id}")
            
            # Wait for current batch to complete
            if check_batch_status(client, batch.id, check_interval):
                logger.info(f"Batch {batch.id} completed successfully")
                # download results and rename the file
                download_results(client, batch.id, output_dir, os.path.basename(json_file).replace('slice', 'result'))
                break  # Success, move to next file
            else:
                if retry_count < max_retries:
                    logger.warning(f"Batch {batch.id} failed. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_count += 1
                else:
                    logger.error(f"Batch {batch.id} failed after {max_retries} retries. Exiting...")
                    exit()
