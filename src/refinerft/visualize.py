import os
from typing import List, Dict, Any
import json

def load_jsonl_by_image_path(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads a JSONL file into a dictionary keyed by the image path from the prompt.

    Args:
        filepath: Path to the JSONL file.

    Returns:
        A dictionary mapping the image path to the corresponding data item.
    """
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                image_path = None
                # Extract image path from the prompt structure
                if 'prompt' in item and isinstance(item['prompt'], list) and item['prompt']:
                    content = item['prompt'][0].get('content')
                    if isinstance(content, list):
                        for content_item in content:
                            if isinstance(content_item, dict) and content_item.get('type') == 'image':
                                image_path = content_item.get('image')
                                break
                
                if image_path:
                    data[image_path] = item
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                # Silently ignore lines that are malformed or don't have the expected structure
                pass
    return data

def compare_ours_vs_baseline(ours_path: str, baseline_path: str, baseline_name: str, share_models_dir: str) -> List[Dict[str, Any]]:
    """
    Compares 'ours' model against a single baseline model based on image path.
    Finds samples where 'ours' is correct and the baseline is not.

    Args:
        ours_path: Directory name for 'ours' model.
        baseline_path: Directory name for the baseline model.
        baseline_name: Name of the baseline (e.g., 'cot_sft' or 'grpo').
        share_models_dir: Directory where model results are stored.

    Returns:
        A list of dictionaries with details about inconsistent predictions.
    """
    ours_filepath = os.path.join(share_models_dir, ours_path, "conversations_step200.jsonl")
    baseline_filepath = os.path.join(share_models_dir, baseline_path, "conversations_step200.jsonl")

    if not os.path.exists(ours_filepath):
        print(f"Error: File not found for 'ours': {ours_filepath}")
        return []
    if not os.path.exists(baseline_filepath):
        print(f"Error: File not found for '{baseline_name}': {baseline_filepath}")
        return []

    print("Comparing files:")
    print(f"- OURS:        {os.path.basename(os.path.dirname(ours_filepath))}")
    print(f"- {baseline_name.upper()}:      {os.path.basename(os.path.dirname(baseline_filepath))}")
    
    ours_data = load_jsonl_by_image_path(ours_filepath)
    baseline_data = load_jsonl_by_image_path(baseline_filepath)

    inconsistent_results = []
    
    common_image_paths = set(ours_data.keys()).intersection(set(baseline_data.keys()))

    if not common_image_paths:
        print("Warning: No common image paths found between the two files.")
        return []

    for image_path in sorted(list(common_image_paths)):
        ours_item = ours_data.get(image_path)
        baseline_item = baseline_data.get(image_path)

        if not ours_item or not baseline_item:
            continue

        ours_correct = ours_item.get('is_correct')
        baseline_correct = baseline_item.get('is_correct')

        if ours_correct is True and baseline_correct is False:
            result = {
                'image_path': image_path,
                'label': ours_item.get('label'),
                'ours_correct': ours_correct,
                'ours_response': ours_item.get('response'),
            }
            result[f'{baseline_name}_correct'] = baseline_correct
            result[f'{baseline_name}_response'] = baseline_item.get('response')
            inconsistent_results.append(result)

    return inconsistent_results

def compare_ours_vs_all(ours_path: str, cot_sft_path: str, grpo_path: str, share_models_dir: str) -> List[Dict[str, Any]]:
    """
    Compares 'ours' model against both 'cot_sft' and 'grpo' models.
    Finds samples where 'ours' is correct but BOTH baselines are incorrect.
    """
    ours_filepath = os.path.join(share_models_dir, ours_path, "conversations_step200.jsonl")
    cot_sft_filepath = os.path.join(share_models_dir, cot_sft_path, "conversations_step200.jsonl")
    grpo_filepath = os.path.join(share_models_dir, grpo_path, "conversations_step200.jsonl")

    if not (os.path.exists(ours_filepath) and os.path.exists(cot_sft_filepath) and os.path.exists(grpo_filepath)):
        print(f"Skipping comparison for {ours_path}, {cot_sft_path}, and {grpo_path} due to missing files.")
        return []

    ours_data = load_jsonl_by_image_path(ours_filepath)
    cot_sft_data = load_jsonl_by_image_path(cot_sft_filepath)
    grpo_data = load_jsonl_by_image_path(grpo_filepath)
    
    inconsistent_results = []
    common_image_paths = set(ours_data.keys()).intersection(set(cot_sft_data.keys())).intersection(set(grpo_data.keys()))

    for image_path in common_image_paths:
        ours_item = ours_data[image_path]
        cot_sft_item = cot_sft_data[image_path]
        grpo_item = grpo_data[image_path]

        if ours_item.get('is_correct') is True and cot_sft_item.get('is_correct') is False and grpo_item.get('is_correct') is False:
            inconsistent_results.append({
                'image_path': image_path, 'label': ours_item.get('label'),
                'ours_correct': True, 'ours_response': ours_item.get('response'),
                'cot_sft_correct': False, 'cot_sft_response': cot_sft_item.get('response'),
                'grpo_correct': False, 'grpo_response': grpo_item.get('response')
            })
    return inconsistent_results


cot_sft_paths = ["fgvc_aircraft_4_shot-sft-lora-r64a128-cot_1106_0058", "flowers_4_shot-sft-lora-r64a128-cot_1105_1628", "cars_4_shot-sft-lora-r64a128-cot_1105_0128", "pets_4_shot-sft-lora-r64a128-cot_1105_2002"]
grpo_paths = ["aircrafts-4-shot-grpo-lora-r64a128_1106_0233", "flower-4-shot-grpo-lora-r64a128_1106_0936", "cars-4-shot-grpo-lora-r64a128_1106_0455", "pets-4-shot-grpo-lora-r64a128_1106_0706"]
ours_paths = ["aircrafts-4-shot-mrpo-lora-r64a128-RMQwen2VL_7B_1106_2058", "flower-4-shot-mrpo-lora-r64a128-RMQwen2VL_7B_1106_1540", "cars-4-shot-mrpo-lora-r64a128-RMQwen2VL_7B_1108_0010", "pets-4-shot-mrpo-lora-r64a128-RMQwen2VL_7B_1108_1714"]

# <<< 在这里选择要比较的基线模型 >>>
# 可选项: 'cot_sft' 或 'grpo'
baseline_to_compare = 'grpo' 
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

if baseline_to_compare == 'cot_sft':
    baseline_paths = cot_sft_paths
elif baseline_to_compare == 'grpo':
    baseline_paths = grpo_paths
else:
    raise ValueError("Invalid baseline_to_compare value. Choose 'cot_sft' or 'grpo'.")

datasets = ['aircrafts', 'flowers', 'cars', 'pets']
share_models_dir = "/research/cvlshare/cvl-zhujie4/Refine-RFT/share_models"

# ==============================================================================
# 1. OURS vs ALL (COT_SFT 和 GRPO)
# ==============================================================================
print("="*20 + " Running OURS vs ALL Comparison " + "="*20)
all_inconsistent_results_vs_all = {}
for i, dataset in enumerate(datasets):
    print(f"\n--- Processing dataset: {dataset.upper()} ---")
    inconsistent_data = compare_ours_vs_all(
        ours_path=ours_paths[i],
        cot_sft_path=cot_sft_paths[i],
        grpo_path=grpo_paths[i],
        share_models_dir=share_models_dir
    )
    all_inconsistent_results_vs_all[dataset] = inconsistent_data
    print(f"Found {len(inconsistent_data)} samples where 'ours' is correct but BOTH baselines are not for '{dataset}'.")



print("="*20 + " Pets" + "="*20)
for i in range(len(all_inconsistent_results_vs_all['pets'])):
    item = all_inconsistent_results_vs_all['cars'][i]
    for key, value in item.items():
        print(f"{key}: {value}")
    print("=" * 40)