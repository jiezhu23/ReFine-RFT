# Define the reward function
import time
import requests
import inspect
import asyncio
import re
import os
from math_verify import parse, verify
from datetime import datetime

from utils.eval_utils import parse_generated_prediction, is_match
from utils.img_utils import pil_image_to_base64, resize_image
from utils.api_usage import async_process_queries
from utils.prompt_utils import get_reasoning_reward_prompt
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

API_URL = "http://127.0.0.1:8001/v1"

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def attribute_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<attribute>.*?</attribute>\s*<answer>.*?</answer>$"
    rewards = []

    for content in completions:
        try:
            text = content[0]['content']
            match = re.search(pattern, text, flags=re.DOTALL)
            rewards.append(1 if match else 0)
        except (IndexError, KeyError, TypeError):
            rewards.append(0)

    return rewards

def think_length_reward(completions, min_len=10, max_len=20, margin=10, **kwargs):
    """
    Reward function that encourages think content length to be within a specific range [min_len, max_len].
    - Reward is 1.0 if length is within [min_len, max_len].
    - Reward decays linearly from 1.0 to 0.0 in the ranges [min_len - margin, min_len) and (max_len, max_len + margin].
    - Reward is 0.0 otherwise.
    """
    rewards = []
    for content in completions:
        try:
            text = content[0]['content']
            match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
            if match:
                think_content = match.group(1).strip()
                L = len(think_content)
                
                if min_len <= L <= max_len:
                    reward = 1.0
                # if min_len <= L <= max_len:
                #     reward = 1.0
                # elif min_len - margin < L < min_len:
                #     reward = (L - (min_len - margin)) / margin
                # elif max_len < L <= max_len + margin:
                #     reward = ((max_len + margin) - L) / margin
                else:
                    reward = 0.0
                
                rewards.append(reward)
            else:
                rewards.append(0.0)
        except (IndexError, KeyError, TypeError):
            rewards.append(0.0)
    return rewards


def think_punishment_reward(completions, max_len=10, **kwargs):
    """
    Reward function that encourages think content length to be within a specific range [min_len, max_len].
    - Reward is 1.0 if length is longer than min_len.

    """
    rewards = []
    for content in completions:
        try:
            text = content[0]['content']
            match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL)
            if match:
                think_content = match.group(1).strip()
                L = len(think_content)
                
                if L < max_len:
                    reward = 1.0
                else:
                    reward = 0.0
                
                rewards.append(reward)
            else:
                rewards.append(0.0)
        except (IndexError, KeyError, TypeError):
            rewards.append(0.0)
    return rewards


def accuracy_reward(completions, solution, prompt_type, **kwargs):
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

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            if prompt_type == "cot":
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                    
                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()
                    if "vqa-rad" in kwargs.get('eval_dataset_name') or "RobustAD" in kwargs.get('eval_dataset_name'):
                        ground_truth = ground_truth.lower()
                        student_answer = student_answer.lower()
                    else:
                        ground_truth = ground_truth.replace(' ','').replace('_','').lower()
                        student_answer = student_answer.replace(' ','').replace('_','').lower()

                    # Compare the extracted answers
                    if ground_truth in student_answer or student_answer in ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            elif prompt_type == "ao":
                try:
                    # Extract answer from content if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                    student_answer = content.strip()
                    ground_truth = ground_truth.replace(' ','').replace('_','').lower()
                    student_answer = student_answer.replace(' ','').replace('_','').lower()
                    # Compare the extracted answers
                    if ground_truth in student_answer or student_answer in ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            else:
                raise ValueError(f"Invalid prompt type: {prompt_type}")
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        # if os.getenv("DEBUG_MODE") == "true":
        #     log_path = os.getenv("LOG_PATH")
        #     # local_rank = int(os.getenv("LOCAL_RANK", 0))
        #     with open(log_path, "a") as f:
        #         f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
        #         f.write(f"content: {content}\n")
        #         f.write(f"sol: {sol}\n")
    return rewards


def embedding_similarity_reward(completions, solution, **kwargs):
    embedding_model = kwargs.get('embedding_model')
    accelerator = kwargs.get('accelerator')

    if embedding_model is None:
        if accelerator and accelerator.is_main_process:
            print("Warning: Embedding model not provided for embedding_similarity_reward. Returning 0 rewards.")
        return [0.0] * len(completions)

    contents = [completion[0]["content"] for completion in completions]
    
    pred_answers = []
    for content in contents:
        content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        student_answer = content_match.group(1).strip() if content_match else content.strip()
        pred_answers.append(student_answer)
    
    gt_answers = []
    for sol in solution:
        sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        gt_answers.append(ground_truth)
    
    with torch.no_grad():
        pred_embeddings = embedding_model.encode(pred_answers, convert_to_tensor=True)
        gt_embeddings = embedding_model.encode(gt_answers, convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarities = F.cosine_similarity(pred_embeddings, gt_embeddings, dim=1)
    
    return similarities.tolist()

async def accuracy_reward_openai(completions, solution, **kwargs):
    """
    Reward function that checks if the completion is correct using either symbolic verification or exact string matching.
    Call api to get the accuracy reward from teacher model.
    NOTE: We do not parse the <think></think> tags in the completion.
    """
    images = kwargs['images']
    num_generations = kwargs['num_generations']
    accelerator = kwargs['accelerator']
    max_score = 10
    
    reasoning_reward_prompt_template = r'''
        You are a scoring assistant. Based on the similarity between the "Predicted Answer" and the "Correct Answer", provide a score from 0 to {max_score}. A score of {max_score} means a perfect match, and 0 means a complete mismatch. You must output only the numerical score.

        ---
        [Example 1]
        Predicted Answer: "2007 Dodge Dakota Club Cab"
        Correct Answer: "2007 Dodge Dakota Club Cab"
        Score: 10
        ---
        [Example 2]
        Predicted Answer: "Boeing 707"
        Correct Answer: "707-320"
        Score: 6
        ---
        [Example 3]
        Predicted Answer: "Nasturtium"
        Correct Answer: "watercress"
        Score: 0
        ---
        [Your Task]
        Predicted Answer: "{pred}"
        Correct Answer: "{gt}"
        Score:
    '''.strip()

    assert len(images) * num_generations == len(completions)
    
    # Prepare messages for API call
    answer_contents = []
    for completion in completions:
        match = re.search(r'<answer>(.*?)</answer>', completion[0]['content'])
        if match:
            answer_contents.append(match.group(1).strip())    # remove the <answer> tags
        else:
            answer_contents.append(completion[0]['content'])
    gts = [re.search(r'<answer>(.*?)</answer>', solution[idx]).group(1).strip() for idx in range(len(completions))]
    queries = [reasoning_reward_prompt_template.format(pred=answer_content, gt=gts[idx], max_score=str(max_score)) for idx, answer_content in enumerate(answer_contents)]

    # images = [resize_image(images[idx // num_generations], max_size=448, min_size=224) for idx in range(len(completions))]

    try:
        # Only evaluate the accuracy of the answer, no need the image
        all_responses = await async_process_queries(API_URL, queries, None)
    except Exception as e:
        print(f"Error in API call: {str(e)}, return 0 reward for reasoning.")
        all_responses = [0] * len(queries)
    # Process responses and compute rewards
    rewards = []
    for response in all_responses:
        try:
            score_pattern = r'(?i)(\d+(?:\.\d+)?)'
            match = re.search(score_pattern, response)
            if match:
                score = float(match.group(1))
                score = max(0, min(max_score, score))
                rewards.append(score / max_score)
            else:
                print(f"Warning: Could not find score in response: {response}...")
                rewards.append(0)
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            rewards.append(0)
    print(f'Reasoning rewards: {rewards}') if accelerator.is_main_process else None
    return rewards

async def reasoning_reward_openai(completions, **kwargs):
    """
    Reward function that checks if the reasoning part in the completion is reasonable and accurate.
    Call api to get the reasoning reward from teacher model.
    NOTE: We do not parse the <think></think> tags in the completion.
    """
    images = kwargs['images']
    num_generations = kwargs['num_generations']
    accelerator = kwargs['accelerator']
    max_score = 100
    
    reasoning_reward_prompt_template = r'''
    You are tasked with evaluating the reasoning usefulness and accuracy of visual description in identifying an object from an image. 
    Based on the image and the reasoning content between the <think> and <attribute> tags, focus only on the usefulness and accuracy of visual description.
    Do not give points for simply stating a answer in <think>.
    Output only a score between 0 and {max_score} (number only, no additional text):
    - {max_score} means the reasoning contains fine-grained, correct, and helpful visual details that could support identification of the object in the image.
    - ≥ 0.5 × {max_score}: Some correct elements but notable gaps, vagueness, or one incorrect feature.
    - ≥ 0.3 × {max_score}: Barely any specific evidence; mostly generic yet not directly wrong.  
    - 0: No analysis, or reasoning or attributes are largely incorrect, irrelevant, or contradictory.
    
    Here is the reasoning from the model:
    '''.strip()
    
    reasoning_reward_prompt_template_v2 = r'''
    You are tasked with evaluating the reasoning usefulness and accuracy of visual description in identifying an object from an image. 
    Based on the image and the reasoning content between the <think> and <attribute> tags, focus only on the usefulness and accuracy of visual description.
    Do not give points for simply stating a answer in <think>.
    Output only a score between 0 and {max_score} (number only, no additional text):
    - {max_score} means the reasoning contains fine-grained, correct, and helpful visual details that could support identification of the object in the image.
    - ≥ 0.5 × {max_score}: Some correct elements but notable gaps, vagueness, or one incorrect feature.
    - ≥ 0.3 × {max_score}: Barely any specific evidence; mostly generic yet not directly wrong.  
    - 0: No analysis, or reasoning or attributes are largely incorrect, irrelevant, or contradictory.
    
    Here are examples of how to score:

    **Example 1:**
    <think>This image shows a large commercial passenger plane with the Air Berlin logo.</think>
    Score: 0.2
    Reasoning for Score: This is a high-level summary, not a reasoning process. It states the conclusion without providing the specific visual features (e.g., text style, logo shape, color pattern) that led to it.

    **Example 2:**
    <think> I see a large aircraft, consistent with a commercial passenger jet. The fuselage is primarily white, with a distinct red section covering the top and the tail fin. The text "airberlin.com" is visible on the side in a specific font. The tail logo is a red circle/oval.
    Based on these specific features (red top, tail logo, and text), this is a plane from Air Berlin.</think>
    Score: 1.0
    Reasoning for Score: The reasoning provides a step-by-step analysis of specific, fine-grained visual features (color, text, logo) and correctly links them to the final identification.
    
    **Example 3:**
    <think>...</think>
    Score: 0.0
    Reasoning for Score: The reasoning contains nothing.
    
    Here is the reasoning from the model:
    '''.strip()
    
    reasoning_reward_prompt_template_v3 = r'''
    You are a strict grader of visual–reasoning quality. 
    Your task: give a single integer score from 0 to {max_score} that reflects **how informative, specific, and accurate the content inside <think> and <attribute> is for identifying the depicted object** (assume the class in <answer> itself is correct).
    Based on the image and the reasoning content between the <think> and <attribute> tags, focus only on the usefulness and accuracy of visual description.
    Do not give points for simply stating a answer in <think>.
    Output only a score between 0 and {max_score} (number only, no additional text):
    - {max_score}: All four criteria fully met.
    - ≥ 0.7 × {max_score}: Minor omissions or one small mismatch.  
    - ≥ 0.5 × {max_score}: Some correct elements but notable gaps, vagueness, or one incorrect feature.
    - ≥ 0.3 × {max_score}: Barely any specific evidence; mostly generic yet not directly wrong.  
    - 0: No analysis, or reasoning or attributes are largely incorrect, irrelevant, or contradictory.
    Response from the model:
    '''.strip()

    assert len(images) * num_generations == len(completions)
    # Prepare messages for API call
    # queries = [reasoning_reward_prompt_template_v2.format(max_score=str(max_score)) + completion[0]['content'] for completion in completions]

    # extract <think> content
    think_contents = []
    for completion in completions:
        match = re.search(r'<think>(.*?)</think>', completion[0]['content'])
        if match:
            think_contents.append(f"<think>{match.group(1)}</think>")
        else:
            think_contents.append(completion[0]['content'])
    queries = [reasoning_reward_prompt_template_v2.format(max_score=str(max_score)) + think_content for think_content in think_contents]
    images = [images[idx // num_generations] for idx in range(len(completions))]

    try:
        all_responses = await async_process_queries(API_URL, queries, images)
    except Exception as e:
        print(f"Error in API call: {str(e)}, return 0 reward for reasoning.")
        all_responses = [0] * len(queries)
    # Process responses and compute rewards
    rewards = []
    for response in all_responses:
        try:
            score_pattern = r'(?i)(\d+(?:\.\d+)?)'
            match = re.search(score_pattern, response)
            if match:
                score = float(match.group(1))
                score = max(0, min(max_score, score))
                rewards.append(score / max_score)
            else:
                print(f"Warning: Could not find score in response: {response}...")
                rewards.append(0)
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            rewards.append(0)
    print(f'Reasoning rewards: {rewards}') if accelerator.is_main_process else None
    return rewards

#########################################################
# Old Reward Functions
#########################################################

def cls_reward(completions, **kwargs):
    """
    Reward function that checks if the completion has a specific format.
    1. The completion should be in <answer>...</answer> format.
    2. The answer should be start with one of the choices first.
    3. The answer should be end with the choice's category name.
    """
    solutions = kwargs['solution']
    # get the answer from the completion
    predictions = [content[0]['content'] for content in completions]
    # parse the answer
    parsed_answers = parse_generated_prediction(predictions)
    # reward the answer if it is correct
    corrects = [is_match(p, l, use_choice=True, use_label=True, use_fuzzy=True) 
                for p, l in zip(parsed_answers, solutions)]
    return [1 if correct else 0 for correct in corrects]

def cls_reward_ao(completions, **kwargs):
    """
    Reward function that checks if the completion has a specific format.
    1. The completion should be in <answer>...</answer> format.
    2. The answer should be start with one of the choices first.
    3. The answer should be end with the choice's category name.
    """
    solutions = kwargs['solution']
    # get the answer from the completion
    predictions = [content[0]['content'] for content in completions]
    # reward the answer if it is correct
    corrects = [is_match(p, l, use_choice=True, use_label=False, use_fuzzy=True) 
                for p, l in zip(predictions, solutions)]
    return [1 if correct else 0 for correct in corrects]

def soft_overlong_punishment(completion_length, L_max=50, L_cache=15, **kwargs):
    rewards = []
    for cl in completion_length:
        if cl <= L_max - L_cache:
            rewards.append(0)
        elif L_max - L_cache < cl <= L_max:
            rewards.append((L_max - L_cache - cl) / L_cache)
        else:
            rewards.append(-1)
    return rewards


def reasoning_reward(completions, images, num_generations, accelerator, scale_factor=1, max_score=100, **kwargs):
    """
    Reward function that checks if the reasoning part in the completion is reasonable and accurate.
    Call api to get the reasoning reward from teacher model.
    NOTE: We do not parse the <think></think> tags in the completion.
    """
    # start_time = time.time()
    # max_score = 100
    
    messages = []
    # reasoning_reward_prompt_template = r'''
    # You are tasked with evaluating the reasoning provided by a model in identifying an object from an image. 
    # Based on the image and the reasoning content between the <think> tags, assess whether the reasoning is factually correct, relevant, and helpful for identifying the object. 
    # Output only a score between 0 and {max_score} (number only, no additional text):
    # - {max_score} means the reasoning is completely accurate, relevant, and helpful for classification.
    # - 0 means the reasoning is completely inaccurate or irrelevant.
    # - Values between 0 and {max_score} reflect partial correctness or partial relevance.
    
    # Response from the model:
    # '''.strip()
    reasoning_reward_prompt_template = get_reasoning_reward_prompt()

    assert len(images) * num_generations == len(completions)
    # Prepare messages for API call
    for idx, (completion) in enumerate(completions):
        completion_text = completion[0]['content']
        image = images[idx // num_generations]
        base64_image = pil_image_to_base64(image)
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image":  f"data:image/jpeg;base64,{base64_image}"},
                {"type": "text", "text": reasoning_reward_prompt_template.format(max_score=str(max_score)) + completion_text}
            ]
        })

    try:
        response = requests.post(API_URL, json={"messages": messages})
        if response.status_code == 200:
            all_responses = response.json()['responses']
        else:
            print(f"Error calling API: {response.status_code}, return 0 reward for reasoning.")
            all_responses = [0] * len(messages)
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        all_responses = [0] * len(messages)
    # Process responses and compute rewards
    rewards = []
    for response in all_responses:
        try:
            score_pattern = r'(?i)(\d+(?:\.\d+)?)'
            match = re.search(score_pattern, response)
            if match:
                score = float(match.group(1))
                score = max(0, min(max_score, score))
                rewards.append(score / max_score * scale_factor)
            else:
                print(f"Warning: Could not find score in response: {response}...")
                rewards.append(0)
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            rewards.append(0)
    print(f'Reasoning rewards: {rewards}') if accelerator.is_main_process else None
    return rewards


# Requires transformers>=4.51.0
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def embedding_similarity_reward_qwen3(completions, solution, **kwargs):
    embedding_model = kwargs.get('embedding_model')
    embedding_tokenizer = kwargs.get('embedding_tokenizer')
    accelerator = kwargs.get('accelerator')

    if embedding_model is None or embedding_tokenizer is None:
        if accelerator and accelerator.is_main_process:
            print("Warning: Embedding model or tokenizer not provided for embedding_similarity_reward. Returning 0 rewards.")
        return [0.0] * len(completions)

    contents = [completion[0]["content"] for completion in completions]
    
    pred_answers = []
    for content in contents:
        content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        student_answer = content_match.group(1).strip() if content_match else content.strip()
        pred_answers.append(student_answer)
    
    gt_answers = []
    for sol in solution:
        sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        gt_answers.append(ground_truth)
    
    input_texts = pred_answers + gt_answers
    
    max_length = 8192

    # Tokenize the input texts
    batch_dict = embedding_tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    batch_dict = {k: v.to(embedding_model.device) for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = embedding_model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

    num_samples = len(pred_answers)
    pred_embeddings = embeddings[:num_samples]
    gt_embeddings = embeddings[num_samples:]
    
    # Calculate cosine similarity
    with torch.no_grad():
        similarities = F.cosine_similarity(pred_embeddings, gt_embeddings, dim=1)
    
    return similarities.tolist()

