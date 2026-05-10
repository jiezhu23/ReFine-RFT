import re
from bs4 import BeautifulSoup
import difflib

def normalize_text(text):
    if not text:
        return ""
    text = text.strip().replace('\n', ' ').replace('_', ' ').replace('-', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def extract_choice_and_name(text):
    # only for choices in ABCD.
    if not text:
        return '', ''
    text = text.strip()
    match = re.match(r'([A-D])\.?\s*([^\n]*)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), normalize_text(match.group(2))
    else:
        return '', normalize_text(text)

def is_match(pred, gold, use_choice=False, use_label=False, use_fuzzy=True):
    ratio_threshold = 0.85
    pred_choice, pred_name = extract_choice_and_name(pred)
    gold_choice, gold_name = extract_choice_and_name(gold)

    choice_match = (pred_choice == gold_choice) if gold_choice else True
    name_match = (pred_name == gold_name)
    
    # english name fuzzy match
    if use_fuzzy:
        ratio = difflib.SequenceMatcher(None, pred_name, gold_name).ratio()

    if use_choice and use_label:
        if gold_name in pred_name:
            return True
        if use_fuzzy:
            return ratio > ratio_threshold and choice_match
        else:
            return choice_match and name_match
    elif use_choice:
        return choice_match
    elif use_label:
        if use_fuzzy:
            return ratio > ratio_threshold
        else:
            return name_match
    else:
        assert use_choice or use_label, "must set ture at least one of use_choice or use_label"

def parse_generated_prediction(predictions, restrict_format=True):
    """
    Parse the generated predictions and extract <answer> content.
    
    If restrict_format is True, expects the format is HTML-like:
    <think>...</think> <answer>...</answer>
    
    If parsing fails, returns: "NA ### {original_prediction}"
    """
    parsed = []
    
    for prediction in predictions:
        try:
            if not isinstance(prediction, str) or not prediction.strip():
                parsed.append(f"NA ### EMPTY OR INVALID INPUT")
                continue

            if restrict_format:
                soup = BeautifulSoup(prediction, "html.parser")
                answer_tag = soup.find("answer")

                if answer_tag and answer_tag.text.strip():
                    parsed.append(answer_tag.text.strip())
                else:
                    parsed.append(f"NA ### {prediction}")
            else:
                parsed.append(prediction)

        except Exception as e:
            print(f"Error parsing prediction {prediction}: {str(e)}")
            parsed.append(f"NA ### {prediction}")
    return parsed

def merge_lora_model(model, lora_checkpoint_path):
    """
    Merge a LoRA model checkpoint with the base model for evaluation.
    
    Args:
        model: The base model object
        lora_checkpoint_path: Path to the LoRA checkpoint
        
    Returns:
        model: The merged model for inference
    """
    from peft import PeftModel
    
    print(f"Loading and merging LoRA weights from {lora_checkpoint_path}")
    
    # Load the LoRA model
    model.model = PeftModel.from_pretrained(
        model.model,
        lora_checkpoint_path,
        is_trainable=False,
    )
    
    # Merge weights for evaluation
    model.model = model.model.merge_and_unload()
    
    print("Successfully merged LoRA weights with base model")
    
    return model