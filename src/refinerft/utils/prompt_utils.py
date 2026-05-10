SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

DATASET_MAPPING = {
    "fgvc_aircraft": "aircraft",
    "oxford_flowers": "plant",
    "stanford_cars": "car",
    "pets": "pet",
}

KEYWORD_MAPPING = {
    "fgvc_aircraft": "model",
    "oxford_flowers": "species",
    "stanford_cars": "model",
    "pets": "species",
}

def get_user_prompt_cot(dataset_name):
    return (
        f"This is an image containing an {DATASET_MAPPING[dataset_name]}. Please identify the {KEYWORD_MAPPING[dataset_name]} of the {DATASET_MAPPING[dataset_name]} based on the image.\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "The output answer format should be as follows:\n"
        "<think> ... </think> <answer>species name</answer>\n"
        "Please strictly follow the format."
    )

def get_user_prompt_ao(dataset_name):
    return (
        f"This is an image containing an {DATASET_MAPPING[dataset_name]}. "
        f"Please identify the {KEYWORD_MAPPING[dataset_name]} of the {DATASET_MAPPING[dataset_name]} based on the image. "
        "Only provide the final answer directly, without any explanation or special formatting."
    )

def get_reasoning_reward_prompt():
    return r'''
    You are tasked with evaluating the reasoning provided by a model in identifying an object from an image. 
    Based on the image and the reasoning content between the <think> tags, assess whether the reasoning is factually correct, relevant, and helpful for identifying the object. 
    Output only a score between 0 and {max_score} (number only, no additional text):
    - {max_score} means the reasoning is completely accurate, relevant, and helpful for classification.
    - 0 means the reasoning is completely inaccurate or irrelevant.
    - Values between 0 and {max_score} reflect partial correctness or partial relevance.
    
    Response from the model:
    '''.strip()