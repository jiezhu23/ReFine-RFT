"""
Credited from https://www.kaggle.com/code/alphadua/finetune-qwen2-5vl-grpo

# modified by Jie Zhu

Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import textwrap
import json
import copy
import gc
import random
from argparse import Namespace
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Dict, List
from tqdm import tqdm
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
    BitsAndBytesConfig
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from transformers.trainer import EvalLoopOutput
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from qwen_vl_utils import process_vision_info
from accelerate.utils import gather_object
import asyncio
import inspect

from src.refinerft.utils.prompt_utils import get_user_prompt_cot, get_user_prompt_ao


try:
    # Add classification evaluation path
    sys.path.append("./classification")
    from Qwen2_VL_classification_utils import run as evaluate
    # from Qwen2_VL_classification_infere import run as evaluate
    CLASSIFICATION_AVAILABLE = True
except Exception as e:
    print(e)
    CLASSIFICATION_AVAILABLE = False
    print("Warning: Aircraft classification evaluation modules not found. Aircraft classification evaluation will be disabled.")

try:
    # Add evaluation path
    sys.path.append("./coco_evaluation")
    # from Qwen2_VL_classification_utils import run as evaluate
    from Qwen2_VL_coco_infere import run as evaluate
    COCO_EVALUATION_AVAILABLE = True
except Exception as e:
    print(e)
    COCO_EVALUATION_AVAILABLE = False
    print("Warning: coco evaluation modules not found. Coco evaluation will be disabled.")

if is_peft_available():
    from peft import PeftConfig, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

dataset_name_mapping = {
    "laolao77/ViRFT_CLS_fgvc_aircraft_4_shot": "fgvc_aircraft",
    "laolao77/ViRFT_CLS_flower_4_shot": "oxford_flowers",
    "laolao77/ViRFT_CLS_car196_4shot": "stanford_cars",
    "laolao77/ViRFT_CLS_pets37_4shot": "pets",
    "flaviagiammarino/vqa-rad": "vqa-rad",
    "AmazonScience/RobustAD": "RobustAD",
    "laolao77/ViRFT_COCO_8_cate_4_shot": "coco",
}

class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        model: Union[str, PreTrainedModel] = None,
        model_cache_dir: Optional[str] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        reward_weights: Optional[list[float]] = None,
        custom_args: Optional[Any] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs,
                                                                        torch_dtype=torch.bfloat16,
                                                                        cache_dir=model_cache_dir)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs,
                                                                           cache_dir=model_cache_dir)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            print(peft_config)
            model.enable_input_require_grads()
            model = get_peft_model(model, peft_config)
            print(model.print_trainable_parameters())

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = int(max_pixels)
                    processing_class.image_processor.min_pixels = int(min_pixels)
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs
        
        # Reward weights
        if reward_weights is not None:
            if len(reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)
            
        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.generation_config_val = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=False,  
            num_return_sequences=1, # only one completion for validation
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta
        self.custom_args = custom_args

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def override_prompt(self, prompt_dict: dict[str, Union[torch.Tensor, Any]], dataset_name: str, prompt_type: str) -> dict[str, Union[torch.Tensor, Any]]:
        if "vqa-rad" in dataset_name or "RobustAD" in dataset_name or "coco" in dataset_name:
            return None
        if prompt_type == "cot":
            prompt =  get_user_prompt_cot(dataset_name)
        elif prompt_type == "ao":
            prompt =  get_user_prompt_ao(dataset_name)
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        prompt_dict[0]['content'][1]['text'] = prompt
        
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        for x in inputs:
            self.override_prompt(x["prompt"], dataset_name_mapping[self.custom_args.dataset_name], self.custom_args.prompt_type)
        prompts = [x["prompt"] for x in inputs]
        
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        images = [x["image"] for x in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]

        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)

        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        #------------------insert a ground truth into completions----------------
        # TODO: debug
        # gt = inputs[0]['solution']
        # random_index = random.randint(0, self.num_generations - 1)
        # completions[random_index][0]['content'] = gt
        # completions[random_index][0]['content'] = completions[random_index][0]['content'].split(r'<answer>')[0] + gt

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                # prepare for some reward calculations
                reward_kwargs['prompts'] = prompts
                reward_kwargs['completion_length'] = completion_mask.sum(1).float().cpu().tolist()
                reward_kwargs['images'] = images
                reward_kwargs['num_generations'] = self.num_generations
                reward_kwargs['accelerator'] = self.accelerator
                reward_kwargs['prompt_type'] = self.custom_args.prompt_type
                reward_kwargs['min_len'] = self.custom_args.min_len if hasattr(self.custom_args, 'min_len') else None
                reward_kwargs['max_len'] = self.custom_args.max_len if hasattr(self.custom_args, 'max_len') else None
                reward_kwargs['margin'] = self.custom_args.margin if hasattr(self.custom_args, 'margin') else None
                reward_kwargs['eval_dataset_name'] = self.custom_args.dataset_name
                _result = reward_func(completions=completions, **reward_kwargs)
                # detect if the reward function is async function (api call)
                if inspect.isawaitable(_result):
                    output_reward_func = asyncio.run(_result)
                else:
                    output_reward_func = _result
                # for combined reward function, return 3 values
                if isinstance(output_reward_func, tuple):
                    output_reward_func, _func_names, _reward_per_func = output_reward_func
                    _reward_per_func = _reward_per_func.to(device)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        # print the last sample group for visualization
        if self.accelerator.is_main_process:

            # Select a random sample from the last batch
            selected_idx = torch.randint(0, len(prompts), (1,)).item()
            
            # Log to wandb at specific steps
            # if wandb.run is not None:
            if wandb.run is not None and self.state.global_step % 10 == 0:
            
                train_sample_table = wandb.Table(columns=["Prompt", "Label", "Response", "Image"])
                # Calculate the original batch index
                selected_img_idx = selected_idx // self.num_generations
                
                # Extract prompt, label and response
                prompt_text = prompts[selected_idx][-1]['content'][-1]['text']
                label_text = reward_kwargs['solution'][selected_idx]
                response_text = completions[selected_idx][0]['content']
                
                # Handle image if available
                wandb_image = None
                if images is not None and images[selected_img_idx] is not None:
                    wandb_image = wandb.Image(images[selected_img_idx])
                
                # Add data to table
                train_sample_table.add_data(
                    prompt_text,
                    label_text,
                    response_text,
                    wandb_image
                )
                
                # Log the table
                wandb.log({
                    "visual/train_sample": train_sample_table,
                }, commit=False)
                
                # Print to console
                print('/---------------Random Selected Response-----------------/')
                print(f"[Prompt]:\n{prompts[selected_idx][-1]['content'][-1]['text']}\n")
                print(f"[Label]:\n{reward_kwargs['solution'][selected_idx]}")
                print(f"[Response]:\n{completions[selected_idx][0]['content']}")
                print('/--------------------------------------------------------/\n')
            
        #------------------NEW advantage calculation----------------
        # rewards = rewards_per_func.sum(dim=1) # (B*G,)

        # # Compute grouped-wise rewards based on the rewards of each reward function
        # mean_grouped_rewards = rewards_per_func.view(-1, self.num_generations, len(self.reward_funcs)).mean(dim=1) # (B, num_reward_funcs)
        # std_grouped_rewards = rewards_per_func.view(-1, self.num_generations, len(self.reward_funcs)).std(dim=1) # (B, num_reward_funcs)

        # # Normalize the rewards to compute the advantages based on the rewards of each reward function
        # mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # (B*G, num_reward_funcs)
        # std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # (B*G, num_reward_funcs)
        # advantages = (rewards_per_func - mean_grouped_rewards) / (std_grouped_rewards + 1e-4) # (B*G, num_reward_funcs)
        # advantages = advantages.sum(dim=1) # (B*G,)
        #-----------------------------------------------------------
        
        #------------------old advantage calculation----------------
        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1) # (B*G,)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        #-----------------------------------------------------------
        
        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        torch.cuda.empty_cache()
        
        return loss


    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Evaluation loop for GRPO trainer.
        """
        import re
        import math
        
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        
        # Set model to evaluation mode
        self.model.eval()
        if self.ref_model is not None:
            self.ref_model.eval()
            
        eval_results = {}
        error_count = 0
        right_count = 0
        total_count = 0
        local_conversations = []
        total_response_token_length = 0
        
        # Check if we should run aircraft classification evaluation
        if CLASSIFICATION_AVAILABLE:
            with torch.inference_mode():
                for batch in tqdm(dataloader, desc=f"Evaluating on rank {self.accelerator.process_index}"):
                    # Handle both single examples and batches
                    if isinstance(batch, list):
                        examples = batch
                    else:
                        examples = [batch]
                    
                    for example in examples:
                        messages = example["prompt"]
                        
                        true_category = example["category"]
                        eval_dataset_name = example["eval_dataset_name"]
                        if "vqa-rad" in eval_dataset_name or "RobustAD" in eval_dataset_name:
                            prompts_text = maybe_apply_chat_template(example, self.processing_class)["prompt"]
                            inputs = self.processing_class(
                                text=prompts_text,
                                images=example["image"],
                                return_tensors="pt",
                                padding=True,
                                padding_side="left",
                                add_special_tokens=False,
                            )
                            inputs = super()._prepare_inputs(inputs)             
                            
                        else:
                            # debug to make sure the messages is correct
                            del messages[0]['content'][0]['text']
                            del messages[0]['content'][1]['image']
                            # Preparation for inference
                            text = self.processing_class.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                            image_inputs, video_inputs = process_vision_info(messages)

                            inputs = self.processing_class(
                                text=[text],
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensors="pt",
                            )
                        inputs = inputs.to(self.accelerator.device)
                        
                        # Generate response
                        generated_ids = self.model.generate(
                            **inputs, 
                            max_new_tokens=1024, 
                            use_cache=True,
                            temperature=0.0,
                            do_sample=False  # Use greedy decoding for evaluation
                        )
                            
                        generated_ids_trimmed = [
                            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        
                        response = self.processing_class.batch_decode(
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                        total_response_token_length += len(generated_ids_trimmed[0])
                        
                        total_count += 1
                        
                        # Extract answer and evaluate
                        if self.custom_args.prompt_type == "cot":
                            try:
                                match = re.search(r"<answer>(.*?)</answer>", response)
                                if match:
                                    answer_content = match.group(1)
                                    
                                    # Normalize for comparison
                                    if "vqa-rad" in eval_dataset_name or "RobustAD" in eval_dataset_name:
                                        true_category_norm = true_category.lower()
                                        answer_content_norm = answer_content.lower()
                                    else:
                                        true_category_norm = true_category.replace(' ', '').replace('_', '').lower()
                                        answer_content_norm = answer_content.replace(' ', '').replace('_', '').lower()
                                    
                                    # Check if answer is correct
                                    if true_category_norm in answer_content_norm or answer_content_norm in true_category_norm:
                                        right_count += 1
                                        is_correct = True
                                    else:
                                        error_count += 1
                                        is_correct = False
                                else:
                                    error_count += 1
                                    is_correct = False
                            except Exception as e:
                                error_count += 1
                                is_correct = False
                        elif self.custom_args.prompt_type == "ao":                                    # Normalize for comparison
                            true_category_norm = true_category.replace(' ', '').replace('_', '').lower()
                            answer_content_norm = response.replace(' ', '').replace('_', '').lower()
                            if true_category_norm in answer_content_norm or answer_content_norm in true_category_norm:
                                right_count += 1
                                is_correct = True
                            else:
                                error_count += 1
                                is_correct = False
                                
                        conversation_data = {
                            "prompt": messages,
                            "response": response,
                            "label": true_category,
                            "is_correct": is_correct,
                        }
                        if "index" in example:
                            conversation_data["index"] = example["index"]
                        local_conversations.append(conversation_data)
                
            # Gather conversations from all processes and save to file
            gathered_conversations = gather_object(local_conversations)

            if self.accelerator.is_main_process and gathered_conversations:
                if hasattr(self, "_globalstep_last_logged"):
                    output_path = os.path.join(self.args.output_dir, f"conversations_step{self._globalstep_last_logged}.jsonl")
                else:
                    output_path = os.path.join(self.args.output_dir, f"conversations_zeroshot.jsonl")

                # Sort conversations if index is available
                if all("index" in conv for conv in gathered_conversations):
                    all_conversations = sorted(gathered_conversations, key=lambda x: x['index'])
                else:
                    all_conversations = gathered_conversations
                
                with open(output_path, 'w') as f:
                    for item in all_conversations:
                        f.write(json.dumps(item) + '\n')

            # Gather results from all processes using accelerator
            local_stats = torch.tensor([error_count, right_count, total_count, total_response_token_length], 
                                     dtype=torch.float32, device=self.accelerator.device)
            gathered_stats = self.accelerator.gather(local_stats).reshape(-1, 4)
            
            # Handle both single process (1D tensor) and multi-process (2D tensor) cases
            if gathered_stats.dim() == 1:
                # Single process case - tensor is [error_count, right_count, total_count]
                global_error_count = gathered_stats[0].item()
                global_right_count = gathered_stats[1].item()
                global_total_count = gathered_stats[2].item()
                global_total_response_token_length = gathered_stats[3].item()
            else:
                # Multi-process case - tensor is [num_processes, 4]
                global_error_count = gathered_stats[:, 0].sum().item()
                global_right_count = gathered_stats[:, 1].sum().item()
                global_total_count = gathered_stats[:, 2].sum().item()
                global_total_response_token_length = gathered_stats[:, 3].sum().item()
            
            avg_response_length = global_total_response_token_length / global_total_count if global_total_count > 0 else 0
            global_accuracy = global_right_count / global_total_count if global_total_count > 0 else 0
            
            eval_results = {
                f"{metric_key_prefix}_accuracy": global_accuracy,
                f"{metric_key_prefix}_{eval_dataset_name}_accuracy": global_accuracy,
                f"{metric_key_prefix}_avg_response_length": avg_response_length,
                # f"{metric_key_prefix}_error_count": int(global_error_count),
                # f"{metric_key_prefix}_right_count": int(global_right_count),
                # f"{metric_key_prefix}_total_count": int(global_total_count)
            }
            if self.accelerator.is_main_process:
                print(f"Global Classification Results:")
                print(f"  Accuracy: {global_accuracy:.1%} ({int(global_right_count)}/{int(global_total_count)})")
                print(f"  Average Response Length: {avg_response_length:.2f}")
                print(f"  Error count: {int(global_error_count)}")
                
                # Log to wandb if available  
                # if wandb.run is not None and hasattr(gathered_stats, '__len__') and len(gathered_stats) > 0:
                #     # Get the dataset name from the first example for logging
                #     first_example_dataset = None
                #     try:
                #         if hasattr(dataloader, 'dataset') and len(dataloader.dataset) > 0:
                #             first_example_dataset = dataloader.dataset[0].get("eval_dataset_name", "unknown")
                #     except:
                #         first_example_dataset = "unknown"
                    
                #     wandb.log({
                #         f"eval/{first_example_dataset}_accuracy": global_accuracy,
                #     }, commit=False)
            else:
                # Log local results for debugging
                local_accuracy = right_count / total_count if total_count > 0 else 0
                print(f"Local accuracy (rank {self.accelerator.process_index}): {local_accuracy:.1%} ({right_count}/{total_count})")
            
            total_count = int(global_total_count)
            # Wait for all processes to complete
            self.accelerator.wait_for_everyone()
        
        return EvalLoopOutput(
            predictions=[],  # No predictions needed for classification evaluation
            label_ids=[],   # No label ids needed for classification evaluation
            metrics=eval_results,
            num_samples=total_count
        )