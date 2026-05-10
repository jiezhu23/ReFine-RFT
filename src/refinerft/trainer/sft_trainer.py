import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Dict
from tqdm import tqdm
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    TrainingArguments,
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
)
from transformers.utils import logging
from transformers.utils import is_peft_available
from transformers.trainer import EvalLoopOutput
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from collections import defaultdict
from qwen_vl_utils import process_vision_info
from accelerate.utils import gather_object

import sys
import json
import copy
from typing import Dict, List, Optional

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

logger = logging.get_logger(__name__)

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    
    
dataset_name_mapping = {
    "laolao77/ViRFT_CLS_fgvc_aircraft_4_shot": "fgvc_aircraft",
    "laolao77/ViRFT_CLS_flower_4_shot": "oxford_flowers",
    "laolao77/ViRFT_CLS_car196_4shot": "stanford_cars",
    "laolao77/ViRFT_CLS_pets37_4shot": "pets",
}

class Qwen2VLSFTTrainer(Trainer):
    """
    Trainer for Supervised Fine-tuning (SFT) of Qwen2-VL models.
    This trainer handles the same input format as Qwen2VLGRPOTrainer but performs standard supervised training.
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel] = None,
        model_cache_dir: Optional[str] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        custom_args: Optional[Any] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = TrainingArguments(f"{model_name}-SFT")

        # Models
        # Trained model
        model_init_kwargs = {}
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
                    "Invalid `torch_dtype` passed to `TrainingArguments`. Expected either 'auto' or a string representing "
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
                                                                           torch_dtype=torch.bfloat16,
                                                                           cache_dir=model_cache_dir)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        elif model is None:
            pass # instantiate model in super().__init__
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `TrainingArguments`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
                
        if peft_config is not None:
            print(peft_config)
            model.enable_input_require_grads()
            model = get_peft_model(model, peft_config)
            print(model.print_trainable_parameters())
    
        # Processing class
        pad_token_id = processing_class.tokenizer.pad_token_id if processing_class is not None else None
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Data collator
        def data_collator(features):  # No data collation is needed
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length if hasattr(args, 'max_prompt_length') else None
        self.max_completion_length = args.max_completion_length if hasattr(args, 'max_completion_length') else None
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )
        self.custom_args = custom_args


        # Initialize metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            model_init=model_init,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # Set model warnings to avoid FLOPs computation warning
        self.model.warnings_issued["estimate_tokens"] = True
        # Gradient accumulation requires scaled loss
        self.model_accepts_loss_kwargs = False
        # if peft_config is not None:
        #     self.model = get_peft_model(self.model, peft_config)
        
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Match GRPO trainer's signature columns
            self._signature_columns = ["prompt"]
            
    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Skip default input preparation as we handle it in compute_loss
        """
        return inputs
    
    def override_prompt(self, prompt_dict: dict[str, Union[torch.Tensor, Any]], dataset_name: str, prompt_type: str) -> dict[str, Union[torch.Tensor, Any]]:
        if prompt_type == "cot":
            prompt =  get_user_prompt_cot(dataset_name)
        elif prompt_type == "ao":
            prompt =  get_user_prompt_ao(dataset_name)
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        prompt_dict[0]['content'][1]['text'] = prompt
        
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss for supervised fine-tuning
        """
        # Extract prompts, solutions and images from inputs
        for x in inputs:
            self.override_prompt(x["prompt"], dataset_name_mapping[self.custom_args.dataset_name], self.custom_args.prompt_type)
        prompts = [x["prompt"] for x in inputs]
        
        prompts_text = [maybe_apply_chat_template(x, self.processing_class)["prompt"] for x in inputs]
        images = [x["image"] for x in inputs]
        # for cot dataset, the solution is in the reference key
        if self.custom_args.prompt_type == "cot":
            completions = [x["reference"] for x in inputs]
        else:
            completions = [x["solution"] for x in inputs]

        # Handle case where no images are provided
        if all(i is None for i in images):
            images = None
            
        # Process full conversations (prompt + completion)
        model_inputs = self.processing_class(
            text=[prompt+completion+self.processing_class.tokenizer.eos_token for prompt, completion in zip(prompts_text, completions)],
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        # Move inputs to appropriate device
        model_inputs = {k: v.to(self.args.device) for k, v in model_inputs.items()}
        
        # Create labels for supervised training
        labels = model_inputs["input_ids"].clone()
        # Mask prompt tokens with -100 to only compute loss on completion tokens
        batch_size, seq_len = labels.shape
        eos_token_id = self.processing_class.tokenizer.eos_token_id
        
        # Create mask for prompt tokens
        for i in range(batch_size):
            input_ids = labels[i]
            eos_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]

            # Detect the interval between the last <eos> and the previous <eos>
            if len(eos_positions) > 1:
                last_eos = eos_positions[-1].item()
                prev_eos = eos_positions[-2].item() + 1  # Include the <eos> itself
            else:
                prev_eos = 0  # If there's only one <eos>, start from the beginning
                last_eos = eos_positions[-1].item() + 1  # Include the <eos> itself

            # Mask everything except the interval between the last and previous <eos>
            # add 3 as we concat "<|im_start|>assistant\n" to the end of the prompt to get the completion
            labels[i, :prev_eos+3] = -100
            labels[i, last_eos:] = -100
            
        model_inputs["labels"] = labels
                
        # Compute loss using the model
        outputs = model(**model_inputs)
        loss = outputs.loss
        
        # Calculate completion length
        # Get the attention mask for the completion part (after prompt)
        # completion_mask = model_inputs["attention_mask"].clone()
        # Set prompt tokens to 0 in the mask
        # completion_mask[prompt_mask] = 0
        # Sum the mask to get completion length and gather across devices
        # completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        
        # Log metrics
        self._metrics["loss"].append(loss.item())
        # self._metrics["completion_length"].append(completion_length)       
         
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics.clear()


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
        # if self.ref_model is not None:
        #     self.ref_model.eval()
            
        eval_results = {}
        error_count = 0
        right_count = 0
        total_count = 0
        local_conversations = []
        
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
                        
                        total_count += 1
                        
                        # Extract answer and evaluate
                        if self.custom_args.prompt_type == "cot":
                            try:
                                match = re.search(r"<answer>(.*?)</answer>", response)
                                if match:
                                    answer_content = match.group(1)
                                    
                                    # Normalize for comparison
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
            local_stats = torch.tensor([error_count, right_count, total_count], 
                                     dtype=torch.float32, device=self.accelerator.device)
            gathered_stats = self.accelerator.gather(local_stats).reshape(-1, 3)
            
            # Handle both single process (1D tensor) and multi-process (2D tensor) cases
            if gathered_stats.dim() == 1:
                # Single process case - tensor is [error_count, right_count, total_count]
                global_error_count = gathered_stats[0].item()
                global_right_count = gathered_stats[1].item()
                global_total_count = gathered_stats[2].item()
            else:
                # Multi-process case - tensor is [num_processes, 3]
                global_error_count = gathered_stats[:, 0].sum().item()
                global_right_count = gathered_stats[:, 1].sum().item()
                global_total_count = gathered_stats[:, 2].sum().item()
            
            global_accuracy = global_right_count / global_total_count if global_total_count > 0 else 0
            
            eval_results = {
                f"{metric_key_prefix}_accuracy": global_accuracy,
                f"{metric_key_prefix}_{eval_dataset_name}_accuracy": global_accuracy,
                # f"{metric_key_prefix}_error_count": int(global_error_count),
                # f"{metric_key_prefix}_right_count": int(global_right_count),
                # f"{metric_key_prefix}_total_count": int(global_total_count)
            }
            if self.accelerator.is_main_process:
                print(f"Global Classification Results:")
                print(f"  Accuracy: {global_accuracy:.1%} ({int(global_right_count)}/{int(global_total_count)})")
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