## ViRFT for reasoning grounding

## training
1. Download [LISA dataset](https://github.com/dvlab-research/LISA)
2. use `gen_box_ann.py` to generate box from mask.
3. use `gen_sft.py` to generate SFT/Visual-RFT training annotations.
4. use `src/scripts/2B_lisa_grounding.sh` to train the model, with annotation path changed to step.3 generated annotations.

After training model, replace model path in `Qwen2_VL_lisa_infere.py` with your own ckpt.

```python
# Load Qwen2-VL-2B model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/path/to/your/checkpoint-498", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
).eval()

processor = AutoProcessor.from_pretrained("/path/to/your/checkpoint-498")
```

to compute gIoU, follow the process bellow.
1. Use `box2mask.py` to extract mask from [SAM](https://github.com/facebookresearch/segment-anything)
2. Use `mask_iou` to comput mask IoU.

```shell
cd lisa_evaluation
bash Qwen2_VL_lisa_infere.sh
```
