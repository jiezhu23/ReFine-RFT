import json
import os
import torch
import numpy as np
import cv2
from pycocotools import mask as maskUtils
from segment_anything import sam_model_registry, SamPredictor

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def scale_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]

def mask_to_rle(mask):
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def save_rle_to_json(rle_data_list, save_path):
    with open(save_path, 'w') as f:
        json.dump(rle_data_list, f)
    print(f"Saved all RLE data to {save_path}")

def main(json_path, sam_checkpoint, output_json_path, model_type="vit_h"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    data = load_json(json_path)

    all_rle_data = []

    for idx, item in enumerate(data):
        image_pth = item["image_pth"]
        pred_bbox = item["pred_bbox"]

        image = load_image(image_pth)
        height, width, _ = image.shape

        abs_bbox = scale_bbox(pred_bbox, width, height)

        predictor.set_image(image)

        transformed_bbox = predictor.transform.apply_boxes_torch(
            torch.tensor([abs_bbox], device=device), image.shape[:2]
        )
        masks, scores, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_bbox,
            multimask_output=False,
        )

        mask = masks[0][0].cpu().numpy()

        rle = mask_to_rle(mask)

        rle_data = {
            "image_pth": image_pth,
            "pred_bbox": pred_bbox,
            "rle_mask": rle
        }
        all_rle_data.append(rle_data)

    save_rle_to_json(all_rle_data, output_json_path)

if __name__ == "__main__":
    json_path = f"/path/to/resbox" 
    sam_checkpoint = "sam_vit_h_4b8939.pth" 
    output_json_path = f"./all_masks_rle.json"

    main(json_path, sam_checkpoint, output_json_path)
