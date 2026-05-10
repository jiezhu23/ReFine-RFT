import os
import json
import numpy as np
from pycocotools import mask as maskUtils
from PIL import Image, ImageDraw
from tqdm import tqdm

def polygon_to_mask(polygon, size):
    mask = Image.new('L', (size[1], size[0]), 0)
    ImageDraw.Draw(mask).polygon([tuple(point) for point in polygon], outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)

def rle_to_mask(rle):
    rle_decoded = maskUtils.decode(rle)
    return rle_decoded.astype(np.uint8)

def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / (union + 1e-6) 
    return intersection, union, iou

res = []
base_path = "/path/to/LISA-main/data/test"

test_res = {}

with open(f"all_masks_rle.json", 'r') as f:
    for item in json.load(f):
        test_res[item['image_pth']] = item['rle_mask']

total_intersection = 0
total_union = 0
ious = []

for pth in tqdm(os.listdir(base_path)):
    if pth.endswith(".json"):
        json_path = os.path.join(base_path, pth)

        with open(json_path, 'r') as f:
            item = json.load(f)

        try:
            gt_polygon = item["shapes"][0]['points']
        except:
            print("no res")
            continue

        image_pth = f"/path/to/LISA-main/data/test/{pth.replace('.json', '.jpg')}"
        pred_rle = test_res.get(image_pth, None)


        with Image.open(image_pth) as img:
            img_width, img_height = img.size
            real_size = [img_height, img_width]

        if pred_rle is not None:
            size = pred_rle['size']

            if size != real_size:
                print("shape mismatch")
                pred_mask = np.zeros((size[0], size[1]), dtype=np.uint8)
                continue
            pred_mask = rle_to_mask(pred_rle)
        else:
            print(f"Warning: No prediction found for {image_pth}. Setting IoU to 0.")
            pred_mask = np.zeros((size[0], size[1]), dtype=np.uint8)

        gt_mask = polygon_to_mask(gt_polygon, real_size)
        intersection, union, iou = compute_iou(pred_mask, gt_mask)

        total_intersection += intersection
        total_union += union

        ious.append(iou)

gIoU = np.mean(ious) if ious else 0
cIoU = total_intersection / (total_union + 1e-6) 

print(f"gIoU (Global IoU): {gIoU:.4f}")
print(f"cIoU (Cumulative IoU): {cIoU:.4f}")
