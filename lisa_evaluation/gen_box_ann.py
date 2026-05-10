import os
import json
from PIL import Image, ImageDraw

res = []
base_path = "/path/to/your/LISA-main/data/train"

for pth in os.listdir(base_path):
    if pth.endswith(".json"):
        json_path = os.path.join(base_path, pth)

        with open(json_path, 'r') as f:
            item = json.load(f)

        instruct = item["text"]
        shapes = item["shapes"]

        boxes = []
        for shape in shapes[:1]:
            points = shape["points"]
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            boxes.append((x_min, y_min, x_max, y_max))

        img_path = json_path.replace(".json", ".jpg")
        if os.path.exists(img_path):
            res.append({
                "image_path": img_path,
                "instruction": instruct,
                "boxes": boxes
            })

json.dump(res, open("lisa_train.json", 'w'), indent=4)
