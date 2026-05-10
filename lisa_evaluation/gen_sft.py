import os
import json
from PIL import Image
res = []
index = 0
for i, item in enumerate(json.load(open("path/to/your/lisa_train.json", 'r'))):
    for instruct in item['instruction']:
        w, h= Image.open(item['image_path']).size
        res.append({
            "id": f"lisa_{index}",
            "conversations": [
                {
                    "from": "user",
                    "value": f"<img>{item['image_path']}</img>\n Output the bounding box in the image corresponding to the instruction: {instruct}"
                },
                {
                    "from": "assistant",
                    "value": f"({int(item['boxes'][0][0] / w * 1000)},{int(item['boxes'][0][1] / h * 1000)}),({int(item['boxes'][0][2] / w * 1000)},{int(item['boxes'][0][3] / h * 1000)})"
                }
            ]
        })
        index += 1
json.dump(res, open("lisa_train_sft.json", 'w'), indent=4)
