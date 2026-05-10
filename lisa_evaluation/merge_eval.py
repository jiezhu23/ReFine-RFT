import json
import os
merged = []
for i in range(int(os.environ['SPLIT_NUM'])):
    data = json.load(open(f"tmp/res_{i}.json", 'r'))
    merged += data
print(f"mIoU: {sum(merged) / len(merged)}")
