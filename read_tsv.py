import csv
import base64
import numpy as np
import sys

# Increase max field size for CSV (important for large base64 strings)
csv.field_size_limit(sys.maxsize)

# Path to your TSV file (adjust as needed)
tsv_file = "../data/outputs/coco_precomp/dev_features.tsv"

# Fieldnames in the TSV (same as used during conversion)
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']



with open(tsv_file, "r") as f:
    reader = csv.DictReader(f, delimiter="\t", fieldnames=FIELDNAMES)
    # Count rows

    # Read only the first row (first image)
    item = next(reader)

    # Convert fields to proper types
    item['image_id'] = int(item['image_id'])
    item['num_boxes'] = int(item['num_boxes'])

    # Decode features (base64 → bytes → float32 → reshape)
    feat_buf = base64.b64decode(item['features'])
    feats = np.frombuffer(feat_buf, dtype=np.float32).reshape(item['num_boxes'], -1)

    print("Image ID:", item['image_id'])
    print("Features shape:", feats.shape)  # (num_boxes, feat_dim)
    print("First 5 region features:\n", feats[:5])  # print first 2 region vectors
