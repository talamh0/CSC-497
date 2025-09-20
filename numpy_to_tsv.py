import numpy as np
import base64
import csv

# Paths
ids_file = "../data/coco_precomp/dev_ids.txt"
npy_file = "../data/coco_precomp/dev_ims.npy"
tsv_out = "../data/outputs/coco_precomp/dev_features.tsv"

# Load IDs
with open(ids_file, "r") as f:
    ids = [int(line.strip()) for line in f]

# Load features
feats = np.load(npy_file)   # shape (N, K, D)
N, K, D = feats.shape
print("Loaded features:", feats.shape)

assert len(ids) == N, "IDs and feature count must match!"

# Write TSV
FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']

with open(tsv_out, "w", newline="") as f_out:
    writer = csv.DictWriter(f_out, delimiter="\t", fieldnames=FIELDNAMES)
    for i, img_id in enumerate(ids):
        feat = feats[i]  # (K, D)
        # Encode features
        feat_str = base64.b64encode(feat.astype(np.float32).tobytes()).decode("utf-8")
        # Here we don't have boxes or image size → put dummy values
        row = {
            'image_id': img_id,
            'image_w': -1,         # unknown
            'image_h': -1,         # unknown
            'num_boxes': K,
            'boxes': '',           # leave blank or dummy
            'features': feat_str
        }
        writer.writerow(row)

print("Saved TSV:", tsv_out)
