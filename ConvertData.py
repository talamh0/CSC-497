import numpy as np
import base64
import csv

features_path = 'data/data/f30k_precomp/test_ims.npy'
features = np.load(features_path)  # شكلها (N, 2048)

with open('output_features.tsv', 'w', newline='') as tsv_file:
    writer = csv.writer(tsv_file, delimiter='\t')

    writer.writerow(['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features'])

    for idx, feat in enumerate(features):
        image_id = f'image_{idx}'
        image_w = 640 
        image_h = 480
        num_boxes = 1

        boxes = np.array([[0, 0, image_w, image_h]], dtype=np.float32)

        feat_bytes = base64.b64encode(feat.astype(np.float32).tobytes()).decode('utf-8')
        boxes_bytes = base64.b64encode(boxes.astype(np.float32).tobytes()).decode('utf-8')

        writer.writerow([image_id, image_w, image_h, num_boxes, boxes_bytes, feat_bytes])
