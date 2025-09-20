import numpy as np

# Load the numpy array
data_out = np.load('../data/coco_precomp/dev_ims.npy')

print("Array shape:", data_out.shape)   # (N, K, D)

# Get the features of the first image (index 0)
first_image_feats = data_out[0]         # shape (K, D)

print("First image features shape:", first_image_feats.shape)
print("First image features (first 5 rows):")
print(first_image_feats[:5])            # print only first 5 region features
