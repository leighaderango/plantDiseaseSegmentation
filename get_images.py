import os
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import torch

def get_cropped_data(folder, target_size=(192, 192)):

    images = []
    masks = []
    random.seed(1234)
    
    for filename in os.listdir(folder):

        img_path = os.path.join(folder, filename).replace('\\', '/')
        mask_path = os.path.join('data/masks', filename).replace('\\', '/').replace('jpg', 'png')

        img = load_img(img_path)
        mask = load_img(mask_path)

        img = img_to_array(img)
        mask = img_to_array(mask)

        if (img.shape[0] >= target_size[0]) & (img.shape[1] >= target_size[1]):
            combined = tf.concat([img, mask], axis = -1)
            combined = tf.image.random_crop(combined, size = [target_size[0], target_size[1], 6])

            img_crop = combined[:, :, :3] / 255.0 # normalize on input
            mask_crop = combined[:, :, 3:]

            gray_mask = np.dot(mask_crop[...,:3], [0.2989, 0.5870, 0.1140])
            binary_mask = (gray_mask > 38).astype(np.uint8)
            binary_mask = np.expand_dims(binary_mask, axis=-1)

            images.append(img_crop)
            masks.append(binary_mask)

    return [np.array(images), np.array(masks)]




images = get_cropped_data('data/images')
# image has dimension (192, 192, 3)
# mask has dimension (192, 192, 1)

# Unpack
image_data = images[0]
mask_data = images[1]

# Shuffle indices
total_samples = len(image_data)
indices = np.random.permutation(total_samples)

# Define split sizes
train_split = int(0.75 * total_samples)
val_split = int(0.85 * total_samples)

# Get splits
train_idx = indices[:train_split]
val_idx = indices[train_split:val_split]
test_idx = indices[val_split:]

# Slice data
train_images, train_masks = image_data[train_idx], mask_data[train_idx]
val_images, val_masks = image_data[val_idx], mask_data[val_idx]
test_images, test_masks = image_data[test_idx], mask_data[test_idx]


def get_normalizing_constants(train_images):
    mean = np.zeros(3)
    std = np.zeros(3)
    num_pixels = 0

    for img in train_images:
        num_pixels += img.shape[0] * img.shape[1]

        # Sum over height and width for each channel
        mean += img.sum(axis=(0, 1))
        std += (img ** 2).sum(axis=(0, 1))

    mean /= num_pixels
    std = np.sqrt(std / num_pixels - mean ** 2)
    return [mean, std]


train_constants = get_normalizing_constants(train_images)
val_constants = get_normalizing_constants(val_images)
test_constants = get_normalizing_constants(test_images)

def normalize_datasets(data, constants):
    normalized_images = []
    for img in data:
        standardized = (img - constants[0])/constants[1]
        normalized_images.append(standardized)

    return normalized_images

norm_train_images = normalize_datasets(train_images, train_constants) 
norm_val_images = normalize_datasets(val_images, val_constants) 
norm_test_images = normalize_datasets(test_images, test_constants) 


# Create tf.data datasets
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((norm_train_images, train_masks)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((norm_val_images, val_masks)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((norm_test_images, test_masks)).batch(batch_size)



def show_imageandmask(images, n):
    fig, axes = plt.subplots(nrows = n, ncols = 2, figsize = (8, 2.5*n))
    
    for i in range(0, n):
        
        axes[i, 0].imshow(images[0][i])
        axes[i, 1].imshow(images[1][i])


    plt.tight_layout()

#show_imageandmask(images, 3)
  
