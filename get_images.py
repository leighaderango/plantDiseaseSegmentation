import os
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def get_cropped_data(folder, target_size=(192, 192)):

    images = []
    masks = []
    
    for filename in os.listdir(folder):

        img_path = os.path.join(folder, filename).replace('\\', '/')
        mask_path = os.path.join('data/masks', filename).replace('\\', '/').replace('jpg', 'png')

        img = load_img(img_path)
        mask = load_img(mask_path)

        img = img_to_array(img)
        mask = img_to_array(mask)

        if (img.shape[0] >= 200) & (img.shape[1] >= 200):
            combined = tf.concat([img, mask], axis = -1)
            combined = tf.image.random_crop(combined, size = [target_size[0], target_size[1], 6])

            img_crop = combined[:, :, :3] / 255.0 # normalize on input
            mask_crop = combined[:, :, 3:]

        
            images.append(img_crop)
            masks.append(mask_crop)
  
    return [np.array(images), np.array(masks)]



images = get_cropped_data('data/images')
# image and mask have dimension (224, 224, 3)


def show_imageandmask(images, n):
    fig, axes = plt.subplots(nrows = n, ncols = 2, figsize = (8, 2.5*n))
    
    for i in range(0, n):
        
        axes[i, 0].imshow(images[0][i])
        axes[i, 1].imshow(images[1][i])


    plt.tight_layout()

show_imageandmask(images, 3)
  
