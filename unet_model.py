import tensorflow as tf
from tensorflow.keras import layers
from get_images import train_dataset, val_dataset, test_dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class UNet(tf.keras.Model):
    def __init__(self, input_shape=(192, 192, 3), num_classes=1):
        super(UNet, self).__init__()
        self.num_classes = num_classes

        # Encoder / Downsampling
        self.down1 = self.down_block(64)
        self.down2 = self.down_block(128)
        self.down3 = self.down_block(256)
        self.down4 = self.down_block(512)

        # Bottleneck
        self.bottleneck_conv1 = layers.Conv2D(1024, 3, padding='same', activation='relu')
        self.bottleneck_conv2 = layers.Conv2D(1024, 3, padding='same', activation='relu')

        # Decoder / Upsampling
        self.up1 = self.up_block(512)
        self.up2 = self.up_block(256)
        self.up3 = self.up_block(128)
        self.up4 = self.up_block(64)

        # Output
        self.output_conv = layers.Conv2D(num_classes, 1, padding='same',
                                         activation='sigmoid' if num_classes == 1 else 'softmax')

    def down_block(self, filters):
        return {
            "conv1": layers.Conv2D(filters, 3, padding='same', activation='relu'),
            "conv2": layers.Conv2D(filters, 3, padding='same', activation='relu'),
            "pool": layers.MaxPooling2D(pool_size=(2, 2))
        }

    def up_block(self, filters):
        return {
            "upsample": layers.Conv2DTranspose(filters, 2, strides=2, padding='same'),
            "conv1": layers.Conv2D(filters, 3, padding='same', activation='relu'),
            "conv2": layers.Conv2D(filters, 3, padding='same', activation='relu')
        }

    def call(self, inputs):
        # Downsampling path
        c1 = self.down1["conv2"](self.down1["conv1"](inputs))
        p1 = self.down1["pool"](c1)

        c2 = self.down2["conv2"](self.down2["conv1"](p1))
        p2 = self.down2["pool"](c2)

        c3 = self.down3["conv2"](self.down3["conv1"](p2))
        p3 = self.down3["pool"](c3)

        c4 = self.down4["conv2"](self.down4["conv1"](p3))
        p4 = self.down4["pool"](c4)

        # Bottleneck
        b = self.bottleneck_conv2(self.bottleneck_conv1(p4))

        # Upsampling path
        u1 = self.up1["upsample"](b)
        u1 = tf.concat([u1, c4], axis=-1)
        u1 = self.up1["conv2"](self.up1["conv1"](u1))

        u2 = self.up2["upsample"](u1)
        u2 = tf.concat([u2, c3], axis=-1)
        u2 = self.up2["conv2"](self.up2["conv1"](u2))

        u3 = self.up3["upsample"](u2)
        u3 = tf.concat([u3, c2], axis=-1)
        u3 = self.up3["conv2"](self.up3["conv1"](u3))

        u4 = self.up4["upsample"](u3)
        u4 = tf.concat([u4, c1], axis=-1)
        u4 = self.up4["conv2"](self.up4["conv1"](u4))

        return self.output_conv(u4)




def iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Flatten the tensors
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # Compute the intersection and union
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    return (intersection + smooth) / (union + smooth)



unet = UNet(num_classes = 1)

unet.compile(optimizer = tf.keras.optimizers.Adam(1e-4),  
                      loss = 'binary_crossentropy',
                      metrics=['accuracy', iou])


history = unet.fit(train_dataset,
                       epochs = 50,
                       validation_data = val_dataset)

history.save_weights('unet_model.weights.h5')

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

hist.tail(1)

def plot_history(hist_):
    plt.figure(figsize=(12, 10))
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.plot(hist['epoch'], hist['accuracy'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_accuracy'], label = 'Val Error')
    plt.legend(["train", "validation"], loc="upper left", prop={'size': 20})

plot_history(hist)

