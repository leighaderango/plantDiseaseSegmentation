import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from get_images import train_dataset, val_dataset, test_dataset, test_images, test_masks
import numpy as np

class VGGBlock(tf.keras.layers.Layer):
    def __init__(self, filters, layers):
        super(VGGBlock, self).__init__()
        self.conv_layers = [
            tf.keras.layers.Conv2D(filters, 3, padding='same', activation='relu')
            for _ in range(layers)
        ]
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

    def call(self, x):
        for conv in self.conv_layers:
            x = conv(x)
        return self.pool(x), x  # pooled output and last conv output (for later use in skip connections)


class FCN16s(tf.keras.Model):
    def __init__(self, num_classes):
        super(FCN16s, self).__init__()

        self.block1 = VGGBlock(64, 2)
        self.block2 = VGGBlock(128, 2)
        self.block3 = VGGBlock(256, 4)
        self.block4 = VGGBlock(512, 4)
        self.block5 = VGGBlock(512, 4)

        self.conv6 = tf.keras.layers.Conv2D(4096, 7, padding='same', activation='relu')
        self.conv7 = tf.keras.layers.Conv2D(4096, 1, activation='relu')

        self.score5 = tf.keras.layers.Conv2D(num_classes, 1)
        self.score4 = tf.keras.layers.Conv2D(num_classes, 1)

        self.upsample_2x = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')   # 6 → 12
        self.upsample_16x = tf.keras.layers.UpSampling2D(size=16, interpolation='bilinear') # 12 → 192

        self.final_activation = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs):
        x1_out, _ = self.block1(inputs)        # 192 → 96
        x2_out, _ = self.block2(x1_out)        # 96 → 48
        x3_out, _ = self.block3(x2_out)        # 48 → 24
        x4_out, feat4 = self.block4(x3_out)    # 24 → 12
        x5_out, _ = self.block5(x4_out)        # 12 → 6

        x = self.conv6(x5_out)
        x = self.conv7(x)
        s5 = self.score5(x)                   # 6×6 → num_classes
        s4 = self.score4(x4_out)               # 12×12 → num_classes

        s5_up = self.upsample_2x(s5)          # 6 → 12
        merged = tf.add(s5_up, s4)            # 12×12 skip connections

        final = self.upsample_16x(merged)     # 12 → 192

        return self.final_activation(final)

    def build_model(self):
        inputs = tf.keras.Input(shape=(192, 192, 3))

        # forward pass
        x1_out, _ = self.block1(inputs)
        x2_out, _ = self.block2(x1_out)
        x3_out, _ = self.block3(x2_out)
        x4_out, feat4 = self.block4(x3_out)
        x5_out, _ = self.block5(x4_out)

        x = self.conv6(x5_out)
        x = self.conv7(x)
        s5 = self.score5(x)
        s4 = self.score4(x4_out)

        s5_up = self.upsample_2x(s5)
        merged = tf.keras.layers.Add()([s5_up, s4]) # skip connection
        final = self.upsample_16x(merged)

        output = self.final_activation(final)

        return tf.keras.Model(inputs=inputs, outputs=output)


def iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # flatten the tensors
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    # compute the intersection and union
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    return (intersection + smooth) / (union + smooth)



fcn = FCN16s(1)

fcn.compile(optimizer = tf.keras.optimizers.Adam(1e-4),  
                      loss = 'binary_crossentropy',
                      metrics=['accuracy', iou])


fcn.build_model()
fcn.summary()



history = fcn.fit(train_dataset,
                       epochs = 50,
                       validation_data = val_dataset)


fcn.save_weights('fcn_model.weights.h5')

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

hist.to_csv('fcn_model_history.csv')

hist.tail(1)

def plot_history(hist):
    plt.figure(figsize=(12, 10))
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('IoU',fontsize=20)
    plt.plot(hist['epoch'], hist['iou'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_iou'], label = 'Val Error')
    plt.legend(["train", "validation"], loc="upper left", prop={'size': 20})

plot_history(hist)


fcn.load_weights('fcn_model.weights.h5')


test_preds = fcn.predict(test_dataset)
pred_masks = (test_preds >= 0.5).astype(np.uint8)


accuracy = np.mean([np.mean(mask == test_mask) for mask, test_mask in zip(pred_masks, test_masks)])
ious = np.mean([iou(mask, test_mask) for mask, test_mask in zip(pred_masks, test_masks)])
print(ious)


i = 2
plt.imshow(pred_masks[i])
plt.show()
plt.imshow(test_images[i]) 
plt.show()
plt.imshow(test_masks[i])
plt.show()


history = pd.read_csv('fcn_model_history.csv')
history.tail(1)