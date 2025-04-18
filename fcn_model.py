import tensorflow as tf
import numpy as np
import pandas as pd 
import os
import seaborn as sns
import matplotlib.pyplot as plt
import random
import time


# filters = channels

class VGGBlock(tf.keras.layers.Layer):

    def __init__(self, filters, layers):
        super(VGGBlock, self).__init__()
        self.filters = filters
        self.layers = layers
        self.conv_layers = [tf.keras.layers.Conv2D(filters, 3,
                                                    padding = 'same',
                                                      activation = 'relu') for _ in range(layers)]
        self.max_pool = tf.keras.layers.MaxPooling2D(2, strides = 2)

    def call(self, inputs):
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)
        x = self.max_pool(x)
        return x
    

# VGG Architecture
class VGG19(tf.keras.Model):

    def __init__(self, num_classes, input_shape):
        super(VGG19, self).__init__()

        self.num_classes = num_classes
        self.input_shape = input_shape

        self.block1 = VGGBlock(64, 2)
        self.block2 = VGGBlock(128, 2)
        self.block3 = VGGBlock(256, 4) # if 3 then vgg16
        self.block4 = VGGBlock(512, 4)
        self.block5 = VGGBlock(512, 4)

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(4096, activation = 'relu')
        self.fc2 = tf.keras.layers.Dense(4096, activation = 'relu')
        self.fc3 =  tf.keras.layers.Dense(num_classes, activation = 'softmax')

    def call(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


    def build_model(self):
        x = tf.keras.layers.Input(shape = self.input_shape)
        return tf.keras.Model(inputs = [x], outputs = self.call(x))


vgg = VGG19(10, (224, 224, 3))
vgg.build_model().summary()