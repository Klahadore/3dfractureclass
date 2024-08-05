
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, concatenate, UpSampling3D, BatchNormalization, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from einops.layers.tensorflow import Reduce


import numpy as np


from groupy.gconv.tensorflow_gconv.splitgconv3d import gconv3d, gconv3d_util

kernel_initializer = 'he_uniform'

class GConv3D(tf.keras.layers.Layer):
        super(GConv3D, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h_input = h_input
        self.h_output = h_output
        self.ksize = ksize
        self.padding = padding
        self.activation = activation
        self.groups = 16

        # Initialize gconv3d utility
        self.gconv_indices, self.gconv_shape_info, self.w_shape = gconv3d_util(
            h_input=self.h_input, h_output=self.h_output, in_channels=self.in_channels,
            out_channels=self.out_channels, ksize=self.ksize)

    def build(self, input_shape):
        # Create the weights for the convolution
        self.w = self.add_weight(shape=self.w_shape,
                                 initializer=kernel_initializer,
                                 trainable=True,
                                 name='kernel')

    def call(self, inputs):
        # Perform the group convolution
        print(f"Input shape: {inputs.shape}, Expected in_channels: {self.in_channels}")
        conv_output = gconv3d(inputs, self.w, strides=[1, 1, 1, 1, 1], padding=self.padding,
                       gconv_indices=self.gconv_indices, gconv_shape_info=self.gconv_shape_info)
        if self.activation:
            conv_output = Activation(self.activation)(conv_output)
        print(f"Output shape: {conv_output.shape}, Produced out_channels: {self.out_channels}")
        return conv_output

def group_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs

    c1 = Dropout(0.1)(c1)
    print(c1.shape)
    p1 = MaxPooling3D((2,2,2))(c1)
    print(p1.shape)

    c2 = Dropout(0.1)(c2)
    p2 = MaxPooling3D((2,2,2))(c2)

    c3 = Dropout(0.1)(c3)
    p3 = MaxPooling3D((2,2,2))(c3)

    c4 = Dropout(0.1)(c4)
    p4 = MaxPooling3D((2,2,2))(c4)

    c5 = Dropout(0.1)(c5)


    c6 = UpSampling3D((2,2,2))(c6)
    c6 = Dropout(0.2)(c6)
    print(c6.shape)
    c7 = UpSampling3D((2,2,2))(c7)
    c7 = Dropout(0.2)(c7)

    c8 = UpSampling3D((2,2,2))(c8)
    c8 = Dropout(0.2)(c8)

    c9 = UpSampling3D((2,2,2))(c9)
    c9 = Dropout(0.2)(c9)


    outputs = Conv3D(num_classes, (1,1,1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

# model = group_unet_model(128,128,128,3,4)
# print(model.input_shape)
# print(model.output_shape)
