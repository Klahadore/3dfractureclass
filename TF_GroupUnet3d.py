
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
    def __init__(self, in_channels, out_channels, h_input, h_output, ksize=3, padding='SAME', activation='relu', **kwargs):
        super(GConv3D, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h_input = h_input
        self.h_output = h_output
        self.ksize = ksize
        self.padding = padding
        self.activation = activation
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
        conv_output = gconv3d(inputs, self.w, strides=[1, 1, 1, 1, 1], padding=self.padding,
                       gconv_indices=self.gconv_indices, gconv_shape_info=self.gconv_shape_info)
        if self.activation:
            conv_output = Activation(self.activation)(conv_output)
        return conv_output

def group_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs

    c1 = GConv3D(3, 16, "Z3", "OH")(s)
    c1 = Dropout(0.1)(c1)
    c1 = GConv3D(16, 16, "OH", "OH")(c1)
    print(c1.shape)
    p1 = MaxPooling3D((2,2,2))(c1)
    print(p1.shape)

    c2 = GConv3D(16,32,"OH", "OH")(p1)
    c2 = Dropout(0.1)(c2)
    c2 = GConv3D(32,32, "OH", "OH")(c2)
    p2 = MaxPooling3D((2,2,2))(c2)

    c3 = GConv3D(32, 64, "OH", "OH")(p2)
    c3 = Dropout(0.1)(c3)
    c3 = GConv3D(64, 64, "OH", "OH")(c3)
    p3 = MaxPooling3D((2,2,2))(c3)

    c4 = GConv3D(64, 128, "OH", "OH")(p3)
    c4 = Dropout(0.1)(c4)
    c4 = GConv3D(128, 128, "OH", "OH")(c4)
    p4 = MaxPooling3D((2,2,2))(c4)

    c5 = GConv3D(128, 256, "OH", "OH")(p4)
    c5 = Dropout(0.1)(c5)
    c5 = GConv3D(256, 256, "OH", "OH")(c5)


    c6 = GConv3D(256, 128, "OH", "OH")(c5)
    c6 = UpSampling3D((2,2,2))(c6)
    c6 = concatenate([c6, c4])
    c6 = Dropout(0.2)(c6)
    c6 = GConv3D(128, 128, "OH", "OH")(c6)

    print(c6.shape)
    c7 = GConv3D(128, 64, "OH", "OH")(c6)
    c7 = UpSampling3D((2,2,2))(c7)
    c7 = concatenate([c7, c3])
    c7 = Dropout(0.2)(c7)
    c7 = GConv3D(64, 64, "OH", "OH")(c7)

    c8 = GConv3D(64, 32, "OH", "OH")(c7)
    c8 = UpSampling3D((2,2,2))(c8)
    c8 = concatenate([c8,c2])
    c8 = Dropout(0.2)(c8)
    c8 = GConv3D(32,32, "OH", "OH")(c8)

    c9 = GConv3D(32, 16, "OH", "OH")(c8)
    c9 = UpSampling3D((2,2,2))(c9)
    c9 = concatenate([c9, c1])
    c9 = Dropout(0.2)(c9)
    c9 = GConv3D(16, 16, "OH", "OH")(c9)

    c9 = Reduce("b h w d (g c) -> b h w d c", g=48, reduction="sum")(c9)

    outputs = Conv3D(num_classes, (1,1,1), activation='softmax')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model

# model = group_unet_model(128,128,128,3,4)
# print(model.input_shape)
# print(model.output_shape)
