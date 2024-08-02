import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, concatenate, UpSampling3D, BatchNormalization, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from einops.layers.tensorflow import Reduce


from TF_GroupUnet3d import group_unet_model
from dataloader import imageLoader

def train_model():
    img_dir = './data/train/images/'
    mask_dir = './data/train/masks/'
    img_list = os.listdir(img_dir) # List of image filenames
    mask_list = os.listdir(mask_dir)  # List of mask filenames
    batch_size = 4

    # Create the model
    model = group_unet_model(128, 128, 128, 3, 4)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[MeanIoU(num_classes=4)])
    print(model.summary())
    # Create data generators
    train_gen = imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size)

    # Number of training samples
    steps_per_epoch = len(img_list) // batch_size

    # Train the model
    model.fit(
        train_gen, 
        steps_per_epoch=steps_per_epoch, 
        epochs=10, 
        verbose=1, 
        validation_data=val_img_datagen, 
        validation_steps=val_steps_per_epoch,)
