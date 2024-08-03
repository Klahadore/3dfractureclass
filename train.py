import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, concatenate, UpSampling3D, BatchNormalization, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU
from einops.layers.tensorflow import Reduce


from TF_GroupUnet3d import group_unet_model
from dataloader import imageLoader
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def train_model():
    train_img_dir = '/Users/quanhuynh/Desktop/data/train/images'
    train_mask_dir = '/Users/quanhuynh/Desktop/data/train/masks'
    train_img_list = os.listdir(train_img_dir) # List of image filenames
    train_mask_list = os.listdir(train_mask_dir)  # List of mask filenames
    
    val_img_dir = '/Users/quanhuynh/Desktop/data/val/images'
    val_mask_dir = '/Users/quanhuynh/Desktop/data/val/masks'
    val_img_list = os.listdir(val_img_dir)
    val_mask_list = os.listdir(val_mask_dir)

    
    batch_size = 1

    # Create data generators
    train_gen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, batch_size)
    val_gen = imageLoader(val_img_dir, val_img_list, val_mask_dir, val_mask_list, batch_size)
    # Number of training samples
    
    with tf.device("/gpu:0"):   
        model = group_unet_model(128, 128, 128, 3, 4)
        
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[MeanIoU(num_classes=4)])
        print(model.summary())
        
        steps_per_epoch = len(train_img_list) // batch_size

        # Train the model
        model.fit(
            train_gen, 
            steps_per_epoch=steps_per_epoch, 
            epochs=20, 
            verbose=1, 
            validation_data=val_gen, 
            validation_steps=len(val_img_list)//batch_size
            )

if __name__ == "__main__":
    train_model()