<<<<<<< HEAD
def load_img(img_dir, img_list):
    images=[]
    for i, image_name in enumerate(img_list):
        if (image_name.split('.')[1] == 'npy'):

            image = np.load(img_dir+image_name)
=======
import os
import numpy as np


def load_img(img_dir, img_list):
    images=[]
    for i, image_name in enumerate(img_list):    
        if (image_name.split('.')[1] == 'npy'):

            image = np.load(img_dir + "/" + image_name)
>>>>>>> d1a276aa77b48fe9525b11a8488dc5874ab1273b

            images.append(image)
    images = np.array(images)

    return(images)

<<<<<<< HEAD



=======
>>>>>>> d1a276aa77b48fe9525b11a8488dc5874ab1273b
def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):

    L = len(img_list)

<<<<<<< HEAD
    #keras needs the generator infinite, so we will use while true
=======
    #keras needs the generator infinite, so we will use while true  
>>>>>>> d1a276aa77b48fe9525b11a8488dc5874ab1273b
    while True:

        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

<<<<<<< HEAD
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples

            batch_start += batch_size
            batch_end += batch_size
=======
            yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

            batch_start += batch_size   
            batch_end += batch_size

if __name__ == '__main__':
    train_img_dir = "/Users/quanhuynh/Desktop/data/train/images"
    train_mask_dir = "/Users/quanhuynh/Desktop/data/train/masks"
    train_img_list=os.listdir(train_img_dir)
    train_mask_list = os.listdir(train_mask_dir)
    train_img_datagen = imageLoader(train_img_dir, train_img_list, train_mask_dir, train_mask_list, 3)

    img, msk = train_img_datagen.__next__()
    print(img.shape)
>>>>>>> d1a276aa77b48fe9525b11a8488dc5874ab1273b
