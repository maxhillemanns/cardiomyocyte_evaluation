# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:24:28 2020

@author: User
"""

import numpy as np

from keras.utils import to_categorical
from DataPrep_2D import data_prep_2D, data_augmentation_2D


file_location = "../data/Ground Truth/"
image_location = "../data/Images/"

task_flag = 1
in_shape = [1024,1024, 1]
number_of_examples = 1

files = []
images = np.zeros((number_of_examples, in_shape[0], in_shape[1], in_shape[2]), dtype=np.uint8)

for r,d,f in os.walk(image_loc):
    for file in f:
        if '.tif' in file and 'MAX' in file:
            files.append(os.path.join(r, file))
        
selected_img = random.choices(files, k=number_of_examples)

for i in range(len(selected_img)):
    
    image = io.imread(selected_img[i])
    image = resize(image, in_shape)
    image *= 255/np.amax(image)
    image = image.reshape([1, in_shape[0], in_shape[1], in_shape[2]])
    image = image.astype(np.uint8)
    images[i,:,:,:] = image
    
a[0].imshow(images[i].reshape([in_shape[0], in_shape[1]]), cmap='gray', norm=Normalize())
    a[0].axis('off')
    a[0].set_title('original image: '+ selected_img[i].split('/')[-1][4:-4])