# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 10:24:39 2020

@author: User
"""

import keras
from keras import metrics, losses, optimizers
from keras.models import model_from_json
from keras_explain.integrated_gradients import IntegratedGrad
from keras_explain.grad_cam import GradCam
from LRP_funcs import LRP

import numpy as np
import os
import random

from skimage import io
from skimage.transform import resize

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

task_flag = 2
in_shape = [1024,1024, 1]

model_loc = "../results/"
image_loc = "../data/Images/"

my_loss = losses.categorical_crossentropy
my_metrics = [metrics.mae, metrics.categorical_accuracy]
my_opt = optimizers.adam(lr = 0.00001, decay=0.0000001)

if task_flag == 0:
    
    json_file = open(model_loc+'testall/sarc_data/sarc_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_loc+'testall/sarc_data/sarc_model.h5')
    
    save_loc = model_loc+'testall/sarc_data/'
    
    classes = ['NULL', '1', '2', '3', '4']
    
elif task_flag == 1:
    
    json_file = open(model_loc+'testall/dir_data/dir_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_loc+'testall/dir_data/dir_model.h5')
    
    save_loc = model_loc+'testall/dir_data/'
    
    classes = ['NULL', '1', '2', '3', '4']
    
elif task_flag == 2:
    
    json_file = open(model_loc+'testall/cell_diff/cell_diff.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_loc+'testall/cell_diff/cell_diff.h5')
    
    save_loc = model_loc+'testall/cell_diff/'
    
    classes = ['NULL', 'adult', 'cor4u', 'ipsc', 'neonatal', 'w4esc']
    
model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)

#LRP_explainer = LRP(model)
igrad_explainer = IntegratedGrad(model)
grad_cam_explainer = GradCam(model, layer=None)

### random image picker

#files = []

#
#for r,d,f in os.walk(image_loc):
#    for file in f:
#        if '.tif' in file and 'MAX' in file:
#            files.append(os.path.join(r, file))
        
selected_img = [image_loc+"adult/MAX_adult_116_1.tif", \
                image_loc+"cor4u/MAX_cor4u_9_2.tif", \
                image_loc+"IPSCs/MAX_ipsc_150_3.tif", \
                image_loc+"neonatal/MAX_neonatal_50_1.tif", \
                image_loc+"w4ESC/MAX_w4esc_9_1.tif" ]

  
number_of_examples = len(selected_img)

images = np.zeros((number_of_examples, in_shape[0], in_shape[1], in_shape[2]), dtype=np.uint8)    
          
for i in range(len(selected_img)):
    
    image = io.imread(selected_img[i])
    image = resize(image, in_shape)
    image *= 255/np.amax(image)
    image = image.reshape([1, in_shape[0], in_shape[1], in_shape[2]])
    image = image.astype(np.uint8)
    images[i,:,:,:] = image
    
### predict image

y = model.predict(images, batch_size=1)

### zeig her das lrp
    
for i in range(len(selected_img)):
#    lrp_exp = LRP_explainer.explain(images[i], y[i].argmax())
    igrad_exp = igrad_explainer.explain(images[i], y[i].argmax())
    igrad_exp = igrad_exp[0] * 255/np.amax(igrad_exp[0])
    grad_cam_exp = grad_cam_explainer.explain(images[i], y[i].argmax())
    grad_cam_exp = grad_cam_exp[0] * 255/np.amax(grad_cam_exp[0])
    
#    fig, a = plt.subplots(1,2)
#    a[0].imshow(images[i].reshape([in_shape[0], in_shape[1]]), cmap='gray', norm=Normalize())
#    a[0].axis('off')
#    a[0].set_title('original image: '+ selected_img[i].split('/')[-1][4:-4])
#    
#    a[1].imshow(lrp_exp[0], cmap='viridis', norm=Normalize())
#    a[1].axis('off')
#    a[1].set_title('LRP for class ' + classes[y[i].argmax()])
    
#    plt.savefig(save_loc+'LRP_'+selected_img[i].split('/')[-1][4:-4]+'.png')
    
    fig, a = plt.subplots(1,2)
    a[0].imshow(images[i].reshape([in_shape[0], in_shape[1]]), cmap='gray', norm=Normalize())
    a[0].axis('off')
    a[0].set_title('original image:\n')
    
    a[1].imshow(igrad_exp, cmap='viridis', norm=Normalize())
    a[1].axis('off')
    a[1].set_title('IGrad for class ' + classes[y[i].argmax()]+'\n')
    
    #a[1].imshow(grad_cam_exp, cmap='viridis', norm=Normalize())
    #a[1].axis('off')
    #a[1].set_title('Grad-CAM for class ' + classes[y[i].argmax()]+'\n')
    
    plt.savefig(save_loc+'igrad_exp_heatmap_'+selected_img[i].split('/')[-1][4:-4]+'.png')
    

