
import keras
from keras import metrics, losses, optimizers
from keras.models import model_from_json
from keras_explain.integrated_gradients import IntegratedGrad
from keras_explain.grad_cam import GradCam

import numpy as np
import os
import random

from skimage import io
from skimage.transform import resize

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

task_flag = 0
in_shape = [1024,1024, 1]

number_of_examples = 4

model_loc = "../results/"
image_loc = "../data/Images/"

### random image picker

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

my_loss = losses.binary_crossentropy
my_metrics = [metrics.mae, metrics.binary_accuracy]
my_opt = optimizers.adam(lr = 0.00001, decay=0.0000001)

if task_flag == 0:
    
    json_file_1 = open(model_loc+'onevsall/sarc_data/sarc_model_1.json', 'r')
    loaded_model_json_1 = json_file_1.read()
    json_file_1.close()
    model_1 = model_from_json(loaded_model_json_1)
    model_1.load_weights(model_loc+'onevsall/sarc_data/sarc_model_1.h5')
    
    model_1.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    y_1 = model_1.predict(images)
    
    json_file_2 = open(model_loc+'onevsall/sarc_data/sarc_model_2.json', 'r')
    loaded_model_json_2 = json_file_2.read()
    json_file_2.close()
    model_2 = model_from_json(loaded_model_json_2)
    model_2.load_weights(model_loc+'onevsall/sarc_data/sarc_model_2.h5')
    
    model_2.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    y_2 = model_2.predict(images)
    
    json_file_3 = open(model_loc+'onevsall/sarc_data/sarc_model_3.json', 'r')
    loaded_model_json_3 = json_file_3.read()
    json_file_3.close()
    model_3 = model_from_json(loaded_model_json_3)
    model_3.load_weights(model_loc+'onevsall/sarc_data/sarc_model_3.h5')
    
    model_3.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    y_3 = model_3.predict(images)
    
    json_file_4 = open(model_loc+'onevsall/sarc_data/sarc_model_4.json', 'r')
    loaded_model_json_4 = json_file_4.read()
    json_file_4.close()
    model_4 = model_from_json(loaded_model_json_4)
    model_4.load_weights(model_loc+'onevsall/sarc_data/sarc_model_4.h5')
    
    model_4.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    y_4 = model_4.predict(images)
    
    y = np.concatenate((np.zeros_like(y_1), y_1, y_2, y_3, y_4), axis=1)
    y = np.transpose(np.argmax(y, axis=1))
    
    save_loc = model_loc+'onevsall/sarc_data/'
    
    classes = ['NULL', '1', '2', '3', '4']
    
    for i in range(len(selected_img)):
        
        if y[i] == 1:
            igrad_explainer = IntegratedGrad(model_1)
            grad_cam_explainer = GradCam(model_1, layer=None)
        elif y[i] == 2:
            igrad_explainer = IntegratedGrad(model_2)
            grad_cam_explainer = GradCam(model_2, layer=None)
        elif y[i] == 3:
            igrad_explainer = IntegratedGrad(model_3)
            grad_cam_explainer = GradCam(model_3, layer=None)
        elif y[i] == 4:
            igrad_explainer = IntegratedGrad(model_4)
            grad_cam_explainer = GradCam(model_4, layer=None)
        elif y[i] == 0:
            continue
        
        igrad_exp = igrad_explainer.explain(images[i], 1)
        igrad_exp = igrad_exp[0] * 255/np.amax(igrad_exp[0])
        grad_cam_exp = grad_cam_explainer.explain(images[i], 1)
        grad_cam_exp = grad_cam_exp[0] * 255/np.amax(grad_cam_exp[0])
        
        fig, a = plt.subplots(1,2)
        a[0].imshow(images[i].reshape([in_shape[0], in_shape[1]]), cmap='gray', norm=Normalize())
        a[0].axis('off')
        a[0].set_title('original image: '+ selected_img[i].split('/')[-1][4:-4])
        
        a[1].imshow(igrad_exp, cmap='viridis', norm=Normalize())
        a[1].axis('off')
        a[1].set_title('Integrated Gradients for class ' + classes[y])
        
        plt.savefig(save_loc+'igrad_'+selected_img[i].split('/')[-1][4:-4]+'.png') 
        
        fig, a = plt.subplots(1,2)
        a[0].imshow(images[i].reshape([in_shape[0], in_shape[1]]), cmap='gray', norm=Normalize())
        a[0].axis('off')
        a[0].set_title('original image: '+ selected_img[i].split('/')[-1][4:-4])
        
        a[1].imshow(grad_cam_exp, cmap='viridis', norm=Normalize())
        a[1].axis('off')
        a[1].set_title('Grad-CAM for class ' + classes[y])
        
        plt.savefig(save_loc+'grad_cam_'+selected_img[i].split('/')[-1][4:-4]+'.png')
    
elif task_flag == 1:
    
    json_file_1 = open(model_loc+'onevsall/dir_data/dir_model_1.json', 'r')
    loaded_model_json_1 = json_file_1.read()
    json_file_1.close()
    model_1 = model_from_json(loaded_model_json_1)
    model_1.load_weights(model_loc+'onevsall/dir_data/dir_model_1.h5')
    
    model_1.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    y_1 = model_1.predict(images)
    
    json_file_2 = open(model_loc+'onevsall/dir_data/dir_model_2.json', 'r')
    loaded_model_json_2 = json_file_2.read()
    json_file_2.close()
    model_2 = model_from_json(loaded_model_json_2)
    model_2.load_weights(model_loc+'onevsall/dir_data/dir_model_2.h5')
    
    model_2.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    y_2 = model_2.predict(images)
    
    json_file_3 = open(model_loc+'onevsall/dir_data/dir_model_3.json', 'r')
    loaded_model_json_3 = json_file_3.read()
    json_file_3.close()
    model_3 = model_from_json(loaded_model_json_3)
    model_3.load_weights(model_loc+'onevsall/dir_data/dir_model_3.h5')
    
    model_3.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    y_3 = model_3.predict(images)
    
    json_file_4 = open(model_loc+'onevsall/dir_data/dir_model_4.json', 'r')
    loaded_model_json_4 = json_file_4.read()
    json_file_4.close()
    model_4 = model_from_json(loaded_model_json_4)
    model_4.load_weights(model_loc+'onevsall/dir_data/dir_model_4.h5')
    
    model_4.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    y_4 = model_4.predict(images)
    
    y = np.concatenate((np.zeros_like(y_1), y_1, y_2, y_3, y_4), axis=1)
    y = np.transpose(np.argmax(y, axis=1))
    
    save_loc = model_loc+'onevsall/dir_data/'
    
    classes = ['NULL', '1', '2', '3', '4']
    
    for i in range(len(selected_img)):
        
        if y[i] == 1:
            igrad_explainer = IntegratedGrad(model_1)
            grad_cam_explainer = GradCam(model_1, layer=None)
        elif y[i] == 2:
            igrad_explainer = IntegratedGrad(model_2)
            grad_cam_explainer = GradCam(model_2, layer=None)
        elif y[i] == 3:
            igrad_explainer = IntegratedGrad(model_3)
            grad_cam_explainer = GradCam(model_3, layer=None)
        elif y[i] == 4:
            igrad_explainer = IntegratedGrad(model_4)
            grad_cam_explainer = GradCam(model_4, layer=None)
        elif y[i] == 0:
            continue
        
        igrad_exp = igrad_explainer.explain(images[i], 1)
        igrad_exp = igrad_exp[0] * 255/np.amax(igrad_exp[0])
        grad_cam_exp = grad_cam_explainer.explain(images[i], 1)
        grad_cam_exp = grad_cam_exp[0] * 255/np.amax(grad_cam_exp[0])
        
        fig, a = plt.subplots(1,2)
        a[0].imshow(images[i].reshape([in_shape[0], in_shape[1]]), cmap='gray', norm=Normalize())
        a[0].axis('off')
        a[0].set_title('original image: '+ selected_img[i].split('/')[-1][4:-4])
        
        a[1].imshow(igrad_exp, cmap='viridis', norm=Normalize())
        a[1].axis('off')
        a[1].set_title('Integrated Gradients for class ' + classes[y])
        
        plt.savefig(save_loc+'igrad_'+selected_img[i].split('/')[-1][4:-4]+'.png') 
        
        fig, a = plt.subplots(1,2)
        a[0].imshow(images[i].reshape([in_shape[0], in_shape[1]]), cmap='gray', norm=Normalize())
        a[0].axis('off')
        a[0].set_title('original image: '+ selected_img[i].split('/')[-1][4:-4])
        
        a[1].imshow(grad_cam_exp, cmap='viridis', norm=Normalize())
        a[1].axis('off')
        a[1].set_title('Grad-CAM for class ' + classes[y])
        
        plt.savefig(save_loc+'grad_cam_'+selected_img[i].split('/')[-1][4:-4]+'.png')
    
elif task_flag == 2:
    
    json_file_adult = open(model_loc+'onevsall/cell_diff/cell_diff_adult.json', 'r')
    loaded_model_json_adult = json_file_adult.read()
    json_file_adult.close()
    model_adult = model_from_json(loaded_model_json_adult)
    model_adult.load_weights(model_loc+'onevsall/cell_diff/cell_diff_adult.h5')
    
    model_adult.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    y_adult = model_adult.predict(images)
    
    json_file_cor4u = open(model_loc+'onevsall/cell_diff/cell_diff_cor4u.json', 'r')
    loaded_model_json_cor4u = json_file_cor4u.read()
    json_file_cor4u.close()
    model_cor4u = model_from_json(loaded_model_json_cor4u)
    model_cor4u.load_weights(model_loc+'onevsall/cell_diff/cell_diff_cor4u.h5')
    
    model_cor4u.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    y_cor4u = model_cor4u.predict(images)
    
    json_file_ipsc = open(model_loc+'onevsall/cell_diff/cell_diff_ipsc.json', 'r')
    loaded_model_json_ipsc = json_file_ipsc.read()
    json_file_ipsc.close()
    model_ipsc = model_from_json(loaded_model_json_ipsc)
    model_ipsc.load_weights(model_loc+'onevsall/cell_diff/cell_diff_ipsc.h5')
    
    model_ipsc.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    y_ipsc = model_ipsc.predict(images)
    
    json_file_neonatal = open(model_loc+'onevsall/cell_diff/cell_diff_neonatal.json', 'r')
    loaded_model_json_neonatal = json_file_neonatal.read()
    json_file_neonatal.close()
    model_neonatal = model_from_json(loaded_model_json_neonatal)
    model_neonatal.load_weights(model_loc+'onevsall/cell_diff/cell_diff_neonatal.h5')
    
    model_neonatal.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    y_neonatal = model_neonatal.predict(images)
    
    json_file_w4esc = open(model_loc+'onevsall/cell_diff/cell_diff_w4esc.json', 'r')
    loaded_model_json_w4esc = json_file_w4esc.read()
    json_file_w4esc.close()
    model_w4esc = model_from_json(loaded_model_json_w4esc)
    model_w4esc.load_weights(model_loc+'onevsall/cell_diff/cell_diff_w4esc.h5')
    
    model_w4esc.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    y_w4esc = model_w4esc.predict(images)
    
    y = np.concatenate((np.zeros_like(y_adult), y_adult, y_cor4u, y_ipsc, y_neonatal, y_w4esc), axis=1)
    y = np.transpose(np.argmax(y, axis=1))
    
    save_loc = model_loc+'onevsall/cell_diff/'
    
    classes = ['NULL', 'adult', 'cor4u', 'ipsc', 'neonatal', 'w4esc']
    
    for i in range(len(selected_img)):
        
        if y[i] == 1:
            igrad_explainer = IntegratedGrad(model_adult)
            grad_cam_explainer = GradCam(model_adult, layer=None)
        elif y[i] == 2:
            igrad_explainer = IntegratedGrad(model_cor4u)
            grad_cam_explainer = GradCam(model_cor4u, layer=None)
        elif y[i] == 3:
            igrad_explainer = IntegratedGrad(model_ipsc)
            grad_cam_explainer = GradCam(model_ipsc, layer=None)
        elif y[i] == 4:
            igrad_explainer = IntegratedGrad(model_neonatal)
            grad_cam_explainer = GradCam(model_neonatal, layer=None)
        elif y[i] == 5:
            igrad_explainer = IntegratedGrad(model_w4esc)
            grad_cam_explainer = GradCam(model_w4esc, layer=None)
        elif y[i] == 6:
            continue
        
        igrad_exp = igrad_explainer.explain(images[i], 1)
        igrad_exp = igrad_exp[0] * 255/np.amax(igrad_exp[0])
        grad_cam_exp = grad_cam_explainer.explain(images[i], 1)
        grad_cam_exp = grad_cam_exp[0] * 255/np.amax(grad_cam_exp[0])
        
        fig, a = plt.subplots(1,2)
        a[0].imshow(images[i].reshape([in_shape[0], in_shape[1]]), cmap='gray', norm=Normalize())
        a[0].axis('off')
        a[0].set_title('original image: '+ selected_img[i].split('/')[-1][4:-4])
        
        a[1].imshow(igrad_exp, cmap='viridis', norm=Normalize())
        a[1].axis('off')
        a[1].set_title('Integrated Gradients for class ' + classes[y])
        
        plt.savefig(save_loc+'igrad_'+selected_img[i].split('/')[-1][4:-4]+'.png') 
        
        fig, a = plt.subplots(1,2)
        a[0].imshow(images[i].reshape([in_shape[0], in_shape[1]]), cmap='gray', norm=Normalize())
        a[0].axis('off')
        a[0].set_title('original image: '+ selected_img[i].split('/')[-1][4:-4])
        
        a[1].imshow(grad_cam_exp, cmap='viridis', norm=Normalize())
        a[1].axis('off')
        a[1].set_title('Grad-CAM for class ' + classes[y])
        
        plt.savefig(save_loc+'grad_cam_'+selected_img[i].split('/')[-1][4:-4]+'.png')
  
