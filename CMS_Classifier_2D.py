# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:28:52 2020

@author: mhill
"""

import pandas as pd
import numpy as np
import os

from os import path
from DataPrep_2D import data_prep_2D, data_augmentation_2D, split_2D
from model_builder import build_2D_model
from info import print_info



from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

file_location = "../data/Ground Truth/"
image_location = "../data/Images/"
save_location = "../results/"

task_flag = 0
run = 2
in_shape = [1024,1024, 1]
status_flag = 0
batches = 4

print_info(task_flag, run, status_flag)

if path.isdir(save_location+"sarc_data/2D_Run "+str(run)+"/") == False:
    os.mkdir(save_location+"sarc_data/2D_Run "+str(run)+"/")
if path.isdir(save_location+"dir_data/2D_Run "+str(run)+"/") == False:
    os.mkdir(save_location+"dir_data/2D_Run "+str(run)+"/")
if path.isdir(save_location+"cell_diff/2D_Run "+str(run)+"/") == False:
    os.mkdir(save_location+"cell_diff/2D_Run "+str(run)+"/")


[data, max_dim] = data_prep_2D(file_location, image_location, task_flag, (in_shape[0], in_shape[1]))

status_flag += 1
print_info(task_flag, run, status_flag)

images = data[:,2]
im_temp = np.zeros([len(images), in_shape[0], in_shape[1]], dtype=np.uint8)

for i in range(images.shape[0]):
    images[i] *= 255/np.amax(images[i])
    im_temp[i] = images[i]

images = im_temp.reshape(len(images),in_shape[0],in_shape[1],in_shape[2])
images = images.astype(np.uint8)


labels = data[:,1]
if task_flag == 0 or task_flag == 1:
    l = np.zeros([len(labels),5])
    for i in range(labels.shape[0]):
        l[i] = to_categorical(labels[i], num_classes = 5)
elif task_flag == 2:
    l = np.zeros([len(labels),6])
    for i in range(labels.shape[0]):
        l[i] = to_categorical(labels[i], num_classes = 6)

labels = l.astype(np.uint8)

status_flag += 1
print_info(task_flag, run, status_flag)

#images, labels = data_augmentation_2D(images, labels, 14, in_shape)

status_flag += 1
print_info(task_flag, run, status_flag)

images_train, labels_train, images_test, labels_test = split_2D(images, labels, 0.7)

status_flag += 1
print_info(task_flag, run, status_flag)

model = build_2D_model(task_flag, run, in_shape)

status_flag += 1
print_info(task_flag, run, status_flag)

train_history = model.fit(x=images_train,y=labels_train,batch_size=batches,epochs=500,validation_split=0.3,shuffle=True)

status_flag += 1
print_info(task_flag, run, status_flag)

model_json = model.to_json()
if task_flag == 0:
    with open(save_location+"sarc_data/2D_Run "+str(run)+"/sarc_model_"+str(run)+".json", "w") as json_file:
        json_file.write(model_json)
elif task_flag == 1:
    with open(save_location+"dir_data/2D_Run "+str(run)+"/dir_model_"+str(run)+".json", "w") as json_file:
        json_file.write(model_json)
elif task_flag == 2:
    with open(save_location+"cell_diff/2D_Run "+str(run)+"/cell_diff_model_"+str(run)+".json", "w") as json_file:
        json_file.write(model_json)

if task_flag == 0:
    model.save_weights(save_location+"sarc_data/2D_Run "+str(run)+"/sarc_model_"+str(run)+".h5")
elif task_flag == 1:
    model.save_weights(save_location+"dir_data/2D_Run "+str(run)+"/dir_model_"+str(run)+".h5")
elif task_flag == 2:
    model.save_weights(save_location+"cell_diff/2D_Run "+str(run)+"/cell_diff_model_"+str(run)+".h5")

hist_df = pd.DataFrame(train_history.history)

if task_flag == 0:
    hist_df.to_csv(save_location+"sarc_data/2D_Run "+str(run)+"/sarc_training_metrics_"+str(run)+".csv")
elif task_flag == 1:
    hist_df.to_csv(save_location+"dir_data/2D_Run "+str(run)+"/dir_training_metrics_"+str(run)+".csv")
elif task_flag == 2:
    hist_df.to_csv(save_location+"cell_diff/2D_Run "+str(run)+"/cell_diff_training_metrics_"+str(run)+".csv")

status_flag += 1
print_info(task_flag, run, status_flag)

scores = model.evaluate(x=images_test,y=labels_test,batch_size=batches)

scores_df = pd.DataFrame(scores)

if task_flag == 0:
    scores_df.to_csv(save_location+"sarc_data/2D_Run "+str(run)+"/sarc_testing_metrics_"+str(run)+".csv")
elif task_flag == 1:
    scores_df.to_csv(save_location+"dir_data/2D_Run "+str(run)+"/dir_testing_metrics_"+str(run)+".csv")
elif task_flag == 2:
    scores_df.to_csv(save_location+"cell_diff/2D_Run "+str(run)+"/cell_diff_testing_metrics_"+str(run)+".csv")

print("%s: %.2f%%" % (model.metrics_names[2], scores[2]*100))

labels_pred = model.predict(x=images_test,batch_size=batches)

y_true = np.zeros((len(labels_test),1))
y_pred = np.zeros((len(labels_pred),1))

for i in range(len(labels_test)):
    y_true[i] = labels_test[i].argmax()
    y_pred[i] = labels_pred[i].argmax()

conf_matrix = confusion_matrix(y_true, y_pred)

if task_flag == 0:
    np.save(save_location+"sarc_data/2D_Run "+str(run)+"/sarc_conf_matrix_"+str(run), conf_matrix)
elif task_flag == 1:
    np.save(save_location+"dir_data/2D_Run "+str(run)+"/dir_conf_matrix_"+str(run), conf_matrix)
elif task_flag == 2:
    np.save(save_location+"cell_diff/2D_Run "+str(run)+"/cell_diff_conf_matrix_"+str(run), conf_matrix)
    
status_flag += 1
print_info(task_flag, run, status_flag)

status_flag += 1
print_info(task_flag, run, status_flag)