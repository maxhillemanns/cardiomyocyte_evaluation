# -*- coding: utf-8 -*-
"""
Created on Mon May 18 21:19:45 2020

@author: User
"""

import pandas as pd
import numpy as np

from os import path
from DataPrep import data_prep, data_augmentation, split
from model_builder import build_3D_model
from info import print_info

from keras.utils import to_categorical
from keras.models import model_from_json
from keras import metrics, losses, optimizers, initializers, regularizers, constraints
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPooling3D, SpatialDropout3D, Dropout, BatchNormalization, LeakyReLU, SpatialDropout2D
from keras.layers.core import Flatten
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

task_flag = 1
in_shape = [512, 512, 32, 1]
batches = 1
ep = 500

my_loss = losses.categorical_crossentropy
my_metrics = [metrics.mae, metrics.categorical_accuracy]
my_opt = optimizers.adam(lr = 0.00001, decay=0.00000001)

save_location = "../results/"

if task_flag == 0:
    
    json_file = open(save_location+"testall_3D/sarc_data/sarc_model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(save_location+"testall_3D/sarc_data/sarc_model.h5")
    images_test = np.load(save_location+"testall_3D/sarc_data/images_test.npy")
    labels_test = np.load(save_location+"testall_3D/sarc_data/labels_test.npy")
    
elif task_flag == 1:
    
    json_file = open(save_location+"testall_3D/dir_data/dir_model.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(save_location+"testall_3D/dir_data/dir_model.h5")
    images_test = np.load(save_location+"testall_3D/dir_data/images_test.npy")
    labels_test = np.load(save_location+"testall_3D/dir_data/labels_test.npy")
    
elif task_flag == 2:
    
    json_file = open(save_location+"testall_3D/cell_diff/cell_diff.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(save_location+"testall_3D/cell_diff/cell_diff.h5")
    images_test = np.load(save_location+"testall_3D/cell_diff/images_test.npy")
    labels_test = np.load(save_location+"testall_3D/cell_diff/labels_test.npy")
    
model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)

labels_pred = model.predict(x=images_test,batch_size=batches)

scores = model.evaluate(images_test, labels_test, batch_size=batches)

labels_pred = np.transpose(np.argmax(labels_pred, axis=1))

labels_test = np.transpose(np.argmax(labels_test, axis=1))

conf_matrix = confusion_matrix(labels_test, labels_pred)

scores_df = pd.DataFrame(scores)

if task_flag == 0:
    scores_df.to_csv(save_location+"testall_3D/sarc_data/sarc_test.csv")
elif task_flag == 1:
    scores_df.to_csv(save_location+"testall_3D/dir_data/dir_test.csv")
elif task_flag == 2:
    scores_df.to_csv(save_location+"testall_3D/cell_diff/cell_test.csv")
    
if task_flag == 0:
    np.save(save_location+"testall_3D/sarc_data/sarc_conf_matrix", conf_matrix)
elif task_flag == 1:
    np.save(save_location+"testall_3D/dir_data/dir_conf_matrix", conf_matrix)
elif task_flag == 2:
    np.save(save_location+"testall_3D/cell_diff/cell_diff_conf_matrix", conf_matrix)