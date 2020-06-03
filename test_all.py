# -*- coding: utf-8 -*-
"""
Created on Fri May  1 17:32:38 2020

@author: User
"""

import pandas as pd
import numpy as np
import os

from os import path
from DataPrep_2D import data_prep_2D, data_augmentation_2D, split_2D
from model_builder import build_2D_model
from info import print_info

from keras.utils import to_categorical
from keras import metrics, losses, optimizers, initializers, regularizers, constraints
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, LeakyReLU, SpatialDropout2D
from keras.layers.core import Flatten
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

file_location = "../data/Ground Truth/"
image_location = "../data/Images/"
save_location = "../results/"

task_flag = 2
in_shape = [1024,1024, 1]
batches = 4
ep = 500

if path.isdir(save_location+"testall/sarc_data/") == False:
    os.mkdir(save_location+"testall/sarc_data/")
if path.isdir(save_location+"testall/dir_data/") == False:
    os.mkdir(save_location+"testall/dir_data/")
if path.isdir(save_location+"testall/cell_diff/") == False:
    os.mkdir(save_location+"testall/cell_diff/")

[data, max_dim] = data_prep_2D(file_location, image_location, task_flag, (in_shape[0], in_shape[1]))

images = data[:,2]
im_temp = np.zeros([len(images), in_shape[0], in_shape[1]], dtype=np.uint8)

for i in range(images.shape[0]):
    images[i] *= 255/np.amax(images[i])
    im_temp[i] = images[i]

images = im_temp.reshape(len(images),in_shape[0],in_shape[1],in_shape[2])
images = images.astype(np.float32)


labels = data[:,1]
if task_flag == 0 or task_flag == 1:
    l = np.zeros([len(labels),5])
    for i in range(labels.shape[0]):
        l[i] = to_categorical(labels[i], num_classes = 5)
elif task_flag == 2:
    l = np.zeros([len(labels),6])
    for i in range(labels.shape[0]):
        l[i] = to_categorical(labels[i], num_classes = 6)
        
labels = to_categorical(labels)
labels = l.astype(np.uint8)

images, labels = data_augmentation_2D(images, labels, 14, in_shape)

images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.3)

my_loss = losses.categorical_crossentropy
my_metrics = [metrics.mae, metrics.categorical_accuracy]
my_opt = optimizers.adam(lr = 0.00001, decay=0.00000001)

kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
bias_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), \
                 data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), \
                 kernel_initializer=kernel_init, bias_initializer=bias_init, \
                 activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(SpatialDropout2D(rate=0.6))

model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation="relu", \
                 kernel_initializer=kernel_init, bias_initializer=bias_init))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(SpatialDropout2D(rate=0.4))

model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), activation="relu", \
                 kernel_initializer=kernel_init, bias_initializer=bias_init))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(SpatialDropout2D(rate=0.2))

model.add(Flatten())

model.add(Dense(units=100,activation="relu",  kernel_initializer=kernel_init, bias_initializer=bias_init))
model.add(Dropout(rate=0.1))

model.add(Dense(units=20, activation="relu",  kernel_initializer=kernel_init, bias_initializer=bias_init))


if task_flag == 0 or task_flag == 1:
    model.add(Dense(units=5, activation="softmax", kernel_initializer=kernel_init, bias_initializer=bias_init))
elif task_flag == 2:
    model.add(Dense(units=6, activation="softmax", kernel_initializer=kernel_init, bias_initializer=bias_init))
    
model.summary()

model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)

train_history = model.fit(x=images_train,y=labels_train,batch_size=batches,epochs=ep,validation_split=0.3,shuffle=True)

labels_pred = model.predict(x=images_test,batch_size=batches)

scores = model.evaluate(images_test, labels_test)

labels_pred = np.transpose(np.argmax(labels_pred, axis=1))

labels_test = np.transpose(np.argmax(labels_test, axis=1))

conf_matrix = confusion_matrix(labels_test, labels_pred)

model_json = model.to_json()
if task_flag == 0:
    with open(save_location+"testall/sarc_data/sarc_model.json", "w") as json_file:
        json_file.write(model_json)
elif task_flag == 1:
    with open(save_location+"testall/dir_data/dir_model.json", "w") as json_file:
        json_file.write(model_json)
elif task_flag == 2:
    with open(save_location+"testall/cell_diff/cell_diff.json", "w") as json_file:
        json_file.write(model_json)
        
if task_flag == 0:
    model.save_weights(save_location+"testall/sarc_data/sarc_model.h5")
elif task_flag == 1:
    model.save_weights(save_location+"testall/dir_data/dir_model.h5")
elif task_flag == 2:
    model.save_weights(save_location+"testall/cell_diff/cell_diff.h5")
    
hist_df = pd.DataFrame(train_history.history)    

if task_flag == 0:
    hist_df.to_csv(save_location+"testall/sarc_data/sarc_train.csv")
elif task_flag == 1:
    hist_df.to_csv(save_location+"testall/dir_data/dir_train.csv")
elif task_flag == 2:
    hist_df.to_csv(save_location+"testall/cell_diff/cell_train.csv")
    
scores_df = pd.DataFrame(scores)

if task_flag == 0:
    scores_df.to_csv(save_location+"testall/sarc_data/sarc_test.csv")
elif task_flag == 1:
    scores_df.to_csv(save_location+"testall/dir_data/dir_test.csv")
elif task_flag == 2:
    scores_df.to_csv(save_location+"testall/cell_diff/cell_test.csv")
    
if task_flag == 0:
    np.save(save_location+"testall/sarc_data/sarc_conf_matrix", conf_matrix)
elif task_flag == 1:
    np.save(save_location+"testall/dir_data/dir_conf_matrix", conf_matrix)
elif task_flag == 2:
    np.save(save_location+"testall/cell_diff/cell_diff_conf_matrix", conf_matrix)