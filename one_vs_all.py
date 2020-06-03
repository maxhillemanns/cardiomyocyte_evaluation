# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:07:55 2020

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
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

file_location = "../data/Ground Truth/"
image_location = "../data/Images/"
save_location = "../results/"

task_flag = 1
in_shape = [1024,1024, 1]
batches = 4
ep = 500

if path.isdir(save_location+"onevsall/sarc_data/") == False:
    os.mkdir(save_location+"onevsall/sarc_data/")
if path.isdir(save_location+"onevsall/dir_data/") == False:
    os.mkdir(save_location+"onevsall/dir_data/")
if path.isdir(save_location+"onevsall/cell_diff/") == False:
    os.mkdir(save_location+"onevsall/cell_diff/")

my_loss = losses.binary_crossentropy
my_metrics = [metrics.mae, metrics.binary_accuracy]
my_opt = optimizers.adam(lr = 0.00001, decay=0.0000001)

kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
bias_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), \
                 data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), \
                 kernel_initializer=kernel_init, bias_initializer=bias_init, \
                 activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(SpatialDropout2D(rate=0.7))

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu", \
                 kernel_initializer=kernel_init, bias_initializer=bias_init))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(SpatialDropout2D(rate=0.5))

model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu", \
                 kernel_initializer=kernel_init, bias_initializer=bias_init))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(SpatialDropout2D(rate=0.4))


model.add(Flatten())

model.add(Dense(units=100,activation="relu",  kernel_initializer=kernel_init, bias_initializer=bias_init))
model.add(Dropout(rate=0.2))

model.add(Dense(units=20, activation="relu",  kernel_initializer=kernel_init, bias_initializer=bias_init))

model.add(Dense(units=1, activation="sigmoid", kernel_initializer=kernel_init, bias_initializer=bias_init))
    
model.summary()

model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)

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

images, images_test, labels, labels_test = train_test_split(images, labels, test_size=0.3)

if task_flag == 0 or task_flag == 1:
    
    images_1 = images[labels[:,1]==1]
    labels_1 = labels[labels[:,1]==1]
    images_n1 = images[labels[:,1]==0]
    labels_n1 = labels[labels[:,1]==0]
    
    images_set_1 = np.concatenate((images_1, images_n1))
    labels_set_1 = np.concatenate((labels_1, labels_n1))
    
    labels_set_bin_1 = np.zeros((labels_set_1.shape[0],1))
    
    for i in range(len(labels_set_bin_1)):
    
        labels_set_bin_1[i] = labels_set_1[i].argmax()
        
        if labels_set_bin_1[i] == 1:
            labels_set_bin_1[i] = 1
        else:
            labels_set_bin_1[i] = 0
            
    labels_set_1 = labels_set_bin_1
    
    model_1 = model
    
    train_hist_1 = model_1.fit(x=images_set_1,y=labels_set_1,batch_size=batches,epochs=ep,validation_split=0.3,shuffle=True)
    
    labels_pred_1 = model_1.predict(x=images_test,batch_size=batches)
    
    model_json = model_1.to_json()
    if task_flag == 0:
        with open(save_location+"onevsall/sarc_data/sarc_model_1.json", "w") as json_file:
            json_file.write(model_json)
    elif task_flag == 1:
        with open(save_location+"onevsall/dir_data/dir_model_1.json", "w") as json_file:
            json_file.write(model_json)
            
    if task_flag == 0:
        model_1.save_weights(save_location+"onevsall/sarc_data/sarc_model_1.h5")
    elif task_flag == 1:
        model_1.save_weights(save_location+"onevsall/dir_data/dir_model_1.h5")
        
    hist_df = pd.DataFrame(train_hist_1.history)    
    
    if task_flag == 0:
        hist_df.to_csv(save_location+"onevsall/sarc_data/sarc_train_1.csv")
    elif task_flag == 1:
        hist_df.to_csv(save_location+"onevsall/dir_data/dir_train_1.csv")
    
    images_2 = images[labels[:,2]==1]
    labels_2 = labels[labels[:,2]==1]
    images_n2 = images[labels[:,2]==0]
    labels_n2 = labels[labels[:,2]==0]
    
    images_set_2 = np.concatenate((images_2, images_n2))
    labels_set_2 = np.concatenate((labels_2, labels_n2))
    
    labels_set_bin_2 = np.zeros((labels_set_2.shape[0],1))
    
    for i in range(len(labels_set_bin_2)):
    
        labels_set_bin_2[i] = labels_set_2[i].argmax()
        
        if labels_set_bin_2[i] == 2:
            labels_set_bin_2[i] = 1
        else:
            labels_set_bin_2[i] = 0
            
    labels_set_2 = labels_set_bin_2
    
    model_2 = model
    
    train_hist_2 = model_2.fit(x=images_set_2,y=labels_set_2,batch_size=batches,epochs=ep,validation_split=0.3,shuffle=True)
    
    labels_pred_2 = model_2.predict(x=images_test,batch_size=batches)
    
    model_json = model_2.to_json()
    if task_flag == 0:
        with open(save_location+"onevsall/sarc_data/sarc_model_2.json", "w") as json_file:
            json_file.write(model_json)
    elif task_flag == 1:
        with open(save_location+"onevsall/dir_data/dir_model_2.json", "w") as json_file:
            json_file.write(model_json)
            
    if task_flag == 0:
        model_2.save_weights(save_location+"onevsall/sarc_data/sarc_model_2.h5")
    elif task_flag == 1:
        model_2.save_weights(save_location+"onevsall/dir_data/dir_model_2.h5")
        
    hist_df = pd.DataFrame(train_hist_2.history)    
    
    if task_flag == 0:
        hist_df.to_csv(save_location+"onevsall/sarc_data/sarc_train_2.csv")
    elif task_flag == 1:
        hist_df.to_csv(save_location+"onevsall/dir_data/dir_train_2.csv")
    
    images_3 = images[labels[:,3]==1]
    labels_3 = labels[labels[:,3]==1]
    images_n3 = images[labels[:,3]==0]
    labels_n3 = labels[labels[:,3]==0]
    
    images_set_3 = np.concatenate((images_3, images_n3))
    labels_set_3 = np.concatenate((labels_3, labels_n3))
    
    labels_set_bin_3 = np.zeros((labels_set_3.shape[0],1))
    
    for i in range(len(labels_set_bin_3)):
    
        labels_set_bin_3[i] = labels_set_3[i].argmax()
        
        if labels_set_bin_3[i] == 3:
            labels_set_bin_3[i] = 1
        else:
            labels_set_bin_3[i] = 0
            
    labels_set_3 = labels_set_bin_3
    
    model_3 = model
    
    train_hist_3 = model_3.fit(x=images_set_3,y=labels_set_3,batch_size=batches,epochs=ep,validation_split=0.3,shuffle=True)
    
    labels_pred_3 = model_3.predict(x=images_test,batch_size=batches)
    
    model_json = model_3.to_json()
    if task_flag == 0:
        with open(save_location+"onevsall/sarc_data/sarc_model_3.json", "w") as json_file:
            json_file.write(model_json)
    elif task_flag == 1:
        with open(save_location+"onevsall/dir_data/dir_model_3.json", "w") as json_file:
            json_file.write(model_json)
            
    if task_flag == 0:
        model_3.save_weights(save_location+"onevsall/sarc_data/sarc_model_3.h5")
    elif task_flag == 1:
        model_3.save_weights(save_location+"onevsall/dir_data/dir_model_3.h5")
        
    hist_df = pd.DataFrame(train_hist_3.history)    
    
    if task_flag == 0:
        hist_df.to_csv(save_location+"onevsall/sarc_data/sarc_train_3.csv")
    elif task_flag == 1:
        hist_df.to_csv(save_location+"onevsall/dir_data/dir_train_3.csv")

    images_4 = images[labels[:,4]==1]
    labels_4 = labels[labels[:,4]==1]
    images_n4 = images[labels[:,4]==0]
    labels_n4 = labels[labels[:,4]==0]
    
    images_set_4 = np.concatenate((images_4, images_n4))
    labels_set_4 = np.concatenate((labels_4, labels_n4))
    
    labels_set_bin_4 = np.zeros((labels_set_4.shape[0],1))
    
    for i in range(len(labels_set_bin_4)):
    
        labels_set_bin_4[i] = labels_set_4[i].argmax()
        
        if labels_set_bin_4[i] == 4:
            labels_set_bin_4[i] = 1
        else:
            labels_set_bin_4[i] = 0
            
    labels_set_4 = labels_set_bin_4
    
    model_4 = model
    
    train_hist_4 = model_4.fit(x=images_set_4,y=labels_set_4,batch_size=batches,epochs=ep,validation_split=0.3,shuffle=True)
    
    labels_pred_4 = model_4.predict(x=images_test,batch_size=batches)
    
    model_json = model_4.to_json()
    if task_flag == 0:
        with open(save_location+"onevsall/sarc_data/sarc_model_4.json", "w") as json_file:
            json_file.write(model_json)
    elif task_flag == 1:
        with open(save_location+"onevsall/dir_data/dir_model_4.json", "w") as json_file:
            json_file.write(model_json)
            
    if task_flag == 0:
        model_4.save_weights(save_location+"onevsall/sarc_data/sarc_model_4.h5")
    elif task_flag == 1:
        model_4.save_weights(save_location+"onevsall/dir_data/dir_model_4.h5")
        
    hist_df = pd.DataFrame(train_hist_4.history)    
    
    if task_flag == 0:
        hist_df.to_csv(save_location+"onevsall/sarc_data/sarc_train_4.csv")
    elif task_flag == 1:
        hist_df.to_csv(save_location+"onevsall/dir_data/dir_train_4.csv")
    
    labels_pred = np.concatenate((np.zeros_like(labels_pred_1), labels_pred_1, labels_pred_2, labels_pred_3, labels_pred_4), axis=1)
    
    labels_pred = np.transpose(np.argmax(labels_pred, axis=1))
    
elif task_flag == 2:
    
    images_adult = images[labels[:,1]==1]
    labels_adult = labels[labels[:,1]==1]
    images_nadult = images[labels[:,1]==0]
    labels_nadult = labels[labels[:,1]==0]
    
    images_set_adult = np.concatenate((images_adult, images_nadult))
    labels_set_adult = np.concatenate((labels_adult, labels_nadult))
    
    labels_set_bin_adult = np.zeros((labels_set_adult.shape[0],1))
    
    for i in range(len(labels_set_bin_adult)):
    
        labels_set_bin_adult[i] = labels_set_adult[i].argmax()
        
        if labels_set_bin_adult[i] == 1:
            labels_set_bin_adult[i] = 1
        else:
            labels_set_bin_adult[i] = 0
            
    labels_set_adult = labels_set_bin_adult
    
    model_adult = model
    
    train_hist_adult = model_adult.fit(x=images_set_adult,y=labels_set_adult,batch_size=batches,epochs=ep,validation_split=0.3,shuffle=True)
    
    labels_pred_adult = model_adult.predict(x=images_test,batch_size=batches)
    
    model_json = model_adult.to_json()
    with open(save_location+"onevsall/cell_diff/cell_diff_adult.json", "w") as json_file:
        json_file.write(model_json)
            
    model_adult.save_weights(save_location+"onevsall/cell_diff/cell_diff_adult.h5")
        
    hist_df = pd.DataFrame(train_hist_adult.history)    
    
    hist_df.to_csv(save_location+"onevsall/cell_diff/cell_train_adult.csv")
    
    images_cor4u = images[labels[:,2]==1]
    labels_cor4u = labels[labels[:,2]==1]
    images_ncor4u = images[labels[:,2]==0]
    labels_ncor4u = labels[labels[:,2]==0]
    
    images_set_cor4u = np.concatenate((images_cor4u, images_ncor4u))
    labels_set_cor4u = np.concatenate((labels_cor4u, labels_ncor4u))
    
    labels_set_bin_cor4u = np.zeros((labels_set_cor4u.shape[0],1))
    
    for i in range(len(labels_set_bin_cor4u)):
    
        labels_set_bin_cor4u[i] = labels_set_cor4u[i].argmax()
        
        if labels_set_bin_cor4u[i] == 2:
            labels_set_bin_cor4u[i] = 1
        else:
            labels_set_bin_cor4u[i] = 0
            
    labels_set_cor4u = labels_set_bin_cor4u
    
    model_cor4u = model
    
    train_hist_cor4u = model_cor4u.fit(x=images_set_cor4u,y=labels_set_cor4u,batch_size=batches,epochs=ep,validation_split=0.3,shuffle=True)
    
    labels_pred_cor4u = model_cor4u.predict(x=images_test,batch_size=batches)
    
    model_json = model_cor4u.to_json()
    with open(save_location+"onevsall/cell_diff/cell_diff_cor4u.json", "w") as json_file:
        json_file.write(model_json)
            
    model_cor4u.save_weights(save_location+"onevsall/cell_diff/cell_diff_cor4u.h5")
        
    hist_df = pd.DataFrame(train_hist_cor4u.history)    
    
    hist_df.to_csv(save_location+"onevsall/cell_diff/cell_train_cor4u.csv")
    
    images_ipsc = images[labels[:,3]==1]
    labels_ipsc = labels[labels[:,3]==1]
    images_nipsc = images[labels[:,3]==0]
    labels_nipsc = labels[labels[:,3]==0]
    
    images_set_ipsc = np.concatenate((images_ipsc, images_nipsc))
    labels_set_ipsc = np.concatenate((labels_ipsc, labels_nipsc))
    
    labels_set_bin_ipsc = np.zeros((labels_set_ipsc.shape[0],1))
    
    for i in range(len(labels_set_bin_ipsc)):
    
        labels_set_bin_ipsc[i] = labels_set_ipsc[i].argmax()
        
        if labels_set_bin_ipsc[i] == 3:
            labels_set_bin_ipsc[i] = 1
        else:
            labels_set_bin_ipsc[i] = 0
            
    labels_set_ipsc = labels_set_bin_ipsc
    
    model_ipsc = model
    
    train_hist_ipsc = model_ipsc.fit(x=images_set_ipsc,y=labels_set_ipsc,batch_size=batches,epochs=ep,validation_split=0.3,shuffle=True)
    
    labels_pred_ipsc = model_ipsc.predict(x=images_test,batch_size=batches)
    
    model_json = model_ipsc.to_json()
    with open(save_location+"onevsall/cell_diff/cell_diff_ipsc.json", "w") as json_file:
        json_file.write(model_json)
            
    model_ipsc.save_weights(save_location+"onevsall/cell_diff/cell_diff_ipsc.h5")
        
    hist_df = pd.DataFrame(train_hist_ipsc.history)    
    
    hist_df.to_csv(save_location+"onevsall/cell_diff/cell_train_ipsc.csv")
    
    images_neonatal = images[labels[:,4]==1]
    labels_neonatal = labels[labels[:,4]==1]
    images_nneonatal = images[labels[:,4]==0]
    labels_nneonatal = labels[labels[:,4]==0]
    
    images_set_neonatal = np.concatenate((images_neonatal, images_nneonatal))
    labels_set_neonatal = np.concatenate((labels_neonatal, labels_nneonatal))
    
    labels_set_bin_neonatal = np.zeros((labels_set_neonatal.shape[0],1))
    
    for i in range(len(labels_set_bin_neonatal)):
    
        labels_set_bin_neonatal[i] = labels_set_neonatal[i].argmax()
        
        if labels_set_bin_neonatal[i] == 4:
            labels_set_bin_neonatal[i] = 1
        else:
            labels_set_bin_neonatal[i] = 0
            
    labels_set_neonatal = labels_set_bin_neonatal
    
    model_neonatal = model
    
    train_hist_neonatal = model_neonatal.fit(x=images_set_neonatal,y=labels_set_neonatal,batch_size=batches,epochs=ep,validation_split=0.3,shuffle=True)
    
    labels_pred_neonatal = model_neonatal.predict(x=images_test,batch_size=batches)
    
    model_json = model_neonatal.to_json()
    with open(save_location+"onevsall/cell_diff/cell_diff_neonatal.json", "w") as json_file:
        json_file.write(model_json)
            
    model_neonatal.save_weights(save_location+"onevsall/cell_diff/cell_diff_neonatal.h5")
        
    hist_df = pd.DataFrame(train_hist_neonatal.history)    
    
    hist_df.to_csv(save_location+"onevsall/cell_diff/cell_train_neonatal.csv")
    
    images_w4esc = images[labels[:,5]==1]
    labels_w4esc = labels[labels[:,5]==1]
    images_nw4esc = images[labels[:,5]==0]
    labels_nw4esc = labels[labels[:,5]==0]
    
    images_set_w4esc = np.concatenate((images_w4esc, images_nw4esc))
    labels_set_w4esc = np.concatenate((labels_w4esc, labels_nw4esc))
    
    labels_set_bin_w4esc = np.zeros((labels_set_w4esc.shape[0],1))
    
    for i in range(len(labels_set_bin_w4esc)):
    
        labels_set_bin_w4esc[i] = labels_set_w4esc[i].argmax()
        
        if labels_set_bin_w4esc[i] == 5:
            labels_set_bin_w4esc[i] = 1
        else:
            labels_set_bin_w4esc[i] = 0
            
    labels_set_w4esc = labels_set_bin_w4esc
    
    model_w4esc = model
    
    train_hist_w4esc = model_w4esc.fit(x=images_set_w4esc,y=labels_set_w4esc,batch_size=batches,epochs=ep,validation_split=0.3,shuffle=True)
    
    labels_pred_w4esc = model_w4esc.predict(x=images_test,batch_size=batches)
    
    model_json = model_w4esc.to_json()
    with open(save_location+"onevsall/cell_diff/cell_diff_w4esc.json", "w") as json_file:
        json_file.write(model_json)
            
    model_w4esc.save_weights(save_location+"onevsall/cell_diff/cell_diff_w4esc.h5")
        
    hist_df = pd.DataFrame(train_hist_w4esc.history)    
    
    hist_df.to_csv(save_location+"onevsall/cell_diff/cell_train_w4esc.csv")
    
    labels_pred = np.concatenate((np.zeros_like(labels_pred_adult), labels_pred_adult, labels_pred_cor4u, labels_pred_ipsc, labels_pred_neonatal, labels_pred_w4esc), axis=1)
    
    labels_pred = np.transpose(np.argmax(labels_pred, axis=1))
    
labels_true = np.transpose(np.argmax(labels_test, axis=1))

conf_matrix = confusion_matrix(labels_true, labels_pred)

if task_flag == 0:
    np.save(save_location+"onevsall/sarc_data/sarc_conf_matrix", conf_matrix)
elif task_flag == 1:
    np.save(save_location+"onevsall/dir_data/dir_conf_matrix", conf_matrix)
elif task_flag == 2:
    np.save(save_location+"onevsall/cell_diff/cell_diff_conf_matrix", conf_matrix)
