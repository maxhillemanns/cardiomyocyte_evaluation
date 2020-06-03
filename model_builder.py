# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 10:41:38 2020

@author: mhill
"""

import keras
from keras import metrics, losses, optimizers, initializers, regularizers, constraints
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, SpatialDropout2D, Dropout, Conv3D, MaxPooling3D, SpatialDropout3D, BatchNormalization, AveragePooling3D, LeakyReLU
from keras.layers.core import Flatten
#from keras.engine.input_layer import Input
from keras.applications import mobilenet, vgg16

def build_2D_model(task_flag, run, in_shape):
    
    if run == 0:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.001, momentum=0, nesterov=False)
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.3))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.3))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
    
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 1:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.3))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.3))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
    
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 2:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.3))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.3))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
    
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    
    elif run == 3:
    
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        model = Sequential()
    
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.3))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=16, kernel_size=(2,2), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.3))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0.3))
    
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    
    elif run == 4:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.3))
        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(SpatialDropout2D(0.3))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
    
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 5:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.Zeros()
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None, input_shape=(in_shape[0], in_shape[1], in_shape[2])))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Dropout(rate=0.5))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Dropout(rate=0.5))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Dropout(rate=0.5))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Dropout(rate=0.4))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Dropout(rate=0.4))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Dropout(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Dropout(rate=0.3))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Dropout(rate=0.3))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Dropout(rate=0.3))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(units=50, activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        model.add(Dropout(rate=0.2))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 6:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.Zeros()
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None, input_shape=(in_shape[0], in_shape[1], in_shape[2])))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Dropout(rate=0.5))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Dropout(rate=0.5))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Dropout(rate=0.5))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Dropout(rate=0.4))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Dropout(rate=0.4))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Dropout(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Dropout(rate=0.3))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Dropout(rate=0.3))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Dropout(rate=0.3))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(units=50, activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        model.add(Dropout(rate=0.2))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 7:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.Zeros()
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None, input_shape=(in_shape[0], in_shape[1], in_shape[2])))
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.3))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(units=50, activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        model.add(Dropout(rate=0.1))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 8:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.Zeros()
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None, input_shape=(in_shape[0], in_shape[1], in_shape[2])))
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.3))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(units=50, activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        model.add(Dropout(rate=0.1))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 9:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.Zeros()
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None, input_shape=(in_shape[0], in_shape[1], in_shape[2])))
        model.add(SpatialDropout2D(rate=0.8, data_format="channels_last"))
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.8, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.6, data_format="channels_last"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.6, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.3))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.3))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.3))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(units=50, activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        model.add(Dropout(rate=0.2))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 10:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.Zeros()
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None, input_shape=(in_shape[0], in_shape[1], in_shape[2])))
        model.add(SpatialDropout2D(rate=0.8, data_format="channels_last"))
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.8, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.6, data_format="channels_last"))
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.6, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.3))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.3))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.3))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(units=50, activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        model.add(Dropout(rate=0.2))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 11:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.Zeros()
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None, input_shape=(in_shape[0], in_shape[1], in_shape[2])))
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.3))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(units=50, activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        model.add(Dropout(rate=0.1))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 12:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.Zeros()
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None, input_shape=(in_shape[0], in_shape[1], in_shape[2])))
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.3))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(units=50, activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        model.add(Dropout(rate=0.1))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 13:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.Zeros()
        
        model = Sequential()
        
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None, input_shape=(in_shape[0], in_shape[1], in_shape[2])))
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.5, data_format="channels_last"))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.4))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", data_format="channels_last", dilation_rate=(1,1), \
                         activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                         bias_constraint=None))
        model.add(SpatialDropout2D(rate=0.3))
        model.add(MaxPooling2D(pool_size=(2,2), padding="valid", strides=(2,2), data_format="channels_last"))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(units=50, activation="relu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        model.add(Dropout(rate=0.1))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 14:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(lr=0.01, momentum=0, nesterov=False)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.Zeros()
        
        model = Sequential()
        
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(0.3))
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(0.3))
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0.3))
    
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 15:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(lr = 0.001, momentum=0.1, nesterov=False)
        
        model = Sequential()
        
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
    
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 16:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(lr = 0.001, momentum=0.1, nesterov=False)
        
        model = Sequential()
        
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.7))
        
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.6))
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.5))
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.4))
        
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(rate=0.3))
    
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 17:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(lr = 0.001, momentum=0.1, nesterov=False)
        
        model = Sequential()
        
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.5))
        
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.4))
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.3))
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
    
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 18:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(lr = 0.001, momentum=0.1, nesterov=False)
        
        model = Sequential()
        
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.7))
        
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.6))
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.5))
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 19:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(lr = 0.001, momentum=0.1, nesterov=False)
        
        model = Sequential()
        
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.6))
        
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.5))
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.4))
        
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(SpatialDropout2D(rate=0.3))
        
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
        
        model.summary()
        
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 20:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(lr = 0.001, momentum=0.1, nesterov=False)
        
        model = Sequential()
        
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2])))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2)))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(100))
        model.add(LeakyReLU(alpha=0.3))
    
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 21:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(lr = 0.001, momentum=0.1, nesterov=False)
        
        model = Sequential()
        
        model.add(Conv2D(filters=32, kernel_size=(10,10), strides=(2,2), data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2])))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization())
        model.add(Dropout(rate=0.3))
        model.add(Dense(64))
        model.add(LeakyReLU(alpha=0.3))
        model.add(BatchNormalization())
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 22 or run == 23:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.categorical_accuracy]
        my_opt = optimizers.adam(lr = 0.001)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        
        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), \
                         data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), \
                         kernel_initializer=kernel_init, bias_initializer=bias_init))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        
        model.add(Flatten())
        
        model.add(Dense(units=100, kernel_initializer=kernel_init, bias_initializer=bias_init))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(units=5, activation="softmax", kernel_initializer=kernel_init, bias_initializer=bias_init))
        elif task_flag == 2:
            model.add(Dense(units=6, activation="softmax", kernel_initializer=kernel_init, bias_initializer=bias_init))
            
        model.summary()
        
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 24:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.categorical_accuracy]
        my_opt = optimizers.adam(lr = 0.001)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        
        model = Sequential()
        
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), \
                         data_format="channels_last", input_shape=(in_shape[0], in_shape[1], in_shape[2]), \
                         kernel_initializer=kernel_init, bias_initializer=bias_init))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), \
                         kernel_initializer=kernel_init, bias_initializer=bias_init))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), \
                         kernel_initializer=kernel_init, bias_initializer=bias_init))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), \
                         kernel_initializer=kernel_init, bias_initializer=bias_init))
        model.add(LeakyReLU(alpha=0.3))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(units=100, kernel_initializer=kernel_init, \
                        bias_initializer=bias_init))
        model.add(LeakyReLU(alpha=0.3))
    
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(units=5, activation="softmax", \
                         kernel_initializer=kernel_init, bias_initializer=bias_init))
        elif task_flag == 2:
            model.add(Dense(units=6, activation="softmax", \
                         kernel_initializer=kernel_init, bias_initializer=bias_init))
    
        model.summary()
    
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    
    return model

def build_transfer_2D_model(task_flag, run, in_shape):
        
    if run == 1:
            
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
            
        transfer_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=in_shape)
            
        x = transfer_model.output
            
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
            
        if task_flag == 0 or task_flag == 1:
            out = Dense(5, activation="softmax")(x)
        elif task_flag == 2:
            out = Dense(6, activation="softmax")(x)
    
        model = Model(inputs=transfer_model.input, outputs=out)
            
        model.summary()
    
        for layer in model.layers[:19]:
            layer.trainable = False
        for layer in model.layers[19:]:
            layer.trainable = True

        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 2:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
        
        transfer_model = mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=in_shape)
        
        x = transfer_model.output
        
        x = Flatten()(x)
        x = Dense(50, activation="relu")(x)
        
        if task_flag == 0 or task_flag == 1:
            out = Dense(5, activation="softmax")(x)
        elif task_flag == 2:
            out = Dense(6, activation="softmax")(x)
    
        model = Model(inputs=transfer_model.input, outputs=out)
        
        model.summary()
    
        for layer in model.layers[:86]:
            layer.trainable = False
        for layer in model.layers[86:]:
            layer.trainable = True

        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    
    return model


def build_3D_model(task_flag, run, in_shape):
    
    if run == 1:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.001, momentum=0, nesterov=False)
    
        model = Sequential()

        model.add(Conv3D(64, [3,3,3], activation="relu", input_shape= in_shape, data_format="channels_last", padding="same"))
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(SpatialDropout3D(rate = 0.3))
        model.add(Conv3D(32, [3,3,3], activation="relu"))
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(Conv3D(32, [3,3,3], activation="relu"))
        model.add(SpatialDropout3D(rate = 0.3))
        model.add(MaxPooling3D(pool_size = [3,3,1]))
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
        
        model.summary()
        
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 2:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
    
        model = Sequential()

        model.add(Conv3D(64, [3,3,3], activation="relu", input_shape= in_shape, data_format="channels_last", padding="same"))
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(SpatialDropout3D(rate = 0.3))
        model.add(Conv3D(32, [3,3,3], activation="relu"))
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(Conv3D(32, [3,3,3], activation="relu"))
        model.add(SpatialDropout3D(rate = 0.3))
        model.add(MaxPooling3D(pool_size = [3,3,1]))
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
        
        model.summary()
        
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 3:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
    
        model = Sequential()

        model.add(Conv3D(32, [3,3,3], activation="relu", input_shape= in_shape, data_format="channels_last", padding="same"))
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(Conv3D(16, [3,3,3], activation="relu"))
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(Conv3D(16, [3,3,3], activation="relu"))
        model.add(MaxPooling3D(pool_size = [3,3,1]))
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
        
        model.summary()
        
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 4:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
    
        model = Sequential()

        model.add(Conv3D(16, [32,32,8], activation="relu", input_shape= in_shape, data_format="channels_last", padding="same"))
        model.add(MaxPooling3D(pool_size = [10,10,5]))
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
        
        model.summary()
        
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 5:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
    
        model = Sequential()

        model.add(Conv3D(32, [5,5,5], activation="relu", input_shape= in_shape, data_format="channels_last", padding="same"))
        model.add(MaxPooling3D(pool_size = [5,5,3]))
        model.add(SpatialDropout3D(rate = 0.4))
        model.add(Conv3D(16, [5,5,5], activation="relu"))
        model.add(MaxPooling3D(pool_size = [5,5,3]))
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
        
        model.summary()
        
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    
    elif run == 6:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
    
        model = Sequential()

        model.add(Conv3D(32, [3,3,3], activation="relu", input_shape= in_shape, data_format="channels_last", padding="same"))
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(SpatialDropout3D(rate = 0.3))
        model.add(Conv3D(16, [3,3,3], activation="relu"))
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(Conv3D(16, [3,3,3], activation="relu"))
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(SpatialDropout3D(rate = 0.3))
        
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
        
        model.summary()
        
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 7:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
    
        model = Sequential()

        model.add(Conv3D(32, [3,3,3], activation="relu", input_shape= in_shape, data_format="channels_last", padding="same"))
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(SpatialDropout3D(rate = 0.3))
        model.add(Conv3D(16, [3,3,3], activation="relu"))
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(Conv3D(16, [3,3,3], activation="relu"))
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(SpatialDropout3D(rate = 0.3))
        
        model.add(Flatten())
        model.add(Dense(50, activation="relu"))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
        
        model.summary()
        
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    
    elif run == 8:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.01, momentum=0, nesterov=False)
    
        model = Sequential()

        model.add(Conv3D(32, [3,3,3], activation="relu", input_shape= in_shape, data_format="channels_last", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(SpatialDropout3D(rate = 0.3))
        model.add(Conv3D(16, [3,3,3], activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(Conv3D(16, [3,3,3], activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling3D(pool_size = [3,3,2]))
        model.add(SpatialDropout3D(rate = 0.3))
        
        model.add(Flatten())
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0.3))
        
        if task_flag == 0 or task_flag == 1:
            model.add(Dense(5, activation="softmax"))
        elif task_flag == 2:
            model.add(Dense(6, activation="softmax"))
        
        model.summary()
        
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 9:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.accuracy, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.Zeros()
        
        in_layer = Input(shape=in_shape)
        x = AveragePooling3D(pool_size=(2,2,2), padding="same", data_format="channels_last")(in_layer)
        x = Conv3D(filters=128, kernel_size=(3,3,3), strides=(1,1,1), padding="same", data_format="channels_last", dilation_rate=(1,1,1), \
                         activation="elu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None)(x)
        x = SpatialDropout3D(rate=0.2, data_format="channels_last")(x)
        x = Conv3D(filters=32, kernel_size=(3,3,3), strides=(1,1,1), padding="same", data_format="channels_last", dilation_rate=(1,1,1), \
                         activation="elu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None)(x)
        x = SpatialDropout3D(rate=0.2, data_format="channels_last")(x)
        x = MaxPooling3D(pool_size=(2,2,2), padding="same", data_format="channels_last")(x)
        
        x = Flatten()(x)
        
        x = Dense(units=32, activation="elu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None)(x)
        x = Dropout(rate=0.2)(x)
        
        
        if task_flag == 0 or task_flag == 1:
            out = Dense(units=5, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                            bias_constraint=None)(x)
        elif task_flag == 2:
            out = Dense(units=6, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                            bias_constraint=None)(x)
            
        model = Model(inputs=in_layer, outputs=out)
        
        model.summary()
        
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
        
    elif run == 10:
        
        my_loss = losses.mean_squared_error
        my_metrics = [metrics.mae, metrics.categorical_accuracy]
        my_opt = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False)
        
        kernel_init = initializers.VarianceScaling(scale=1, mode="fan_avg", distribution="uniform", seed=None)
        bias_init = initializers.Zeros()
        
        in_layer = Input(shape=in_shape)
        x = AveragePooling3D(pool_size=(2,2,2), padding="same", data_format="channels_last")(in_layer)
        x = Conv3D(filters=128, kernel_size=(3,3,3), strides=(1,1,1), padding="same", data_format="channels_last", dilation_rate=(1,1,1), \
                         activation="elu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None)(x)
        x = SpatialDropout3D(rate=0.2, data_format="channels_last")(x)
        x = Conv3D(filters=32, kernel_size=(3,3,3), strides=(1,1,1), padding="same", data_format="channels_last", dilation_rate=(1,1,1), \
                         activation="elu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                         kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                         bias_constraint=None)(x)
        x = SpatialDropout3D(rate=0.2, data_format="channels_last")(x)
        x = MaxPooling3D(pool_size=(2,2,2), padding="same", data_format="channels_last")(x)
        
        x = Flatten()(x)
        
        x = Dense(units=32, activation="elu", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                        kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                        bias_constraint=None)(x)
        x = Dropout(rate=0.2)(x)
        
        
        if task_flag == 0 or task_flag == 1:
            out = Dense(units=5, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                            bias_constraint=None)(x)
        elif task_flag == 2:
            out = Dense(units=6, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
                            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
                            bias_constraint=None)(x)
            
        model = Model(inputs=in_layer, outputs=out)
        
        model.summary()
        
        model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)
    
    return model
