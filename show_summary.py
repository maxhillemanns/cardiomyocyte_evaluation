# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:51:05 2020

@author: User
"""

from keras import metrics, losses, optimizers, initializers, regularizers, constraints
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, SpatialDropout2D, Dropout, Conv3D, MaxPooling3D, SpatialDropout3D, BatchNormalization, AveragePooling3D, LeakyReLU
from keras.layers.core import Flatten
from keras.applications import mobilenet, vgg16

in_shape = [512,512,32,1]
        
my_loss = losses.mean_squared_error
my_metrics = [metrics.mae, metrics.categorical_accuracy]
my_opt = optimizers.SGD(lr=0.01, momentum=0, nesterov=False)

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

out = Dense(units=5, activation="softmax", use_bias=True, kernel_initializer=kernel_init, bias_initializer=bias_init, \
            kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, \
            bias_constraint=None)(x)

model = Model(inputs=in_layer, outputs=out)

model.compile(optimizer=my_opt, loss=my_loss, metrics=my_metrics)    

model.summary()








    