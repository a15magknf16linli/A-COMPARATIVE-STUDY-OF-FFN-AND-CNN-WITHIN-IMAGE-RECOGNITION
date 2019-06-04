# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 08:11:46 2019

@author: Linus Lindahl
"""

import random as rn
import numpy as np
import tensorflow as tf

def createModel(random_seed_value, input_x, input_y, color_depth):
    
    #seed random
    rn.seed(random_seed_value)
    np.random.seed(random_seed_value)
    tf.set_random_seed(random_seed_value)
    
    #create the model
    model = tf.keras.models.Sequential()
    #Convolution 1
    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = 5, padding = 'same', strides = 1, input_shape = (input_x, input_y, color_depth), name = 'Convolution_1'))
    #Leaky ReLU 2
    model.add(tf.keras.layers.LeakyReLU(name = 'Leaky_ReLU_2'))
    #Max pooling 3
    model.add(tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, name = 'Max_pooling_3'))
    #Convolution 4
    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = 5, padding = 'same', strides = 1, name = 'Convolution_4'))
    #Leaky ReLU 5
    model.add(tf.keras.layers.LeakyReLU(name = 'Leaky_ReLU_5'))
    #Max pooling 6
    model.add(tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2, name = 'Max_pooling_6'))
    #Flatten 7
    model.add(tf.keras.layers.Flatten(name = 'Flatten_7'))
    #Fully connected 8
    model.add(tf.keras.layers.Dense(units = 1024, name = 'Fully_connected_8'))
    #Leaky ReLU 9
    model.add(tf.keras.layers.LeakyReLU(name = 'Leaky_ReLU_9'))
    #Fully connected 10
    model.add(tf.keras.layers.Dense(units = 10, name = 'Fully_connected_10'))
    #Soft Max 11
    model.add(tf.keras.layers.Activation(tf.nn.softmax, name = 'Soft_max_11'))
    
    #Compile model
    model.compile(loss = tf.keras.losses.sparse_categorical_crossentropy, optimizer = 'sgd', metrics = ['accuracy'])
    return model
    