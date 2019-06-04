# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:05:05 2019

@author: Magnus Knutsson
"""

import numpy as np
import random as rn
import tensorflow as tf

def createModel(random_seed_value, input_x, input_y, color_depth):
    
    rn.seed(random_seed_value)
    np.random.seed(random_seed_value)
    tf.set_random_seed(random_seed_value)
    
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Flatten(input_shape = (input_x, input_y, color_depth), name = 'Flatten_1'))
    
    model.add(tf.keras.layers.Dense(3000, name = 'Fully_connected_2'))
    model.add(tf.keras.layers.Activation(tf.nn.relu, name = 'ReLu_3'))
    
    model.add(tf.keras.layers.Dense(900, name = 'Fully_connected_4'))
    model.add(tf.keras.layers.Activation(tf.nn.relu, name = 'ReLu_5'))
    
    model.add(tf.keras.layers.Dense(10, name = 'Fully_connected_6'))
    model.add(tf.keras.layers.Activation(tf.nn.softmax, name = 'Softmax_7'))
    
    model.compile(loss = tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer = 'sgd',
                  metrics = ['accuracy'])
    
    return model