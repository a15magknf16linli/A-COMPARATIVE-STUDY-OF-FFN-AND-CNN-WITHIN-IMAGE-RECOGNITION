# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 08:02:09 2019

@author: Linus Lindahl & Magnus Knutsson
"""

#Enum for choosing behavior
from enum import Enum
class model_type(Enum):
    FFN_base = 0
    FFN_deep = 1
    FFN_wide = 2
    FFN_Sigmoid = 3
    FFN_TanH = 4
    FFN_Leaky_ReLU = 5
    CNN_base = 6
    CNN_deep = 7
    CNN_wide = 8
    CNN_Sigmoid = 9
    CNN_TanH = 10
    CNN_Leaky_ReLU = 11
    
############################
# parameters for the model #
############################
input_x = 32
input_y = 32
color_depth = 3
batch_size = 32
epochs = 100

##############################################
# parameters for training and choosing model #
##############################################
k_fold_split = 10
model_architecture = model_type.FFN_Leaky_ReLU

#support libraries
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import pickle

#import models
import sys
sys.path.insert(0, '../ModelTemplates')
import FFNBaseModel
import FFNDeepModel
import FFNWideModel
import FFNLeakyReLuModel
import FFNSigmoidModel
import FFNTanHModel
import CNNBaseModel
import CNNDeepModel
import CNNWideModel
import CNNSigmoidModel
import CNNTanHModel
import CNNLeakyReLUModel

#seed random
import random as rn
random_seed = 123
rn.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

def mergeDataSet():
    image_data_set = np.concatenate((x_train, x_test))
    image_labels = np.concatenate((y_train, y_test))
    
    return (image_data_set, image_labels)

def createFilePath(epoch, folder, k, fileEnding):
    #Create the filename. <model_architecture>_epoch(<epoch>)k(<k>)seed(<seed>).pkl
    filePath = '../' + str(folder) + '/'
    filePath += str(model_architecture).split('.')[1]
    filePath += '_epoch(' + str(epoch) + ')'
    filePath += 'k(' + str(k) + ')'
    filePath += 'seed(' + str(random_seed) + ')'
    filePath += '.' + str(fileEnding)
    
    return filePath

def kFoldCrossValidation(model_architecture):
    kFolds = StratifiedKFold(k_fold_split)
    print('### ' + str(k_fold_split) + ' fold crossvalidation')
    print('### ' + str(model_architecture))
    print('### epochs ' + str(epochs))
    print('### batch size ' + str(batch_size) + '\n')
    
    #create new model
    model = createModel(model_architecture)(random_seed, input_x, input_y, color_depth)
    filePath = createFilePath('untrained', 'models', k_fold_split, 'HDF5')
    model.save(filePath)
    print('\n### untrained model saved\n')
    del model
    
    model_performance = []
    k = 0
    #K fold cross validation
    for train_index, test_index in kFolds.split(X, Y):
        x_train = X[train_index]
        y_train = Y[train_index]
        x_test = X[test_index]
        y_test = Y[test_index]
        
        #load the untrained model
        model = tf.keras.models.load_model(filePath)
        print('### untrained model loaded')
        print('### Fold = ' + str(k) + ' ###\n')
        
        #save only the best performing model after each epoch
        bestModelFilePath = createFilePath(epochs, 'models', k, 'HDF5')
        callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath = bestModelFilePath,
                                                      monitor = 'val_loss',
                                                      verbose= 1,
                                                      save_best_only = True,
                                                      period = 1)]
        
        #train the model
        history = model.fit(x = x_train,
                  y = y_train,
                  validation_data = (x_test, y_test),
                  epochs = epochs,
                  batch_size = batch_size,
                  callbacks = callbacks)
        
        #load the best trained model
        del model
        model = tf.keras.models.load_model(bestModelFilePath)
        
        #find the predicted class for each image in the test set
        predictions = model.predict(X[test_index])
        predictions = np.argmax(predictions, axis = 1)
        
        #find the actual class in the test set and reshape it to the same format as predictions 
        actual = Y[test_index].reshape(Y[test_index].shape[0],)
        
        #save the design of the model as a string
        modelSummary = []
        model.summary(print_fn = lambda x : modelSummary.append(x))
        modelSummary = "\n".join(modelSummary)
        
        #store the training history, prediction of the test set on trained model and actual class of the test set, and the design of the model
        model_performance.append({'history' : history.history, 'predictions' : predictions, 'actual' : actual, 'summary' : modelSummary})
        k = k + 1
   
        #dealocate memory
        del history
        del predictions
        del actual
        del modelSummary
        del model

    return model_performance

def createModel(model_architecture):
    #returns a functions
    switcher = {
            model_type.FFN_base : FFNBaseModel.createModel,
            model_type.FFN_deep : FFNDeepModel.createModel,
            model_type.FFN_wide : FFNWideModel.createModel,
            model_type.FFN_Sigmoid : FFNSigmoidModel.createModel,
            model_type.FFN_TanH : FFNTanHModel.createModel,
            model_type.FFN_Leaky_ReLU : FFNLeakyReLuModel.createModel,
            model_type.CNN_base : CNNBaseModel.createModel,
            model_type.CNN_deep : CNNDeepModel.createModel,
            model_type.CNN_wide : CNNWideModel.createModel,
            model_type.CNN_Sigmoid : CNNSigmoidModel.createModel,
            model_type.CNN_TanH : CNNTanHModel.createModel,
            model_type.CNN_Leaky_ReLU : CNNLeakyReLUModel.createModel
            }
    
    return switcher.get(model_architecture)

def saveResults(model_architecture, results):
    filePath = createFilePath(epochs, 'ModelOutputs', k_fold_split, 'pkl')
    
    #write to disk    
    results = np.array(results)
    with open (filePath, 'wb') as output:
        pickle.dump(results, output)
    
    print(str(filePath) + ' saved!')

print('*********')
print('* start *')
print('*********')

#load the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#normalize the data
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)
 
#merge the train and test sets and merge the corresponing labels
(X, Y) = mergeDataSet()

#Train the model architecture
results = kFoldCrossValidation(model_architecture)
saveResults(model_architecture, results)
    
print('done')
