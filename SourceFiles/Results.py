# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:14:14 2019

@author: Linus Lindahl & Magnus Knutsson
"""

#Enum for choosing behavior and loading all the models
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

###########################################
# Parameters to specify the models to use #
###########################################
epoch = 100
k = 10
seed = 123

#######################################
# Statistics Two tailed paired t test #
#######################################
# significance level 0.05 / 132 = 0.0000378 value (column 0.001 degree of freedome 9 used)
tKFold = 4.781

#support libraries
import sys
sys.path.insert(0, '../ModelOutputs')
import numpy as np
import pickle
from scipy import stats

def getConfusionMatrix(histories):
    #create empty table
    table = np.zeros((10,11))
    
    #fill table with the predicted and actual results for all k-folds
    for model in histories:
        
        actual = model['actual']
        predictions = model['predictions']
        for i in range(0, len(actual)):
            row = actual[i]
            column = predictions[i]
            table[row][column] += 1
            #total amount of images of the given category
            table[row][10] += 1
    
    #convert the table entries into pecentages
    for row in range(0, 10):
        for column in range(0,10):
            table[row][column] /= table[row][10] / 100
            table[row][column] = round(table[row][column], 2)
            
    #Predicted categories (Cifar10 categories) + the total amount of images of each category
    columns = 'P.Airplane,P.Automobile,P.Bid,P.Cat,P.Deer,P.Dog,P.Frog,P.Horse,P.Ship,P.Truck,Total'
    
    #Actual categories (Cifar10 categories)
    rows =    ['Airplane', 'Automobile', 'Bid', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    #build the table with comma separation
    results = ',' + columns + '\n'
    for i in range(0, 10):
        results += rows[i]
        for j in range(0, 11):
            results += ',' + str(table[i][j])
        results += '\n'
    
    return results

def getModelHistory(epoch, k, seed, model_architecture):
    #history dataStructure
    #    [k] k-fold model
    #        ['history'] training loss, acc, val_loss, val_acc
    #        ['predictions'] predicted classes on the test set once trained
    #        ['actual'] actual classes of the test set
    #        ['summary'] string with model design
    
    model = str(model_architecture).split('.')[1]
    filePath = '../ModelOutputs/'
    filePath += model + '_epoch(' + str(epoch) + ')k(' 
    filePath += str(k) + ')seed(' + str(seed) + ').pkl'

    with open (filePath, 'rb') as file:
        modelHistory = pickle.load(file)
        
    return modelHistory

def getAverageTrainingHistory(history):
    averageHistory = {'acc' : np.zeros(epoch), 'val_acc' : np.zeros(epoch), 'loss' : np.zeros(epoch), 'val_loss' : np.zeros(epoch)}
    
    #sum acc, val_acc, loss, val_loss
    for model in history:
        averageHistory['acc'] += np.array(model['history']['acc'])
        averageHistory['val_acc'] += np.array(model['history']['val_acc'])
        averageHistory['loss'] += np.array(model['history']['loss'])
        averageHistory['val_loss'] += np.array(model['history']['val_loss'])
    
    #average accuracy, validation accuracy, loss, validation loss
    averageHistory['acc'] /= k
    averageHistory['val_acc'] /= k
    averageHistory['loss'] /= k
    averageHistory['val_loss'] /= k
    
    return averageHistory

def saveStatistics(results, model_architecture):
    model = str(model_architecture).split('.')[1]
    filePath = '../ModelOutputs/' + str(model) + '(statistics).txt'
    with open (filePath, 'w') as output:
        output.write(results)

def getResults(model_architecture):
    #get the model history for each k
    histories = getModelHistory(epoch, k, seed, model_architecture)
    
    #get a confusion matrix with actual image label vs predicted image label
    confusionMatrix = getConfusionMatrix(histories)
    
    #get the average training performance 
    averageHistory = getAverageTrainingHistory(histories)
    
    trainingAcc = averageHistory['acc']
    trainingValAcc = averageHistory['val_acc']
    trainingLoss = averageHistory['loss']
    trainingValLoss = averageHistory['val_loss']
    
    #create a table epochs as columns, rows: training acc, val acc, training loss, val loss
    training = 'epohs'
    for i in range(0,100):
        training += ',' + str(i)
        
    training += '\nTraing accuracy'
    for i in trainingAcc:
        training += ',' + str(i)
    
    training += '\nValidation accuracy'
    for i in trainingValAcc:
        training += ',' + str(i)
    
    training += '\nTraing loss'
    for i in trainingLoss:
        training += ',' + str(i)
    
    training += '\nValidation loss'
    for i in trainingValLoss:
        training += ',' + str(i)
    
    output = str(model_architecture) + '\n\n'
    output += training + '\n'
    output += '\n\n' + confusionMatrix
    
    #save the performance and confusion matrix to disk
    saveStatistics(output, model_architecture)
    
    return {'histories' : histories, 'averageHistory' : averageHistory}

def pairedTTest(sampleA, sampleB):    
    #A two tailed paired t-test returning a tuple <t-value, p-value>
    tp = stats.ttest_rel(sampleA, sampleB)
    
    results = {'t*' : tp[0], 'p*' : tp[1]}
    
    return results

def getEpochsDoneTraining(models, archType):
    modelTypes = ['base', 'deep', 'wide', 'Sigmoid', 'TanH', 'Leaky']
    
    bestKFoldEpochs = {}
    for i in range(0, len(modelTypes)):
        allKFolds = []
        for j in range(0, k):
            epochs = models[archType][modelTypes[i]]['histories'][j]['history']['val_loss']
            bestEpoch = np.argmin(epochs)
            allKFolds.append(bestEpoch)
        bestKFoldEpochs[modelTypes[i]] = allKFolds
        
    return bestKFoldEpochs

def getTrainingValAcc(models, archType, bestKFoldEpochs):
    modelTypes = ['base', 'deep', 'wide', 'Sigmoid', 'TanH', 'Leaky']
    
    trainedKFoldValAcc = {}
    for i in modelTypes:
        allKFolds = []
        for j in range(0, k):
            valAcc = models[archType][i]['histories'][j]['history']['val_acc']
            valAcc = valAcc[bestKFoldEpochs[i][j]]
            allKFolds.append(valAcc)
        trainedKFoldValAcc[i] = allKFolds
        
    return trainedKFoldValAcc

def getTTestTable(title, samples, measure):
    title = title + '\n'
    
    header = ''
    for key in samples:
        header += ',' + key
    header += ',' + measure + '\n'
    
    rows = ''
    for i in samples:
        row = i
        for j in samples:
            modelA = samples[i]
            modelB = samples[j]
            t = pairedTTest(modelA, modelB)['t*']
            row += ',' + str(t)
        average = float(np.sum(samples[i])/k)
        rows += row + ',' + str(average) + '\n'
        
    table = title + header + rows + '\n\n'
    return table

def allModels():
    #models datastructure
    #   ['FFN'] -> same as CNN
    #   ['CNN']
    #       ['base'] -> same as Leaky
    #       ['wide'] -> same as Leaky
    #       ['deep'] -> same as Leaky
    #       ['Sigmoid] -> same as Leaky
    #       ['TanH'] -> same as Leaky
    #       ['Leaky']
    #           ['averageHistory']          !! average training among all k folds !!
    #               ['acc']
    #               ['val_acc']
    #               ['loss']
    #               ['val_loss']
    #           ['histories']               !! k-fold training k = 10 !!
    #               [k]                     !! training history for fold k (0-9) !!
    #                   ['history']
    #                       ['acc']
    #                       ['val_acc']
    #                       ['loss']
    #                       ['val_loss']
    #                   ['predictions'] predicted classes on the test set once trained
    #                   ['actual'] actual classes of the test set
    #                   ['summary'] string with model design

    #build models datastructe by loading the results from files
    models = {'CNN' : {}, 'FFN' : {}}
    for archType in model_type:
        modelType = str(archType).split('.')[1].split('_')
        temp = getResults(archType)
        
        #networktype = FFN/CNN, variant = base, wide, deep, Sigmoid, TanH, Leaky
        networkType = modelType[0]
        variant = modelType[1]
        models[networkType][variant] = temp
    
    #Training statistics    
    #FFN
    ffnTrain = getEpochsDoneTraining(models, 'FFN')
    ffnValAcc = getTrainingValAcc(models, 'FFN', ffnTrain)
    ffnTrainTable = getTTestTable('FFN training epochs', ffnTrain, 'epochs')
    ffnValAccTable = getTTestTable('FFN accuracy', ffnValAcc, 'acc')
    
    #CNN
    cnnTrain = getEpochsDoneTraining(models, 'CNN')
    cnnValAcc = getTrainingValAcc(models, 'CNN', cnnTrain)
    cnnTrainTable = getTTestTable('CNN Training epochs', cnnTrain, 'epochs')
    cnnValAccTable = getTTestTable('CNN accuracy', cnnValAcc, 'acc')
    
    #FFN and CNN training
    allTraining = {}
    for i in ffnTrain:
        key = 'f.' + i
        allTraining[key] = ffnTrain[i]
    for i in cnnTrain:
        key = 'c.' + i
        allTraining[key] = cnnTrain[i]
    allTrainingTable = getTTestTable('all training', allTraining, 'epochs')
    
    #FFN and CNN val acc
    allAccuracy = {}
    for i in ffnValAcc:
        key = 'f.' + i
        allAccuracy[key] = ffnValAcc[i]
    for i in cnnValAcc:
        key = 'c.' + i
        allAccuracy[key] = cnnValAcc[i]
    allAccuracyTable = getTTestTable('all accuracy', allAccuracy, 'acc')
    
    #t value
    content = 't value' + ',' + str(tKFold) + ',' + str(-tKFold) + '\n\n'
    #FFN train
    content += ffnTrainTable
    #FFN acc
    content += ffnValAccTable
    #CNN train
    content += cnnTrainTable
    #CNN acc
    content += cnnValAccTable
    #FFN & CNN train
    content += allTrainingTable
    #FFN & CNN acc
    content += allAccuracyTable
    
    saveStatistics(content, 'output.Training')
    
allModels()
print('output calculated')