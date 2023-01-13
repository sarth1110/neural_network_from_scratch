# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:49:12 2022

@author: kanan
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import random
import sys
from sklearn.metrics import accuracy_score


def readDataset():
    global trainXDataset
    global trainYDataset

    #trainXDataset = pd.read_csv(sys.argv[1], names=['X', 'Y'], header = None)
    trainXDataset = pd.read_csv('C:\\Users\\kanan\\OneDrive\\Desktop\\Study Material\\MS\\AI\\HW3\\dataSet\\gaussian_train_data.csv', names=['X', 'Y'], header = None)
    trainXDataset['X2'] = np.square(trainXDataset['X'])
    trainXDataset['Y2'] = np.square(trainXDataset['Y'])
    trainXDataset['XY'] = np.multiply(trainXDataset['X'], trainXDataset['Y'])
    trainXDataset['sinX'] = np.sin(trainXDataset['X'])
    trainXDataset['sinY'] = np.sin(trainXDataset['Y'])

    #trainYDataset = pd.read_csv(sys.argv[2], header = None)
    trainYDataset = pd.read_csv("C:\\Users\\kanan\\OneDrive\\Desktop\\Study Material\\MS\\AI\\HW3\\dataSet\\gaussian_train_label.csv", header = None)
    trainXDataset = np.array(trainXDataset, dtype = float)
    trainYDataset = np.array(trainYDataset, dtype = float)
    print(trainYDataset.shape)

def initialize():
    global weight1
    global weight2
    global bias1
    global bias2
    
    weight1 = np.random.rand(7,7)/np.sqrt(7)
    weight2 = np.random.rand(7,1)/np.sqrt(7)
    bias1 = np.zeros([7,])/np.sqrt(7)
    bias2 = np.zeros([1,])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def forwardPropagation(trainData):
    global weight1
    global weight2
    global bias1
    global bias2
    global hiddenOutput
    global hiddenActivated
    global outputOutput
    global outputActivated
    ip = trainData
    hiddenOutput = np.matmul(ip, weight1) + bias1
    hiddenActivated = sigmoid(hiddenOutput)
    outputOutput = np.matmul(hiddenActivated, weight2) + bias2
    outputActivated = sigmoid(outputOutput)
    return outputActivated

def backPropagation(learningRate, trainData, trainLabel, batchSize):
    global weight1
    global weight2
    global outputActivated
    global hiddenActivated
    global bias1
    global bias2
    weight1 = np.subtract(weight1, learningRate * ((np.matmul(np.multiply(np.matmul(np.subtract(outputActivated, trainLabel), weight2.T), np.multiply(hiddenActivated, np.subtract(1,hiddenActivated))).T, trainData))/batchSize).T)
    weight2 = np.subtract(weight2, learningRate * ((np.matmul(np.subtract(outputActivated, trainLabel).T, hiddenActivated).T) / batchSize))
    bias1 = np.subtract(bias1, learningRate * (np.reshape((np.matmul(np.multiply(np.matmul(np.subtract(outputActivated, trainLabel), weight2.T), np.multiply(hiddenActivated, np.subtract(1,hiddenActivated))).T, np.ones([batchSize, 1]))), newshape=(7,))/batchSize))
    bias2 = np.subtract(bias2, learningRate * ((np.matmul(np.subtract(outputActivated, trainLabel).T, np.ones([batchSize,1])).T) / batchSize))

def predict():
    testYDataset = pd.read_csv("C:\\Users\\kanan\\OneDrive\\Desktop\\Study Material\\MS\\AI\\HW3\\dataSet\\gaussian_test_label.csv", header = None)
    #testXDataset = pd.read_csv(sys.argv[3], names=['X', 'Y'], header = None)
    testXDataset = pd.read_csv("C:\\Users\\kanan\\OneDrive\\Desktop\\Study Material\\MS\\AI\\HW3\\dataSet\\gaussian_test_data.csv", names=['X', 'Y'], header = None)
    testXDataset["X2"] = np.square(testXDataset["X"])
    testXDataset["Y2"] = np.square(testXDataset["Y"])
    testXDataset["XY"] = np.multiply(testXDataset["X"],testXDataset["Y"])
    testXDataset["sinX"] = np.sin(testXDataset["X"])
    testXDataset["sinY"] = np.sin(testXDataset["Y"])
    testXDataset = np.array(testXDataset,dtype=float)
    prediction = forwardPropagation(testXDataset)
    prediction = np.around(prediction)
    print("Accuracy : ",accuracy_score(testYDataset.values,prediction))
    df = pd.DataFrame(prediction)
    df.to_csv("test_predictions.csv",index=False,header=False)
    
def binary_cross_entropy(actual,predicted):
    return -(np.sum((actual*np.log(predicted)) + (1-actual)*np.log(1-predicted)))/actual.shape[0]

def trainModel(epochs, batchSize, learningRate):
    global trainXDataset
    global trainYDataset
    initialize()
    for itr in range(epochs):
        if(epochs>150):
            learningRate = 0.15
        elif(epochs>250):
            learningRate = 0.1
            
        b=0
        bSize = batchSize
        permSequence = np.random.permutation(trainXDataset.shape[0])
        X_train = trainXDataset[permSequence]
        Y_train = trainYDataset[permSequence]
        loss=[]
        while b < X_train.shape[0]:
            trainData = X_train[b:min(b+bSize, trainXDataset.shape[0])]
            trainLabel = Y_train[b:min(b+bSize, trainYDataset.shape[0])]
            bSize = len(trainData)
            prediction = forwardPropagation(trainData)
            loss.append(binary_cross_entropy(trainLabel, prediction))
            backPropagation(learningRate, trainData, trainLabel, bSize)
            b+=bSize
        #print(itr)
        #print("Avg loss: " + str(sum(loss)/len(loss)))

###########Trigger############
readDataset()
epochs = 300
batchSize = 15
learningRate = 0.3
trainModel(epochs, batchSize, learningRate)
predict()