#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 22:31:33 2017

@author: sanketh
"""

import numpy as np
import csv
from numpy import genfromtxt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Perceptron
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import random
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

def shuffle_in_unison(a, b, state):
    #randomly shuffles a and b arrays.
    rng_state = state
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a,b


#data = np.load("../DataSets/DRREDDYNS.csv")
#data = open("C:\CMPSCI_589\Project\DataSets\DRREDDYNS.csv",'r')
# data = genfromtxt("C:\CMPSCI_589\Project\DataSets\DRREDDYNS.csv", delimiter = ',')

data = genfromtxt("test.csv", delimiter = ',')

data_X = data[1:data.shape[0], 1:-1]
data_Y = data[1:data.shape[0], -1]
#train_x, train_y = shuffle_in_unison(data_X, data_Y)
np.random.seed(24)
state = np.random.get_state()
data_x, data_y = shuffle_in_unison(data_X, data_Y, state)
half = data_x.shape[0]/3

# train = data[1:half+1,1:]
# test = data[half+1:data.shape[0]+1,1:]
# #test = data[half+1:half+half,1:]
# print "train data shape:",train.shape
# print "test data shape:",test.shape
# print train[0]
# print test[0]
print data_x
train_X = data_x[0:half,:]
print "train_X", train_X

# print train_X

train_Y = data_y[0:half]
#
#
test_X = data_x[half:,:]
test_Y = data_y[half:]

# #================================KNeighborsRegression=================================
#clf = KNeighborsRegressor(n_neighbors = 30)
#clf.fit(train_X,train_Y)
#predictions = clf.predict(test_X)
#MSE = mean_squared_error(test_Y,predictions)
#RMSE = math.sqrt(MSE)
#print RMSE
#
# #===============================Decision Tree Regression==============================
#
#clf = DecisionTreeRegressor(criterion = 'mse')
#clf.fit(train_X,train_Y)
#predictions = clf.predict(test_X)
#MSE = mean_squared_error(test_Y,predictions)
#RMSE = math.sqrt(MSE)
#print RMSE
#print test_Y[0], predictions[0]
#
# #===============================Neural Networks==============================
#
#clf = Perceptron()
#clf.fit(train_X,train_Y)
#predictions = clf.predict(test_X)
#MSE = mean_squared_error(test_Y,predictions)
#RMSE = math.sqrt(MSE)
#print RMSE

 #===============================Support Vector Regression==============================
 #with linear kernel running indefinitely, with poly giving error "Process finished with exit code ...."
#clf = SVR(kernel = 'linear')
#clf.fit(train_X,train_Y)
#predictions = clf.predict(test_X)
#MSE = mean_squared_error(test_Y,predictions)
#RMSE = math.sqrt(MSE)
#print RMSE
#
# #==============================Linear Regression=================================
#
#clf = LinearRegression(fit_intercept=True, normalize = True)
#clf.fit(train_X,train_Y)
#predictions = clf.predict(test_X)
#MSE = mean_squared_error(test_Y,predictions)
#RMSE = math.sqrt(MSE)
#print RMSE
#
# #==============================Ridge Regression=================================
#
#clf = Ridge(alpha = 100)
#clf.fit(train_X,train_Y)
#predictions = clf.predict(test_X)
#MSE = mean_squared_error(test_Y,predictions)
#RMSE = math.sqrt(MSE)
#print RMSE

# #==============================Lasso Regression=================================
#
#clf = Lasso(alpha = 100, max_iter=10000, selection = 'cyclic')
#clf.fit(train_X,train_Y)
#predictions = clf.predict(test_X)
#MSE = mean_squared_error(test_Y,predictions)
#RMSE = math.sqrt(MSE)
#print RMSE

#================================Random Forest Regressor==============================

clf = RandomForestRegressor(n_estimators = 30, random_state = 10, max_depth = 30, )
clf.fit(train_X,train_Y)
predictions = clf.predict(test_X)
MSE = mean_squared_error(test_Y,predictions)
RMSE = math.sqrt(MSE)
print RMSE