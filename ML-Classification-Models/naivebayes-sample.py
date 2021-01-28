#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 18:33:07 2021

@author: Batuhan Duzgun (batux)
"""

import numpy as np
import DataPreProcessingLibrary as dpt
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB


"""
dtloader = dpt.RawDataLoader()
rawdata = dtloader.load("./data/veriler.csv")
xValues = rawdata.iloc[:, 1:4].values
yValues = rawdata.iloc[:, 4:].values

catproc = dpt.CategoricalDataProcessor()
yValues = catproc.makeLabelEncode(yValues)
yValues = np.ravel(yValues)
"""

rawData = datasets.load_iris()

ds = dpt.DataScaler()
scaled_data = ds.normalize(rawData.data)


dsp = dpt.DataSplitter()
x_train, x_test, y_train, y_test = dsp.split(scaled_data, rawData.target, 0.20)


gnb = GaussianNB()
gnb.fit(x_train, y_train)
predicted_y_test = gnb.predict(x_test)
print("GaussianNaiveBayes Accuracy Rate: " + str(accuracy_score(y_test, predicted_y_test))) 
cm = confusion_matrix(y_test, predicted_y_test)
print("*** Confusion Matrix ***")
print(cm)


bnb = BernoulliNB()
bnb.fit(x_train, y_train)
predicted_y_test = bnb.predict(x_test)
print("BernoulliNaiveBayes Accuracy Rate: " + str(accuracy_score(y_test, predicted_y_test))) 
cm = confusion_matrix(y_test, predicted_y_test)
print("*** Confusion Matrix ***")
print(cm)


mnb = MultinomialNB()
mnb.fit(x_train, y_train)
predicted_y_test = mnb.predict(x_test)
print("MultinomialNaiveBayes Accuracy Rate: " + str(accuracy_score(y_test, predicted_y_test))) 
cm = confusion_matrix(y_test, predicted_y_test)
print("*** Confusion Matrix ***")
print(cm)
