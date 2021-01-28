#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 00:17:48 2021

@author: Batuhan Duzgun (batux)
"""

import numpy as np
import DataPreProcessingLibrary as dpt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


dtloader = dpt.RawDataLoader()
rawdata = dtloader.load("./data/veriler.csv")
xValues = rawdata.iloc[:, 1:4].values
yValues = rawdata.iloc[:, 4:].values

catproc = dpt.CategoricalDataProcessor()
yValues = catproc.makeLabelEncode(yValues)
yValues = np.ravel(yValues)

dsp = dpt.DataSplitter()
x_train, x_test, y_train, y_test = dsp.split(xValues, yValues, 0.20)


dct = DecisionTreeClassifier(criterion="entropy")
dct.fit(x_train, y_train)

predicted_y_values = dct.predict(x_test)

print("DecisinTreeClassifier Accuracy Rate: " + str(accuracy_score(y_test, predicted_y_values))) 
cm = confusion_matrix(y_test, predicted_y_values)
print("*** Confusion Matrix ***")
print(cm)