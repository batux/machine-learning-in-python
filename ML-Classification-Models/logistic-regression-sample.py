#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:42:06 2021

@author: Batuhan Duzgun (batux)
"""

import numpy as np
import DataPreProcessingLibrary as dpt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, confusion_matrix


dtloader = dpt.RawDataLoader()
rawdata = dtloader.load("./data/veriler.csv")
xValues = rawdata.iloc[:, 1:4].values
yValues = rawdata.iloc[:, 4:].values

catproc = dpt.CategoricalDataProcessor()
yValues = catproc.makeLabelEncode(yValues)
yValues = np.ravel(yValues)

ds = dpt.DataScaler()
xValues = ds.scale(xValues)

dsp = dpt.DataSplitter()
x_train, x_test, y_train, y_test = dsp.split(xValues, yValues, 0.10)


logr = LogisticRegression(random_state=0)
logr.fit(x_train, y_train)

predicted_y_values = logr.predict(x_test)
print("Logistic Regression >> R2 Value: " + str( r2_score(y_test, predicted_y_values) ))

cm = confusion_matrix(y_test, predicted_y_values)
print("*** Confusion Matrix ***")
print(cm)