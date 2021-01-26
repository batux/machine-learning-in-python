#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:01:06 2021

@author: Batuhan Duzgun (batux)
"""

import numpy as np
import DataPreProcessingToolkit as dpt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score


rawDataLoader = dpt.RawDataLoader()
rawData = rawDataLoader.load("./data/salary-data.csv")
rowSize = rawData.shape[0]

xValues = rawData.iloc[:, 2:5].values
yValues = rawData.iloc[:, 5:].values


# Data Scaler
dsca = dpt.DataScaler()
xValues = dsca.scale(xValues)
yValues = dsca.scale(yValues)


# Data Dimension Reduction
drt = dpt.DimensionReductionTool(0.10)
xValues = drt.performBackwardElimination(yValues, xValues, False)


# Data Concat Process
di = dpt.DataImporter()
axis1Size = xValues.shape[1]
reductedData = di.createDataFrame(xValues, rowSize, di.createColumnNames("f_", axis1Size))
allData = di.concat([reductedData], 1)

axis1Size = yValues.shape[1]
reductedYData = di.createDataFrame(yValues, rowSize, di.createColumnNames("y_", axis1Size))
allYData = di.concat([reductedYData], 1)


# Train-Test Data Split
ds = dpt.DataSplitter()
x_train, x_test, y_train, y_test = ds.split(allData, allYData, 0.10)

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values



# 1- LinearRegression Model
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_predicted_values = lr.predict(x_test)
print("Linear Regression >> R2 Value: " + str( r2_score(y_test, lr_y_predicted_values) ))


# 2- PolynomialRegression Model
forthDegreePolyFeature = PolynomialFeatures(degree = 4)
forthDegreeXValues = forthDegreePolyFeature.fit_transform(x_train)
forthDegreeXTestValues = forthDegreePolyFeature.fit_transform(x_test)

plr4 = LinearRegression()
plr4.fit(forthDegreeXValues, y_train)

pr_y_predicted_values = plr4.predict(forthDegreeXTestValues)
print("Polynomial Regression >> R2 Value: " + str( r2_score(y_test, pr_y_predicted_values) ))


# 3- Support Vector Regressor Model (RBF)
svr_poly = SVR(kernel="rbf")
svr_poly.fit(x_train, np.ravel(y_train))

svr_poly_y_predicted_values = svr_poly.predict(x_test)
print("SVR >> R2 Value: " + str( r2_score(y_test, svr_poly_y_predicted_values) ))


# 4- Decision Tree Model
dcst = DecisionTreeRegressor(random_state=0)
dcst.fit(x_train, y_train)

dcst_y_predicted_values = dcst.predict(x_test)
print("Decision Tree >> R2 Value: " + str( r2_score(y_test, dcst_y_predicted_values) ))


# 5- Random Forest Model
rndfrst = RandomForestRegressor()
rndfrst.fit(x_train, np.ravel(y_train))

rndfrst_y_predicted_values = rndfrst.predict(x_test)
print("Random Forest >> R2 Value: " + str( r2_score(y_test, rndfrst_y_predicted_values) ))

# Results for each ML Model
# Linear Regression >> R2 Value: 0.8003564455001666
# Polynomial Regression >> R2 Value: 0.5641489621110274
# SVR >> R2 Value: 0.718743387090854
# Decision Tree >> R2 Value: 0.9791666666666667
# Random Forest >> R2 Value: 0.8454533098958336


