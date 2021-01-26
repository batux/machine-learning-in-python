#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 20:30:03 2020

@author: Batuhan Duzgun (batux)
"""

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import sklearn.model_selection as ms
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.impute import SimpleImputer


class RawDataLoader:
    
    raw_data = None
    
    def load(self, path):
        self.raw_data = pd.read_csv(path)
        return self.raw_data
    
    def getData(self):
        return self.raw_data
    

class MissingValueImputer:
    
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    
    def fill_missing_values(self, raw_data, startColumnIndex, endColumnIndex):
        
        # iloc ile verileri dizi olarak alabiliyorsun. : ve 1:4 çalışması için iloc üzerinden yapman lazım
        numeric_values = raw_data.iloc[:, startColumnIndex : endColumnIndex].values
        #print(numeric_values)
        
        
        # aldığın veri kümesinin ortalamasını aldırmak için fit fonksiyonu kullan
        self.imputer = self.imputer.fit(numeric_values)
        
        # nan olan değerlere ortalama değeri atıyoruz.
        completed_numeric_values = self.imputer.transform(numeric_values)
        #print(completed_numeric_values)
        
        return completed_numeric_values


class CategoricalDataProcessor:
    
    le = preprocessing.LabelEncoder()
    ohe = preprocessing.OneHotEncoder()
    
    def makeLabelEncode(self, data):
        encoded_labels = self.le.fit_transform(data)
        #print(encoded_labels)
        return encoded_labels
    
    def makeOneShotEncoding(self, data):
        ohe_encoded_matrix = self.ohe.fit_transform(data).toarray()
        #print(ohe_encoded_matrix)
        return ohe_encoded_matrix
        

class DataImporter:
    
    def createDataFrame(self, data, rowSize, columns):
        
        result = pd.DataFrame(data = data, index = range(rowSize), columns = columns)
        return result
    
    def concat(self, dataParts, axisWay):
        return pd.concat(dataParts, axis = axisWay)
    
    def createColumnNames(self, prefix, size):
        
        columns = []
        for i in range(size):
           columns.append(prefix + str(i))
        
        return columns
   


class DataSplitter:
    
    def split(self, data, target_labels, testSize):
        
        return ms.train_test_split(data, target_labels, test_size = testSize, random_state = 0)
    
        
class DataScaler:
    
    def scale(self, data):
        standardScaler = preprocessing.StandardScaler()
        return standardScaler.fit_transform(data)
    
    def normalize(self, data):
        minMaxScaler = preprocessing.MinMaxScaler()
        return minMaxScaler.fit_transform(data)
    
class DimensionReductionTool:
    
    model = None
    significanceLevel = 0.05
    
    def __init__(self, significanceLevel):
        self.significanceLevel = significanceLevel

    def fit(self, targetLabels, fulldata):
        ols = sm.OLS(targetLabels, fulldata)
        self.model = ols.fit()
        return self.model
    
    def getPValues(self):
        return self.model.pvalues
    
    def summary(self):
        print(self.model.summary())
        
    def performBackwardElimination(self, targetLabels, fulldata, onestep=False):
        
        tmpFulldata = np.array(fulldata, dtype=float)
        #print(tmpFulldata)
        
        while True:
            
            olsAlgo = sm.OLS(targetLabels, tmpFulldata)
            tmpModel = olsAlgo.fit()
            print(tmpModel.summary())
        
            pValues = self.formatPValues(tmpModel.pvalues)     
            valid = self.isValidSignificanceLevel(pValues)
            
            if onestep:
                
                while self.isValidSignificanceLevel(pValues) != True:
                    maxValueIndex = np.argmax(pValues)
                    #print(maxValueIndex)
                    tmpFulldata = np.delete(tmpFulldata, maxValueIndex, 1) 
                    #print(tmpFulldata)
                    pValues = np.delete(pValues, maxValueIndex, 0)
                break
                
            
            if valid:
                break
            
            maxValueIndex = np.argmax(pValues)
            #print(maxValueIndex)
            tmpFulldata = np.delete(tmpFulldata, maxValueIndex, 1) 
            #print(tmpFulldata)
        
        return tmpFulldata
    
    def stepbystep(self, targetLabels, fulldata):
        olsAlgo = sm.OLS(targetLabels, fulldata)
        return olsAlgo.fit()
    
    def formatPValues(self, pvalues):
        pValues = []
        for pVal in pvalues:
            pValues.append(float(format(pVal, 'f')))
            #print(pVal)
        return pValues
    
    def isValidSignificanceLevel(self, pValues):
        
        flag = True
        for pVal in pValues:
            if pVal > self.significanceLevel:
                flag = False
                break
        return flag