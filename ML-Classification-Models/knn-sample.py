#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:16:36 2021

@author: Batuhan Duzgun (batux)
"""

import numpy as np
import DataPreProcessingLibrary as dpt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier, VALID_METRICS
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, recall_score

# !!! Set value to 1 if you would like to get detailed logs.
DETAIL_LOG = 0

def run_knn(x_train, y_train, x_test, y_test, n = 5, algorithm="auto", metric="minkowski", metric_params=None, leaf_size=30, n_jobs=None):
    
    knn = KNeighborsClassifier(n_neighbors = n, algorithm=algorithm, metric=metric, metric_params=metric_params, leaf_size=leaf_size, n_jobs=n_jobs)
    knn.fit(x_train, y_train)

    predicted_y_values = knn.predict(x_test)
    
    print("---------------------------------------------")
    print("n = " + str(n) +
          ", algorithm: " + algorithm +
          ", leaf_size: " + str(leaf_size) +
          ", metric: " + metric)
    if DETAIL_LOG == 1:
        print("KNN >> R2 Value: " + str( r2_score(y_test, predicted_y_values) ))
        
        cm = confusion_matrix(y_test, predicted_y_values)
        print("*** Confusion Matrix ***")
        print(cm)
    
    print("KNN Accuracy Rate: " + str(accuracy_score(y_test, predicted_y_values))) 
    
    if DETAIL_LOG == 1:
        print("KNN Recall Rate: " + str(recall_score(y_test, predicted_y_values, average="macro"))) 


print(VALID_METRICS['kd_tree'])

# Load Data
rawData = datasets.load_iris()

# Scale Data Fields
ds = dpt.DataScaler()
scaled_data = ds.scale(rawData.data)

# Create Train and Test Data Sets
dsp = dpt.DataSplitter()
x_train, x_test, y_train, y_test = dsp.split(scaled_data, rawData.target, 0.25)


# Run Different KNN Models,
# We change algorithm, metric types, neighbors size and leaf_sizes
dim = x_train.shape[1]
weights = np.random.choice(dim, dim, replace=False)
metric_params = None
algorithms = [ "auto", "ball_tree", "kd_tree" ]
distance_metrics = [ "euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "mahalanobis" ]
leaf_sizes = [30, 40, 50]

for algo in algorithms:
    for dist in distance_metrics:
        
        if dist == "wminkowski":
            metric_params={'w': weights}
        elif dist == "mahalanobis":
            metric_params={'V': np.cov(x_train.T)}
        else:
            metric_params=None
            
        if algo == "kd_tree" and (dist == "wminkowski" or dist == "mahalanobis"):
            # wminkowski or mahalanobis metric not valid for 'kd_tree' algorithm
            continue
            
        for n in range(4,8):
            
            if algo == "ball_tree" or algo == "kd_tree":
                # try leaf_size!
                for lfs in leaf_sizes:
                    run_knn(x_train, y_train, x_test, y_test, n=n, algorithm=algo, metric=dist, metric_params=metric_params, leaf_size=lfs, n_jobs=-1)

            else:    
                run_knn(x_train, y_train, x_test, y_test, n=n, algorithm=algo, metric=dist, metric_params=metric_params, n_jobs=-1)

