#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 18:54:14 2021

@author: Batuhan Duzgun (batux)
"""

import DataPreProcessingLibrary as dpt
from sklearn import datasets
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score, recall_score

def print_results(svm_type_text, y_test, predicted_y_values, C, gamma, degree):
    
    print("-------------------------------------------")
    print("C: " + str(C) + ", gamma: " + str(gamma) + ", degree: " + str(degree))
    print(svm_type_text + " >> R2 Value: " + str( r2_score(y_test, predicted_y_values) ))
    cm = confusion_matrix(y_test, predicted_y_values)
    print("*** Confusion Matrix ***")
    print(cm)
    print(svm_type_text + " Accuracy Rate: " + str(accuracy_score(y_test, predicted_y_values))) 
    print(svm_type_text + " Recall Rate: " + str(recall_score(y_test, predicted_y_values, average="macro"))) 

def create_svm_model(linear=False, kernel="rbf", degree=3, tol=1e-3, gamma="scale", C=1, class_weight="balanced", decision_function_shape="ovo", loss="squared_hinge"):
    
    if not linear:
        svm = SVC(
            C=C,
            gamma=gamma,
            kernel=kernel,
            degree=degree,
            tol=tol, 
            class_weight=class_weight, 
            decision_function_shape=decision_function_shape)
    else:
        svm = LinearSVC(C=C, loss=loss)
    return svm


rawData = datasets.load_iris()

ds = dpt.DataScaler()
scaled_data = ds.scale(rawData.data)

dsp = dpt.DataSplitter()
x_train, x_test, y_train, y_test = dsp.split(scaled_data, rawData.target, 0.25)



# 1- With default C and gamma values
print(">>>>>>>>>>> Default RBF Model <<<<<<<<<<<<<<")
const_C=1
const_gamma="scale"

svm = create_svm_model(kernel="rbf", C=const_C, gamma=const_gamma)
svm.fit(x_train, y_train)
predicted_y_values = svm.predict(x_test)
print_results("SVM (RBF)", y_test, predicted_y_values, const_C, const_gamma, 3)



# 2- With dynamic C and gamma values
print(">>>>>>>>>>> Dynamic RBF Model <<<<<<<<<<<<<<")
C_range = [1e-2, 1, 1e2]
gamma_range = [1e-1, 1, 1e1]
degree_range = [3,5,7]

for C in C_range:
    for gamma in gamma_range:
        for degree in degree_range:
            svm = create_svm_model(kernel="rbf", C=C, gamma=gamma, degree=degree)
            svm.fit(x_train, y_train)
            predicted_y_values = svm.predict(x_test)
            print_results("SVM (RBF)", y_test, predicted_y_values, C, gamma, degree)


# Best results for dynamic C and gamma pairs
# C: 1, gamma: 0.1
# C: 1, gamma: 1
# C: 100.0, gamma: 0.1
# C: 100.0, gamma: 1


# 3- Polynomial Kernel SVM
print(">>>>>>>>>>> Dynamic Poly Model <<<<<<<<<<<<<<")
for C in C_range:
    for gamma in gamma_range:
        for degree in degree_range:
            svm = create_svm_model(kernel="poly", degree=degree, C=C, gamma=gamma)
            svm.fit(x_train, y_train)
            predicted_y_values = svm.predict(x_test)
            print_results("SVM (Poly)", y_test, predicted_y_values, C, gamma, degree)

# Best results for dynamic C and gamma pairs
# C: 0.01, gamma: 10.0
# C: 1, gamma: 1
# C: 1, gamma: 10.0
# C: 100.0, gamma: 1
# C: 100.0, gamma: 10.0


# 4- LinearSVM Model
print(">>>>>>>>>>> Linear Model <<<<<<<<<<<<<<")
const_C=2
lsvm = create_svm_model(linear=True, C=const_C)

lsvm.fit(x_train, y_train)
predicted_y_values = lsvm.predict(x_test)
print_results("LinearSVM", y_test, predicted_y_values, const_C, "None", 1) 



