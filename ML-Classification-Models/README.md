
# Basic Classification Models on "iris" data set with SVM and KNN

# KNN Machine Learning Model

Details of KNN Algorithm in Sci-learn Library:
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

KNN Algorithm Details:
https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

KNN Results
```console
---------------------------------------------
n = 4, algorithm: auto, leaf_size: 30, metric: euclidean
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 5, algorithm: auto, leaf_size: 30, metric: euclidean
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 6, algorithm: auto, leaf_size: 30, metric: euclidean
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 7, algorithm: auto, leaf_size: 30, metric: euclidean
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 4, algorithm: auto, leaf_size: 30, metric: manhattan
KNN Accuracy Rate: 0.9473684210526315
---------------------------------------------
n = 5, algorithm: auto, leaf_size: 30, metric: manhattan
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 6, algorithm: auto, leaf_size: 30, metric: manhattan
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 7, algorithm: auto, leaf_size: 30, metric: manhattan
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 4, algorithm: auto, leaf_size: 30, metric: chebyshev
KNN Accuracy Rate: 0.9473684210526315
---------------------------------------------
n = 5, algorithm: auto, leaf_size: 30, metric: chebyshev
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 6, algorithm: auto, leaf_size: 30, metric: chebyshev
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 7, algorithm: auto, leaf_size: 30, metric: chebyshev
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 4, algorithm: auto, leaf_size: 30, metric: minkowski
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 5, algorithm: auto, leaf_size: 30, metric: minkowski
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 6, algorithm: auto, leaf_size: 30, metric: minkowski
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 7, algorithm: auto, leaf_size: 30, metric: minkowski
KNN Accuracy Rate: 0.9736842105263158
---------------------------------------------
n = 4, algorithm: auto, leaf_size: 30, metric: wminkowski
KNN Accuracy Rate: 0.9473684210526315
---------------------------------------------
n = 5, algorithm: auto, leaf_size: 30, metric: wminkowski
KNN Accuracy Rate: 0.9473684210526315
---------------------------------------------
n = 6, algorithm: auto, leaf_size: 30, metric: wminkowski
KNN Accuracy Rate: 0.9473684210526315
---------------------------------------------
n = 7, algorithm: auto, leaf_size: 30, metric: wminkowski
KNN Accuracy Rate: 0.9473684210526315
---------------------------------------------
n = 4, algorithm: auto, leaf_size: 30, metric: mahalanobis
KNN Accuracy Rate: 0.9210526315789473
---------------------------------------------
n = 5, algorithm: auto, leaf_size: 30, metric: mahalanobis
KNN Accuracy Rate: 0.9210526315789473
---------------------------------------------
n = 6, algorithm: auto, leaf_size: 30, metric: mahalanobis
KNN Accuracy Rate: 0.9210526315789473
---------------------------------------------
n = 7, algorithm: auto, leaf_size: 30, metric: mahalanobis
KNN Accuracy Rate: 0.9210526315789473
...
```

# SVM Machine Learning Model

We tried 3 types of SVM kernels.

- Linear Kernel
- Polynomial Kernel
- RBF Kernel

Note: When you find the optimal C and Gamma values, after that increasing the degree of model might not change the accuracy rate anymore on this dataset for RBF and Polynomial models.

Details of SVM Algorithm in Sci-learn Library:
https://scikit-learn.org/stable/modules/svm.html

SVM Algorithm Details:
https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47


Best results with dynamic C, Gamma and Degree values on SVM (RBF)
```console
-------------------------------------------
C: 1, gamma: 0.1
SVM (RBF) >> R2 Value: 0.9536585365853658
*** Confusion Matrix ***
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]
SVM (RBF) Accuracy Rate: 0.9736842105263158
SVM (RBF) Recall Rate: 0.9791666666666666
-------------------------------------------
C: 1, gamma: 1
SVM (RBF) >> R2 Value: 0.9536585365853658
*** Confusion Matrix ***
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]
SVM (RBF) Accuracy Rate: 0.9736842105263158
SVM (RBF) Recall Rate: 0.9791666666666666
```

Best results with dynamic C, Gamma and Degree values on SVM (Poly)
```console
-------------------------------------------
C: 1, gamma: 1, degree: 3
SVM (Poly) >> R2 Value: 0.9536585365853658
*** Confusion Matrix ***
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]
SVM (Poly) Accuracy Rate: 0.9736842105263158
SVM (Poly) Recall Rate: 0.9791666666666666
-------------------------------------------
C: 100.0, gamma: 1, degree: 3
SVM (Poly) >> R2 Value: 0.9536585365853658
*** Confusion Matrix ***
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]
SVM (Poly) Accuracy Rate: 0.9736842105263158
SVM (Poly) Recall Rate: 0.9791666666666666
-------------------------------------------
C: 100.0, gamma: 1, degree: 7
SVM (Poly) >> R2 Value: 0.9536585365853658
*** Confusion Matrix ***
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]
SVM (Poly) Accuracy Rate: 0.9736842105263158
SVM (Poly) Recall Rate: 0.9791666666666666
```

Best result for LinearSVM
```console
-------------------------------------------
C: 2, gamma: None
LinearSVM >> R2 Value: 0.9536585365853658
*** Confusion Matrix ***
[[13  0  0]
 [ 0 15  1]
 [ 0  0  9]]
LinearSVM Accuracy Rate: 0.9736842105263158
LinearSVM Recall Rate: 0.9791666666666666
```

Other observations

When C is 0.01 and Gamma is 1, if we increase the degree of model, we can get better accuracy on iris dataset.
```console
-------------------------------------------
C: 0.01, gamma: 1, degree: 3
SVM (Poly) >> R2 Value: 0.7682926829268293
*** Confusion Matrix ***
[[13  0  0]
 [ 0 16  0]
 [ 0  5  4]]
SVM (Poly) Accuracy Rate: 0.868421052631579
SVM (Poly) Recall Rate: 0.8148148148148149
-------------------------------------------
C: 0.01, gamma: 1, degree: 5
SVM (Poly) >> R2 Value: 0.8146341463414635
*** Confusion Matrix ***
[[13  0  0]
 [ 0 16  0]
 [ 0  4  5]]
SVM (Poly) Accuracy Rate: 0.8947368421052632
SVM (Poly) Recall Rate: 0.8518518518518517
-------------------------------------------
C: 0.01, gamma: 1, degree: 7
SVM (Poly) >> R2 Value: 0.8609756097560975
*** Confusion Matrix ***
[[13  0  0]
 [ 0 16  0]
 [ 0  3  6]]
SVM (Poly) Accuracy Rate: 0.9210526315789473
SVM (Poly) Recall Rate: 0.8888888888888888
```

But on the other hand, when C is 1 and Gamma is 0.1, if we increase the degree of model, we can get worse accuracy on iris dataset.
It shows that changing parameters of ML model might provide better or worse accuracy rates!
```console
-------------------------------------------
C: 1, gamma: 0.1, degree: 3
SVM (Poly) >> R2 Value: 0.6756097560975609
*** Confusion Matrix ***
[[13  0  0]
 [ 0 16  0]
 [ 0  7  2]]
SVM (Poly) Accuracy Rate: 0.8157894736842105
SVM (Poly) Recall Rate: 0.7407407407407408
-------------------------------------------
C: 1, gamma: 0.1, degree: 5
SVM (Poly) >> R2 Value: 0.4439024390243902
*** Confusion Matrix ***
[[10  3  0]
 [ 0 16  0]
 [ 0  9  0]]
SVM (Poly) Accuracy Rate: 0.6842105263157895
SVM (Poly) Recall Rate: 0.5897435897435898
-------------------------------------------
C: 1, gamma: 0.1, degree: 7
SVM (Poly) >> R2 Value: 0.07317073170731703
*** Confusion Matrix ***
[[ 2 11  0]
 [ 0 16  0]
 [ 0  9  0]]
SVM (Poly) Accuracy Rate: 0.47368421052631576
SVM (Poly) Recall Rate: 0.3846153846153846
```
