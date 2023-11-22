#!/usr/bin/env python3

# Machine Learning Basics
# Assignment 01: Terminology

from sklearn import datasets, neighbors

# Dataset of 3 species of Iris
iris = datasets.load_iris()
print(f"Targets: {[name for name in iris.target_names]}")
# Consisting of 4 features
print(f"Features: {[name for name in iris.feature_names]}")

# The algorithm used is KNN (k nearest neighbors), classification
yourAlgorithm = neighbors.KNeighborsClassifier()
# In this example it uses the dataset minus the first row to fit the algorithm
yourAlgorithm.fit(iris.data[1:], iris.target[1:])
# To test the data I took the first row, which was not used in the training data
test_data = [iris.data[0]]
print(f"Test data: {test_data}, target: {iris.target[0]}, label: {iris.target_names[0]}")
my_prediction = yourAlgorithm.predict(test_data)
# This means it should predict setosa, which it does.
print(f"Prediction: {iris.target_names[my_prediction]}")

# To better train the data, the available data could be split randomly in, for example
# 10 parts where 9 are used to train and 1 is used to test.
