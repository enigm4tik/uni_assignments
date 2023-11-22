# Machine Learning Basics
# Assignment 02: Implement knn algorithm

from sklearn import datasets, preprocessing, model_selection
import matplotlib.pyplot as plt 
import numpy as np

def knn(X, y, k=1):
    """
        Implementation of k nearest neighbors
        1. Calculate euclidean distance to all training data points
        2. Choose first k elements
        3. Find majority of classes 
        3.1. Resolve at tie: take first class
    """
    if k > len(X):
        error_message = f'Error: k ({k}) is greater than sum of samples for sample size {len(X)}.'
        return error_message

    def _calculate_performance_metrics(metrics):
        true_positive, false_positive, true_negative, false_negative = metrics

        negative = true_negative + false_negative
        positive = true_positive + false_positive
        accuracy = (true_positive + true_negative) / (positive + negative)
        if positive > 0:
            tpr = true_positive / positive
        else:
            tpr = None
        if negative > 0:
            tnr = true_negative / negative
        else:
            tnr = None
        return tpr, tnr, accuracy

    def _find_majority_class(matrix, k, y):
        smallest_values = []
        sorted_matrix = dict(sorted(matrix.items(), key=lambda item: item[1]))
        sorted_keys = [index for index in sorted_matrix.keys()]

        for i in range(k):
            smallest_values.append(y[sorted_keys[i]])

        majority_class = np.bincount(smallest_values).argmax()
        return majority_class

    true_positive = true_negative = false_negative = false_positive = 0
    predicted = []
    for index, row in enumerate(X):
        closest = {}
        matrix = X
        matrix_removed = matrix - row
        matrix_squared = matrix_removed * matrix_removed
        matrix_sqrt = np.sqrt(matrix_squared)
        for index2, distance_row in enumerate(matrix_sqrt):
            closest[index2] = abs(np.sum(distance_row))
        predicted_class = _find_majority_class(closest, k, y)
        actual_class = y[index]
        predicted.append((predicted_class, actual_class))
        if predicted_class == actual_class == 0:
            true_positive += 1
        elif predicted_class == actual_class == 1:
            true_negative += 1
        elif predicted_class == 1 and actual_class == 0:
            false_negative += 1
        elif predicted_class == 0 and actual_class == 1:
            false_positive += 1
    metrics = (true_positive, false_positive, true_negative, false_negative)
    tpr, tnr, accuracy = _calculate_performance_metrics(metrics)
    return tpr, tnr, accuracy


def perform_knn_x_fold_cross_validation(fold, X, y, k):
    runs = {"tpr": 0, "tnr": 0, "accuracy": 0}
    valid_ks = fold
    for i in range(fold):
        tpr, tnr, accuracy = knn(X, y, k)
        if not tpr is None and not tnr is None:
            runs["tpr"] += tpr
            runs["tnr"] += tnr
        else:
            valid_ks -= 1
        runs["accuracy"] += accuracy
    if valid_ks > 0:
        runs = {key: value / valid_ks for key, value in runs.items()}
    else:
        runs["accuracy"] /= fold
    return runs

# Dataset Breast Cancer
breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

# Split the training set from the test set
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, shuffle=True, train_size=0.8)

#10 fold cross validation with hyperaparameter tuning values 1, 300 in interval of 25
result = {}
for i in range(1, 300, 25):
    result[i] = perform_knn_x_fold_cross_validation(10, X_train, y_train, i)

x_values = list(result.keys())
y_values = [value['accuracy'] for value in result.values()]
plt.figure(0)
plt.plot(x_values, y_values, label="Training Set")
plt.ylabel("Accuracy")
plt.xlabel("Hyperparameter k")
plt.legend(loc="best")

tpr = [value['tpr'] for value in result.values()]
fpr = [1-value['tnr'] for value in result.values()]
plt.figure(1)
plt.plot(x_values, tpr, label="True Positive Rate")
plt.xlabel("k")
plt.ylabel("TPR/FPR")
plt.plot(x_values, fpr, label="False Positive Rate")
plt.legend(loc="best")

# Scale data to see if there is a difference
X_scaler = preprocessing.StandardScaler().fit(X)
X_train_scaled = X_scaler.transform(X_train)

scaled_result = {}
for i in range(1, 300, 25):
    scaled_result[i] = perform_knn_x_fold_cross_validation(10, X_train_scaled, y_train, i)

x_scaled_values = list(scaled_result.keys())
y_scaled_values = [value['accuracy'] for value in scaled_result.values()]
plt.figure(0)
plt.plot(x_scaled_values, y_scaled_values, label="Scaled Training Set")
plt.ylabel("Accuracy")
plt.xlabel("Hyperparameter k")
plt.legend(loc="best")

## The scaled data performs slightly better than the original data.

# The best value that is not 100% (with k = 1) is k=26.
# Use k = 26 on the test data.
k = 26
tpr, tnr, accuracy = knn(X_test, y_test, k)

print(f"Final accuracy of self-implemented k-nearest neighbor with k = {k} has the accuracy of {round(accuracy*100, 2)}%. True positive rate = {round(tpr*100, 2)}% and false positive rate = {round((1-tnr)*100, 2)}%")

plt.show()
