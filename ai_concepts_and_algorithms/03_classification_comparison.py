# Machine Learning 2: AI Concepts and Algorithms
# Assignment #03: Comparison of Classification using 4 algorithms
# Decision Tree vs. K-nearest neighbors vs. Artificial Neural Network vs. Logistic Regression

import warnings

import matplotlib.pyplot as plt
from mlxtend.preprocessing import minmax_scaling
from seaborn import distplot
from sklearn import datasets, model_selection, tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')  # "distplot is deprecated" Warning suppressed


# Helper Functions

def create_data_set(number_of_dataset, save=False):
    """
    Create a random dataset using make_classification() and scale the data.
    Prepare 2 plots per dataset (and its scaled counterpart).
    :param number_of_dataset: int, number of current dataset
    :param save: boolean, True if plots should be saved
    :return features, scaled features, target of dataset
    """
    X, y = datasets.make_classification()
    number_of_dataset += 1  # offset to start with 1

    plt.figure(str(number_of_dataset) + 'original')
    plt.boxplot(X)
    plt.title(f'Original dataset: {number_of_dataset}')
    if save:
        plt.savefig(f"reg_log/{number_of_dataset}_original.png")
    plt.figure(str(number_of_dataset) + 'density')
    distplot(X)
    plt.title(f'Original dataset: {number_of_dataset}')
    if save:
        plt.savefig(f"reg_log/{number_of_dataset}_original_density.png")

    scaled_X = minmax_scaling(X, [i for i in range(20)])

    plt.figure(str(number_of_dataset) + 'scaled')
    plt.boxplot(scaled_X)
    plt.title(f'Scaled dataset: {number_of_dataset}')
    if save:
        plt.savefig(f"reg_log/{number_of_dataset}_scaled.png")
    plt.figure(str(number_of_dataset) + 'density_scaled')
    distplot(scaled_X)
    plt.title(f'Scaled dataset: {number_of_dataset}')
    if save:
        plt.savefig(f"reg_log/{number_of_dataset}_scaled_density.png")

    return X, scaled_X, y


def prepare_result_set(result, iteration):
    """
    Helper function that prepares the result dictionary for each iteration
    :param result: dictionary, results to be updated
    :param iteration: int, number of iterations run
    :return: dictionary, updated result dictionary
    """
    result[iteration] = {
        "original": {
            "tree": [],
            "knn": [],
            "ann": [],
            "log_reg": []
        },
        "scaled": {
            "tree": [],
            "knn": [],
            "ann": [],
            "log_reg": []
        }
    }
    return result


def hyper_parameter_tuning_for_knn(X, y):
    """
    Helper function to determine the best k for k-nearest neighbor per dataset.
    :param X: features of the dataset
    :param y: target of the dataset
    :return: int, the best parameter found
    """

    X_train, X_valid, y_train, y_valid = train_test_split_helper(X, y)
    base = KNeighborsClassifier()
    base_random = RandomizedSearchCV(estimator=base, param_distributions={"n_neighbors": [i for i in range(1, 10)]},
                                     n_iter=9, cv=5, n_jobs=-1)
    base_random.fit(X_train, y_train)
    best_k = base_random.best_params_['n_neighbors']
    return best_k


def hyper_parameter_tuning_for_ann(X, y):
    """
    Helper function to determine the best hyper parameters (selection) for ANN.
    :param X: features of the dataset
    :param y: target of the dataset
    :return: dictionary of the best hyper parameters
    """
    X_train, X_valid, y_train, y_valid = train_test_split_helper(X, y)
    base = MLPClassifier()
    param_distributions = {
        "solver": ['lbfgs', 'sgd', 'adam'],
        "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "hidden_layer_sizes": [(i, j) for j in range(1, 6) for i in range(1, 6)]
    }
    base_random = RandomizedSearchCV(estimator=base, param_distributions=param_distributions, cv=10)
    base_random.fit(X_train, y_train)
    best_params = base_random.best_params_
    return best_params


def train_test_split_helper(X, y, size=0.8):
    """
    Apply train test split function on targets and features given size percentage.
    :param X: features of the dataset
    :param y: target of the dataset
    :param size: float, percentage of train vs. test data, default 0.8
    :return: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, shuffle=True, train_size=size)
    return X_train, X_test, y_train, y_test


def perform_one_iteration(X, y, k, ann_hyperparameters):
    """
    Calculate one cross validation: calculate the accuracy for all 4 algorithms:
    * Decision Tree
    * K-nearest Neighbors
    * Artificial Neural Network
    * Logistic Regression

    :param X: features of a dataset
    :param y: target of a dataset
    :param k: k used for k-nearest neighbors
    :param ann_hyperparameters: dictionary of best determined hyperparameters
    return: accuracy of one run
    """

    X_train, X_test, y_train, y_test = train_test_split_helper(X, y)

    # Decision Tree
    decision_tree = classify_with_decision_tree(X_train, y_train, X_test, y_test)

    # K-nearest Neighbors
    knn = classify_with_knn(X_train, y_train, X_test, y_test, k)

    # Artificial Neural Network
    ann = classify_with_ann(X_train, y_train, X_test, y_test, ann_hyperparameters)

    # Logistic Regression
    log_reg = classify_with_logistic_regression(X_train, y_train, X_test, y_test)
    return decision_tree, knn, ann, log_reg


def calculate_accuracy(prediction, reality):
    """
    Calculate the accuracy based on predicted vs. actual classes.
    :param prediction: array, values 0 or 1 for classes
    :param reality: array, values 0 or 1 for classes
    :return calculated accuracy
    """
    occurences = len(prediction)
    correct = 0
    for i in range(occurences):
        if prediction[i] == reality[i]:
            correct += 1
    accuracy = correct / len(prediction)
    return accuracy


def update_results(result, iteration, accuracies, scaled=False):
    """
    Updates the result dictionary with accuracies
    :param result: dictionary, contains current accuracies
    :param iteration: int, current iteration
    :param accuracies: current calculated accuracies of all 4 algorithms
    :param scaled: boolean, whether the data is scaled
    :return: updated result dictionary
    """
    data_type = "original"
    if scaled:
        data_type = "scaled"
    decision_tree, knn, ann, log_reg = accuracies

    result[iteration][data_type]["tree"].append(decision_tree)
    result[iteration][data_type]["knn"].append(knn)
    result[iteration][data_type]["ann"].append(ann)
    result[iteration][data_type]["log_reg"].append(log_reg)
    return result


def calculate_mean_accuracy(result):
    """
    Calculate the mean accuracy for all Cross validation runs.
    :param result: dictionary, containing all information of all accuracies
    :return: dictionary, calculated mean accuracies
    """
    print("Calculating mean accuracy...")
    results = {}
    for dataset in result:
        results[dataset] = {}
        for datatype in result[dataset]:
            results[dataset][datatype] = {}
            for algorithm in result[dataset][datatype]:
                data_array = result[dataset][datatype][algorithm]
                results[dataset][datatype][algorithm] = round(sum(data_array) / len(data_array), 2)
    return results


def pretty_print_results(results, n, k):
    """
    Print the results in an aesthetically pleasing way using a table.
    :param results: dictionary, containing all mean accuracies
    :param n: amount of datasets used
    :param k: amount of cross validations performed
    """
    print(f"Results for {n} datasets with {k}-fold Cross-Validation.")
    print(f"{'':_^70}")
    for result in results:
        original_data = results[result]['original']
        scaled_data = results[result]['scaled']
        print(f"Dataset {result + 1}")
        print(f"{'Algorithm'} | {'Original Data'} | {'Scaled Data'}")
        for algorithm in original_data:
            print(f"{algorithm: ^9} | {original_data[algorithm]: ^13} | {scaled_data[algorithm]: ^8}")
        print(f"{'':_^70}")


# Classification Functions

def classify_with_decision_tree(X_train, y_train, X_test, y_test):
    """
    Perform classification using sklearn's DecisionTreeClassifier
    :param X_train: training features of the dataset
    :param y_train: training target of the dataset
    :param X_test: test features of the dataset
    :param y_test: test target of the dataset
    :return: calculated accuracy
    """
    classification = tree.DecisionTreeClassifier()
    classification = classification.fit(X_train, y_train)
    predict = classification.predict(X_test)
    accuracy = calculate_accuracy(predict, y_test)
    return accuracy


def classify_with_knn(X_train, y_train, X_test, y_test, k):
    """
    Perform classification using sklearn's KNeighborsClassifier
    :param X_train: training features of the dataset
    :param y_train: training target of the dataset
    :param X_test: test features of the dataset
    :param y_test: test target of the dataset
    :param k: number of neighbors
    :return: calculated accuracy
    """
    classification = KNeighborsClassifier(n_neighbors=k)
    classification.fit(X_train, y_train)
    predict = classification.predict(X_test)
    accuracy = calculate_accuracy(predict, y_test)
    return accuracy


def classify_with_ann(X_train, y_train, X_test, y_test, hyperparameters):
    """
    Perform classification using sklearn's MLPClassifier
    :param X_train: training features of the dataset
    :param y_train: training target of the dataset
    :param X_test: test features of the dataset
    :param y_test: test target of the dataset
    :return: calculated accuracy
    """
    classification = MLPClassifier(**hyperparameters)
    classification.fit(X_train, y_train)
    predict = classification.predict(X_test)
    accuracy = calculate_accuracy(predict, y_test)
    return accuracy


def classify_with_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Perform classification using sklearn's LogisticRegression
    :param X_train: training features of the dataset
    :param y_train: training target of the dataset
    :param X_test: test features of the dataset
    :param y_test: test target of the dataset
    :return: calculated accuracy
    """
    classification = LogisticRegression().fit(X_train, y_train)
    predict = classification.predict(X_test)
    accuracy = calculate_accuracy(predict, y_test)
    return accuracy


# Assignment 03


def ml_workflow(number_of_datasets=4, k_fold=10, save_plots=False):
    """
    Perform Machine Learning Workflow
    1. Create (or use) a dataset
    2. Scaling
    3. Hyperparameter tuning (in this case performed for k in k-nearest neighbors)
    4. K-fold Cross Validation

    Performed for number_of_datasets datasets, k_fold cross validations and 4 algorithms:
    * Decision Tree
    * K-nearest Neighbors
    * Artificial Neural Network
    * Logistic Regression

    :param number_of_datasets: int, amount of datasets to analyze
    :param k_fold: int, amount of cross validations
    :param save_plots: boolean, save plots to disk
    :return: dictionary, mean accuracy for all datasets and each algorithm used
    """
    result = {}
    for i in range(number_of_datasets):
        print(f"Analysis of dataset {i+1}")
        original_X, scaled_X, y = create_data_set(i, save=save_plots)
        result = prepare_result_set(result, i)

        print(f"Start Hyper parameter Tuning for dataset {i + 1}")
        best_hyper_original = hyper_parameter_tuning_for_ann(original_X, y)
        best_hyper_scaled = hyper_parameter_tuning_for_ann(scaled_X, y)
        print(f"Best hyper parameters for ANN original: {best_hyper_original}")
        print(f"Best hyper parameters for ANN scaled: {best_hyper_scaled}")

        best_k_original = hyper_parameter_tuning_for_knn(original_X, y)
        best_k_scaled = hyper_parameter_tuning_for_knn(scaled_X, y)
        print(f"Best k for KNN original: {best_k_original}")
        print(f"Best k for KNN scaled: {best_k_scaled}")

        for j in range(k_fold):
            original_data = perform_one_iteration(original_X, y, best_k_original, best_hyper_original)
            scaled_data = perform_one_iteration(scaled_X, y, best_k_scaled, best_hyper_scaled)
            result = update_results(result, i, original_data)
            result = update_results(result, i, scaled_data, True)

    mean_accuracy = calculate_mean_accuracy(result)
    return mean_accuracy


# Initialization
classification_datasets = 5
k_fold_cross_validation = 10

mean_accuracies = ml_workflow(classification_datasets, k_fold_cross_validation, save_plots=True)
pretty_print_results(mean_accuracies, classification_datasets, k_fold_cross_validation)

plt.show()
