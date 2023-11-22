# Machine Learning 2: AI Concepts and Algorithms
# Assignment #02: Multiple Regression, forward selection vs. backward selection

import pandas as pd
import statsmodels.api as sm
from sklearn import model_selection, linear_model, metrics

# Set Pandas Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def find_highest_pvalue(pvalues):
    """
    Finds the highest pvalue out of pandas series
    Determines if all pvalues are lower than 0.5
    :param pvalues: pvalues from ols model
    :return: name of feature with biggest pvalue and boolean again
    """
    again = False
    pvalue_dict = dict(pvalues)
    biggest_pvalue = 0
    biggest_index = ""
    for pvalue in pvalue_dict:
        if pvalue_dict[pvalue] > biggest_pvalue:
            biggest_pvalue = pvalue_dict[pvalue]
            biggest_index = pvalue
    if biggest_pvalue > 0.05:
        again = True
    return biggest_index, again


def backwards_feature_selection(data, target, list_of_features, end_values={}, iteration=0):
    """
    Backwards feature selection:
    Add all features and recursively remove feature with lowest pvalue
    until all remaining pvalues are < 0.05
    :param data: Pandas dataframe
    :param target: Target column
    :param list_of_features: list of strings containing feature names
    :param end_values: result container (dict)
    :param iteration: number of runs
    :return dictionary of end values
    """
    features = pd.DataFrame(data, columns=[*list_of_features])
    ols = sm.OLS(target, features).fit()
    biggest_index, again = find_highest_pvalue(ols.pvalues)
    print(ols.summary())
    end_values[iteration] = {
        "r2": ols.rsquared,
        "pvalues": ols.pvalues,
    }
    if again:
        feature_columns.remove(biggest_index)
        backwards_feature_selection(data, target, feature_columns, end_values, iteration + 1)
    return end_values

def forward_feature_selection(data, target, list_of_features, resulting_features = [], iteration = 0):
    """
    Forward feature selection:
    Find the feature with the lowest MSE
    Add that to the resulting model until the pvalues of the features > 0.05
    Remove the last added feature
    :param data: Pandas dataframe
    :param target: Target column
    :param list_of_features: list of strings containing feature names
    :param resulting_features: list of features selected
    :param iteration: number of runs
    :return dictionary of end values
    """
    smallest_mse = float("inf")
    best_feature = ""
    for feature in list_of_features:
        feature_data = pd.DataFrame(data, columns=[feature])
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            feature_data, target, shuffle=True, train_size=0.8)
        regression = linear_model.LinearRegression().fit(X_train, y_train)
        prediction = regression.predict(X_test)
        mse = metrics.mean_squared_error(y_test, prediction)
        if mse < smallest_mse:
            smallest_mse = mse
            best_feature = feature
    resulting_features.append(best_feature)
    list_of_features.remove(best_feature)
    ols = sm.OLS(target, pd.DataFrame(data, columns=resulting_features)).fit()
    biggest_index, again = find_highest_pvalue(ols.pvalues)
    print(ols.summary())
    if not again:
        forward_feature_selection(data, target, list_of_features, resulting_features, iteration + 1)
    else:
        resulting_features.remove(best_feature)
    return resulting_features

dataset = pd.read_csv("Walmart.csv")
target_data = dataset.Weekly_Sales
feature_columns = ["Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]

print("\n############ BACKWARD FEATURE SELECTION ############\n")
results = backwards_feature_selection(dataset, target_data, feature_columns)

# for run, result in results.items(): ## Collection of results, also found in summary
#     print("\nIteration: ", run + 1)
#     for key, value in result.items():
#         print("\n", key, "\n", value)

print("\n############ FORWARD FEATURE SELECTION ############\n")
feature_columns = ["Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment"]
print(forward_feature_selection(dataset, target_data, feature_columns))


### Summary:

## Dataset:
# Walmart Weekly Sales Data

## Question: which features can predict the weekly sales?

## Features:

# Holiday Flag: Is this week a special holiday week or a regular week?
# Temperature: Average Temperature in the area
# Fuel Price: Average Fuel Price in the area
# CPI: Consumer Price Index, Measure for inflation
# Unemployment: Average unemployment rate in the area

## Results:
# Both backwards and forwards feature selection showed that temperature is the
# least significant factor to predict sales.
# Also (maybe counterintuitively) unemployment rate showed low significance.
