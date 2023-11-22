# Machine Learning Basics
# Assignment #05: MSE and R²

import pandas as pd
from sklearn import model_selection, linear_model, metrics
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Set Pandas Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def calculate_mse(y_real, y_predicted):
    """
    Manual calculation of mean squared error
    :param y_real: ground truth of target
    :param y_predicted: predicted target value
    :return: calculated mean squared error
    """
    n = len(y_real)
    mse = 1 / n * (np.sum((y_real - y_predicted)**2))
    return mse


def calculate_r2(y_real, y_predicted):
    """
    Manual calculation of r squared
    :param y_real: ground truth of target
    :param y_predicted: predicted target value
    :return: calculated r squared
    """
    model_rss = np.sum((y_predicted - y_real)**2)
    mean_rss = np.sum((np.mean(y_real) - y_real)**2)
    r2 = 1 - (model_rss / mean_rss)
    return r2


weather = pd.read_csv("weatherHistory.csv")
print(weather.columns)

# Rename column to be easier to use
weather = weather.rename(columns={"Temperature (C)": "temp"})

# Weather dataset: Question - Can I predict weather based on humidity?
#print(weather.describe())

target = weather.temp
feature = pd.DataFrame(weather, columns=["Humidity"])

# Split data in test and train
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    feature, target, shuffle=True, train_size=0.8)

# Use linear regression from sklearn to train and predict with X_test
regression = linear_model.LinearRegression().fit(X_train, y_train)
prediction = regression.predict(X_test)

## Results:
print("--- MSE ---")
print(f"Manually calculated: {calculate_mse(y_test, prediction)}")
print(f"Using metrics from sklearn: {metrics.mean_squared_error(y_test, prediction)}")

print("--- R squared ---")
print(f"Manually calculated: {calculate_r2(y_test, prediction)}")
print(f"Using metrics from sklearn: {metrics.r2_score(y_test, prediction)}")
print(f"Score of regression: {regression.score(X_train, y_train)}")

plt.figure(1)
plt.scatter(X_test, y_test, marker="^", linewidth=1)
plt.plot(X_test, prediction, color="r")
plt.scatter(X_test, prediction-y_test, color="g", marker="o")

model_rss = (prediction - y_test)**2

plt.figure(2)

sns.distplot(model_rss, hist=False)

ols = sm.OLS(y_train, X_train).fit()

# P-value is < 0.000 => significant
print(ols.summary())

plt.show()
