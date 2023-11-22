# Machine Learning 2: AI Concepts and Algorithms
# Assignment #04: DBSCAN

import random
import warnings

import numpy as np
import scipy as sp
from sklearn.cluster import KMeans

warnings.simplefilter(action="ignore", category=FutureWarning)

import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set Pandas Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def dbscan_manual(dataframe, epsilon, minpt, initial=True, list_of_points=None, current_cluster=1):
    """
    Manually perform DBSCAN clustering algorithm
    1) start at a random point
    2) calculate all distances for the other points, sort them
    3) find <minpt> points in <epsilon> range
    3)a) if found -> perform again for those points
    3)b) if not found -> assign point to noise (class 0) and start again at 1)
    4) iterate until all points are assigned

    :param dataframe: pandas DataFrame
    :param epsilon: range of distance
    :param minpt: minimal points needed for cluster formation
    :param initial: boolean, whether to start a new cluster
    :param list_of_points: list, points found when compared against minpt
    :param current_cluster: int, increasing value for cluster
    :return: updated pandas DataFrame
    """
    if initial:
        start_possibilities = dataframe[dataframe["cluster"].isnull()].index.tolist()
        start_point = random.randint(0, len(start_possibilities) - 1)
        checked = start_possibilities[start_point]
        center = dataframe.iloc[checked]
    else:
        checked = list_of_points.pop()
        center = dataframe.iloc[checked]
        dataframe.at[checked, "checked"] = True

    distances = dataframe.apply(lambda row: sp.spatial.distance.euclidean([row[0], row[1]], [center[0], center[1]]),
                                axis=1)
    distances = distances.sort_values(0)
    possible_distances = distances.index[distances <= epsilon].tolist()
    checked_distances = dataframe[dataframe["checked"]].index.tolist()
    list_of_points = list(set(possible_distances) - set(checked_distances))
    assigned_cluster = current_cluster
    if len(possible_distances) < minpt:
        assigned_cluster = 0
        dataframe.at[checked, "cluster"] = assigned_cluster
    else:
        for point in list_of_points:
            dataframe.at[point, "cluster"] = assigned_cluster
    while list_of_points and assigned_cluster:
        dbscan_manual(dataframe, epsilon, minpt, False, list_of_points, current_cluster)
    else:
        if dataframe[dataframe["cluster"].isnull()].index.tolist():
            dbscan_manual(dataframe, epsilon, minpt, True, None, dataframe["cluster"].max() + 1)
    return dataframe


def plot_dbscan(dataframe, amount_of_clusters, position):
    """
    Plot the DBSCAN data
    :param dataframe: pandas DataFrame
    :param amount_of_clusters: int, amount of clusters
    :param position: position for label
    """
    fig, ax = plt.subplots()
    scatter = ax.scatter(dataframe[0], dataframe[1], c=dataframe['cluster'], marker="o", edgecolors="black")
    ax.set_title(f"Manual DBSCAN with {amount_of_clusters} clusters and noise (0)")
    legend = ax.legend(*scatter.legend_elements(),
                       loc=position, title="Classes")
    ax.add_artist(legend)


def calculate_and_plot_kmeans(features, amount_of_clusters, position):
    """
    Perform KMeans from sklearns library using k from dbscan and plot the result
    :param features: array of x, y values
    :param amount_of_clusters: int, amount of clusters based on DBSCAN
    :param position: position for label
    """
    plt.figure('b')
    y_pred = KMeans(n_clusters=amount_of_clusters).fit_predict(features)
    fig, ax = plt.subplots()
    scatter = ax.scatter(features[:, 0], features[:, 1], c=y_pred, marker="o", edgecolors="black")
    ax.set_title(f"KMeans with {amount_of_clusters} clusters")
    legend = ax.legend(*scatter.legend_elements(),
                       loc=position, title="Classes")
    ax.add_artist(legend)


## Dataset 1:
X, true_labels = datasets.make_blobs(n_samples=100, centers=5, cluster_std=1, random_state=1)
X = StandardScaler().fit_transform(X)
X_array = pd.DataFrame(X)
X_array = X_array.assign(cluster=None, checked=False)
dataset = dbscan_manual(X_array, 0.3, 5)
k = dataset["cluster"].max()
plot_dbscan(dataset, k, "upper left")
calculate_and_plot_kmeans(X, k, "upper left")

## Dataset 2:
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_aniso = np.dot(X, transformation)
X_aniso_ = pd.DataFrame(X_aniso)
X_aniso_ = X_aniso_.assign(cluster=None, checked=False)
dataset = dbscan_manual(X_aniso_, 0.3, 7)
k = dataset["cluster"].max()
plot_dbscan(dataset, k, "lower left")
calculate_and_plot_kmeans(X_aniso, k, "lower left")

## Dataset 3:
X_varied, y_varied = datasets.make_blobs(
    n_samples=300, cluster_std=[1.0, 2.5, 0.5], random_state=1
)
X_varied_ = pd.DataFrame(X_varied)
X_varied_ = X_varied_.assign(cluster=None, checked=False)
dataset = dbscan_manual(X_varied_, 0.6, 8)
k = dataset["cluster"].max()
plot_dbscan(dataset, k, "lower left")
calculate_and_plot_kmeans(X_varied, k, "lower left")

plt.show()
