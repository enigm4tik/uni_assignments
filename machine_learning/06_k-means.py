# Machine Learning Basics
# Assignment #06: K-means implementation

import pandas as pd
from sklearn import cluster
from timeit import default_timer as timer
from matplotlib import pyplot as plt
import scipy as sp
from random import randint
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)
# Set Pandas Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def calculate_clusters(dataframe, centroids):
    """
    Calculate clusters based on centroids using euclidean distance
    :param dataframe: Pandas DataFrame
    :param centroids: list of Tuples with x, y coordinates for the centroids
    :return: updated dataframe
    """
    for index in range(len(centroids)):
        dataframe[index] = dataframe.apply(
            lambda row: sp.spatial.distance.euclidean([row[annual], row[spending]],
                                                      [centroids[index][0], centroids[index][1]]),
            axis=1)
    dataframe[clusters] = dataframe[[*[index for index in range(len(centroids))]]].idxmin(axis=1)
    return dataframe


def find_new_centroids(dataframe):
    """
    Calculate new centroids based on dataframe and assigned clusters
    :param dataframe: Updated dataframe with column: "Clusters"
    :return: list of calculated centroids (tuples of x, y coordinates)
    """
    centroids = []
    cluster_groups = dataframe.groupby(clusters).mean()
    for index in range(len(cluster_groups)):
        centroids.append((cluster_groups[annual][index], cluster_groups[spending][index]))
    return centroids


def kmeans_manual(dataframe, k, max_iterations, current_iteration=0, centroids=None):
    """
    Calculate the kmeans manually using a dataframe of 2 features
    Draw plots for each iteration showing the different clusters and their centroids
    :param dataframe: pandas DataFrame
    :param k: amount of clusters
    :param max_iterations: maximum iterations allowed
    :param current_iteration: optional, starts with 0 and increases by 1 for each iteration
    :centroids: optional, list of centroids (tuples of x, y coordinates)
    :return: updated dataframe
    """
    plt.figure(current_iteration)
    if current_iteration == 0:
        # place random centroids
        max = len(dataframe)
        centroids = []
        for i in range(k):
            random_int = (randint(0, max))
            centroids.append((dataframe.iloc[random_int][annual], dataframe.iloc[random_int][spending]))
    dataframe = calculate_clusters(dataframe, centroids)
    grouped = dataframe.groupby(clusters)
    old_centroids = centroids
    centroids = find_new_centroids(dataframe)
    if old_centroids == centroids:
        return dataframe

    plt.title(f"Iteration: {current_iteration+1}")
    for key, group in grouped:
        plt.scatter(group[annual], group[spending], label=key+1)
    plt.xlabel(annual)
    plt.ylabel(spending)
    for x, y in centroids:
        plt.plot(x, y, "k1", linewidth=2)
    plt.legend(loc="best")
    plt.savefig(str(current_iteration) + "_plot.png")
    current_iteration += 1
    if current_iteration < max_iterations:
        dataframe = kmeans_manual(dataframe, k, max_iterations, current_iteration, centroids)
    else:
        print(f"Centers are no longer moving after {current_iteration} iterations.")
    return dataframe


# Describe Dataset
dataset = pd.read_csv('Mall_Customers.csv')
print(dataset.head())
print(dataset.shape)
print(dataset.info())

# Global Variables for this dataset
annual = "Annual Income (k$)"
spending = "Spending Score (1-100)"
clusters = "Cluster"

# Feature selection: annual income & spending score
features = pd.DataFrame(dataset, columns=[annual, spending])

# Take time to compare both version, keep in mind that plots are being drawn
start = timer()
data = kmeans_manual(features, 5, 20)
end = timer()
print(f"Manual calculation: {round(end - start, 3)} seconds.")

# Calculating clusters using sklearn
wcss = []
for i in range(1, 20):
    kmeans = cluster.KMeans(n_clusters=i, init="k-means++", random_state=0)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)

start = timer()
kmeans = cluster.KMeans(n_clusters=5, random_state=0)
y_kmeans = kmeans.fit_predict(features)
end = timer()
print(f"Calculation using sklearn: {round(end - start, 3)} seconds.")
plt.show()
