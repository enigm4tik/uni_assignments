# Machine Learning 2: AI Concepts and Algorithms
# Assignment #04: tSNE

from sklearn import datasets, manifold, decomposition
import matplotlib.pyplot as plt
from matplotlib import ticker


def plot_3d(dataset, color_array, title):
    """
    Create a 3d plot for 3 dimensional data
    :param dataset: 3d dataset
    :param color_array: 1d array of clusters (used for color)
    :param title: str, Title of plot
    """
    x, y, z = dataset.T

    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=color_array, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))

    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


def plot_2d(dataset, color_array, title):
    """
    Plot 2d data based on <dataset> and color with <color_array>.
    :param dataset: 2d dataset
    :param color_array: 1d array
    :param title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    x, y = dataset.T
    ax.scatter(x, y, c=color_array, s=50, alpha=0.8)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    plt.show()


def perform_tsne(dataset, perplexity):
    """
    Perform tSNE on given <dataset> with given <perplexity>
    :param dataset: 3d dataset
    :param perplexity: int, level of perplexity
    :return fit tsne model
    """
    t_sne = manifold.TSNE(
        n_components=2,
        perplexity=perplexity,
        init="random",
        n_iter=500, random_state=1337
    )
    t_sne = t_sne.fit_transform(dataset)
    return t_sne


def perform_pca(dataset):
    """
    Perform PCA on 3d <dataset>
    :param dataset: 3d dataset
    :return: fit pca model
    """
    pca = decomposition.PCA(n_components=2, random_state=1337)
    pca = pca.fit(dataset).transform(dataset)
    return pca


def plot_in_3_ways(dataset, color_array, title):
    """
    Assignment #05:
    Perform tsne in 3 different perplexities and compare with PCA on the same dataset
    Draw a plot for each of these models
    :param dataset: 3d dataset
    :param color_array: 1d dataset
    :param title: list, titles for plots
    """
    plot_3d(dataset, color_array, title)
    perplexities = [5, 30, 100]
    for perplexity in perplexities:
        tsne = perform_tsne(dataset, perplexity)
        plot_2d(tsne, color_array, f"{title}\nTSNE\n Perplexity: {perplexity}")
    pca = perform_pca(dataset)
    plot_2d(pca, color_array, f"{title}\nPCA")


n_samples = 1500
s_curve, color_for_s_curve = datasets.make_s_curve(n_samples)
plot_in_3_ways(s_curve, color_for_s_curve, title="S Curve")

swiss_roll, color_for_swiss_roll = datasets.make_swiss_roll(n_samples)
plot_in_3_ways(swiss_roll, color_for_swiss_roll, title="Swiss Roll")

swiss_roll2, color_for_swiss_roll2 = datasets.make_swiss_roll(n_samples, noise=0.8)
plot_in_3_ways(swiss_roll2, color_for_swiss_roll2, title="Swiss Roll, high noise")
