import argparse
import os
import pickle
import time
import json

import numpy as np
import umap
from matplotlib import cm
from matplotlib import pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

from song_ import SONG

DATA_PATH = "../data"
MAX_DATA_NUM = 10000


def loader():
    dataset = "mnist_784"
    try:
        with open(os.path.join(DATA_PATH, dataset), "rb") as fp:
            mnist = pickle.load(fp)
    except FileNotFoundError:
        mnist = datasets.fetch_openml(dataset, data_home=DATA_PATH)
        with open(os.path.join(DATA_PATH, dataset), "wb") as fp:
            pickle.dump(mnist, fp, -1)

    return mnist


def to_n_class(digit_lst, data, labels):
    """to make a subset of MNIST dataset, which has particular digits
    Parameters
    ----------
    digit_lst : list
        for example, [0,1,2] or [1, 5, 8]
    data : numpy.array, shape (n_samples, n_features)
    labels : numpy.array or list of str
    Returns
    -------
    numpy.array, list of int
    """
    if not set(digit_lst) <= set(range(10)):
        raise ValueError
    indices = []
    new_labels = []
    for i, x in enumerate(data):
        for digit in digit_lst:
            if labels[i] == str(digit):
                indices.append(i)
                new_labels.append(digit)

    return data[indices], new_labels


def savegraph(embeddings, labels, filename, title):
    plt.figure()
    plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        vmin=min(labels),
        vmax=max(labels),
        c=labels,
        cmap=cm.gist_rainbow,
        alpha=0.3,
        marker=".",
    )
    plt.colorbar()
    plt.title(title)
    plt.savefig(filename)
    plt.clf()
    plt.close()


def accuracy(embedding, label):
    classifier = KNeighborsClassifier(n_neighbors=10)
    try:
        pred_y = cross_val_predict(classifier, embedding, label, cv=5)
        acc = sklearn.metrics.accuracy_score(label, pred_y)
    except ValueError:
        acc = -1
    return acc


def song(n_class, data, labels, n_max_epoch, a, b, theta_g, filename):
    model = SONG(n_max_epoch=n_max_epoch, a=a, b=b, theta_g=theta_g)
    print("... embedding")
    start = time.time()
    model.fit(data, labels)
    calc_time = time.time() - start
    print("{} sec".format(calc_time))
    title = "SONG,epoch={},a={},b={},g={}".format(n_max_epoch, a, b, theta_g)
    savegraph(model.raw_embeddings, labels, filename, title)
    print("... evaluation")
    acc = accuracy(model.raw_embeddings, labels)
    print("accuracy: ", acc)
    result = {
        "method": "SONG",
        "class": n_class,
        "accuracy": acc,
        "time": calc_time,
        "hyperparam": {"n_max_epoch": n_max_epoch, "a": a, "b": b, "theta_g": theta_g},
    }
    return result


def tsne(n_class, data, labels, filename):
    model = TSNE(n_components=2, random_state=42)
    start = time.time()
    embedding = model.fit_transform(data)
    calc_time = time.time() - start
    print("{} sec.".format(calc_time))
    title = "t-SNE"
    savegraph(embedding, labels, filename, title)
    print("... evaluation")
    acc = accuracy(embedding, labels)
    print("accuracy:", acc)
    result = {
        "method": "SONG",
        "class": n_class,
        "accuracy": acc,
        "time": calc_time,
    }
    return result


def umap_(n_class, data, labels, filename):
    model = umap.UMAP()
    start = time.time()
    embedding = model.fit_transform(data)
    calc_time = time.time() - start
    print("{} sec.".format(calc_time))
    title = "UMAP"
    savegraph(embedding, labels, filename, title)
    print("... evaluation")
    acc = accuracy(embedding, labels)
    print("accuracy:", acc)
    result = {
        "method": "UMAP",
        "class": n_class,
        "accuracy": acc,
        "time": calc_time,
    }
    return result


def search_param():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="path to the output")
    args = parser.parse_args()

    mnist = loader()
    data = mnist["data"][:MAX_DATA_NUM]
    labels = mnist["target"][:MAX_DATA_NUM]

    params = [
        (g, a, b)
        for g in [20, 40, 60, 80, 100]
        for a in [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
        ]
        for b in [
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
        ]
    ]

    data, labels = to_n_class([0, 1, 2], data, labels)
    data = normalize(data)
    if data.shape[1] > 20:
        prep_model = PCA(n_components=20)
        data = prep_model.fit_transform(data)

    if not os.path.exists("3cls"):
        os.mkdir("3cls")
    if not os.path.exists("5cls"):
        os.mkdir("5cls")

    try:
        with open(args.output, mode="x"):
            pass
    except FileExistsError:
        pass

    for i, param in enumerate(params):
        g = param[0]
        a = param[1]
        b = param[2]

        filename = os.path.join("3cls", "SONG_3cls_{}".format(i))
        result = song(3, data, labels, 100, a, b, g, filename)
        with open(args.output, mode="a") as fp:
            print(json.dumps(result) + "\n", file=fp)

    filename = os.path.join("3cls", "TSNE_3cls")
    result = tsne(3, data, labels, filename)
    with open(args.output, mode="a") as fp:
        print(json.dumps(result) + "\n", file=fp)

    filename = os.path.join("3cls", "UMAP_3cls")
    result = umap_(3, data, labels, filename)
    with open(args.output, mode="a") as fp:
        print(json.dumps(result) + "\n", file=fp)

    data, labels = to_n_class([0, 1, 2, 3, 4], data, labels)
    data = normalize(data)
    if data.shape[1] > 20:
        prep_model = PCA(n_components=20)
        data = prep_model.fit_transform(data)

    for i, param in enumerate(params):
        a = param[0]
        b = param[1]
        g = param[2]

        filename = os.path.join("5cls", "SONG_5cls_{}".format(i))
        result = song(5, data, labels, 100, a, b, g, filename)
        with open(args.output, mode="a") as fp:
            print(json.dumps(result) + "\n", file=fp)

    filename = os.path.join("5cls", "TSNE_5cls")
    result = tsne(5, data, labels, filename)
    with open(args.output, mode="a") as fp:
        print(json.dumps(result) + "\n", file=fp)

    filename = os.path.join("5cls", "UMAP_5cls")
    result = umap_(5, data, labels, filename)
    with open(args.output, mode="a") as fp:
        print(json.dumps(result) + "\n", file=fp)


if __name__ == "__main__":
    mnist = loader()
    data = mnist["data"][:MAX_DATA_NUM]
    labels = mnist["target"][:MAX_DATA_NUM]
    data, labels = to_n_class([0, 1, 2], data, labels)
    data = normalize(data)
    if data.shape[1] > 20:
        prep_model = PCA(n_components=20)
        data = prep_model.fit_transform(data)

    print(json.dumps(song(3, data, labels, 100, 1.0, 0.25, 40, "./mnist.png")))
