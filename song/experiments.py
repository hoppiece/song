import argparse
import os
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

from parameter_search import loader, to_n_class, accuracy

MAX_DATA_NUM = 10000


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
    # plt.colorbar()
    plt.title(title)
    plt.savefig(filename)
    plt.clf()
    plt.close()


def main():
    OUT_DIR = "../figure/result"

    mnist = loader()
    data = mnist["data"][:MAX_DATA_NUM]
    labels = mnist["target"][:MAX_DATA_NUM]

    data, labels = to_n_class([0, 1, 2, 3, 4], data, labels)
    data = normalize(data)

    model = SONG(n_max_epoch=100, a=1.0, b=0.25, theta_g=40)
    start = time.time()
    model.fit(data, labels)
    calc_time = time.time() - start
    acc = accuracy(model.raw_embeddings, labels)
    title = "SONG"
    savegraph(model.raw_embeddings, labels, os.path.join(OUT_DIR, title), title)
    result = {"method": "SONG", "class": 5, "accuracy": acc, "time": calc_time}
    with open(os.path.join(OUT_DIR, "result.json"), "w") as fp:
        print(json.dumps(result), file=fp)

    model = TSNE(n_components=2, random_state=42)
    start = time.time()
    embedding = model.fit_transform(data)
    calc_time = time.time() - start
    acc = accuracy(embedding, labels)
    title = "t-SNE"
    savegraph(embedding, labels, os.path.join(OUT_DIR, title), title)
    result = {"method": "t-SNE", "class": 5, "accuracy": acc, "time": calc_time}
    with open(os.path.join(OUT_DIR, "result.json"), "a") as fp:
        print(json.dumps(result), file=fp)

    model = umap.UMAP()
    start = time.time()
    embedding = model.fit_transform(data)
    calc_time = time.time() - start
    acc = accuracy(embedding, labels)
    title = "UMAP"
    savegraph(embedding, labels, os.path.join(OUT_DIR, title), title)
    result = {"method": "UMAP", "class": 5, "accuracy": acc, "time": calc_time}
    with open(os.path.join(OUT_DIR, "result.json"), "a") as fp:
        print(json.dumps(result), file=fp)


if __name__ == "__main__":
    main()
