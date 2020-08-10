import os.path
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

from song_ import SONG

DATA_PATH = "../data"
MAX_DATA_NUM = 1000


def loader():
    dataset = "mnist_784"
    try:
        with open(os.path.join(DATA_PATH, dataset), "rb") as fp:
            mnist = pickle.load(fp)
    except FileNotFoundError:
        mnist = datasets.fetch_openml(dataset, data_home=DATA_PATH)
        with open(os.path.join(DATA_PATH, dataset), "wb") as fp:
            pickle.dump(mnist, fp, -1)

    print()
    return mnist


def main():
    mnist = loader()
    stdsc = StandardScaler()
    mnist_data = mnist["data"][:MAX_DATA_NUM]
    mnist_labels = mnist["target"][:MAX_DATA_NUM]
    mnist_data_std = stdsc.fit_transform(mnist_data)
    idx = []
    color = []
    for i in range(MAX_DATA_NUM):
        if mnist_labels[i] == "0":
            idx.append(i)
            color.append(0)
        if mnist_labels[i] == "1":
            idx.append(i)
            color.append(1)
        if mnist_labels[i] == "2":
            idx.append(i)
            color.append(2)

    idx = np.array(idx)

    model = SONG()
    model.fit(mnist_data_std[idx], color)

    plt.scatter(model.raw_embeddings[:, 0], model.raw_embeddings[:, 1], c=color)
    plt.show()


if __name__ == "__main__":
    main()
