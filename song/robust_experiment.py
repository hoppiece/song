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

from parameter_search import loader, to_n_class, savegraph, accuracy, savegraph

MAX_DATA_NUM = 10000


def add_noise(data: np.array, eps: float):
    R = np.random.standard_normal(data.shape)
    ret = data + R * eps
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="path to the output")
    args = parser.parse_args()
    OUT_DIR = "robust_test"
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)

    try:
        with open(args.output, mode="x"):
            pass
    except FileExistsError:
        pass

    mnist = loader()
    data = mnist["data"][:MAX_DATA_NUM]
    labels = mnist["target"][:MAX_DATA_NUM]

    data, labels = to_n_class([0, 1, 2, 3, 4], data, labels)
    data = normalize(data)

    eps_lst = [0.1 * i for i in range(20)]
    for i, eps in enumerate(eps_lst):
        noisy_data = add_noise(data, eps)
        if noisy_data.shape[1] > 20:
            prep_model = PCA(n_components=20)
            noisy_data = prep_model.fit_transform(noisy_data)

        model = SONG(n_max_epoch=100, a=1.0, b=0.25, theta_g=20)
        start = time.time()
        model.fit(noisy_data, labels)
        calc_time = time.time() - start
        acc = accuracy(model.raw_embeddings, labels)
        title = "SONG, r={}".format(eps)
        savegraph(
            model.raw_embeddings,
            labels,
            os.path.join(OUT_DIR, "SONG_rbst_{}".format(i)),
            title,
        )
        result = {
            "method": "SONG",
            "class": 5,
            "accuracy": acc,
            "time": calc_time,
            "noise_size": eps,
        }
        with open(args.output, mode="a") as fp:
            print(json.dumps(result), file=fp)


if __name__ == "__main__":
    main()
