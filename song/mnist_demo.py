import os.path
import pickle
import time

from matplotlib import cm
from matplotlib import pyplot as plt
import sklearn
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

import song_
import song_batch

DATA_PATH = "../data"
MAX_DATA_NUM = 10000


def loader():
    """Get the mnist data

    Returns
    -------
    [type]
        sklertn.dataset object.
        This object has a dict-like interface, so we get
        training data by `obj["data"]`, and
        label data by `obj["target"]`
    """
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


def accuracy(embedding, label):
    classifier = KNeighborsClassifier(n_neighbors=10)
    try:
        pred_y = cross_val_predict(classifier, embedding, label, cv=5)
        acc = sklearn.metrics.accuracy_score(label, pred_y)
    except ValueError:
        acc = -1
    return acc


def main():
    mnist = loader()
    data = mnist["data"][:MAX_DATA_NUM]
    labels = mnist["target"][:MAX_DATA_NUM]
    data, labels = to_n_class([0, 1, 2, 3, 4], data, labels)
    data = normalize(data)
    if data.shape[1] > 20:
        prep_model = PCA(n_components=20)
        data = prep_model.fit_transform(data)

    model = song_batch.SONG(
        n_max_epoch=100, a=1.0, b=0.25, theta_g=40, edge_decay_rate=0.9
    )
    start = time.time()
    model.fit(data, labels)
    calc_time = time.time() - start
    acc = accuracy(model.raw_embeddings, labels)

    print("time: ", calc_time, "sec.")
    print("acc:", acc)

    plt.scatter(
        model.raw_embeddings[:, 0],
        model.raw_embeddings[:, 1],
        vmin=min(labels),
        vmax=max(labels),
        c=labels,
        cmap=cm.gist_rainbow,
        alpha=0.3,
        marker=".",
    )
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
