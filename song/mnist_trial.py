import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt

from song_ import SONG

DATA_PATH = "../data"


def main():
    mnist = datasets.fetch_openml("mnist_784", data_home=DATA_PATH)
    mnist_data = mnist["data"][:1000]
    mnist_labels = mnist["target"][:1000]
    idx = []
    color = []
    for i in range(1000):
        if mnist_labels[i] == "0":
            idx.append(i)
            color.append(0)
        if mnist_labels[i] == "1":
            idx.append(i)
            color.append(1)
        if mnist_labels[i] == "2":
            idx.append(i)
            color.append(2)

    print(len(idx), len(color))
    idx = np.array(idx)

    model = SONG()
    model.fit(mnist_data, mnist_labels)
    embedding = model.embeddings

    plt.scatter(embedding[:, 0], embedding[:, 1], c=model.embedding_label)
    plt.show()


if __name__ == "__main__":
    main()

