from os import path

from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from song_ import SONG
from mnist_trial import to_n_class, loader

MAX_DATA_NUM = 5000
SAVEFIG_PATH = "../figure/search_ab3/"


def main():
    mnist = loader()
    data = mnist["data"][:MAX_DATA_NUM]
    labels = mnist["target"][:MAX_DATA_NUM]
    data, labels = to_n_class([0, 1, 2], data, labels)
    data = normalize(data)
    if data.shape[1] > 20:
        prep_model = PCA(n_components=20)
        data = prep_model.fit_transform(data)

    for i, a in enumerate([0.5 + 0.1 * i for i in range(10)]):
    a = 1.0
    for j, b in enumerate([0.4 + 0.01 * j for j in range(20)]):
        print("a={:.2f},b={:.2f}".format(a, b))
        model = SONG(a=a, b=b)
        model.fit(data, labels)
        filname = "a{}b{}".format(0, j)
        title = "a={:.2f},b={:.2f}".format(a, b)
        showgraph(model.raw_embeddings, labels, filname, title)


def showgraph(embeddings, labels, filename, title):
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
    plt.savefig(path.join(SAVEFIG_PATH, filename))


if __name__ == "__main__":
    main()
