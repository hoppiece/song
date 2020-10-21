from typing import Tuple

import numpy as np


class SONG:
    def __init__(
        self,
        n_max_epoch=100,  # t_max: max iteration num,
        n_out_dim=2,  # d: output dim
        n_neighbors=3,  # k: number of neighbors to search
        alpha=1.0,  # learning rate
        negative_sample_rate=1,  # r: not mentioned in the paper
        a=1.0,
        b=0.25,  # hyper parameter, not mentioned in the paper
        theta_g=100,  # not mentioned in the paper.
        min_edge_weight=0.3,  # not mentioned in the paper
        edge_decay_rate=0.9,  # epsilon
    ) -> None:

        self.n_max_epoch = n_max_epoch
        self.n_out_dim = n_out_dim
        self.n_neighbors = n_neighbors
        self.init_alpha = alpha
        self.negative_sample_rate = negative_sample_rate
        self.a = a
        self.b = b
        self.theta_g = theta_g
        self.min_edge_weight = min_edge_weight
        self.edge_decay_rate = edge_decay_rate

        self.alpha = alpha

    def fit(self, X: np.array, X_label=None) -> None:
        """Fit X in t an embedded space.

            Parameters
            ----------
            X : array, shape (n_samples, n_features)
            """

        self.X = X
        self.X_label = X_label

        self.n_in_dim = X.shape[1]
        n_init_coding_vector = self.n_out_dim + 1
        self.coding_vector = np.random.randn(n_init_coding_vector, self.n_in_dim)
        self.topo_indices = np.array(
            [
                [i, j]
                for i in range(n_init_coding_vector)
                for j in range(n_init_coding_vector)
                if i < j
            ],
            dtype=np.int64,
        )
        self.topo_weight = np.ones(n_init_coding_vector, dtype=np.float64)
        self.embeddings = np.random.randn(n_init_coding_vector, self.n_out_dim)
        self.grow_rate = np.zeros(n_init_coding_vector)

        for epoch in range(self.n_max_epoch):
            print("---------", epoch, "epoch --------")
            print("coding vector\n", self.coding_vector)
            print("top indices\n", self.topo_weight)
            print("topo w\n", self.topo_weight)
            print("emb\n", self.embeddings)
            (
                self.coding_vector,
                self.topo_indices,
                self.topo_weight,
                self.embeddings,
                self.grow_rate,
            ) = single_batch(
                self.X,
                self.coding_vector,
                self.topo_indices,
                self.topo_weight,
                self.embeddings,
                self.grow_rate,
                self.alpha,
                self.n_out_dim,
                self.n_neighbors,
                self.negative_sample_rate,
                self.a,
                self.b,
                self.theta_g,
                self.min_edge_weight,
                self.edge_decay_rate,
            )
            self.alpha = self.init_alpha * (1 - epoch / self.n_max_epoch)


def nearest_neighbors(point: np.array, data: np.array, n_neighbors: int) -> np.array:
    """ Compute the ``n_neighbors`` nearest points for each data point.
        TODO Use a more efficient algorithm, such as by referring
        to the UMAP implementation or using Annoy.
    Parameters
    ----------
    point: np.array
        The input point. Assume input x
    data: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor graph of.
        Assume coding vector
    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    Returns
    -------
    knn_indices: array of int
        The indices on the ``n_neighbors`` closest points in the dataset.
    """
    dists = np.linalg.norm(data - np.full(data.shape, point), axis=1)
    if data.shape[0] <= n_neighbors:
        knn_indices = np.argsort(dists)
    else:
        unsorted_min_indices = np.argpartition(dists, n_neighbors)[:n_neighbors]
        min_dists = dists[unsorted_min_indices]
        sorted_min_indices = np.argsort(min_dists)
        knn_indices = unsorted_min_indices[sorted_min_indices]

    return knn_indices


def curate_edge(
    topo_indices: np.array,
    topo_weight: np.array,
    knn_indices: np.array,
    edge_decay_rate: float,
    min_edge_weight: float,
) -> Tuple[np.array, np.array]:
    """Prune, Decay and Renew step of the topology

    Parameters
    ----------
    topo_indices : np.array, int, shape: (n, 2)
    topo_weight : np.array, float, shape: (n,)
    knn_indices : np.array, int, shape: (k, )
    edge_decay_rate : float
    min_edge_weight : float

    Returnes
    --------
    new_topo_indices
    new_topo_weight
    """
    i_1 = knn_indices[0]

    # Decay edges
    """
    In this step, edges which connect to i_1 are decayed.

    For example, suppose that
        topo_indices = [[0, 1], [0, 2], [1, 3], [2, 3]]
        topo_weight = [0.8, 0.6, 0.9, 0.5]
        i_1 = 2
        edge_decay_rate = 0.9

    In this setting, each variable is as below.
        connect_to_i1 = [[False, False], [False, True], [False, False], [True, False]]
        connect_indices = [False, True, False, True]
        new_topo_weight = [0,8, 0.54, 0.9, 0.45]
    """
    new_topo_weight = topo_weight
    connect_to_i1 = topo_indices == i_1  # boolean index
    connect_indices = np.sum(connect_to_i1, axis=1) == 1
    new_topo_weight[connect_indices] = topo_weight[connect_indices] * edge_decay_rate

    # Preprocess of renew edges
    """
    In this step, our purpose is to kill the edge which vertex
    is [i_1, j_1],...,[i_1, j_k]
    Suppose that
        knn_neighbor = [2, 0, 1]
    From the above example,

        connect_indices = [False, True, False, True]
        connect_to_j = [
            [True, True, False, False],
            [True, False, True, False]
        ]
        connect_indices * k + np.sum(connect_to_j, axis=0)
            = [2, 4, 1, 3]
        mask_indices = [False, True, False, False]
    Therefore, we find the [0, 2] or [1, 2] edge
    """
    k = len(knn_indices)
    connect_to_j = [np.sum(topo_indices == j, axis=1) for j in knn_indices[:1]]
    mask_indices = (connect_indices * k + np.sum(connect_to_j, axis=0)) > k
    new_topo_weight[mask_indices] = 0

    # Prune edges
    """
    Edge which weight is lower than min_edge_weight will be deleted.
    """
    remain_indices = new_topo_weight >= min_edge_weight
    new_topo_indices = np.take(topo_indices, remain_indices, axis=0)
    new_topo_weight = new_topo_weight[remain_indices]

    # Renew edges
    """
    We add the edges [i_1, j_1],...,[i_1, j_k], and set their weight 1.0.
    In the above example,
        addition_indices = [[0, 2], [1, 2]]
        addtion_weight = [1.0, 1.0]
    """
    addition_indices = np.array([[i_1, j] for j in knn_indices[1:]], dtype=np.int64)
    addition_weight = np.ones(len(knn_indices[1:]), dtype=np.float64)
    new_topo_indices = np.concatenate([new_topo_indices, addition_indices])
    new_topo_weight = np.concatenate([new_topo_weight, addition_weight])

    return new_topo_indices, new_topo_weight


def organize_codign_vector(
    coding_vector: np.array,
    topo_indices: np.array,
    alpha: float,
    x: np.array,
    knn_indices: np.array,
) -> np.array:
    """
    Optimizing the codign vector.

    Parameters
    ----------
    coding_vector : np.array
    topo_indices : np.array
    alpha : float
        [description]
    x : [type]
        [description]
    knn_indices : [type]
        [description]

    Returns
    -------
    np.array
        [description]
    """
    i_1 = knn_indices[0]
    i_k = knn_indices
    w = np.linalg.norm(x - coding_vector[i_k]) ** 2
    new_coding_vector = coding_vector
    for j in connect_node_indices(topo_indices, i_1):
        dire = x - coding_vector[j]
        new_coding_vector[j] += alpha * np.exp(-np.linalg.norm(dire) / w) * dire

    return new_coding_vector


def connect_node_indices(topo_indices: np.array, i: int) -> np.array:
    return np.sum(topo_indices[np.where(topo_indices == i)[0]], axis=1) - i


def update_embeddings(
    coding_vector: np.array,
    topo_indices: np.array,
    topo_weight: np.array,
    embeddings: np.array,
    knn_indices: np.array,
    alpha: float,
    a: float,
    b: float,
    negative_sample_rate: float,
):
    i_1 = knn_indices[0]
    new_embeddings = embeddings

    for c, idx in enumerate(np.where(topo_indices == i_1)[0]):
        j = np.sum(topo_indices[idx]) - i_1
        e = topo_weight[idx]
        dire = embeddings[i_1] - embeddings[j]
        dist = np.linalg.norm(dire)
        nabla = (2 * a * b * e * dist ** (2 * b - 2)) / (1 + dist ** (2 * b))
        new_embeddings[j] += alpha * nabla * dire
        count = c

    n_negative_sample = int((count + 1) * negative_sample_rate)
    for j in np.random.randint(0, len(coding_vector), n_negative_sample):
        if j == i_1:
            continue

        dire = embeddings[i_1] - embeddings[j]
        dist = np.linalg.norm(dire)
        nabla = 2 * b / (dist ** 2 * (1 + dist ** (2 * b)))
        new_embeddings[j] -= alpha * nabla * dire

    return new_embeddings


def refine_topology(
    coding_vector: np.array,
    topo_indices: np.array,
    topo_weight: np.array,
    embeddings: np.array,
    knn_indices: np.array,
    grow_rate: np.array,
    x: np.array,
):
    n_coding_vector = np.shape(coding_vector)[0]

    i_1 = knn_indices[0]
    addition_vector = (x + np.sum(coding_vector[knn_indices], axis=0)) / (
        len(knn_indices) + 1
    )
    coding_vector = np.vstack([coding_vector, addition_vector])

    addition_embedding = np.sum(embeddings[knn_indices], axis=0) / len(knn_indices)
    embeddings = np.vstack([embeddings, addition_embedding])

    addition_topo_indices = np.array([[j, n_coding_vector] for j in knn_indices])
    addition_topo_weight = np.ones(len(knn_indices), dtype=np.float64)
    topo_indices = np.vstack([topo_indices, addition_topo_indices])
    topo_weight = np.vstack([topo_weight, addition_topo_weight])

    grow_rate = np.append(grow_rate, 0.0)
    grow_rate[i_1] = 0.0

    return coding_vector, topo_indices, topo_weight, embeddings, grow_rate


def single_batch(
    X,
    coding_vector,
    topo_indices,
    topo_weight,
    embeddings,
    grow_rate,
    alpha,
    n_out_dim,
    n_neighbors,
    negative_sample_rate,
    a,
    b,
    theta_g,
    min_edge_weight,
    edge_decay_rate,
):
    n_in_data_num = X.shape[0]
    for idx in np.random.permutation(n_in_data_num):
        print(idx, "idx")
        x = X[idx]
        knn_indices = nearest_neighbors(x, coding_vector, n_neighbors)
        print("knn\n", knn_indices)
        i_1 = knn_indices[0]
        topo_indices, topo_weight = curate_edge(
            topo_indices, topo_weight, knn_indices, edge_decay_rate, min_edge_weight
        )
        coding_vector = organize_codign_vector(
            coding_vector, topo_indices, alpha, x, knn_indices
        )
        embeddings = update_embeddings(
            coding_vector,
            topo_indices,
            topo_weight,
            embeddings,
            knn_indices,
            alpha,
            a,
            b,
            negative_sample_rate,
        )
        grow_rate[i_1] += np.linalg.norm(x - coding_vector[i_1])
        if grow_rate[i_1] > theta_g:
            (
                coding_vector,
                topo_weight,
                topo_weight,
                embeddings,
                knn_indices,
                grow_rate,
            ) = refine_topology(
                coding_vector,
                topo_indices,
                topo_weight,
                embeddings,
                knn_indices,
                grow_rate,
                x,
            )

    return coding_vector, topo_indices, topo_weight, embeddings, grow_rate


if __name__ == "__main__":
    a = np.random.normal(2, 2, (20, 5))
    b = np.random.normal(0, 2, (20, 5))
    c = np.random.normal(-3, 1, (20, 5))
    d = np.random.normal(5, 1, (20, 5))
    X = np.vstack((a, b, c, d))
    print("X\n", X)
    model = SONG()
    model.fit(X)
    print(model.embeddings)
