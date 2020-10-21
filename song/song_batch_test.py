import numpy as np


def curate_edge(
    topo_indices: np.array,
    topo_weight: np.array,
    knn_indices: np.array,
    edge_decay_rate: float,
    min_edge_weight: float,
):
    print("input\n", topo_indices, "\n", topo_weight)
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

    print("decayed\n", topo_indices, "\n", new_topo_weight)
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

    print("renew preprocess\n", topo_indices, "\n", new_topo_weight)
    # Prune edges
    """
    Edge which weight is lower than min_edge_weight will be deleted.
    """
    remain_indices = new_topo_weight >= min_edge_weight
    new_topo_indices = np.take(topo_indices, remain_indices, axis=0)
    new_topo_weight = new_topo_weight[remain_indices]

    print("prune\n", new_topo_indices, "\n", new_topo_weight)
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

    print("renew\n", new_topo_indices, "\n", new_topo_weight)
    return new_topo_indices, new_topo_weight


topo_indices = np.array([[0, 1], [0, 2], [1, 3], [2, 3]], dtype=np.int64)
topo_weight = np.array([0.8, 0.6, 0.9, 0.5])
new_topo_indices, new_topo_weight = curate_edge(
    topo_indices, topo_weight, np.array([2, 0, 1]), 0.9, 0.5
)
print(new_topo_indices, new_topo_weight)

