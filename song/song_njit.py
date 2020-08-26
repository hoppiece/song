import numpy as np
from numba import jit
from numba.typed import Dict


@jit
def _init_topology(n):
    topo = dict()
    for i in range(n):
        topo[i] = dict()

    return topo


@jit
def _update_neighbors(x, code_v):
    dists = np.linalg.norm(code_v - np.full(code_v.shape, x), axis=1)
    neighbor_idxs = np.argsort(dists)
    # TODO partial sort method
    return neighbor_idxs


@jit
def _edge_curation(nbr_idxs, topo, r_decay, min_w):
    i_1 = nbr_idxs[0]
    new_topo = topo
    # new_topo is passed by reference, so parameter topo will be changed.
    for j in list(topo[i_1].keys()):
        new_topo[i_1][j] = r_decay * topo[i_1][j]
        new_topo[j][i_1] = new_topo[i_1][j]
        if topo[i_1][j] < min_w:
            del new_topo[i_1][j]
            del new_topo[j][i_1]
    for j in nbr_idxs[1:]:
        new_topo[i_1][j] = 1.0
        new_topo[j][i_1] = new_topo[i_1][j]

    return new_topo


@jit
def _organize_coding_vector(x, code_v, nbr_idxs, topo, k, alpha):
    i_1 = nbr_idxs[0]
    i_k = nbr_idxs[k - 1]
    w = np.linalg.norm(x - code_v[i_k]) ** 2
    new_code_v = code_v  # passed by reference
    for j in topo[i_1].keys():
        dire = x - code_v[j]
        new_code_v[j] += alpha * np.exp(-np.linalg.norm(dire) / w) * dire

    return new_code_v


@jit
def _update_embeddings(nbr_idxs, topo, emb, alpha, negative_sample_rate, a, b):
    i_1 = nbr_idxs[0]
    connecteds = topo[i_1].keys()
    new_emb = emb
    n_negative_sample = int(negative_sample_rate * len(connecteds))

    for j in connecteds:
        dire = emb[i_1] - emb[j]
        dist = np.linalg.norm(dire)
        e = topo[i_1][j]
        nabla = (2 * a * b * e * dist ** (2 * b - 2)) / (1 + dist ** (2 * b))
        new_emb[j] += alpha * nabla * dire

    for j in np.random.randint(0, len(topo), n_negative_sample):
        if j == i_1:
            continue
        dire = emb[i_1] - emb[j]
        dist = np.linalg.norm(dire)
        nabla = (2 * a * b * e * dist ** (2 * b - 2)) / (1 + dist ** (2 * b))
        new_emb[j] -= alpha * nabla * dire

    return new_emb


@jit
def _refine_topology(x, code_v, nbr_idxs, topo, emb, growrate, k):
    i_1 = nbr_idxs[0]
    k_idxs = nbr_idxs[:k]
    new_code_v = code_v
    new_topo = topo
    new_emb = emb
    new_growrate = growrate

    addit_v = (x + np.sum(code_v[k_idxs], axis=0)) / (k + 1)
    new_code_v = np.vstack([code_v, addit_v])

    addit_emb = np.sum(emb[k_idxs], axis=0) / k
    new_emb = np.vstack([emb, addit_emb])

    n_codev = len(new_code_v) - 1
    new_topo[n_codev] = dict()
    for j in k_idxs:
        new_topo[n_codev][j] = 1.0
        new_topo[j][n_codev] = new_topo[n_codev][j]
    new_growrate = np.append(growrate, 0.0)
    new_growrate[i_1] = 0.0

    return new_code_v, new_topo, new_emb, new_growrate


class SONG:
    def __init__(
        self,
        n_max_epoch=100,
        n_out_dim=2,
        n_neighbors=3,
        alpha=1.0,
        negative_sample_rate=1,
        a=1.0,
        b=0.25,
        theta_g=100,
        min_edge_weight=0.1,
        edge_decay_rate=0.99,
    ):
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

        self.coding_vector = None

    def _set_embbeding_label(self):
        self.raw_embeddings = np.zeros((self.X.shape[0], self.embeddings.shape[1]))
        self.embedding_label = np.zeros(len(self.embeddings))
        for i, x in enumerate(self.X):
            nbr_idxs = _update_neighbors(x, self.coding_vector)
            i_1 = nbr_idxs[0]
            self.raw_embeddings[i] = self.embedding_label[i_1]
            self.embedding_label[i_1] = self.X_label[i]

    def fit(self, X, X_label=None):
        self.X = X
        self.X_label = X_label

        self.coding_vector, self.topology, self.embeddings, self.grow_rate = _fit(
            self.X,
            self.n_max_epoch,
            self.n_out_dim,
            self.n_neighbors,
            self.init_alpha,
            self.negative_sample_rate,
            self.a,
            self.b,
            self.theta_g,
            self.min_edge_weight,
            self.edge_decay_rate,
        )

        if self.X_label is not None:
            self._set_embbeding_label()

        print("finished")


@jit
def _fit(
    X,
    max_epoch,
    out_dim,
    n_neighbors,
    init_alpha,
    negative_sample_rate,
    a,
    b,
    theta_g,
    min_edge_weight,
    edge_decay_rate,
):
    data_num = X.shape[0]
    in_dim = X.shape[1]
    t_max = data_num * max_epoch
    n_code_v = out_dim + 1

    code_v = np.random.randn(n_code_v, in_dim)
    topo = _init_topology(n_code_v)
    emb = np.random.randn(n_code_v, out_dim)
    growrate = np.zeros(n_code_v)

    alpha = init_alpha

    for ite, idx in enumerate(np.random.randint(0, data_num, t_max)):
        if ite % data_num == 0:
            print(
                "epoch: {} n_codev: {} m_grow_rate: {}".format(
                    ite // data_num, n_code_v, np.mean(growrate),
                )
            )
        x = X[idx]
        nbr_idxs = _update_neighbors(x, code_v)
        i_1 = nbr_idxs[0]
        topo = _edge_curation(nbr_idxs, topo, edge_decay_rate, min_edge_weight)
        code_v = _organize_coding_vector(x, code_v, nbr_idxs, topo, n_neighbors, alpha)
        emb = _update_embeddings(nbr_idxs, topo, emb, alpha, negative_sample_rate, a, b)
        growrate[i_1] += np.linalg.norm(x - code_v[i_1])
        if growrate[i_1] > theta_g:
            code_v, topo, emb, growrate = _refine_topology(
                x, code_v, nbr_idxs, topo, emb, growrate, n_neighbors
            )
            n_code_v += 1
        alpha = init_alpha * (1 - ite / t_max)

    return code_v, topo, emb, growrate


if __name__ == "__main__":
    a = np.random.normal(2, 2, (20, 5))
    b = np.random.normal(0, 2, (20, 5))
    c = np.random.normal(-3, 1, (20, 5))
    d = np.random.normal(5, 1, (20, 5))
    X = np.vstack((a, b, c, d))
    model = SONG()
    model.fit(X)
    print(model.embeddings)
