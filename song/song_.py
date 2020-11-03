import copy

import numpy as np


class Topology(dict):
    """Graph E implemented in adjacency list.
    The sturucture like this:
    Topology = {i:{j: w1, k: w2}, j:{i: w1}, k:{i: w2}}
    ,where i,j,k are integers, w1, w2 are float edge weights.

    In our SONG implementation, the topology will keep symmetric.
    """

    def __init__(self, n_coding_vector: int) -> None:
        for i in range(n_coding_vector):
            self[i] = dict()

    def check_topology(self) -> None:
        for key, adjacency in self.items():
            for node, val in adjacency.items():
                if abs(val - self[node][key]) > 0.0001:
                    msg = "Topology is not symmetric. E[{}][{}]={},\
                    but E[{}][{}]={}".format(
                        key, node, val, node, key, self[node][key]
                    )
                    raise ValueError(msg)
                if key == node:
                    msg = "Tere exists a diagonal element. E[{}][{}]={}".format(
                        key, node, val
                    )
                    raise ValueError(msg)


class SONG:
    def __init__(
        self,
        n_max_epoch=100,  # t_max: max iteration num,
        n_out_dim=2,  # d: output dim
        n_neighbors=3,  # k: number of neighbors to search
        alpha=1.0,  # learning rate
        negative_sample_rate=1,  # r: not mentioned in the paper
        a=1.577,
        b=0.895,  # hyper parameter, not mentioned in the paper
        theta_g=100,  # not mentioned in the paper.
        min_edge_weight=0.1,  # not mentioned in the paper
        edge_decay_rate=0.99,  # epsilon
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

        self.X = None  # input dataset
        self.embeddings = None  # Y: output
        self.coding_vector = None  # C
        self.topology = None
        self.grow_rate = None  # G
        self.alpha = None
        self.old_topology = None

        self.neighbor_idxs = None

    def _update_neighbors(self, x: np.array) -> None:
        """Update I^(k) TODO this step can be faster, via KD tree. see
        umap implementation.
        """
        dists = np.linalg.norm(
            self.coding_vector - np.full(self.coding_vector.shape, x), axis=1,
        )
        if self.n_coding_vector <= self.n_neighbors:
            self.neighbor_idxs = np.argsort(dists)
        else:
            unsorted_min_indices = np.argpartition(dists, self.n_neighbors)[
                : self.n_neighbors
            ]
            min_dists = dists[unsorted_min_indices]
            sorted_min_indices = np.argsort(min_dists)
            self.neighbor_idxs = unsorted_min_indices[sorted_min_indices]

    def _edge_curation(self) -> None:
        """Updating the directional edge. Algorithem 2 in the paper.
        In this implementation, the graph always keep symmetric.
        """
        i_1 = self.neighbor_idxs[0]
        for j in list(self.topology[i_1].keys()):
            self.topology[i_1][j] *= self.edge_decay_rate  # Decay edges
            self.topology[j][i_1] = self.topology[i_1][j]
            if self.topology[i_1][j] < self.min_edge_weight:
                del self.topology[i_1][j]  # Prune edges
                del self.topology[j][i_1]
        for j in self.neighbor_idxs[1:]:
            self.topology[i_1][j] = 1.0  # Renew edges
            self.topology[j][i_1] = self.topology[i_1][j]

    def _organize_coding_vector(self, x: np.array) -> None:
        i_1 = self.neighbor_idxs[0]
        i_k = self.neighbor_idxs[-1]
        w = np.linalg.norm(x - self.coding_vector[i_k]) ** 2
        for j in self.topology[i_1].keys():
            dire = x - self.coding_vector[j]
            self.coding_vector[j] += (
                self.alpha * np.exp(-np.linalg.norm(dire) ** 2 / w) * dire
            )

    def _update_embeddings(self) -> None:
        i_1 = self.neighbor_idxs[0]
        connecteds = self.topology[i_1].keys()
        negative_edges = list({i for i in range(len(self.topology))} - connecteds)
        n_negative_edges = len(negative_edges)
        n_negative_sample = int(self.negative_sample_rate * len(connecteds))

        a = self.a
        b = self.b
        for j in connecteds:
            dire = self.embeddings[i_1] - self.embeddings[j]
            dist = np.linalg.norm(dire)
            e = self.topology[i_1][j]
            nabla = (2 * a * b * e * dist ** (2 * b - 2)) / (1 + dist ** (2 * b))
            self.embeddings[j] += self.alpha * nabla * dire

        for idx in np.random.randint(0, n_negative_edges, n_negative_sample):
            j = negative_edges[idx]
            if j == i_1:
                continue
            dire = self.embeddings[i_1] - self.embeddings[j]
            dist = np.linalg.norm(dire)
            nabla = 2 * b / (dist ** 2 * (1 + dist ** (2 * b)))
            self.embeddings[j] -= self.alpha * nabla * dire

    def _refine_topology(self, x) -> None:
        """append the element in C, E, Y, G
        """
        i_1 = self.neighbor_idxs[0]
        k_idxs = self.neighbor_idxs
        new_vector = (x + np.sum(self.coding_vector[k_idxs], axis=0)) / (
            self.n_neighbors + 1
        )
        self.coding_vector = np.vstack([self.coding_vector, new_vector])

        new_embedding = np.sum(self.embeddings[k_idxs], axis=0) / self.n_neighbors
        self.embeddings = np.vstack([self.embeddings, new_embedding])
        self.topology[self.n_coding_vector] = dict()
        for j in k_idxs:
            self.topology[self.n_coding_vector][j] = 1.0
            self.topology[j][self.n_coding_vector] = self.topology[
                self.n_coding_vector
            ][j]
        self.grow_rate = np.append(self.grow_rate, 0.0)
        self.grow_rate[i_1] = 0.0
        # Above step does not mention in the paper, but I think this is need.
        self.n_coding_vector += 1

    def _set_embedding_label(self) -> None:
        self.raw_embeddings = np.zeros((self.X.shape[0], self.embeddings.shape[1]))
        self.embedding_label = np.zeros(len(self.embeddings))
        for i, x in enumerate(self.X):
            self._update_neighbors(x)
            i_1 = self.neighbor_idxs[0]
            self.raw_embeddings[i] = self.embeddings[i_1]
            self.embedding_label[i_1] = self.X_label[i]

    def fit(self, X: np.array, X_label=None) -> None:
        """Fit X in t an embedded space.

            Parameters
            ----------
            X : array, shape (n_samples, n_features)
            """

        self.X = X
        self.X_label = X_label
        self.n_in_data_num = X.shape[0]
        self.n_in_dim = X.shape[1]
        self.n_coding_vector = self.n_out_dim + 1
        self.coding_vector = np.random.randn(self.n_coding_vector, self.n_in_dim)
        self.embeddings = np.random.randn(self.n_coding_vector, self.n_out_dim)
        self.topology = Topology(self.n_coding_vector)
        self.grow_rate = np.zeros(self.n_coding_vector)
        self.alpha = self.init_alpha

        is_execute = False
        epoch = 0
        while not is_execute:
            print(
                "epoch: {} n_codev: {} m_grow_rate: {}".format(
                    epoch, self.n_coding_vector, np.mean(self.grow_rate)
                )
            )
            is_execute = True
            for idx in np.random.permutation(self.n_in_data_num):
                self.idx = idx
                x = self.X[idx]
                self._update_neighbors(x)
                i_1 = self.neighbor_idxs[0]
                self._edge_curation()
                if self.old_topology is not None:
                    if i_1 in self.old_topology:
                        old_connecteds = set(self.old_topology[i_1].keys())
                    else:
                        old_connecteds = None
                else:
                    old_connecteds = None
                connecteds = set(self.topology[i_1].keys())
                if old_connecteds != connecteds:
                    is_execute = False

                self._organize_coding_vector(x)
                self._update_embeddings()
                self.grow_rate[i_1] += np.linalg.norm(x - self.coding_vector[i_1])
                if self.grow_rate[i_1] > self.theta_g:
                    self._refine_topology(x)

            self.old_topology = copy.deepcopy(self.topology)
            self.alpha = self.init_alpha * (1 - epoch / self.n_max_epoch)
            epoch += 1
            if epoch >= self.n_max_epoch:
                is_execute = True

        self.topology.check_topology()

        if self.X_label is not None:
            self._set_embedding_label()
        print("finished")

    def transform(self, x: np.array) -> np.array:
        """Fit X into an embedded space and return that transformed
        output.

        Parameters
        ----------
        x : np.array, shape(n_features,)

        Returns
        -------
        embedding: np.array, shape(n_out_dim, )
        """
        self._update_neighbors(x)
        i_1 = self.neighbor_idxs[0]
        return self.embeddings[i_1]


if __name__ == "__main__":
    a = np.random.normal(2, 2, (20, 5))
    b = np.random.normal(0, 2, (20, 5))
    c = np.random.normal(-3, 1, (20, 5))
    d = np.random.normal(5, 1, (20, 5))
    X = np.vstack((a, b, c, d))
    model = SONG()
    model.fit(X)
    print(model.embeddings)
