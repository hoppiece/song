import copy

import numpy as np
import numba

import sklearn.neighbors


@numba.jit
def _coding_vector_sgd(x, coding_vector, knn_indices, connects, alpha):
    i_k = knn_indices[-1]
    w = np.linalg.norm(x - coding_vector[i_k]) ** 2
    dire = x - coding_vector[connects]
    coding_vector[connects] += alpha * (
        np.exp(-np.sum(dire ** 2, axis=1) / w).reshape(-1, 1) * dire
    )


@numba.njit
def embedding_sgd_attr(embeddings, i_1, connecteds, e_ary, a, b, alpha):
    dire = embeddings[i_1] - embeddings[connecteds]
    embeddings[connecteds] += (
        alpha
        * dire
        * (2 * a * b * e_ary * np.sum(dire ** 2, axis=1) ** (b - 1)).reshape(-1, 1)
        / (1 + np.sum(dire ** 2, axis=1) ** b).reshape(-1, 1)
    )


@numba.njit
def embedding_sgd_rep(embeddings, i_1, negative_edges, n_negative_sample, a, b, alpha):
    neg_J = np.random.choice(negative_edges, n_negative_sample)
    dire = embeddings[i_1] - embeddings[neg_J]
    embeddings[neg_J] -= (
        alpha
        * 2
        * b
        * dire
        / (np.sum(dire ** 2, axis=1) * (1 + np.sum(dire ** 2, axis=1) ** b)).reshape(
            -1, 1
        )
    )


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

    def get_degree(self):
        ret = []
        for key, adjacency in self.items():
            ret.append(len(adjacency))

        return ret

    def get_max_degree(self):
        return max(self.get_degree())


class SONG:
    def __init__(
        self,
        n_max_epoch=100,  # t_max: max iteration num,
        n_out_dim=2,  # d: output dim
        n_neighbors=3,  # k: number of neighbors to search
        alpha=1.0,  # learning rate
        negative_sample_rate=1.5,  # r: not mentioned in the paper
        a=1.0,
        b=0.3,  # hyper parameter, not mentioned in the paper
        theta_g=None,  # not mentioned in the paper.
        min_edge_weight=0.1,  # not mentioned in the paper
        edge_decay_rate=0.9,  # epsilon
        knn_method="brute_force",
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
        self.knn_method = knn_method

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
        if self.knn_method == "brute_force":
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
        elif self.knn_method == "kdtree":  # TODO something is wrong
            self.tree = sklearn.neighbors.KDTree(self.coding_vector)
            dist, ids = self.tree.query(x.reshape(1, -1), k=self.n_neighbors)
            self.neighbor_idxs = ids.reshape(-1,)
        elif self.knn_method == "balltree":
            self.tree = sklearn.neighbors.BallTree(self.coding_vector)
            dist, ids = self.tree.query(x.reshape(1, -1), k=self.n_neighbors)
            self.neighbor_idxs = ids.reshape(-1,)

    def _update_neighbors_bat(self, x_batch: np.array):
        self.tree = sklearn.neighbors.KDTree(self.coding_vector, leaf_size=1000)
        self.bknn_ind = self.tree.query(
            x_batch, k=self.n_neighbors, return_distance=False
        )

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

    def _organize_coding_vector_old(self, x):  # not using JIT version
        i_1 = self.neighbor_idxs[0]
        i_k = self.neighbor_idxs[-1]
        w = np.linalg.norm(x - self.coding_vector[i_k]) ** 2
        J = np.array(list(self.topology[i_1].keys()))
        dire = x - self.coding_vector[J]
        self.coding_vector[J] += self.alpha * (
            np.exp(-np.sum(dire ** 2, axis=1) / w).reshape(-1, 1) * dire
        )

    def _organize_coding_vector(self, x):
        i_1 = self.neighbor_idxs[0]
        connects = np.array(list(self.topology[i_1].keys()))
        self._coding_vector_sgd(
            x, self.coding_vector, self.neighbor_idxs, connects, self.alpha
        )

    @staticmethod
    @numba.njit
    def _coding_vector_sgd(x, coding_vector, knn_indices, connects, alpha):
        i_k = knn_indices[-1]
        w = np.linalg.norm(x - coding_vector[i_k]) ** 2
        dire = x - coding_vector[connects]
        coding_vector[connects] += alpha * (
            np.exp(-np.sum(dire ** 2, axis=1) / w).reshape(-1, 1) * dire
        )

    def _organize_coding_vector_bat(self, x_bat):
        # It is difficult to use tensor caluculation because length
        # of `connects` are different with x.
        connects = [
            np.array(list(self.topology[self.bknn_ind[i][0]]))
            for i in range(self.batch_size)
        ]

        self._coding_vector_sgd_bat(
            x_bat,
            self.coding_vector,
            self.bknn_ind,
            connects,
            self.alpha,
            self.batch_size,
        )

    @staticmethod
    # @numba.njit
    def _coding_vector_sgd_bat(
        x_bat, coding_vector, bknn_ind, connects, alpha, batch_size
    ):
        for i in range(batch_size):
            _coding_vector_sgd(x_bat[i], coding_vector, bknn_ind[i], connects[i], alpha)

    def _update_embeddings_old(self) -> None:  # not using jit version
        i_1 = self.neighbor_idxs[0]
        connecteds = self.topology[i_1].keys()

        a = self.a
        b = self.b

        e_ary = np.array(list(self.topology[i_1].values()))
        J = np.array(list(connecteds))
        dire = self.embeddings[i_1] - self.embeddings[J]
        self.embeddings[J] += (
            self.alpha
            * dire
            * (2 * a * b * e_ary * np.sum(dire ** 2, axis=1) ** (self.b - 1)).reshape(
                -1, 1
            )
            / (1 + np.sum(dire ** 2, axis=1) ** b).reshape(-1, 1)
        )

        negative_edges = np.array(
            list({i for i in range(len(self.topology))} - connecteds - {i_1})
        )
        n_negative_edges = len(negative_edges)
        n_negative_sample = int(self.negative_sample_rate * len(connecteds))
        if n_negative_edges > n_negative_sample:

            neg_J = np.random.choice(negative_edges, n_negative_sample, replace=False)
            dire = self.embeddings[i_1] - self.embeddings[neg_J]
            self.embeddings[neg_J] -= (
                self.alpha
                * 2
                * self.b
                * dire
                / (
                    np.sum(dire ** 2, axis=1)
                    * (1 + np.sum(dire ** 2, axis=1) ** self.b)
                ).reshape(-1, 1)
            )

    def _update_embeddings(self):
        i_1 = self.neighbor_idxs[0]
        connecteds = self.topology[i_1].keys()  # a kind of set object
        e_ary = np.array(list(self.topology[i_1].values()))
        negative_edges = np.array(
            list(set(range(len(self.topology))) - connecteds - {i_1})
        )
        connecteds = np.array(list(connecteds))
        n_negative_edges = len(negative_edges)
        n_negative_sample = int(self.negative_sample_rate * len(connecteds))

        self.embedding_sgd_attr(
            self.embeddings, i_1, connecteds, e_ary, self.a, self.b, self.alpha
        )
        if n_negative_edges > n_negative_sample:
            self.embedding_sgd_rep(
                self.embeddings,
                i_1,
                negative_edges,
                n_negative_sample,
                self.a,
                self.b,
                self.alpha,
            )

    @staticmethod
    @numba.njit
    def embedding_sgd_attr(embeddings, i_1, connecteds, e_ary, a, b, alpha):
        dire = embeddings[i_1] - embeddings[connecteds]
        embeddings[connecteds] += (
            alpha
            * dire
            * (2 * a * b * e_ary * np.sum(dire ** 2, axis=1) ** (b - 1)).reshape(-1, 1)
            / (1 + np.sum(dire ** 2, axis=1) ** b).reshape(-1, 1)
        )

    @staticmethod
    @numba.njit
    def embedding_sgd_rep(
        embeddings, i_1, negative_edges, n_negative_sample, a, b, alpha
    ):
        neg_J = np.random.choice(negative_edges, n_negative_sample)
        dire = embeddings[i_1] - embeddings[neg_J]
        embeddings[neg_J] -= (
            alpha
            * 2
            * b
            * dire
            / (
                np.sum(dire ** 2, axis=1) * (1 + np.sum(dire ** 2, axis=1) ** b)
            ).reshape(-1, 1)
        )

    def _update_embeddings_bat(self):
        i_1_bat = self.bknn_ind[:, 0]
        connecteds_bat = [self.topology[i_1].keys for i_1 in i_1_bat]
        e_ary_bat = [list(self.topology[i_1].values()) for i_1 in i_1_bat]
        negative_edges_bat = [
            np.array(
                list(set(range(len(self.topology))) - connecteds_bat[count] - {i_1})
            )
            for count, i_1 in enumerate(i_1_bat)
        ]
        connecteds_bat = [np.array(c) for c in connecteds_bat]
        min_n_negative_edges = min([len(neb) for neb in negative_edges_bat])
        n_negative_sample_bat = [
            int(self.negative_sample_rate * len(con)) for con in connecteds_bat
        ]
        max_n_negative_sample = max(n_negative_sample_bat)
        self.embedding_sgd_attr_bat(
            self.embeddings,
            i_1_bat,
            connecteds_bat,
            e_ary_bat,
            self.a,
            self.b,
            self.alpha,
            self.batch_size,
        )

        if min_n_negative_edges >= max_n_negative_sample:
            self.embedding_sgd_rep_bat(
                self.embeddings,
                i_1_bat,
                negative_edges_bat,
                n_negative_sample_bat,
                self.a,
                self.b,
                self.alpha,
                self.batch_size,
            )

    @staticmethod
    def embedding_sgd_attr_bat(
        embeddings, i_1_bat, connecteds_bat, e_ary_bat, a, b, alpha, batch_size
    ):
        for i in range(batch_size):
            embedding_sgd_attr(
                embeddings, i_1_bat[i], connecteds_bat[i], e_ary_bat[i], a, b, alpha
            )

    @staticmethod
    def embedding_sgd_rep_bat(
        embeddings,
        i_1_bat,
        negative_edges_bat,
        n_negative_sample_bat,
        a,
        b,
        alpha,
        batch_size,
    ):
        for i in range(batch_size):
            embedding_sgd_rep(
                embeddings,
                i_1_bat[i],
                negative_edges_bat[i],
                n_negative_sample_bat[i],
                a,
                b,
                alpha,
            )

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

    def fit(self, X: np.array, X_label=None, batch_size=1) -> None:
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
        self.batch_size = batch_size

        if self.theta_g is None:
            self.theta_g = self.n_in_dim * 1.5

        if self.knn_method is None:
            if self.n_in_dim > 50:
                self.knn_method = "balltree"
            else:
                self.knn_method = "kdtree"

        if self.batch_size >= 2:
            self.single_batch(X, 3)
            self.multi_batch(self.X, self.batch_size)
        else:
            self.single_batch(X, self.n_max_epoch)

        if self.X_label is not None:
            self._set_embedding_label()
        print("finished")

    def single_batch(self, X, max_epoch):
        self.X = X
        self.n_in_data_num = X.shape[0]
        self.n_in_dim = X.shape[1]
        self.n_coding_vector = self.n_out_dim + 1
        self.coding_vector = np.random.randn(self.n_coding_vector, self.n_in_dim)
        self.embeddings = np.random.randn(self.n_coding_vector, self.n_out_dim)
        self.topology = Topology(self.n_coding_vector)
        self.grow_rate = np.zeros(self.n_coding_vector)
        self.alpha = self.init_alpha

        if self.theta_g is None:
            self.theta_g = self.n_in_dim * 1.5

        if self.knn_method is None:
            if self.n_in_dim > 50:
                self.knn_method = "balltree"
            else:
                self.knn_method = "kdtree"

        is_execute = False
        epoch = 0
        while not is_execute:
            print(
                "epoch: {} n_codev: {} m_grow_rate: {}".format(
                    epoch, self.n_coding_vector, np.mean(self.grow_rate)
                )
            )
            for idx in np.random.permutation(self.n_in_data_num):
                self.idx = idx
                x = self.X[idx]
                self._update_neighbors(x)
                i_1 = self.neighbor_idxs[0]
                self._edge_curation()
                self._organize_coding_vector(x)
                self._update_embeddings()
                self.grow_rate[i_1] += np.linalg.norm(x - self.coding_vector[i_1])
                if self.grow_rate[i_1] > self.theta_g:
                    self._refine_topology(x)
            self.alpha = self.init_alpha * (1 - epoch / self.n_max_epoch)
            epoch += 1
            if epoch >= max_epoch:
                is_execute = True

        self.topology.check_topology()

        if self.X_label is not None:
            self._set_embedding_label()
        print("finished")

    def multi_batch(self, X, batch_size):
        is_execute = False
        epoch = 0
        while not is_execute:
            print(
                "epoch: {} n_codev: {} m_grow_rate: {}".format(
                    epoch, self.n_coding_vector, np.mean(self.grow_rate)
                )
            )
            for i in range(int(self.n_in_data_num / self.batch_size)):
                batch_mask = np.random.choice(self.n_in_data_num, self.batch_size)
                x_bat = X[batch_mask]
                self._update_neighbors_bat(x_bat)
                for ind in self.bknn_ind:
                    self.neighbor_idxs = ind
                    self._edge_curation()

                self._organize_coding_vector_bat(x_bat)

                for ind in self.bknn_ind:
                    self.neighbor_idxs = ind
                    self._update_embeddings()

                # self._update_embeddings_bat()

                for i, x in enumerate(x_bat):
                    i_1 = self.bknn_ind[i][0]
                    self.grow_rate[i_1] += np.linalg.norm(x - self.coding_vector[i_1])
                    if self.grow_rate[i_1] > self.theta_g:
                        self._refine_topology(x)

            self.alpha = self.init_alpha * (1 - epoch / self.n_max_epoch)
            epoch += 1
            if epoch >= self.n_max_epoch:
                is_execute = True

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
