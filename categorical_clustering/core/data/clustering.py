import numpy as np
from numba import njit, jit
from sklearn.preprocessing import LabelEncoder
from categorical_clustering.utils.distance_utils import squared_euclidean_distance, hamming_distance


class Clustering:

    def __init__(self, n_clusters, n_categories, size):
        """
        Initializing the clustering
        :param n_clusters: the number of clusters
        :param n_categories: a list of the count of unique categories for each cluster
        """
        self.step_ent = 0
        self.size = size
        self.n_clusters = n_clusters
        self.n_categories = n_categories
        self.clusters = [np.zeros((len(n_categories), max(n_categories))).astype('float') for i in range(self.n_clusters)]
        self.cluster_sizes = [0] * n_clusters

    def reset_clusters(self):
        """
        Resets all clusters to empty ones.
        """
        self.clusters = [np.zeros(self.clusters[0].shape).astype('float') for i in range(self.n_clusters)]
        self.cluster_sizes = [0] * self.n_clusters

    def get_writable_clusters(self):
        """
        Todo
        :return:
        """
        return self.clusters

    def update_clusters_from_writable(self, clusters):
        """
        Todo
        :param clusters:
        :return:
        """
        self.clusters = clusters

    def get_cluster_size(self, cluster: int) -> int:
        """
        Calculates the count of items in a specific cluster
        :param cluster: An int, the cluster to calculate size on.
        :return: A int, the size of the cluster.
        """
        return self.cluster_sizes[cluster]

    def calculate_cluster_impurity(self, cluster: int) -> float:
        """
        Calculates the current entropy for a specific cluster.
        :param cluster: An int, indicating the cluster to calculate the entropy of.
        :return: A float, indicating the impurity of the wanted cluster.
        """
        if self.size == 0:
            return 0.0
        new_data = self.clusters[cluster].reshape(-1,)
        new_data = new_data[new_data.nonzero()[0]] / self.cluster_sizes[cluster]
        ent = -np.sum(new_data * np.log(new_data))
        return ent * (self.cluster_sizes[cluster] / self.size)

    def calculate_overall_impurity(self) -> float:
        """
        Calculates the overall impurity of the clustering.
        :return: A float, the overall impurity of the clustering.
        """
        ent = 0.0
        for cluster in range(self.n_clusters):
            ent += self.calculate_cluster_impurity(cluster)
        return ent

    # This is going to be a rough task :D
    def clusters_to_string(self):
        raise NotImplementedError('Not implemented yet')

    @staticmethod
    @njit
    def _fast_change(cluster, row_multi_index, how):
        for dimension, index in enumerate(row_multi_index):
            cluster[dimension, index] += how

    def assign_row_to_cluster(self, row_multi_index, cluster):
        self.cluster_sizes[cluster] += 1
        self._fast_change(self.clusters[cluster], row_multi_index, +1)

    def remove_row_from_cluster(self, row_multi_index, cluster):
        self.cluster_sizes[cluster] -= 1
        self._fast_change(self.clusters[cluster], row_multi_index, -1)

    @staticmethod
    @jit(parallel=True)
    def calculate_overlap(dataset, cluster_assignments, centroids, multi_indexes):

        for k in range(len(centroids)):
            centroids[k] = centroids[k].ravel()

        dataset = dataset.apply(LabelEncoder().fit_transform).values

        cluster_assignments = np.array(cluster_assignments)
        overlaps = 0
        for index in range(len(dataset)):
            current_cluster = cluster_assignments[index]
            distance_to_centroid = squared_euclidean_distance(centroids[current_cluster], multi_indexes[index])
            item = dataset[index, :]
            for other_index in np.where(cluster_assignments != current_cluster)[0]:
                if hamming_distance(item, dataset[other_index, :]) < distance_to_centroid:
                    overlaps += 1
                    break

        return overlaps / len(dataset)
