import numpy as np
from scipy.stats import entropy


class Clustering:

    def __init__(self, n_clusters, n_categories, size):
        """
        Initializing the clustering
        :param n_clusters: the number of clusters
        :param n_categories: a list of the count of unique categories for each cluster
        """
        self.size = size
        self.n_clusters = n_clusters
        self.n_categories = n_categories
        self.clusters = [np.zeros((len(n_categories), max(n_categories))).astype('float') for i in range(self.n_clusters)]
        self.cluster_sizes = [0] * size

    def reset_clusters(self):
        """
        Resets all clusters to empty ones.
        """
        self.clusters = [np.zeros(self.clusters[0].shape).astype('float') for i in range(self.n_clusters)]

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

    def assign_row_to_cluster(self, row_multi_index, cluster):
        self.cluster_sizes[cluster] += 1
        for dimension, index in enumerate(row_multi_index):
            self.clusters[cluster][dimension, index] += 1

    def remove_row_from_cluster(self, row_multi_index, cluster):
        self.cluster_sizes[cluster] -= 1
        for dimension, index in enumerate(row_multi_index):
            self.clusters[cluster][dimension, index] -= 1
