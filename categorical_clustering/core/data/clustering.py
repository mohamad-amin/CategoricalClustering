import sparse
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
        self.clusters = [sparse.zeros(tuple(self.n_categories)) for i in range(self.n_clusters)]

    def reset_clusters(self):
        """
        Resets all clusters to empty ones.
        """
        self.clusters = [sparse.zeros(tuple(self.n_categories)) for i in range(self.n_clusters)]

    def get_writable_clusters(self):
        """
        Todo
        :return:
        """
        return list(map(lambda cluster: cluster.asformat('dok'), self.clusters))

    def update_clusters_from_writable(self, clusters):
        """
        Todo
        :param clusters:
        :return:
        """
        self.clusters = [cluster.asformat('coo') for cluster in clusters]

    def get_cluster_size(self, cluster: int) -> int:
        """
        Calculates the count of items in a specific cluster
        :param cluster: An int, the cluster to calculate size on.
        :return: A int, the size of the cluster.
        """
        dimensions = list(range(len(self.n_categories)))
        size = int(np.sum(self.clusters[cluster], dimensions))
        return size

    def calculate_cluster_impurity(self, cluster: int) -> float:
        """
        Calculates the current entropy for a specific cluster.
        :param cluster: An int, indicating the cluster to calculate the entropy of.
        :return: A float, indicating the impurity of the wanted cluster.
        """
        if self.size == 0:
            return 0.0
        dimensions = list(range(len(self.n_categories)))
        size = int(np.sum(self.clusters[cluster], dimensions))
        if size == 0:
            return 0.0
        ent = 0.0
        for d in dimensions:
            other_dimensions = dimensions[:d] + dimensions[d+1:]
            ent += entropy(np.sum(self.clusters[cluster], other_dimensions).todense().squeeze())
        return ent * (size / self.size)

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
        from sparse import DOK
        raise NotImplementedError('Not implemented yet')
