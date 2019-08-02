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
        self.clustering.clusters = [np.zeros(self.n_categories) for i in range(self.n_clusters)]

    def reset_clusters(self):
        self.clusters = [np.zeros(self.n_categories) for i in range(self.n_clusters)]

    def get_cluster_size(self, cluster: int) -> int:
        """
        Calculates the count of items in a specific cluster
        :param cluster: An int, the cluster to calculate size on.
        :return: A int, the size of the cluster.
        """
        dimensions = list(range(len(self.n_categories)))
        size = int(np.apply_over_axes(np.sum, self.clusters[cluster], dimensions))
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
        size = int(np.apply_over_axes(np.sum, self.clusters[cluster], dimensions))
        if size == 0:
            return 0.0
        ent = 0.0
        for d in dimensions:
            other_dimensions = dimensions[:d] + dimensions[d+1:]
            # Note: number of attributes might create a problem here! or might not. we should check it (Todo)
            ent += entropy(np.apply_over_axes(np.sum, self.clusters[cluster], other_dimensions).squeeze())
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
        # return pd.DataFrame(
        #     data=self.clusters,
        #     index=['Cluster ' + str(i) for i in range(self.n_clusters)],
        #     columns=self.values
        # ).to_string()
        raise NotImplementedError('Not implemented yet')
