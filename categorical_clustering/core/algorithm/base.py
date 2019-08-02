import numpy as np
import pandas as pd
from abc import abstractmethod

from ..data.clustering import Clustering


class BaseIterativeClustering:

    def __init__(self, n_clusters: int):
        """
        Initiates an iterative clustering algorithm.
        :param n_clusters: Number of clusters to achieve.
        """
        self.n_clusters = n_clusters
        self.dataset = None
        self.clustering = None
        self.category_values = None
        self.cluster_assignments = None

    def set_data(self, dataset: pd.DataFrame) -> None:
        """
        Updates the dataset of the clustering algorithm, this clears the past configurations and results.
        :param dataset: The input dataset for clustering algorithm.
        """
        self.dataset = dataset

        n_categories = [dataset[col].nunique() for col in dataset.columns]
        self.clustering = Clustering(self.n_clusters, n_categories, dataset.shape[0])

        self.category_values = {}
        for col in dataset.columns:
            self.category_values[col] = {k: v for v, k in enumerate(dataset[col].unique())}

        self.cluster_assignments = [-1] * dataset.shape[0]

    def _get_multi_dimensional_index(self, point) -> list:
        """
        Returns the index of the input point in the multi-dimensional array holding the cmf values in Clustering object.
        :param point: A pandas.Series, the data point to return index for (a row of data dataset).
        :return: A list, the index of the input data point, as a list of indexes for each dimension.
        """
        multi_index = []
        for col in self.dataset.columns:
            multi_index += [self.category_values[col][point[col]]]
        return tuple(multi_index)

    def _initialize_clusters(self):
        """
        Initializes the clusters randomly by assigning each data point randomly to a cluster.
        """
        self.clustering.reset_clusters()
        for idx, row in self.dataset.iterrows():
            cluster = np.random.choice(self.n_clusters)
            multi_index = self._get_multi_dimensional_index(row)
            self.clustering.clusters[cluster][multi_index] += 1
            self.cluster_assignments[idx] = cluster

    @abstractmethod
    def _calculate_centroids(self) -> list:
        """
        Calculates the centroids for each cluster.
        :return: A list of centroids for each cluster.
        """
        pass

    @abstractmethod
    def _assign_data_to_clusters(self, centroids: list) -> int:
        """
        Assigns each data point to a cluster, updates the clustering parameter.
        :param centroids: A list of the centroids for each cluster.
        :return: An int, the number of data points who have moved between clusters.
        """
        pass

    def perform_clustering(self, update_proportion_criterion: float = 0.0, verbose=False) -> (Clustering, int):
        """
        Performs the clustering algorithm on the dataset.
        :param verbose: boolean, printing verbose information on the clustering process.
        :param update_proportion_criterion: A float, The criterion to stop the clustering iterations, if the proportion of
            data points moved between clusters is less than this value, the clustering will stop continuing.
        :return: A tuple consisting of a clustering object from type Clustering and an int that indicates the number of
            iterations until convergence according to the criterion.
        """
        self._initialize_clusters()
        iterations = 0
        while True:
            iterations += 1
            centroids = self._calculate_centroids()
            moved_points_count = self._assign_data_to_clusters(centroids)
            if verbose:
                print('Moved points in iteration {}: {}'.format(iterations, moved_points_count))
                print('Overall impurity is now: {}'.format(self.clustering.calculate_overall_impurity()))
            if moved_points_count / self.dataset.shape[0] <= update_proportion_criterion:
                print('Reached criterion, stoping.')
                break
        return self.clustering, iterations

    def get_cluster_as_dataframe(self, cluster=-1):
        """
        Todo
        :param cluster:
        :return:
        """
        if cluster == -1:
            data = np.zeros(self.clustering.n_categories)
            for k in range(self.n_clusters):
                data += self.clustering.clusters[k]
        else:
            data = self.clustering.clusters[cluster]
        column_values = []
        for col in range(self.dataset.shape[1]):
            column_values += [list(self.category_values[self.dataset.columns[col]])]
        if self.dataset.shape[1] == 1:
            df = pd.DataFrame(data=data, index=column_values[0])
        elif self.dataset.shape[1] == 2:
            df = pd.DataFrame(data=data, index=column_values[0], columns=column_values[1])
        else:
            raise AttributeError('Can not have a dataframe for more than two attributes!')
        return df
