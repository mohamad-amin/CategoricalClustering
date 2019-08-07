import sparse
import numpy as np
import pandas as pd
from copy import deepcopy
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

    def _get_multi_dimensional_index(self, point) -> tuple:
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
        writable_clusters = self.clustering.get_writable_clusters()
        for idx, row in self.dataset.iterrows():
            cluster = np.random.choice(self.n_clusters)
            multi_index = self._get_multi_dimensional_index(row)
            writable_clusters[cluster][multi_index] += 1
            self.cluster_assignments[idx] = cluster
        self.clustering.update_clusters_from_writable(writable_clusters)

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

    @abstractmethod
    def _random_swap(self):
        """
        Performs a random swap, removing one of the clusters and then creating a new one.
        """
        pass

    def _perform_iteration(self):
        """
        Combines the steps of calculating centroids and assigning them to the clusters
        :return: An int, the number of data points who have moved between clusters.
        """
        centroids = self._calculate_centroids()
        moved_points_count = self._assign_data_to_clusters(centroids)
        return moved_points_count

    def perform_clustering(self, update_proportion_criterion: float = 0.0, initialize_clusters=True, verbose=False) -> (Clustering, int):
        """
        Performs the clustering algorithm on the dataset.
        :param verbose: boolean, printing verbose information on the clustering process.
        :param update_proportion_criterion: A float, The criterion to stop the clustering iterations, if the proportion
            of data points moved between clusters is less than this value, the clustering will stop continuing.
        :return: A tuple consisting of a clustering object from type Clustering and an int that indicates the number of
            iterations until convergence according to the criterion.
        """
        if initialize_clusters:
            self._initialize_clusters()
        iterations = 0
        while True:
            iterations += 1
            moved_points_count = self._perform_iteration()
            if True:
                print('Moved points in iteration {}: {}'.format(iterations, moved_points_count))
                print('Overall impurity is now: {}'.format(self.clustering.calculate_overall_impurity()))
            if moved_points_count / self.dataset.shape[0] <= update_proportion_criterion:
                print('Reached criterion, stoping.')
                break
        return self.clustering, iterations

    def perform_random_swap_clustering(self, t: int, update_proportion_criterion: float = 0.0, verbose=False) -> (Clustering, int):
        """
        Performs the clustering algorithm + random swap on the dataset.
        :param t: An int, the number of random swap iterations
        :param verbose: boolean, printing verbose information on the clustering process.
        :param update_proportion_criterion: A float, The criterion to stop the clustering iterations, if the proportion
            of data points moved between clusters is less than this value, the clustering will stop continuing.
        :return: A tuple consisting of a clustering object from type Clustering and an int that indicates the total
            number of iterations until convergence according to the criterion (iterations of the original algorithm,
            not random swap).
        """
        _, iterations = self.perform_clustering(update_proportion_criterion, verbose)
        random_swaps = 0

        while random_swaps < t:

            # Getting a backup from the current state
            prev_clustering = deepcopy(self.clustering)
            prev_cluster_assignemnts = deepcopy(self.cluster_assignments)
            prev_overall_impurity = self.clustering.calculate_overall_impurity()

            # Performing a random swap
            self._random_swap()
            _, iters = self.perform_clustering(update_proportion_criterion, False, verbose)
            iterations += iters

            # Checking if we've fucked up and if so, then rolling back
            if self.clustering.calculate_overall_impurity() > prev_overall_impurity:
                if True:
                    print('Random swap rejected!')
                self.clustering = prev_clustering
                self.cluster_assignments = prev_cluster_assignemnts
            else:
                if True:
                    print('Random swap accepted!')
                random_swaps += 1

        return self.clustering, iterations

    def get_cluster_as_dataframe(self, cluster=-1):
        """
        Todo
        :param cluster:
        :return:
        """
        if cluster == -1:
            data = sparse.zeros(tuple(self.clustering.n_categories))
            for k in range(self.n_clusters):
                data += self.clustering.clusters[k]
        else:
            data = self.clustering.clusters[cluster]
        column_values = []
        for col in range(self.dataset.shape[1]):
            column_values += [list(self.category_values[self.dataset.columns[col]])]
        if self.dataset.shape[1] == 1:
            df = pd.DataFrame(data=data.todense(), index=column_values[0])
        elif self.dataset.shape[1] == 2:
            df = pd.DataFrame(data=data.todense(), index=column_values[0], columns=column_values[1])
        else:
            raise AttributeError('Can not have a dataframe for more than two attributes!')
        return df
