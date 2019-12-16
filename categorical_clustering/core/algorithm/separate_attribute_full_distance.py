import numpy as np
from numba import jit, njit
from copy import deepcopy

from ...utils.logger import log, LEVEL
from .base import BaseIterativeClustering


class SeparateAttributeFullDistance(BaseIterativeClustering):

    def _random_swap(self, victim_cluster=-1, prototype_index=-1):

        print('Impurity before random swap: {}'.format(self.clustering.calculate_overall_impurity()))
        # log('Performing random swap...', level=LEVEL.VERBOSE)

        # Choosing the new prototype
        if prototype_index == -1:
            prototype_cluster = np.random.choice(self.n_clusters)
            items = np.where(np.array(self.cluster_assignments) == prototype_cluster)[0]
            while len(items) == 0:
                prototype_cluster = np.random.choice(self.n_clusters)
                items = np.where(np.array(self.cluster_assignments) == prototype_cluster)[0]
            prototype_choice = np.random.choice(len(items))
            prototype_index = items[prototype_choice]
        prototype = self.dataset.loc[prototype_index]
        multi_index = self.multi_indexes[prototype_index]
        prototype_prev_cluster = self.cluster_assignments[prototype_index]

        # log('Chosen data point: ' + ''.join(prototype) + ' in cluster ' + str(prototype_prev_cluster), tabs=1, level=LEVEL.VERBOSE)

        # Choosing a victim cluster to be sacrificed
        if victim_cluster == -1:
            victim_cluster = np.random.choice(self.n_clusters)
            while prototype_prev_cluster == victim_cluster:
                victim_cluster = np.random.choice(self.n_clusters)

        # log('Chosen victim cluster: ' + str(victim_cluster), tabs=1, level=LEVEL.VERBOSE)

        # Creating the new cluster
        new_cluster = np.zeros(self.clustering.clusters[0].shape)
        for dimension, index in enumerate(multi_index):
            new_cluster[dimension, index] += 1

        # Updating the clustering cmfs
        self.clustering.remove_row_from_cluster(multi_index, prototype_prev_cluster)
        self.clustering.clusters[victim_cluster] = new_cluster
        self.clustering.cluster_sizes[victim_cluster] = 1

        # Updating the cluster assignments
        self.cluster_assignments = list(map(lambda x: -1 if x == victim_cluster else x, self.cluster_assignments))
        self.cluster_assignments[prototype_index] = victim_cluster

    @jit
    def _calculate_centroids(self) -> list:

        cluster_matrices = []
        dimensions = list(range(len(self.clustering.n_categories)))

        for k in range(self.n_clusters):
            cluster_dimensions = np.zeros(self.clustering.clusters[0].shape)
            cluster_size = self.clustering.clusters[k][0, :].sum()
            if cluster_size == 0:
                print('Cluster size 0!')
                self.should_abort = True
                return None
            for d in dimensions:
                cluster_dimensions[d, :] = deepcopy(self.clustering.clusters[k][d, :] / cluster_size)
            cluster_matrices += [cluster_dimensions]

        return cluster_matrices

    @staticmethod
    @njit
    def _indexed_get_sum(centroid, indices, max_cats):
        result = 0
        for i in range(len(indices)):
            for j in range(max_cats):
                if j == indices[i]:
                    result += (1 - centroid[indices[i]]) ** 2
                else:
                    result += (centroid[indices[i]]) ** 2
        return result

    @jit(parallel=True)
    def _assign_data_to_clusters(self, centroids: list) -> int:

        max_cats = centroids[0].shape[1]
        dims_array = np.arange(self.dataset.shape[1]) * max_cats

        for k in range(self.n_clusters):
            centroids[k] = centroids[k].ravel()

        movements = 0
        for index in range(self.n_points):

            prev_cluster = self.cluster_assignments[index]
            multi_index = self.multi_indexes[index]
            indices = dims_array + multi_index

            least_distance = 1e10
            selected_cluster = -1
            for k in range(self.n_clusters):
                result = self._indexed_get_sum(centroids[k], indices, max_cats)
                if result < least_distance:
                    least_distance = result
                    selected_cluster = k

            if prev_cluster != selected_cluster:
                if prev_cluster != -1:
                    self.clustering.remove_row_from_cluster(multi_index, prev_cluster)
                self.clustering.assign_row_to_cluster(multi_index, selected_cluster)
                movements += 1
                self.cluster_assignments[index] = selected_cluster

        return movements
