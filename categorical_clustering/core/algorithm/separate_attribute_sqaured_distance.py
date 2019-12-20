import numpy as np
import numba as nb
from copy import deepcopy

from ...utils.logger import log, LEVEL
from .base import BaseIterativeClustering


@nb.njit(parallel=True)
def calculate_point_destinations(n_points, n_clusters, multi_indexes, multi_index_size, dims_array, centroids, centroid_size):
    dests = np.zeros(n_points, dtype=np.int32)
    for index in nb.prange(n_points):

        multi_index = multi_indexes[index * multi_index_size:(index + 1) * multi_index_size]
        indices = dims_array + multi_index

        least_distance = 1e7
        selected_cluster = -1
        for k in range(n_clusters):
            result = 0.0
            centroid = centroids[k * centroid_size:(k + 1) * centroid_size]
            for i in range(len(indices)):
                result += (1 - centroid[indices[i]]) ** 2
            if result < least_distance:
                least_distance = result
                selected_cluster = k

        dests[index] = selected_cluster

    return dests


class SeparateAttributeSquaredDistance(BaseIterativeClustering):

    def _random_swap(self, victim_cluster=-1, prototype_index=-1):

        # print('Impurity before random swap: {}'.format(self.clustering.calculate_overall_impurity()))
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

    def _assign_data_to_clusters(self, centroids: list) -> int:

        dims_array = np.arange(self.dataset.shape[1]) * centroids[0].shape[1]

        for k in range(self.n_clusters):
            centroids[k] = centroids[k].ravel()

        centroid_size = len(centroids[0])
        centroids = np.array(centroids).ravel()

        dests = calculate_point_destinations(
            self.n_points, self.n_clusters, self.r_multi_indexes, self.multi_index_size, dims_array, centroids, centroid_size)

        movements = 0
        for index in range(self.n_points):

            prev_cluster = self.cluster_assignments[index]
            multi_index = self.multi_indexes[index]
            selected_cluster = dests[index]

            if prev_cluster != selected_cluster:
                if prev_cluster != -1:
                    self.clustering.remove_row_from_cluster(multi_index, prev_cluster)
                self.clustering.assign_row_to_cluster(multi_index, selected_cluster)
                movements += 1
                self.cluster_assignments[index] = selected_cluster

        return movements
