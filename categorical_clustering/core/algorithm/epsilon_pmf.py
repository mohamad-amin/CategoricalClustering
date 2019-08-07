from copy import deepcopy
from sys import maxsize

from .base import BaseIterativeClustering


class EpsilonPmf(BaseIterativeClustering):

    def _random_swap(self):
        raise NotImplementedError()

    def _calculate_centroids(self) -> list:
        cluster_pmfs = [deepcopy(cmf) for cmf in self.clustering.clusters]
        for k, pmf in enumerate(cluster_pmfs):
            size = self.clustering.get_cluster_size(k)
            pmf /= size
        return cluster_pmfs

    def _assign_data_to_clusters(self, centroids: list) -> int:

        movements = 0
        writable_clusters = self.clustering.get_writable_clusters()
        for index, row in self.dataset.iterrows():

            candidate_clusters = []
            most_pmf, selected_cluster = 0, -1
            prev_cluster = self.cluster_assignments[index]
            multi_index = self._get_multi_dimensional_index(row)

            for k in range(self.n_clusters):
                cluster_probability = centroids[k][multi_index]
                if cluster_probability > most_pmf:
                    most_pmf = cluster_probability
                    selected_cluster = k
                    candidate_clusters.clear()
                elif cluster_probability == most_pmf:
                    candidate_clusters += [k]

            if len(candidate_clusters) != 0:
                candidate_clusters += [selected_cluster]
                least_cluster_size = maxsize
                for k in candidate_clusters:
                    size = self.clustering.get_cluster_size(k)
                    if size < least_cluster_size:
                        selected_cluster = k

            if prev_cluster != selected_cluster:
                writable_clusters[prev_cluster][multi_index] -= 1
                writable_clusters[selected_cluster][multi_index] += 1
                movements += 1
                self.cluster_assignments[index] = selected_cluster

        self.clustering.update_clusters_from_writable(writable_clusters)
        return movements
