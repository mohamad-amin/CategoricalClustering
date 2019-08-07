import numpy as np
from copy import deepcopy
from scipy.stats import entropy

from .base import BaseIterativeClustering


class SeparateAttributePMF(BaseIterativeClustering):

    def _random_swap(self):
        raise NotImplementedError()

    def _calculate_centroids(self) -> list:
        cluster_dimensions = []
        dimensions = list(range(len(self.clustering.n_categories)))
        for k in range(self.n_clusters):
            dimension_counts = []
            for d in dimensions:
                other_dimensions = dimensions[:d] + dimensions[d + 1:]
                dimension_values = deepcopy(np.sum(self.clustering.clusters[k], other_dimensions).todense().squeeze())
                dimension_counts += [dimension_values]
            cluster_dimensions += [dimension_counts]
        return cluster_dimensions

    def _assign_data_to_clusters(self, centroids: list) -> int:

        movements = 0
        writable_clusters = self.clustering.get_writable_clusters()
        for index, row in self.dataset.iterrows():

            least_entropy = float('inf')
            selected_cluster = -1
            prev_cluster = self.cluster_assignments[index]
            multi_index = self._get_multi_dimensional_index(row)

            for k in range(self.n_clusters):
                sum_entropy = 0.0
                for dim, attr_idx in enumerate(multi_index):
                    ent = entropy(centroids[k][dim])
                    if not np.isnan(ent):
                        sum_entropy += float('inf')
                if least_entropy > sum_entropy:
                    least_entropy = sum_entropy
                    selected_cluster = k

            if prev_cluster != selected_cluster:
                writable_clusters[prev_cluster][multi_index] -= 1
                writable_clusters[selected_cluster][multi_index] += 1
                movements += 1
                self.cluster_assignments[index] = selected_cluster
            else:
                pass

        self.clustering.update_clusters_from_writable(writable_clusters)
        return movements
