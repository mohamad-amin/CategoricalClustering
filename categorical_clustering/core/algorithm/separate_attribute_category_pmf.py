import numpy as np
from copy import deepcopy

from .base import BaseIterativeClustering


class SeparateAttributeCategoryPMF(BaseIterativeClustering):

    def _calculate_centroids(self) -> list:
        cluster_dimensions = []
        dimensions = list(range(len(self.clustering.n_categories)))
        for k in range(self.n_clusters):
            dimension_counts = []
            for d in dimensions:
                other_dimensions = dimensions[:d] + dimensions[d + 1:]
                dimension_values = deepcopy(np.apply_over_axes(np.sum, self.clustering.clusters[k], other_dimensions).squeeze())
                dimension_counts += [dimension_values]
            cluster_dimensions += [dimension_counts]
        return cluster_dimensions

    def _assign_data_to_clusters(self, centroids: list) -> int:

        movements = 0
        for index, row in self.dataset.iterrows():

            least_entropy = float('inf')
            selected_cluster = -1
            prev_cluster = self.cluster_assignments[index]
            multi_index = self._get_multi_dimensional_index(row)

            for k in range(self.n_clusters):
                sum_entropy = 0.0
                for dim, attr_idx in enumerate(multi_index):
                    category_values = centroids[k][dim]
                    current_category = category_values[attr_idx]
                    items_in_this_dimension = sum(category_values)
                    if items_in_this_dimension == 0:
                        # Empty cluster
                        sum_entropy = 100  # Todo: fix this with epsilon idea
                        break
                    attr_probability = current_category / items_in_this_dimension
                    # Applying epsilon idea
                    if attr_probability == 0:
                        attr_probability = 1 / items_in_this_dimension
                    ent = np.log(attr_probability)
                    sum_entropy += -ent
                if least_entropy > sum_entropy:
                    least_entropy = sum_entropy
                    selected_cluster = k

            if prev_cluster != selected_cluster:
                self.clustering.clusters[prev_cluster][multi_index] -= 1
                self.clustering.clusters[selected_cluster][multi_index] += 1
                movements += 1
                self.cluster_assignments[index] = selected_cluster
            else:
                pass

        return movements
