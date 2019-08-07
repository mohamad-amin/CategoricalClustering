import numpy as np
from sparse import DOK
from copy import deepcopy

from ...utils.logger import log, LEVEL
from .base import BaseIterativeClustering


class SeparateAttributeCategoryPMF(BaseIterativeClustering):

    def _random_swap(self):

        log('Performing random swap...', level=LEVEL.VERBOSE)

        # Choosing the new prototype
        prototype_index = np.random.choice(self.dataset.shape[0])
        prototype = self.dataset.loc[prototype_index]
        multi_index = self._get_multi_dimensional_index(prototype)
        prototype_prev_cluster = self.cluster_assignments[prototype_index]

        log('Chosen data point: ' + ''.join(prototype) + ' in cluster ' + str(prototype_prev_cluster), tabs=1, level=LEVEL.VERBOSE)

        # Choosing a victim cluster to be sacrificed
        victim_cluster = np.random.choice(self.n_clusters)
        while prototype_prev_cluster == victim_cluster:
            victim_cluster = np.random.choice(self.n_clusters)

        log('Chosen victim cluster: ' + str(victim_cluster), tabs=1, level=LEVEL.VERBOSE)

        # Creating the new cluster
        new_cluster = DOK(tuple(self.clustering.n_categories))
        new_cluster[multi_index] += 1

        # Updating the clustering cmfs
        writable_clusters = self.clustering.get_writable_clusters()
        writable_clusters[prototype_prev_cluster][multi_index] -= 1
        writable_clusters[victim_cluster] = new_cluster
        self.clustering.update_clusters_from_writable(writable_clusters)

        # Updating the cluster assignments
        self.cluster_assignments = list(map(lambda x: -1 if x == victim_cluster else x, self.cluster_assignments))
        self.cluster_assignments[prototype_index] = victim_cluster

    def _calculate_centroids(self) -> list:

        cluster_dimensions = []
        dimensions = list(range(len(self.clustering.n_categories)))
        log('Calculating centroids...', level=LEVEL.VERBOSE)

        for k in range(self.n_clusters):

            # Todo: think about empty clusters

            log('Cluster: ' + str(k), tabs=1, level=LEVEL.VERBOSE)
            dimension_counts = []
            empty_categories_count = 0
            full_categories_count = 0
            min_nonzero_category_count = float('inf')

            for d in dimensions:

                other_dimensions = dimensions[:d] + dimensions[d + 1:]
                dimension_values = deepcopy(
                    np.sum(self.clustering.clusters[k], other_dimensions).todense().squeeze().reshape(-1,))

                full_categories_count += np.count_nonzero(dimension_values)
                empty_categories_count += (len(dimension_values) - np.count_nonzero(dimension_values))
                min_nonzero_category = dimension_values[dimension_values.nonzero()[0]].min()
                if min_nonzero_category < min_nonzero_category_count:
                    min_nonzero_category_count = min_nonzero_category

                dimension_counts += [dimension_values]
                log('Dimension: ' + str(d) + ', Category counts: ' + str(dimension_values), tabs=2, level=LEVEL.VERBOSE)

            if empty_categories_count > 0:
                log('Found ' + str(empty_categories_count) + ' empty categories.', tabs=2, level=LEVEL.VERBOSE)
                sized_epsilon = min(1, min_nonzero_category_count / (2*empty_categories_count))
                cluster_size = dimension_counts[0].sum()
                epsilon_count = (sized_epsilon / cluster_size) * empty_categories_count
                log('Sized epsilon: ' + str(sized_epsilon) + ', min_nonzero: ' + str(min_nonzero_category_count), tabs=2, level=LEVEL.VERBOSE)
                # Todo: we can avoid this loop if we store the number of
                # Todo: epsilon idea is tough!
                for d in dimensions:
                    for category_index in range(len(dimension_counts[d])):
                        if dimension_counts[d][category_index] == 0:
                            dimension_counts[d][category_index] = sized_epsilon
                        else:
                            dimension_counts[d][category_index] *= (1 - epsilon_count)
                    log('Final dimension ' + str(d) + ': ' + str(dimension_counts[d]), tabs=3, level=LEVEL.VERBOSE)

            cluster_dimensions += [dimension_counts]

        return cluster_dimensions

    def _assign_data_to_clusters(self, centroids: list) -> int:

        log('Assigning items to clusters...', level=LEVEL.VERBOSE)

        movements = 0
        writable_clusters = self.clustering.get_writable_clusters()
        for index, row in self.dataset.iterrows():

            least_entropy = float('inf')
            selected_cluster = -1
            prev_cluster = self.cluster_assignments[index]
            multi_index = self._get_multi_dimensional_index(row)

            log('Data point: ' + ''.join(row) + ' in cluster ' + str(prev_cluster), tabs=1, level=LEVEL.VERBOSE)

            for k in range(self.n_clusters):
                log('Cluster: ' + str(k), tabs=2, level=LEVEL.VERBOSE)
                sum_entropy = 0.0
                for dim, attr_idx in enumerate(multi_index):
                    log('Attribute: ' + str(row[dim]) + ' with index: ' + str(attr_idx) + ' in dimension: ' + str(dim), tabs=3, level=LEVEL.VERBOSE)
                    category_values = centroids[k][dim]
                    current_category = category_values[attr_idx]
                    items_in_this_dimension = sum(category_values)
                    if items_in_this_dimension == 0:
                        # Empty cluster
                        log('No items in this cluster! Using entropy 100 for cluster', tabs=3, level=LEVEL.VERBOSE)
                        sum_entropy = 100  # Todo: fix this with epsilon idea
                        break
                    attr_probability = current_category / items_in_this_dimension
                    # Applying epsilon idea
                    if attr_probability == 0:
                        attr_probability = 1 / items_in_this_dimension
                        log('No item in this category! Using epsilon as 1 / ' + str(items_in_this_dimension) + ' : ' + str(attr_probability), tabs=4, level=LEVEL.VERBOSE)
                    else:
                        log('Probability in this cluster: ' + str(current_category) + ' / ' + str(items_in_this_dimension) + ' = ' + str(attr_probability), tabs=4, level=LEVEL.VERBOSE)
                    ent = -np.log(attr_probability)
                    log('Entropy in this cluster: -log(probability) = ' + str(ent), tabs=4, level=LEVEL.VERBOSE)
                    sum_entropy += ent
                log('Sum entropy for this cluster: ' + str(sum_entropy), tabs=3, level=LEVEL.VERBOSE)
                if least_entropy > sum_entropy:
                    least_entropy = sum_entropy
                    selected_cluster = k

            if prev_cluster != selected_cluster:
                log('Moving from cluster ' + str(prev_cluster) + ' to ' + str(selected_cluster), tabs=2, level=LEVEL.VERBOSE)
                if prev_cluster != -1:
                    writable_clusters[prev_cluster][multi_index] -= 1
                writable_clusters[selected_cluster][multi_index] += 1
                movements += 1
                self.cluster_assignments[index] = selected_cluster
            else:
                log('No movement!', tabs=2, level=LEVEL.VERBOSE)
                pass

        self.clustering.update_clusters_from_writable(writable_clusters)
        return movements
