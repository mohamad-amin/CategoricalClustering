import numpy as np
from copy import deepcopy
import time

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
        new_cluster = np.zeros(self.clustering.clusters[0].shape)
        for dimension, index in enumerate(multi_index):
            new_cluster[dimension, index] += 1

        # Updating the clustering cmfs
        self.clustering.remove_row_from_cluster(multi_index, prototype_prev_cluster)
        self.clustering.clusters[victim_cluster] = new_cluster

        # Updating the cluster assignments
        self.cluster_assignments = list(map(lambda x: -1 if x == victim_cluster else x, self.cluster_assignments))
        self.cluster_assignments[prototype_index] = victim_cluster

    def _calculate_centroids(self) -> list:

        cluster_matrices = []
        dimensions = list(range(len(self.clustering.n_categories)))
        log('Calculating centroids...', level=LEVEL.VERBOSE)

        for k in range(self.n_clusters):

            # Todo: think about empty clusters

            log('Cluster: ' + str(k), tabs=1, level=LEVEL.VERBOSE)
            cluster_dimensions = []
            empty_categories_count = 0
            full_categories_count = 0
            min_nonzero_category_prob = float('inf')
            cluster_size = self.clustering.clusters[k][0, :].sum()
            if cluster_size == 0:
                self.should_abort = True
                return None

            for d in dimensions:

                categories_count = len(self.category_values[self.dataset.columns[d]])
                dimension_values = deepcopy(self.clustering.clusters[k][d, :categories_count] / cluster_size)

                full_categories_count += np.count_nonzero(dimension_values)
                empty_categories_count += (len(dimension_values) - np.count_nonzero(dimension_values))
                min_nonzero_category = dimension_values[dimension_values.nonzero()[0]].min()
                if min_nonzero_category < min_nonzero_category_prob:
                    min_nonzero_category_prob = min_nonzero_category

                cluster_dimensions += [dimension_values]
                log('Dimension: ' + str(d) + ', Category counts: ' + str(dimension_values), tabs=2, level=LEVEL.VERBOSE)

            if empty_categories_count > 0:
                log('Found ' + str(empty_categories_count) + ' empty categories.', tabs=2, level=LEVEL.VERBOSE)
                epsilon = min(1.0/cluster_size, min_nonzero_category_prob/(5*empty_categories_count))
                sum_epsilons = epsilon * empty_categories_count
                log('Epsilon: ' + str(epsilon) + ', min_nonzero: ' + str(min_nonzero_category_prob), tabs=2, level=LEVEL.VERBOSE)
                # Todo: we can avoid this loop if we store the number of
                # Todo: epsilon idea is tough!
                for d in dimensions:
                    for category_index in range(len(cluster_dimensions[d])):
                        if cluster_dimensions[d][category_index] == 0:
                            cluster_dimensions[d][category_index] = epsilon
                        else:
                            cluster_dimensions[d][category_index] *= (1 - sum_epsilons)
                    log('Final dimension ' + str(d) + ': ' + str(cluster_dimensions[d]), tabs=3, level=LEVEL.VERBOSE)

            cluster_matrices += [cluster_dimensions]

        return cluster_matrices

    def _assign_data_to_clusters(self, centroids: list) -> int:

        move_in = time.time()
        log('Assigning items to clusters...', level=LEVEL.VERBOSE)

        movements = 0
        # for index, row in zip(self.dataset.index, self.dataset.values):
        for index in self.dataset.index.values:

            last_stop = time.time()
            least_entropy = float('inf')
            selected_cluster = -1
            prev_cluster = self.cluster_assignments[index]

            # log('Data point: ' + ''.join(row) + ' in cluster ' + str(prev_cluster), tabs=1, level=LEVEL.VERBOSE)
            self.step_init += time.time() - last_stop
            last_stop = time.time()

            multi_index = self._get_multi_dimensional_index(self.dataset.values[index, :])
            self.step_index += time.time() - last_stop
            last_stop = time.time()

            for k in range(self.n_clusters):
                # log('Cluster: ' + str(k), tabs=2, level=LEVEL.VERBOSE)
                sum_entropy = 0.0
                # items_in_this_dimension = sum(centroids[k][0])
                # if items_in_this_dimension == 0:
                #     # Empty cluster
                #     log('No items in this cluster! Using entropy 100 for cluster', tabs=3, level=LEVEL.VERBOSE)
                #     sum_entropy = 100  # Todo: fix this with epsilon idea
                #     print('WTF')
                #     break
                # for dim, attr_idx in enumerate(multi_index):
                for dim in range(len(multi_index)):
                    # log('Attribute: ' + str(row[dim]) + ' with index: ' + str(attr_idx) + ' in dimension: ' + str(dim), tabs=3, level=LEVEL.VERBOSE)
                    # category_values = centroids[k][dim]
                    # current_category = category_values[attr_idx]
                    # items_in_this_dimension = sum(category_values)
                    # if items_in_this_dimension == 0:
                    #     # Empty cluster
                    #     log('No items in this cluster! Using entropy 100 for cluster', tabs=3, level=LEVEL.VERBOSE)
                    #     sum_entropy = 100  # Todo: fix this with epsilon idea
                    #     break
                    # attr_probability = current_category / items_in_this_dimension
                    # Applying epsilon idea
                    # if attr_probability == 0:
                    #     attr_probability = 1 / items_in_this_dimension
                    #     log('No item in this category! Using epsilon as 1 / ' + str(items_in_this_dimension) + ' : ' + str(attr_probability), tabs=4, level=LEVEL.VERBOSE)
                    # else:
                    #     log('Probability in this cluster: ' + str(current_category) + ' / ' + str(items_in_this_dimension) + ' = ' + str(attr_probability), tabs=4, level=LEVEL.VERBOSE)
                    # ent = np.log(centroids[k][dim][attr_idx])
                    ent = np.log(centroids[k][dim][multi_index[dim]])
                    # log('Entropy in this cluster: -log(probability) = ' + str(ent), tabs=4, level=LEVEL.VERBOSE)
                    sum_entropy += ent
                sum_entropy *= -1
                # log('Sum entropy for this cluster: ' + str(sum_entropy), tabs=3, level=LEVEL.VERBOSE)
                if least_entropy > sum_entropy:
                    least_entropy = sum_entropy
                    selected_cluster = k

            self.step_calc += time.time() - last_stop
            last_stop = time.time()

            if prev_cluster != selected_cluster:
                log('Moving from cluster ' + str(prev_cluster) + ' to ' + str(selected_cluster), tabs=2, level=LEVEL.VERBOSE)
                if prev_cluster != -1:
                    self.clustering.remove_row_from_cluster(multi_index, prev_cluster)
                self.clustering.assign_row_to_cluster(multi_index, selected_cluster)
                movements += 1
                self.cluster_assignments[index] = selected_cluster
            else:
                log('No movement!', tabs=2, level=LEVEL.VERBOSE)
                pass

            self.step_assign += time.time() - last_stop

        self.overall_time += time.time() - move_in
        return movements
