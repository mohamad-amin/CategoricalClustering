import time
from ...utils.logger import log, LEVEL
from .base import BaseIterativeClustering


class KEntropy(BaseIterativeClustering):

    def _random_swap(self):
        raise NotImplementedError()

    def _calculate_centroids(self) -> list:
        return None

    def _assign_data_to_clusters(self, centroids: list) -> int:

        log('Assigning items to clusters...', level=LEVEL.VERBOSE)

        movements = 0
        entropies = [self.clustering.calculate_cluster_impurity(k) for k in range(self.n_clusters)]

        for index in self.dataset.index.values:

            last_stop = time.time()
            prev_cluster = self.cluster_assignments[index]
            most_entropy_decrease = 0
            selected_cluster = prev_cluster
            self.step_init += time.time() - last_stop
            last_stop = time.time()

            # log('Data point: ' + ''.join(row) + ' in cluster ' + str(prev_cluster) + ' with impurity: ' + str(least_entropy), tabs=1, level=LEVEL.VERBOSE)

            multi_index = self._get_multi_dimensional_index(self.dataset.values[index, :])
            self.clustering.remove_row_from_cluster(multi_index, prev_cluster)
            self.step_index += time.time() - last_stop
            last_stop = time.time()

            for k in range(self.n_clusters):
                if k == prev_cluster:
                    continue
                old_entropies = entropies[prev_cluster] + entropies[k]
                self.clustering.assign_row_to_cluster(multi_index, k)
                self.step_calc_1 += time.time() - last_stop
                last_stop = time.time()
                new_entropies = self.clustering.calculate_cluster_impurity(prev_cluster) + \
                                self.clustering.calculate_cluster_impurity(k)
                if new_entropies - old_entropies < most_entropy_decrease:
                    most_entropy_decrease = new_entropies - old_entropies
                    selected_cluster = k
                self.step_calc_2 += time.time() - last_stop
                last_stop = time.time()
                self.clustering.remove_row_from_cluster(multi_index, k)

            self.cluster_assignments[index] = selected_cluster
            self.clustering.assign_row_to_cluster(multi_index, selected_cluster)

            if selected_cluster != prev_cluster:
                # log('Moving from cluster ' + str(prev_cluster) + ' to ' + str(selected_cluster), tabs=2, level=LEVEL.VERBOSE)
                movements += 1
                entropies[prev_cluster] = self.clustering.calculate_cluster_impurity(prev_cluster)
                entropies[selected_cluster] = self.clustering.calculate_cluster_impurity(selected_cluster)
            # else:
            #     log('No movement!', tabs=2, level=LEVEL.VERBOSE)

            self.step_assign += time.time() - last_stop

        return movements
