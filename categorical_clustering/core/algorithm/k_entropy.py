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

        for index, row in self.dataset.iterrows():

            prev_cluster = self.cluster_assignments[index]
            multi_index = self._get_multi_dimensional_index(row)
            least_entropy = self.clustering.calculate_overall_impurity()
            selected_cluster = prev_cluster

            log('Data point: ' + ''.join(row) + ' in cluster ' + str(prev_cluster) + ' with impurity: ' + str(least_entropy), tabs=1, level=LEVEL.VERBOSE)

            writable_clusters = self.clustering.get_writable_clusters()
            writable_clusters[prev_cluster][multi_index] -= 1

            for k in range(self.n_clusters):
                if k == prev_cluster:
                    continue
                writable_clusters[k][multi_index] += 1
                self.clustering.update_clusters_from_writable(writable_clusters)
                impurity = self.clustering.calculate_overall_impurity()
                log('Cluster: ' + str(k) + ', impurity: ' + str(impurity), tabs=2, level=LEVEL.VERBOSE)
                writable_clusters[k][multi_index] -= 1
                if impurity < least_entropy:
                    least_entropy = impurity
                    selected_cluster = k

            self.cluster_assignments[index] = selected_cluster
            writable_clusters[selected_cluster][multi_index] += 1
            self.clustering.update_clusters_from_writable(writable_clusters)

            if selected_cluster != prev_cluster:
                log('Moving from cluster ' + str(prev_cluster) + ' to ' + str(selected_cluster), tabs=2, level=LEVEL.VERBOSE)
                movements += 1
            else:
                log('No movement!', tabs=2, level=LEVEL.VERBOSE)

        return movements
