import numpy as np
import pandas as pd
from scipy.stats import entropy

from categorical_clustering.utils.logger import log

ALPHA = 1


class Clustering:

    def __init__(self, n_clusters, values, data_count):  # Values are sorted, the sort order must hold always
        self.values = values
        self.data_count = data_count
        self.n_clusters = n_clusters
        self.clusters = np.zeros((n_clusters, len(values)))

    def calculate_impurity(self):
        return np.apply_along_axis(lambda x: entropy(x) * sum(x), 1, self.clusters).sum()

    def calculate_conditional_impurity(self, addition_element, cluster):
        self.clusters[cluster][addition_element] += 1
        log('The cluster will be:\n' + self.clusters_to_string())
        impurity = self.calculate_impurity()
        self.clusters[cluster][addition_element] -= 1
        log('The cluster was:\n' + self.clusters_to_string())
        return impurity

    def clusters_to_string(self):
        return pd.DataFrame(
            data=self.clusters,
            index=['Cluster ' + str(i) for i in range(self.n_clusters)],
            columns=self.values
        ).to_string()
