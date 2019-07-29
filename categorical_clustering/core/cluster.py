import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

from functools import reduce
from categorical_clustering.utils.logger import log

ALPHA = 1


class Clustering:

    def __init__(self, n_clusters, values, data_count):  # Values are sorted, the sort order must hold always
        self.values = values
        self.data_count = data_count
        self.n_clusters = n_clusters
        self.clusters = np.zeros((n_clusters, len(values)))

    def calculate_impurity(self):
        impurity = np.apply_along_axis(lambda x: entropy(x) * (sum(x) / self.data_count), 1, self.clusters).sum()
        mi_score = 0.0
        for i in range(self.n_clusters):
            for j in range(i+1, self.n_clusters):
                mi_score += mutual_info_score(self.clusters[i], self.clusters[j])
        impurity += mi_score * ALPHA
        return impurity

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
