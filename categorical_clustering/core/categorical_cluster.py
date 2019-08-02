import numpy as np

from categorical_clustering.utils.logger import log
from categorical_clustering.core.data.clustering import Clustering


def cluster(labels, n_clusters):

    values, counts = np.unique(labels, return_counts=True)
    count_sort_ind = np.argsort(-counts)
    values = values[count_sort_ind]
    counts = counts[count_sort_ind]

    clustering = Clustering(n_clusters, values, sum(counts))

    # Initialization of clusters with large groups
    for i in range(n_clusters):
        clustering.clusters[i][i] = counts[i]

    while sum(counts[n_clusters:]) > 0:
        selections = np.where(counts[n_clusters:] != 0)[0]
        selection = np.random.choice(selections) + n_clusters
        log('Selection for next item to add to clusters: ' + values[selection])
        conditional_impurities = [0.0] * n_clusters
        for c in range(n_clusters):
            conditional_impurities[c] = clustering.calculate_conditional_impurity(selection, c)
            log('Impurity of adding ' + str(values[selection]) + ' to cluster ' + str(c) + ' is: ' + str(conditional_impurities[c]))
        chosen_cluster = np.argmin(conditional_impurities)
        log('Chosen cluster for ' + str(values[selection]) + ' is: ' + str(chosen_cluster))
        clustering.clusters[chosen_cluster][selection] += 1
        log('Clusters now:\n' + clustering.clusters_to_string() + '\n')
        counts[selection] -= 1

    return clustering

# from categorical_clustering.experiment.label_generator import generate_labels
# from categorical_clustering.core.categorical_cluster import cluster
# labels = generate_labels({'A': 10, 'B': 10, 'C': 10, 'D': 10})
# n_clusters = 2
#
# clustering = cluster(labels, n_clusters)
