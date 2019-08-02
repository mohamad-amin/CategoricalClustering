import random


def generate_labels(labels_dict):
    labels = []
    for key in labels_dict.keys():
        labels += [key] * labels_dict[key]
    random.shuffle(labels)
    return labels

# import numpy as np
# import pandas as pd
#
# from categorical_clustering.experiment.label_generator import generate_labels
# from categorical_clustering.core.algorithm.pmf import PMF
# from categorical_clustering.core.algorithm.epsilon_pmf import EpsilonPmf
#
# labels = generate_labels({'A': 20, 'B': 10, 'C': 5, 'D': 15})
# df = pd.DataFrame(data={'Labels': labels})
# model = PMF(n_clusters=3)
# model.set_data(df)


# import numpy as np
# import pandas as pd
#
# from categorical_clustering.experiment.label_generator import generate_labels
# from categorical_clustering.core.algorithm.pmf import PMF
# from categorical_clustering.core.algorithm.epsilon_pmf import EpsilonPmf
#
# L1 = generate_labels({'A': 20, 'B': 10, 'C': 5, 'D': 15})
# L2 = generate_labels({'x': 30, 'y': 10, 'z': 10})
# df = pd.DataFrame(data={'L1': L1, 'L2': L2})
# joint_cols = df['L1'] + df['L2']
# model = PMF(n_clusters=3)
# model.set_data(df)
