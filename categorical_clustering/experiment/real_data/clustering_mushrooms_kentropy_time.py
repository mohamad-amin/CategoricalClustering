import numpy as np
import pandas as pd
from categorical_clustering.core.algorithm.k_entropy import KEntropy

import sys
old_stdout = sys.stdout
# log_file = open("mushrooms_kentropy_100.log", "w")
# sys.stdout = log_file


df = pd.read_csv('../../data/soybean-small.data', header=None).iloc[:, :-1]
model = KEntropy(n_clusters=4)
model.set_data(df)
results = []
for i in range(100):
    print('Iteration {}'.format(i))
    clustering, iters = model.perform_clustering(verbose=True)
    print('Converged in {} iterations!'.format(iters))
    print('Overall impurity: {}'.format(clustering.calculate_overall_impurity()))
    results += [(clustering.calculate_overall_impurity(), iters)]
results = np.array(results)

print('Step init: {}'.format(model.step_init))
print('Step index: {}'.format(model.step_index))
print('Step calc: {}'.format(model.step_calc))
print('Step calc 1: {}'.format(model.step_calc_1))
print('Step calc 2: {}'.format(model.step_calc_2))
print('Step assign: {}'.format(model.step_assign))
print('Overall time: {}'.format(model.overall_time))

print('Shape: {}'.format(results[:, 0].shape))
print('Mean: {}'.format(results[:, 0].mean()))
print('Min: {}'.format(results[:, 0].min()))
print('Max: {}'.format(results[:, 0].max()))
print('Std: {}'.format(results[:, 0].std()))

# sys.stdout = old_stdout
# log_file.close()
