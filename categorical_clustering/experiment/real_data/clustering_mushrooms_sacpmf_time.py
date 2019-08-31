import numpy as np
import pandas as pd

from categorical_clustering.core.algorithm.separate_attribute_category_pmf import SeparateAttributeCategoryPMF as SACPMF

# import sys
# old_stdout = sys.stdout
# log_file = open("message_votes.log", "w")
# sys.stdout = log_file

df = pd.read_csv('../../data/mushrooms.data', header=None)
results = []
for i in range(1):
    print('Iteration {}'.format(i))
    model = SACPMF(n_clusters=16)
    model.set_data(df)
    clustering, iters = model.perform_clustering(update_proportion_criterion=0, verbose=True)
    print('Converged in {} iterations!'.format(iters))
    try:
        print('Overall impurity: {}'.format(clustering.calculate_overall_impurity()))
        results += [(model, clustering.calculate_overall_impurity(), iters)]
    except:
        print('Unexpected results!')
        pass
results = np.array(results)

print('Step init: {}'.format(model.step_init))
print('Step index: {}'.format(model.step_index))
print('Step calc: {}'.format(model.step_calc))
print('Step assign: {}'.format(model.step_assign))
print('Overall time: {}'.format(model.overall_time))

print('Shape: {}'.format(results[:, 0].shape))
print('Mean: {}'.format(results[:, 0].mean()))
print('Min: {}'.format(results[:, 0].min()))
print('Max: {}'.format(results[:, 0].max()))
print('Std: {}'.format(results[:, 0].std()))

# sys.stdout = old_stdout
# log_file.close()

