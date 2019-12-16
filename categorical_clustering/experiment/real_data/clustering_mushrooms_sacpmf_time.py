import numpy as np
import pandas as pd

from categorical_clustering.core.algorithm.separate_attribute_category_pmf import SeparateAttributeCategoryPMF as SACPMF

use_log_file = False
log_file_name = 'mushrooms_sacpmf_2_10000.log'

dataset_path = 'categorical_clustering/data/mushrooms.data'

if use_log_file:
    import sys
    old_stdout = sys.stdout
    log_file = open(log_file_name, "w")
    sys.stdout = log_file

# labels = []

df = pd.read_csv(dataset_path, header=None).iloc[:, 1:]
results = []
for i in range(10):
    print('Iteration {}'.format(i))
    model = SACPMF(n_clusters=16)
    model.set_data(df)
    clustering, iters, accepted_iters, _ = model.perform_random_swap_clustering(t=100, update_proportion_criterion=0, verbose=True)
    # clustering, iters = model.perform_clustering()
    # if clustering.calculate_overall_impurity() < 7.016:
    #     labels.append(model.cluster_assignments)
    print('Converged in {} iterations!'.format(iters))
    try:
        print('Overall impurity: {}'.format(clustering.calculate_overall_impurity()))
        results += [(model, clustering.calculate_overall_impurity(), iters)]
    except:
        print('Unexpected results!')
        pass

results = np.array(results)

print('Time')
print('################################')
print('Step init: {}'.format(model.step_init))
print('Step index: {}'.format(model.step_index))
print('Step calc: {}'.format(model.step_calc))
print('Step calc 1: {}'.format(model.step_calc_1))
print('Step calc 2: {}'.format(model.step_calc_2))
print('Step calc 3: {}'.format(model.step_calc_3))
print('Step calc 4: {}'.format(model.step_calc_4))
print('Step assign: {}'.format(model.step_assign))
print('Overall time: {}'.format(model.overall_time))
print('Entropy time: {}'.format(model.clustering.step_ent))
print('\nStats')
print('################################')
print('Shape: {}'.format(results[:, 0].shape))
print('Mean: {}'.format(results[:, 0].mean()))
print('Min: {}'.format(results[:, 0].min()))
print('Max: {}'.format(results[:, 0].max()))
print('Std: {}'.format(results[:, 0].std()))

if use_log_file:
    sys.stdout = old_stdout
    log_file.close()


