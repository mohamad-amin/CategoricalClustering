import numpy as np
import pandas as pd

from categorical_clustering.core.algorithm.separate_attribute_sqaured_distance import SeparateAttributeSquaredDistance as SASD

use_log_file = False
log_file_name = 'mushrooms_sasd_2_10000.log'

dataset_path = '../../data/mushrooms.data'

if use_log_file:
    import sys
    old_stdout = sys.stdout
    log_file = open(log_file_name, "w")
    sys.stdout = log_file

df = pd.read_csv(dataset_path, header=None).iloc[:, 1:]
results = []
for i in range(1):
    print('Iteration {}'.format(i))
    model = SASD(n_clusters=16)
    model.set_data(df)
    # clustering, iters = model.perform_clustering(update_proportion_criterion=0, verbose=False)
    clustering, iters, accepted_iters, _ = model.perform_random_swap_clustering(t=200, update_proportion_criterion=0, verbose=True)
    print('Converged in {} iterations!'.format(iters))
    try:
        print('Overall impurity: {}'.format(clustering.calculate_overall_impurity()))
        results += [(model, clustering.calculate_overall_impurity(), iters)]
    except:
        print('Unexpected results!')
        pass

results = np.array(results)

# print('Time')
# print('################################')
# print('Step init: {}'.format(model.step_init))
# print('Step index: {}'.format(model.step_index))
# print('Step calc: {}'.format(model.step_calc))
# print('Step calc 1: {}'.format(model.step_calc_1))
# print('Step calc 2: {}'.format(model.step_calc_2))
# print('Step calc 3: {}'.format(model.step_calc_3))
# print('Step calc 4: {}'.format(model.step_calc_4))
# print('Step assign: {}'.format(model.step_assign))
# print('Overall time: {}'.format(model.overall_time))
# print('Entropy time: {}'.format(model.clustering.step_ent))
# print('\nStats')
# print('################################')
# print('Shape: {}'.format(results[:, 0].shape))
# print('Mean: {}'.format(results[:, 0].mean()))
# print('Min: {}'.format(results[:, 0].min()))
# print('Max: {}'.format(results[:, 0].max()))
# print('Std: {}'.format(results[:, 0].std()))

if use_log_file:
    sys.stdout = old_stdout
    log_file.close()


