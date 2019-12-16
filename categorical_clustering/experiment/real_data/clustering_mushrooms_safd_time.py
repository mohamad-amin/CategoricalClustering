import numpy as np
import pandas as pd

from categorical_clustering.core.algorithm.separate_attribute_full_distance import SeparateAttributeFullDistance as SAFD

# import sys
# old_stdout = sys.stdout
# log_file = open("message_full.log", "w")
# sys.stdout = log_file

df = pd.read_csv('../../data/votes.data', header=None)
results = []
for i in range(100):
    print('Iteration {}'.format(i))
    model = SAFD(n_clusters=16)
    model.set_data(df)
    clustering, iters = model.perform_random_swap_clustering(t=1000, update_proportion_criterion=0, verbose=True)
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
print('Step calc 1: {}'.format(model.step_calc_1))
print('Step calc 2: {}'.format(model.step_calc_2))
print('Step calc 3: {}'.format(model.step_calc_3))
print('Step calc 4: {}'.format(model.step_calc_4))
print('Step assign: {}'.format(model.step_assign))
print('Overall time: {}'.format(model.overall_time))
print('Entropy time: {}'.format(model.clustering.step_ent))

# print('Shape: {}'.format(results[:, 0].shape))
# print('Mean: {}'.format(results[:, 0].mean()))
# print('Min: {}'.format(results[:, 0].min()))
# print('Max: {}'.format(results[:, 0].max()))
# print('Std: {}'.format(results[:, 0].std()))

# sys.stdout = old_stdout
# log_file.close()

