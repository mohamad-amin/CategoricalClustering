import pandas as pd
from categorical_clustering.core.algorithm.separate_attribute_category_pmf import SeparateAttributeCategoryPMF as SACPMF

df = pd.read_csv('../../data/mushrooms.data', header=None)
model = SACPMF(n_clusters=16)
model.set_data(df)
clustering, iters = model.perform_random_swap_clustering(t=100, verbose=True)
print('Converged in {} iterations!'.format(iters))
print('Overall impurity: {}'.format(clustering.calculate_overall_impurity()))

print('Step init: {}'.format(model.step_init))
print('Step index: {}'.format(model.step_index))
print('Step calc: {}'.format(model.step_calc))
print('Step assign: {}'.format(model.step_assign))
print('Overall time: {}'.format(model.overall_time))
