import pandas as pd
from categorical_clustering.core.algorithm.k_entropy import KEntropy

df = pd.read_csv('../../data/mushrooms.data', header=None)
model = KEntropy(n_clusters=16)
model.set_data(df)
clustering, iters = model.perform_clustering(verbose=True)
print('Converged in {} iterations!'.format(iters))
print('Overall impurity: {}'.format(clustering.calculate_overall_impurity()))

print('Step init: {}'.format(model.step_init))
print('Step index: {}'.format(model.step_index))
print('Step calc: {}'.format(model.step_calc))
print('Step calc 1: {}'.format(model.step_calc_1))
print('Step calc 2: {}'.format(model.step_calc_2))
print('Step assign: {}'.format(model.step_assign))
print('Overall time: {}'.format(model.overall_time))
