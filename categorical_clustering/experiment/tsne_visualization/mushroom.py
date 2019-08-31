import numpy as np
import pandas as pd
import pickle as pkl

tsne_result = pkl.load(open('categorical_clustering/data/mushroom_tsne.pkl', 'rb'))
tsne_df = pd.DataFrame(tsne_result, columns=['One', 'Two'])
df = pd.read_csv('categorical_clustering/data/mushrooms.data', header=None)
sub_clustering, sub_assignment = pkl.load(open('/Users/hezardastan/results_5000.pkl', 'rb'))
best_clustering, best_assignment = pkl.load(open('/Users/hezardastan/results_701.pkl', 'rb'))

import plotly.express as px
fig = px.scatter(tsne_df, x="One", y="Two")
