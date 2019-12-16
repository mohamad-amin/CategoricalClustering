import pandas as pd
import pickle as pkl
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold.t_sne import TSNE

df = pd.read_csv('categorical_clustering/data/mushrooms.data', header=None).iloc[:, 1:]
df = df.apply(LabelEncoder().fit_transform)

model = TSNE(n_components=2, verbose=1, perplexity=30, metric='manhattan')
tsne_results = model.fit_transform(df)

with open('categorical_clustering/data/mushroom_tsne_reduction.pkl', 'wb') as f:
    pkl.dump(tsne_results, f)
