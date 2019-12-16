# KM-Epsilon
Python implementation of KM-Epsilon: Optimal categorical clustering based on information-theoretic objective function and Random Swap. 
More information about the algorithm are avaiable in the [slides](https://slides.com/mohamadaminmohamadi/categorical_clustering/#/).
Also, some other categorical clustering algorithms (like K-Entropies) have been implemented in the `categorical_clustering.core.algorithm` package
for test purposes and comparison of different algorithms.

The code is implemented using `Numpy`, `Scipy` and `Pandas`. I have also used `Numba` for parallelism and optimized machine code. 
According to the experiments, the code takes about ~0.5 second to run each iteration of KM-Epsilon on [Mushrooms](https://archive.ics.uci.edu/ml/datasets/Mushroom) dataset.

## Usage
Available strategies for convergence are:
* Separate attribute probability distance (`SeparateAttributePMF`)
* Separate attribute category distance (`SeparateAttributeCategoryPMF`)
* Separate attribute squared distance (`SeparateAttributeSquaredDistance`)

Parameters used are described down below in the corresponding table.

### Sample code for using separate attribute squared distance strategy without random swap:
```python
import numpy as np
import pandas as pd
from categorical_clustering.core.algorithm.separate_attribute_sqaured_distance import SeparateAttributeSquaredDistance as SASD

dataset_path = '../../data/mushrooms.data'
df = pd.read_csv(dataset_path, header=None).iloc[:, 1:]

model = SASD(n_clusters=16)
model.set_data(df)
clustering, iterations = model.perform_clustering(update_proportion_criterion=0, verbose=True)

print('Overall impurity: {}'.format(clustering.calculate_overall_impurity()))
```
### Sample code for using separate attribute squared distance strategy with random swap:
```python
import numpy as np
import pandas as pd
from categorical_clustering.core.algorithm.separate_attribute_sqaured_distance import SeparateAttributeSquaredDistance as SASD

dataset_path = '../../data/mushrooms.data'
df = pd.read_csv(dataset_path, header=None).iloc[:, 1:]

model = SASD(n_clusters=16)
model.set_data(df)
clustering, iters, accepted_iters, labels = model.perform_random_swap_clustering(
                                                     t=200, update_proportion_criterion=0, verbose=True)

print('Overall impurity: {}'.format(clustering.calculate_overall_impurity()))
```

Cluster number of each input datapoint can be accessed using `model.cluster_assignments` array.

**Note:** The code is still under development and has not been intesively tested, so there is no guarentee of using it in production. But so far, no problem has been reported, so there also should be no specific problem.

## Visualization
Here is the sample visualization of clustering using 16 clusters on Muhsrooms dataset and 1000 random swaps (SASD convergence strategy) attaining 6.95 overall impurity value.
![TSNE clustering visualization](https://github.com/mohamad-amin/KM-Epsilon/blob/separate-category-clustering/media/mushroom_tsne.png)


## Empirical evaluations

### Overall internal evaluations
![Internal evaluations](https://github.com/mohamad-amin/KM-Epsilon/blob/separate-category-clustering/media/internal_evaluation.png)

### Mushroom samples

Results of running `SASD` on Mushrooms dataset without random swap (100 times)

|          Statistic         	| Value 	|
|:--------------------------:	|:-----:	|
|      Average impurity      	|  7.13 	|
|   Best (lowest) impurity   	|  6.98 	|
|  Worst (highest) impurity  	|  7.90 	|
| Standard error of impurity 	|  0.13 	|

Results of running `SASD` on Mushrooms dataset with random swap (100 times)

|          Statistic         	| Value 	|
|:--------------------------:	|:-----:	|
|      Average impurity      	|  6.99 	|
|   Best (lowest) impurity   	|  6.95 	|
|  Worst (highest) impurity  	|  7.07 	|
| Standard error of impurity 	|  0.02 	|

## Documents
I have not managed to put any documents website up yet. It will be a part of this project in near future and pull requests are wellcome. 
But so far, you can refer to the dosctings (available for most crucial functions) to understand the behaviour, although they seem to be 
pretty straightforward based on their name conventions.

## Credits:
* Mohamad Amin Mohamadi (mohammadi.mohamadamin@gmail.com)
* Prof. Pasi Franti (franti sign_at cs dot uef dot fi)

