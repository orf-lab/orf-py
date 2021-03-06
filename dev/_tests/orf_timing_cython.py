"""
Simulated Example Dataset to test orf
"""

import numpy as np
import pandas as pd
import os
import time
from econml.grf import RegressionForest
import timeit

path = "D:/switchdrive/Projects/ORF_Python/ORFpy"
# path = "/home/okasag/Documents/HSG/ORF/python/ORFpy"
# path = "/Users/okasag/Desktop/HSG/orf/python/ORFpy"
os.chdir(path)

# load the ordered forest
from orf.orf import OrderedForest


# Define function to produce data sets of different size
def example_data(seed, n, p_cont, p_cat, p_binary, noise=True, y_cat=3,
                 cat_cat=3):
    """
    Generate example data to test orf

    Parameters
    ----------
    seed : TYPE: integer
        DESCRIPTION: Set seed for reproducability.
    n : TYPE: integer
        DESCRIPTION: The number of observations.
    p_cont : TYPE: integer
        DESCRIPTION: The number of continuous covariates.
    p_cat : TYPE: integer
        DESCRIPTION: The number of categorical covariates.
    p_binary : TYPE: integer
        DESCRIPTION. The number of binary covariates.
    noise : TYPE_ boolean, optional
        DESCRIPTION. Whether to include a continuous noise variable. The
        default is True.
    y_cat : TYPE: integer, optional
        DESCRIPTION. The number of categories of the outcome variable. The
        default is 3.
    cat_cat : TYPE: integer, optional
        DESCRIPTION. The number of categories of the categorical variable. The
        default is 3.

    Returns
    -------
    Example dataset.

    """
    # Set seed
    seed = np.random.default_rng(seed=seed)
    # Draw covariates
    cont = seed.normal(0, 1, size=(n, p_cont))
    cat = seed.binomial(cat_cat-1, 0.5, size=(n, p_cat))+1
    binary = seed.binomial(1, 0.5, size=(n, p_binary))
    # Combine deterministic covariates
    X_det = np.hstack([cont, cat, binary])
    if noise:
        X = np.hstack([X_det, seed.normal(0, 1, size=(n, 1))])
    else:
        X = X_det
    # Generate continuous outcome with logistic error
    Y = np.sum(X_det, axis=1) + seed.logistic(0, 1, n)
    # Thresholds for continuous outcome
    Y = pd.qcut(Y, y_cat, labels=False)+1
    # Return X and Y
    return pd.DataFrame(X), pd.Series(Y)


sample_sizes = [4000]
time_table = {}

for n_sample in sample_sizes:
    # Generate data set
    features, outcome = example_data(seed=123, n=n_sample, p_cont=1, p_cat=1,
                                     p_binary=1, noise=True, y_cat=3,
                                     cat_cat=3)

    # 0.) How fast is fitting two forests in econML?
    forest = RegressionForest(n_estimators=1000, min_samples_leaf=5,
                              max_features=0.5, max_samples=0.5, honest=False)
    test_0 = %timeit -o forest.fit(X=features.iloc[:np.ceil(n_sample/2).astype(int), :], y=outcome.iloc[:np.ceil(n_sample/2).astype(int)])

    # 1.)
    oforest1 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                             max_features=0.5, replace=False,
                             sample_fraction=0.5, honesty=True,
                             n_jobs=1, pred_method='numpy')
    test_1 = %timeit -o oforest1.fit(X=features, y=outcome)

    # 2.)
    oforest2 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                             max_features=0.5, replace=False,
                             sample_fraction=0.5, honesty=True,
                             n_jobs=1, pred_method='numpy_loop')
    test_2 = %timeit -o oforest2.fit(X=features, y=outcome)
    
    # 3.)
    oforest3 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                             max_features=0.5, replace=False,
                             sample_fraction=0.5, honesty=True,
                             n_jobs=4, pred_method='numpy_loop_mpire')
    test_3 = %timeit -o oforest3.fit(X=features, y=outcome)

    # 4.)
    oforest4 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                             max_features=0.5, replace=False,
                             sample_fraction=0.5, honesty=True,
                             n_jobs=1, pred_method='numpy_sparse')
    test_4 = %timeit -o oforest4.fit(X=features, y=outcome)

    # 5.)
    oforest5 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                             max_features=0.5, replace=False,
                             sample_fraction=0.5, honesty=True,
                             n_jobs=1, pred_method='loop')
    test_5 = %timeit -o oforest5.fit(X=features, y=outcome)

    # 6.)
    oforest6 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                             max_features=0.5, replace=False,
                             sample_fraction=0.5, honesty=True,
                             n_jobs=4, pred_method='loop_multi')
    test_6 = %timeit -o oforest6.fit(X=features, y=outcome)

    # 7.)
# =============================================================================
#     oforest7 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
#                              max_features=0.5, replace=False,
#                              sample_fraction=0.5, honesty=True,
#                              n_jobs=4, pred_method='cython')
#     test_7 = %timeit -o oforest7.fit(X=features, y=outcome)
# =============================================================================

    time_table[n_sample] = [test_0.average*2, test_1.average, test_2.average,
                            test_3.average, test_4.average, test_5.average,
                            test_6.average]

pd.DataFrame(time_table, index=['2 x econML', 'numpy', 'numpy_loop',
                                'numpy_loop_mpire', 'numpy_sparse', 'loop',
                                'loop_multi']).T


"""
n      2 x econML      numpy       loop    loop par     cython  cython par
1250     1.634134   2.538868   3.535324    4.342317   2.494032    2.735010
2500     2.726762   3.085457   5.426605    6.776736   3.077705    3.364190
5000     4.038665   5.720619   8.716242   14.779802   6.601798    6.017505
10000    7.989531  17.709245  19.591807   73.085157  43.520433   36.396375
20000   56.749016  97.639873  60.991867  108.089989  62.723277   53.061945
40000   58.343502        NaN 153.837272  257.034573  192.14841  202.482050
80000   ...takes forever..................................................


      2 x econML      numpy   numpy_loop  numpy_sparse      loop   cython
1000    2.789039   3.385252     4.984847      4.903206  5.007984  4.07698
20000   8.752583  55.862967    29.670385     33.555659  58.27305  59.338289
100000 91.609554              149.250421    169.820621

      2 x econML  numpy_loop  numpy_sparse  numpy_sparse2
1000    1.503623    3.305515      2.658488      2.7687940
20000   7.128304   19.515624     21.770787      20.517271
100000 58.246735  110.120471    135.083053     128.428011
"""

""" on Mac with 11 cores
       2 x econML      numpy  numpy_loop  numpy_sparse       loop  loop_multi_1core  loop_multi_3cores
1000     1.677145   2.168326    3.348213      0.092924   3.295068          5.455551           9.259089
4000     2.273600   4.948216    4.471807      0.249727  16.611614         10.421250          11.758524
16000    3.514005  43.179584    7.439444      0.855642  65.371401         39.043368          47.348264
"""

""" on Mac with 4 cores
       2 x econML      numpy  numpy_loop  numpy_sparse       loop  loop_multi_1core  loop_multi_3cores
1000     1.800045   2.252607    3.225075      0.093093   3.285477          5.546436           5.727344
4000     2.160294   4.980013    4.265472      0.247057   7.995749         10.241013           8.059144
16000    3.448132  43.169730    8.746617      0.833426  32.396961         38.083952          26.027622
"""

""" on Linux with 4 cores
      2 x econML     numpy  numpy_loop  numpy_loop_multi  numpy_sparse       loop  loop_multi
1000    2.578914  3.942329    5.790034          4.898176      4.580967   6.429342    5.172048
4000    3.118779  8.704739    7.855401          7.815587      8.511379  27.203318   11.680522
"""

sample_sizes = [1000]
time_table = {}

for n_sample in sample_sizes:
    # Generate data set
    features, outcome = example_data(seed=123, n=n_sample, p_cont=1, p_cat=1,
                                     p_binary=1, noise=True, y_cat=3,
                                     cat_cat=3)

    # 0.) How fast is fitting two forests in econML?
    forest = RegressionForest(n_estimators=1000, min_samples_leaf=5,
                              max_features=0.5, max_samples=0.5, honest=False)
    test_0 = %timeit -o forest.fit(X=features.iloc[:np.ceil(n_sample/2).astype(int), :], y=outcome.iloc[:np.ceil(n_sample/2).astype(int)])

    """# 1.)
    oforest1 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                             max_features=0.5, replace=False,
                             sample_fraction=0.5, honesty=True,
                             n_jobs=1, pred_method='numpy')
    test_1 = %timeit -o oforest1.fit(X=features, y=outcome)"""

    # 2.)
    oforest2 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                             max_features=0.5, replace=False,
                             sample_fraction=0.5, honesty=True,
                             n_jobs=1, pred_method='numpy_loop')
    test_2 = %timeit -o oforest2.fit(X=features, y=outcome)

    # 3.)
    oforest3 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                             max_features=0.5, replace=False,
                             sample_fraction=0.5, honesty=True,
                             n_jobs=1, pred_method='numpy_sparse')
    test_3 = %timeit -o oforest3.fit(X=features, y=outcome)

    # 4.)
    oforest4 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                             max_features=0.5, replace=False,
                             sample_fraction=0.5, honesty=True,
                             n_jobs=1, pred_method='numpy_sparse2')
    test_4 = %timeit -o oforest4.fit(X=features, y=outcome)

    """# 5.)
    oforest5 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                             max_features=0.5, replace=False,
                             sample_fraction=0.5, honesty=True,
                             n_jobs=1, pred_method='cython')
    test_5 = %timeit -o oforest5.fit(X=features, y=outcome)

    time_table[n_sample] = [test_0.average*2, test_1.average, test_2.average,
                            test_3.average, test_4.average, test_5.average]"""
    time_table[n_sample] = [test_0.average*2, test_2.average,
                            test_3.average, test_4.average]

pd.DataFrame(time_table, index=['2 x econML', 'numpy_loop',
                                'numpy_sparse', 'numpy_sparse2']).T


# Test inference true vs false
features, outcome = example_data(seed=123, n=1000, p_cont=1, p_cat=1,
                                     p_binary=1, noise=True, y_cat=3,
                                     cat_cat=3)
oforest2 = OrderedForest(n_estimators=500, min_samples_leaf=5,
                         max_features=0.5, replace=False,
                         sample_fraction=0.5, honesty=True,
                         n_jobs=1, pred_method='numpy_loop',
                         inference=True, random_state=123)
test_2 = %timeit -o oforest2.fit(X=features, y=outcome)


oforest3 = OrderedForest(n_estimators=500, min_samples_leaf=5,
                         max_features=0.5, replace=False,
                         sample_fraction=0.5, honesty=True,
                         n_jobs=1, pred_method='numpy_loop',
                         inference=False, random_state=123)
test_3 = %timeit -o oforest3.fit(X=features, y=outcome)

"""
      inference=True      inferenc=False
1000    7.3019249857       1.57399261428
10000
"""

# test weights with and without mpire or multiprocessing
sample_sizes = [500]
time_table = {}

for n_sample in sample_sizes:
    # Generate data set
    features, outcome = example_data(seed=123, n=n_sample, p_cont=1, p_cat=1,
                                     p_binary=1, noise=True, y_cat=3,
                                     cat_cat=3)

    oforest_loop = OrderedForest(n_estimators=500, min_samples_leaf=5,
                                 max_features=0.5, replace=False,
                                 sample_fraction=0.5, honesty=True,
                                 n_jobs=1, pred_method='numpy_loop',
                                 weight_method='numpy_loop',
                                 inference=True, random_state=123)
    test_loop = %timeit -o oforest_loop.fit(X=features, y=outcome)


    oforest_mpire = OrderedForest(n_estimators=500, min_samples_leaf=5,
                                 max_features=0.5, replace=False,
                                 sample_fraction=0.5, honesty=True,
                                 n_jobs=4, pred_method='numpy_loop',
                                 weight_method='numpy_loop_mpire',
                                 inference=True, random_state=123)
    test_mpire = %timeit -o oforest_mpire.fit(X=features, y=outcome)


    oforest_shared_multi = OrderedForest(n_estimators=500, min_samples_leaf=5,
                                max_features=0.5, replace=False,
                                sample_fraction=0.5, honesty=True,
                                n_jobs=4, pred_method='numpy_loop',
                                weight_method='numpy_loop_shared_multi',
                                inference=True, random_state=123)
    test_shared_multi = %timeit -o oforest_shared_multi.fit(X=features, y=outcome)


    oforest_shared_mpire = OrderedForest(n_estimators=500, min_samples_leaf=5,
                                 max_features=0.5, replace=False,
                                 sample_fraction=0.5, honesty=True,
                                 n_jobs=4, pred_method='numpy_loop',
                                 weight_method='numpy_loop_shared_mpire',
                                 inference=True, random_state=123)
    test_shared_mpire = %timeit -o oforest_shared_mpire.fit(X=features, y=outcome)

    time_table[n_sample] = [test_loop.average, test_mpire.average,
                            test_shared_multi.average, test_shared_mpire.average]

pd.DataFrame(time_table, index=['numpy_loop', 'numpy_loop_mpire',
                                'numpy_loop_shared_multi', 'numpy_loop_shared_mpire']).T



# =============================================================================
# Tests on windows
# =============================================================================

# test prediction without inference with and without mpire

# compare numpy_loop with n_jobs=1 vs. n_jobs=-1 (= joblib version)

sample_sizes = [1000, 5000, 100000]
time_table = {}

for n_sample in sample_sizes:
    # Generate data set
    features, outcome = example_data(seed=123, n=n_sample, p_cont=1, p_cat=1,
                                     p_binary=1, noise=True, y_cat=3,
                                     cat_cat=3)

    oforest_loop_p1 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                                 max_features=0.5, replace=False,
                                 sample_fraction=0.5, honesty=True,
                                 n_jobs=1, pred_method='numpy_loop',
                                 weight_method='numpy_loop',
                                 inference=False, random_state=123)
    test_loop_p1 = %timeit -r3 -o oforest_loop_p1.fit(X=features, y=outcome)
    
    oforest_loop_m1 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                                 max_features=0.5, replace=False,
                                 sample_fraction=0.5, honesty=True,
                                 n_jobs=-1, pred_method='numpy_loop',
                                 weight_method='numpy_loop',
                                 inference=False, random_state=123)
    test_loop_m1 = %timeit -r3 -o oforest_loop_m1.fit(X=features, y=outcome)


    time_table[n_sample] = [test_loop_p1.average, test_loop_m1.average]

pd.DataFrame(time_table, index=['numpy_loop_p1', 'numpy_loop_m1']).T

# =============================================================================
#        numpy_loop_p1   numpy_loop_m1
# 1000        5.784630        7.271092
# 5000       10.728362       12.359371
# 100000     234.585029     284.790578
# =============================================================================



# test leafmean predictions (without weights) with and without mpire
sample_sizes = [1000, 5000, 10000]
time_table = {}

for n_sample in sample_sizes:
    # Generate data set
    features, outcome = example_data(seed=123, n=n_sample, p_cont=1, p_cat=1,
                                     p_binary=1, noise=True, y_cat=3,
                                     cat_cat=3)

    oforest_loop = OrderedForest(n_estimators=500, min_samples_leaf=5,
                                 max_features=0.5, replace=False,
                                 sample_fraction=0.5, honesty=True,
                                 n_jobs=1, pred_method='numpy_loop',
                                 weight_method='numpy_loop',
                                 inference=False, random_state=123)
    test_loop = %timeit -o oforest_loop.fit(X=features, y=outcome)


    oforest_mpire = OrderedForest(n_estimators=500, min_samples_leaf=5,
                                 max_features=0.5, replace=False,
                                 sample_fraction=0.5, honesty=True,
                                 n_jobs=-1, pred_method='numpy_loop_mpire',
                                 weight_method='numpy_loop_mpire',
                                 inference=False, random_state=123)
    test_mpire = %timeit -o oforest_mpire.fit(X=features, y=outcome)


    time_table[n_sample] = [test_loop.average, test_mpire.average]

pd.DataFrame(time_table, index=['numpy_loop', 'numpy_loop_mpire']).T

# =============================================================================
#        numpy_loop  numpy_loop_mpire
# 1000     2.623892         26.517618
# 5000     3.596580         26.616825
# 10000    6.447010         35.683605
# =============================================================================



# test prediction with inference with and without mpire
sample_sizes = [1000, 5000, 10000]
time_table = {}

for n_sample in sample_sizes:
    # Generate data set
    features, outcome = example_data(seed=123, n=n_sample, p_cont=1, p_cat=1,
                                     p_binary=1, noise=True, y_cat=3,
                                     cat_cat=3)

    oforest_loop = OrderedForest(n_estimators=500, min_samples_leaf=5,
                                 max_features=0.5, replace=False,
                                 sample_fraction=0.5, honesty=True,
                                 n_jobs=1, pred_method='numpy_loop',
                                 weight_method='numpy_loop',
                                 inference=True, random_state=123)
    test_loop = %timeit -o oforest_loop.fit(X=features, y=outcome)


    oforest_mpire = OrderedForest(n_estimators=500, min_samples_leaf=5,
                                 max_features=0.5, replace=False,
                                 sample_fraction=0.5, honesty=True,
                                 n_jobs=-1, pred_method='numpy_loop',
                                 weight_method='numpy_loop_mpire',
                                 inference=True, random_state=123)
    test_mpire = %timeit -o oforest_mpire.fit(X=features, y=outcome)


    oforest_shared_mpire = OrderedForest(n_estimators=500, min_samples_leaf=5,
                                 max_features=0.5, replace=False,
                                 sample_fraction=0.5, honesty=True,
                                 n_jobs=-1, pred_method='numpy_loop',
                                 weight_method='numpy_loop_shared_mpire',
                                 inference=True, random_state=123)
    test_shared_mpire = %timeit -o oforest_shared_mpire.fit(X=features, y=outcome)

    time_table[n_sample] = [test_loop.average, test_mpire.average,
                            test_shared_multi.average, test_shared_mpire.average]

pd.DataFrame(time_table, index=['numpy_loop', 'numpy_loop_mpire',
                                'numpy_loop_shared_multi', 'numpy_loop_shared_mpire']).T

# =============================================================================
# Doesn't stop...
# =============================================================================



# test leafmean predictions (without weights) for different numbers of trees
ntrees = [100, 500, 1000, 2000, 5000, 10000]
time_table = {}

for ntree in ntrees:
    # Generate data set
    features, outcome = example_data(seed=123, n=10000, p_cont=1, p_cat=1,
                                     p_binary=1, noise=True, y_cat=3,
                                     cat_cat=3)

    oforest_loop_p1 = OrderedForest(n_estimators=ntree, min_samples_leaf=5,
                                 max_features=0.5, replace=False,
                                 sample_fraction=0.5, honesty=True,
                                 n_jobs=1, pred_method='numpy_loop',
                                 weight_method='numpy_loop',
                                 inference=False, random_state=123)
    test_loop_p1 = %timeit -r3 -o oforest_loop_p1.fit(X=features, y=outcome)
    
    oforest_loop_m1 = OrderedForest(n_estimators=ntree, min_samples_leaf=5,
                                 max_features=0.5, replace=False,
                                 sample_fraction=0.5, honesty=True,
                                 n_jobs=-1, pred_method='numpy_loop',
                                 weight_method='numpy_loop',
                                 inference=False, random_state=123)
    test_loop_m1 = %timeit -r3 -o oforest_loop_m1.fit(X=features, y=outcome)


# =============================================================================
#     oforest_mpire = OrderedForest(n_estimators=ntree, min_samples_leaf=5,
#                                  max_features=0.5, replace=False,
#                                  sample_fraction=0.5, honesty=True,
#                                  n_jobs=-1, pred_method='numpy_loop_mpire',
#                                  weight_method='numpy_loop',
#                                  inference=False, random_state=123)
#     test_mpire = %timeit -r3 -o oforest_mpire.fit(X=features, y=outcome)
# =============================================================================
    
# =============================================================================
#     oforest_ray = OrderedForest(n_estimators=ntree, min_samples_leaf=5,
#                                  max_features=0.5, replace=False,
#                                  sample_fraction=0.5, honesty=True,
#                                  n_jobs=-1, pred_method='numpy_loop_ray',
#                                  weight_method='numpy_loop',
#                                  inference=False, random_state=123)
#     test_ray = %timeit -r3 -o oforest_ray.fit(X=features, y=outcome)
#     
#     oforest_ray_p1 = OrderedForest(n_estimators=ntree, min_samples_leaf=5,
#                                  max_features=0.5, replace=False,
#                                  sample_fraction=0.5, honesty=True,
#                                  n_jobs=1, pred_method='numpy_loop_ray',
#                                  weight_method='numpy_loop',
#                                  inference=False, random_state=123)
#     test_ray_p1 = %timeit -r3 -o oforest_ray_p1.fit(X=features, y=outcome)
# =============================================================================



    time_table[ntree] = [test_loop_p1.average, test_loop_m1.average,
                            test_ray.average, test_ray_p1.average]

pd.DataFrame(time_table, index=['numpy_loop_p1', 'numpy_loop_m1', 
                                'numpy_loop_ray_m1', 
                                'numpy_loop_ray_p1'])

# =============================================================================
# N=1000             Trees:
#                         100       500       1000       2000        5000
# numpy_loop_p1      0.433521  2.224826   4.315261   9.005083   26.934225
# numpy_loop_m1      0.476460  2.466126   4.632021  10.354410   25.188171
# numpy_loop_ray_m1  1.269955  7.207120  16.610913  35.663074  130.514483
# numpy_loop_ray_p1  1.274609  7.342310  16.560046  35.786295  117.000956
# =============================================================================

# =============================================================================
# N=10000         Trees:
#                      100         500        1000        2000         5000
# numpy_loop_p1   0.741128    5.429042   27.768768   76.145003   779.551439
# numpy_loop_m1   0.664058    7.211475   24.288042   84.645377  1585.548253
# =============================================================================
