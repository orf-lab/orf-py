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


sample_sizes = [80000]
time_table = {}

for n_sample in sample_sizes:
    # Generate data set
    features, outcome = example_data(seed=123, n=n_sample, p_cont=1, p_cat=1,
                                     p_binary=1, noise=True, y_cat=3,
                                     cat_cat=3)

    # 0.) How fast is fitting two forests in econML?
    #forest = RegressionForest(n_estimators=1000, min_samples_leaf=5,
    #                          max_features=0.5, max_samples=0.5, honest=False)
    #test_0 = %timeit -o forest.fit(X=features, y=outcome)

    # 1.) How fast is Python implementation without Parallelization
    #oforest1 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
    #                         max_features=0.5, replace=False,
    #                         sample_fraction=0.5, honesty=True,
    #                         n_jobs=1, pred_method='numpy')
    #test_1 = %timeit -o oforest1.fit(X=features, y=outcome)

    # 2.) How fast is Python implementation without Parallelization
    #oforest2 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
    #                         max_features=0.5, replace=False,
    #                         sample_fraction=0.5, honesty=True,
    #                         n_jobs=1, pred_method='loop')
    #test_2 = %timeit -o oforest2.fit(X=features, y=outcome)

    # 3.) How fast is Python implementation with Parallelization
    #oforest3 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
    #                         max_features=0.5, replace=False,
    #                         sample_fraction=0.5, honesty=True,
    #                         n_jobs=-1, pred_method='loop')
    #test_3 = %timeit -o oforest3.fit(X=features, y=outcome)

    # 4.) How fast is Cython implementation without Parallelization
    oforest4 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                             max_features=0.5, replace=False,
                             sample_fraction=0.5, honesty=True,
                             n_jobs=1, pred_method='cython')
    test_4 = %timeit -o oforest4.fit(X=features, y=outcome)

    # 5.) How fast is Cython implementation with Parallelization
    oforest5 = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                             max_features=0.5, replace=False,
                             sample_fraction=0.5, honesty=True,
                             n_jobs=-1, pred_method='cython')
    test_5 = %timeit -o oforest5.fit(X=features, y=outcome)

    time_table[n_sample] = [test_0.average*2, math.nan, test_2.average,
                            test_3.average, test_4.average, test_5.average]

pd.DataFrame(time_table, index=['2 x econML', 'numpy', 'loop', 'loop par',
                                'cython', 'cython par']).T


"""
n      2 x econML      numpy       loop    loop par     cython  cython par
1250     1.634134   2.538868   3.535324    4.342317   2.494032    2.735010
2500     2.726762   3.085457   5.426605    6.776736   3.077705    3.364190
5000     4.038665   5.720619   8.716242   14.779802   6.601798    6.017505
10000    7.989531  17.709245  19.591807   73.085157  43.520433   36.396375
20000   56.749016  97.639873  60.991867  108.089989  62.723277   53.061945
40000   58.343502        NaN 153.837272  257.034573  192.14841  202.482050
80000   ...takes forever..................................................
"""

