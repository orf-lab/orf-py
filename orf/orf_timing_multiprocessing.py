"""
Ordered Forest Parallelisation Benchmarking.
"""

import os
import time
import timeit
import platform

import numpy as np
import pandas as pd

from econml.grf import RegressionForest

# path = "D:/switchdrive/Projects/ORF_Python/ORFpy"
# path = "/home/okasag/Documents/HSG/ORF/python/ORFpy"
path = "/Users/okasag/Desktop/HSG/orf/python/ORFpy"
os.chdir(path)

# load the ordered forest
from orf.OrderedRandomForest import OrderedRandomForest

# define your operating system
opsystem = platform.system()


# %% Define function to produce data sets of different size
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


# %% Benchmark 1: Parallelisation for the .fit() with honesty, no inference
# define loop values
sample_sizes = [1000, 2500, 5000, 10000, 20000]
pred_methods = ['loop', 'loop_joblib', 'numpy', 'numpy_loop', 'numpy_joblib',
                'numpy_sparse', 'numpy_sparse2', 'numpy_mpire']
core_sizes = [1, 4, 8]
reps = 3

# define storage for results
time_table = {}
# start the loop
for n_sample in sample_sizes:
    # Generate data set
    features, outcome = example_data(seed=123, n=n_sample, p_cont=1, p_cat=1,
                                     p_binary=1, noise=True, y_cat=3,
                                     cat_cat=3)

    # Forest: EconML (only with half size of dataset as we do not use honesty)
    econml = RegressionForest(n_estimators=1000, min_samples_leaf=5,
                              max_features=0.3, max_samples=0.5,
                              honest=False)
    # estimate and time it (2* due to 2 categories in the outcome)
    time_econml = 2*timeit.timeit("""econml.fit(
        X=features.iloc[:np.ceil(n_sample/2).astype(int), :],
        y=outcome.iloc[:np.ceil(n_sample/2).astype(int)])""",
    globals=globals(), number=reps)

    # loop through different number of cores
    for n_core in core_sizes:
        time_orf_all = []
        # Ordered Forest: loop through different prediction methods
        for method_idx in pred_methods:
            print(method_idx)
            # define the ordered forest
            orf = OrderedRandomForest(n_estimators=1000, min_samples_leaf=5,
                                      max_features=0.3, replace=False,
                                      sample_fraction=0.5, honesty=True,
                                      n_jobs=n_core, pred_method=method_idx,
                                      random_state=n_sample)
            # estimate and timeit
            time_orf = timeit.timeit('orf.fit(X=features, y=outcome)',
                                     globals=globals(), number=reps)
            # save it
            time_orf_all.append(time_orf/reps)
        
        # add econml
        time_orf_all.append(time_econml/reps)
        # summarize the computation time by sample size and cores
        time_table[str(n_sample) + '_' + str(n_core)] = time_orf_all

# save as dataframe
pred_methods.append('EconML')
timing_pred = pd.DataFrame(time_table, index=pred_methods).T
# save the timing results
timing_pred.to_csv(path+'/orf/timing/'+opsystem+'_timing_pred_method.csv')


# %% Benchmark 2: Parallelisation for the .fit() with honesty and inference
# define loop values
sample_sizes = [1000, 2500, 5000, 10000, 20000]
weight_methods = ['numpy_loop', 'numpy_loop_shared_joblib',
                  'numpy_loop_joblib_conquer']
core_sizes = [1, 4, 8]
reps = 3

# define storage for results
time_table = {}
# start the loop
for n_sample in sample_sizes:
    # Generate data set
    features, outcome = example_data(seed=123, n=n_sample, p_cont=1, p_cat=1,
                                     p_binary=1, noise=True, y_cat=3,
                                     cat_cat=3)

    # Forest: EconML (full dataset as we use honesty)
    econml = RegressionForest(n_estimators=1000, min_samples_leaf=5,
                              max_features=0.3, max_samples=0.5,
                              honest=True, inference=True)
    # estimate and time it
    time_econml = timeit.timeit("""econml.fit(
        X=features.iloc[:np.ceil(n_sample/2).astype(int), :],
        y=outcome.iloc[:np.ceil(n_sample/2).astype(int)])""",
    globals=globals(), number=reps)

    # loop through different number of cores
    for n_core in core_sizes:
        time_orf_all = []
        # Ordered Forest: loop through different prediction methods
        for method_idx in weight_methods:
            print(method_idx)
            # define the ordered forest
            orf = OrderedRandomForest(n_estimators=1000, min_samples_leaf=5,
                                      max_features=0.3, replace=False,
                                      sample_fraction=0.5, honesty=True,
                                      inference=True, weight_method=method_idx,
                                      n_jobs=n_core, random_state=n_sample)
            # estimate and timeit
            time_orf = timeit.timeit('orf.fit(X=features, y=outcome)',
                                     globals=globals(), number=reps)
            # save it
            time_orf_all.append(time_orf/reps)
        
        # add econml
        time_orf_all.append(time_econml/reps)
        # summarize the computation time by sample size and cores
        time_table[str(n_sample) + '_' + str(n_core)] = time_orf_all

# save as dataframe
weight_methods.append('EconML')
timing_weight = pd.DataFrame(time_table, index=weight_methods).T
# save the timing results
timing_weight.to_csv(path+'/orf/timing/'+opsystem+'_timing_weight_method.csv')
