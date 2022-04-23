# import modules
import pandas as pd
import os
import numpy as np
path="D:\switchdrive\Projects\ORF_Python\ORFpy"
# path = "/home/okasag/Documents/HSG/ORF/python/ORFpy"
os.chdir(path)

# load the ordered forest
from orf.orf import OrderedForest

# read in example data from the orf package in R
odata = pd.read_csv('dev/_data/odata.csv')

# define outcome and features
outcome = odata['Y']
features = odata.drop('Y', axis=1)


# =============================================================================
# Test predict function
# =============================================================================

# Different scenarios:
 # inference = True vs False
 # X = None vs features vs features.loc[1:20,:]
 # prob = True vs. False
boolbool = (True, False)
XType = (None, features, features.loc[0:20,:])

res = {}
for inf in boolbool:
    np.random.seed(999)
    oforest = OrderedForest(n_estimators=1000, min_samples_leaf=5, max_features=0.3,
                            replace=False, sample_fraction=0.5, honesty=True,
                            n_jobs=-1, pred_method='numpy_loop',
                            weight_method='numpy_loop', inference=inf)
    forest_fit = oforest.fit(X=features, y=outcome, verbose=True)
    for prob in boolbool:
        i=0
        for X in XType:
            i = i + 1
            res[np.str0(('inf',inf,'prob',prob,'X',i))] = oforest.predict(
                X=X, prob=prob)


# =============================================================================
# Test parallelization of fit via leaf means
# =============================================================================

# Check predict functionality without parallelization: n_jobs=1
pred_methods = ('loop',
                'loop_multi',
                'numpy',
                'numpy_sparse',
                'numpy_sparse2',
                'numpy_loop',
                'numpy_loop_multi',
                'numpy_loop_mpire')

res_pred_jobs1 = {}
for pred_method  in pred_methods:
    np.random.seed(999)
    print(pred_method)
    oforest = OrderedForest(n_estimators=1000, min_samples_leaf=5, max_features=0.3,
                            replace=False, sample_fraction=0.5, honesty=True,
                            n_jobs=1, pred_method=pred_method,
                            weight_method='numpy_loop', inference=False)
    res_pred_jobs1[pred_method] = %timeit -o oforest.fit(X=features, y=outcome, verbose=True)
list(res_pred_jobs1.items())

# =============================================================================
# [('loop',
#   <TimeitResult : 6.77 s ± 597 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>),
#  ('loop_multi',
#   <TimeitResult : 12.8 s ± 529 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>),
#  ('numpy',
#   <TimeitResult : 4.26 s ± 401 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>),
#  ('numpy_sparse',
#   <TimeitResult : 4.4 s ± 400 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>),
#  ('numpy_sparse2',
#   <TimeitResult : 4.22 s ± 390 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>),
#  ('numpy_loop',
#   <TimeitResult : 5.2 s ± 108 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>),
#  ('numpy_loop_multi',
#   <TimeitResult : 11.4 s ± 358 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>),
#  ('numpy_loop_mpire',
#   <TimeitResult : 13.3 s ± 349 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>)]
# =============================================================================

# Check predict functionality with parallelization: n_jobs= -1
pred_methods_par = ('loop_multi',
                    'numpy_loop_multi',
                    'numpy_loop_mpire')

res_pred_jobsmin1 = {}
for pred_method  in pred_methods_par:
    np.random.seed(999)
    print(pred_method)
    oforest = OrderedForest(n_estimators=1000, min_samples_leaf=5, max_features=0.3,
                            replace=False, sample_fraction=0.5, honesty=True,
                            n_jobs=-1, pred_method=pred_method,
                            weight_method='numpy_loop', inference=False)
    res_pred_jobsmin1[pred_method] = %timeit -o oforest.fit(X=features, y=outcome, verbose=True)
list(res_pred_jobsmin1.items())

# =============================================================================
# [('loop_multi',
#   <TimeitResult : 17.4 s ± 258 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>),
#  ('numpy_loop_multi',
#   <TimeitResult : 16.1 s ± 405 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>),
#  ('numpy_loop_mpire',
#   <TimeitResult : 54.5 s ± 674 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>)]
# =============================================================================

# =============================================================================
# Test parallelization of fit via weights
# =============================================================================

# Check weight functionality without parallelization: n_jobs=1
weight_methods = ('numpy_loop',
                  'numpy_loop_multi',
                  'numpy_loop_mpire')

res_weight_jobs1 = {}
for weight_method  in weight_methods:
    np.random.seed(999)
    print(weight_method)
    oforest = OrderedForest(n_estimators=1000, min_samples_leaf=5, max_features=0.3,
                            replace=False, sample_fraction=0.5, honesty=True,
                            n_jobs=1, pred_method='numpy_loop',
                            weight_method=weight_method, inference=True)
    res_weight_jobs1[weight_method] = %timeit -o oforest.fit(X=features, y=outcome, verbose=True)
list(res_weight_jobs1.items())

# =============================================================================
# [('numpy_loop',
#   <TimeitResult : 24 s ± 900 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)>)
#  ('numpy_loop_multi',
#   <TimeitResult : 1min 23s ± 6.22 s per loop (mean ± std. dev. of 7 runs, 1 loop each)>)
#  ('numpy_loop_mpire',
#   <TimeitResult : 1min 9s ± 3.51 s per loop (mean ± std. dev. of 7 runs, 1 loop each)>)]
# =============================================================================

# Check weight functionality with parallelization: n_jobs=-1
weight_methods_par = ('numpy_loop_shared_multi',
                      'numpy_loop_shared_mpire')

res_weight_jobsmin1 = {}
for weight_method  in weight_methods_par:
    np.random.seed(999)
    print(weight_method)
    oforest = OrderedForest(n_estimators=1000, min_samples_leaf=5, max_features=0.3,
                            replace=False, sample_fraction=0.5, honesty=True,
                            n_jobs=1, pred_method='numpy_loop',
                            weight_method=weight_method, inference=True)
    res_weight_jobsmin1[weight_method] = %timeit -o oforest.fit(X=features, y=outcome, verbose=True)
list(res_weight_jobsmin1.items())

# =============================================================================
# Programmes run forever...
# =============================================================================
