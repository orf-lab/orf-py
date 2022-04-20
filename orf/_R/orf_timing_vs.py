"""
orf: Ordered Random Forests.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Compare Timing of Ordered Forest Estimation with R Implementation.
"""
# %% import modules
import os
import pandas as pd
import numpy as np
import time

path="D:\switchdrive\Projects\ORF_Python\ORFpy"
#path = "/home/okasag/Documents/HSG/ORF/python/ORFpy"
os.chdir(path)

# load the ordered forest
from orf.OrderedForest import OrderedForest

# %% read in data
# read in synthetic test data based on the orf package in R
odata = pd.read_csv('orf/_R/data/odata_test.csv')
# specify response and covariates
X_all = odata.drop('Y', axis=1)
Y_all = odata['Y']

replace_options = [False, True]
honesty_options = [False, True]
inference_options = [False]
n_options = [1000,10000, 50000]
ntrees_options = [100, 1000, 2000]

repetitions = 2

results = pd.DataFrame({'n':[],
                        'ntrees':[],
                        'inference':[],
                        'honesty':[],
                        'replace':[],
                        'time':[]})

# loop through different settings and save results
for n in n_options:
    X = X_all[:n]
    Y = Y_all[:n]
    for ntrees in ntrees_options:
        for inference_idx in inference_options:
            # loop through honesty options
            for honesty_idx in honesty_options:
                # check if the setting is admissible
                if inference_idx and not honesty_idx:
                    continue
                # loop thorugh subsampling options
                for replace_idx in replace_options:
                    # check if the setting is admissible
                    if honesty_idx and replace_idx:
                        continue
                    if inference_idx and honesty_idx and replace_idx:
                        continue
                    # print current iteration
                    print('n:', n,
                          'ntrees:', ntrees,
                          'inference:', inference_idx,
                          'honesty:', honesty_idx,
                          'replace:', replace_idx)
                    # Initialize orf
                    orf_fit = OrderedForest(n_estimators=ntrees, 
                                            min_samples_leaf=5,
                                            max_features=0.3, 
                                            random_state=123,
                                            replace=replace_idx,
                                            honesty=honesty_idx,
                                            inference=inference_idx)
                    # Fit orf and measure timing
                    start = time.time()
                    orf_fit.fit(X=X, y=Y)
                    time_orf = time.time() - start
                    # Compute average execution time
                    np.mean(time_orf)
                    # Save result
                    result_it = pd.DataFrame({'n':[n],
                                            'ntrees':[ntrees],
                                            'inference':[inference_idx],
                                            'honesty':[honesty_idx],
                                            'replace':[replace_idx],
                                            'time':[np.mean(time_orf)]})    
                    results = pd.concat([results, result_it])

results.to_csv(path + '/orf/_R/results/py_timing.csv', index=False) 


"""
time_orf = timeit.repeat('orf_fit.fit(X=X, y=Y)',
                         globals=globals(),
                         number=1,
                         repeat=repetitions)
"""