"""
orf: Ordered Random Forests.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Compare Ordered Forest Estimation with R Implementation.
"""

# %% import modules
import os
import pandas as pd
import numpy as np

from plotnine import ggsave

path="D:\switchdrive\Projects\ORF_Python\ORFpy"
#path = "/home/okasag/Documents/HSG/ORF/python/ORFpy"
os.chdir(path)

# load the ordered forest
from orf.OrderedForest import OrderedForest

# %% read in data
# read in synthetic test data based on the orf package in R
odata = pd.read_csv('orf/_R/data/odata_test.csv')
# read in empirical test data based on the stevedata package in R
dataset = pd.read_csv('orf/_R/data/empdata_test.csv')

# %% benchmark settings
replace_options = [False, True]
honesty_options = [False, True]
inference_options = [False, True]
data_types = ['synth', 'emp']

# start benchmark
for data_idx in data_types:
    # based on data type, determine X and Y for orf estimation
    if data_idx == 'synth':
        # specify response and covariates
        X = odata.drop('Y', axis=1)
        Y = odata['Y']
    else:
        # specify response and covariates
        X = dataset.drop('y', axis=1)
        Y = dataset['y']

    # loop through different settings and save results
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
                print('data:', data_idx,
                      'inference:', inference_idx,
                      'honesty:', honesty_idx,
                      'replace:', replace_idx)

                # fit ORF with at least 2000 trees (set seed for replicability)
                orf_fit = OrderedForest(n_estimators=100, min_samples_leaf=5,
                                        max_features=0.3, random_state=123,
                                        replace=replace_idx,
                                        honesty=honesty_idx,
                                        inference=inference_idx)
                # fit the model
                orf_fit.fit(X=X, y=Y)

                # get in-sample results
                orf_pred = orf_fit.forest_['probs']
                orf_var = orf_fit.forest_['variance']
                orf_rps = np.array(orf_fit.measures['rps'])
                orf_mse = np.array(orf_fit.measures['mse'])
                # wrap into dictionary
                fit_results = {'orf_pred': orf_pred,
                               'orf_var': orf_var,
                               'orf_rps': orf_rps,
                               'orf_mse': orf_mse}

                # get the plot
                orf_plot = orf_fit.plot()

                # get the margins (mean margins for reliable comparison)
                orf_margins = orf_fit.margins(eval_point="mean")
                margins_effects = orf_margins['effects']
                margins_vars = orf_margins['variances']
                # wrap into dictionary
                margins_results = {'margins_effects': margins_effects,
                                   'margins_vars': margins_vars}

                # save the results for plot
                ggsave(plot=orf_plot,
                       filename=('py_' + data_idx + '_plot_I_' +
                                 str(inference_idx).upper() + '_H_' +
                                 str(honesty_idx).upper() + '_R_' +
                                 str(replace_idx).upper() + '.png'),
                       path=path + '/orf/_R/results/')

                # save the results for fit
                for key, value in fit_results.items():
                    if type(value) is dict:
                        if value == {}:
                            value = [0]
                    if value is None:
                        value = [0]
                    # save the results
                    np.savetxt(fname=(path + '/orf/_R/results/py_' + data_idx
                                      + '_' + str(key) + '_I_' +
                                      str(inference_idx).upper() + '_H_' +
                                      str(honesty_idx).upper() + '_R_' +
                                      str(replace_idx).upper() + '.csv'),
                               X=value, delimiter=",")

                # save the results for margins
                for key, value in margins_results.items():
                    if type(value) is dict:
                        if value == {}:
                            value = [0]
                    if value is None:
                        value = [0]
                    # save the results
                    np.savetxt(fname=(path + '/orf/_R/results/py_' + data_idx
                                      + '_' + str(key) + '_I_' +
                                      str(inference_idx).upper() + '_H_' +
                                      str(honesty_idx).upper() + '_R_' +
                                      str(replace_idx).upper() + '.csv'),
                               X=value, delimiter=",")

# %%
