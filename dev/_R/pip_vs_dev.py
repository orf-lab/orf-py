"""
orf: Ordered Random Forests.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Compare Results of the PyPi Implementation with the Development Implementation.
"""

# %% import modules
import os
import pandas as pd
import numpy as np

from scipy import stats

# path="D:\switchdrive\Projects\ORF_Python\ORFpy"
# path = "/home/okasag/Documents/HSG/ORF/python/ORFpy"
path = "/Users/okasag/Desktop/HSG/orf/python/ORFpy"
os.chdir(path)

# load the ordered forest
from orf.OrderedForest import OrderedForest

# %% benchmark settings
replace_options = [False, True]
honesty_options = [False, True]
inference_options = [False, True]
data_types = ['synth', 'emp'] # 'package', 'pip'
# saved results
saved_results = ['orf_pred', 'orf_var', 'orf_rps', 'orf_mse',
                 'margins_effects', 'margins_vars']
# differences and results
diffs = {}
pip_results_all = {}
dev_results_all = {}

# %% read in data
for result_idx in saved_results:
    # loop through saved results
    for data_idx in data_types:
        # loop through different settings and read results
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
                    # check if the data exists
                    if (result_idx == 'orf_var' or
                        result_idx == 'margins_vars') and not inference_idx:
                        continue
                    # get result name
                    result_name = (data_idx + '_' + result_idx + '_I_' +
                                   str(inference_idx).upper() + '_H_' +
                                   str(honesty_idx).upper() + '_R_' +
                                   str(replace_idx).upper())
                    # load data from pip
                    pip_result = np.array(pd.read_csv(
                        (path + '/dev/_R/results/pip/' +
                         'py_' + result_name + '.csv'), header=None))
                    # save the result into a dictionary
                    pip_results_all[result_name] = pip_result
                    # load data from dev
                    dev_result = np.array(pd.read_csv(
                        (path + '/dev/_R/results/dev/' +
                         'py_' + result_name + '.csv'), header=None))
                    # save the result into a dictionary
                    dev_results_all[result_name] = dev_result
                    # compare the average results for predictions and mse, rps
                    if not result_idx in ['margins_effects', 'margins_vars']:
                        # take difference of means
                        result_diff = np.abs(np.mean(pip_result, axis=0) -
                                             np.mean(dev_result, axis=0))
                        # check if the difference is small enough
                        if (np.all(result_diff <= 0.05)):
                            print('Differences between pip and dev for ' +
                                  result_name + ' are small.')
                        else:
                            print('Differences between pip and dev for ' +
                                  result_name + ' are NOT small.')
                    else:
                        # compare the absolute differences for margins
                        result_diff = np.abs(pip_result - dev_result)
                        # check if the difference is small enough
                        if (np.all(result_diff <= 0.05)):
                            print('Differences between pip and dev for ' +
                                  result_name + ' are small.')
                        else:
                            print('Differences between pip and dev for ' +
                                  result_name + ' are NOT small.')
                    # save the difference into a dictionary
                    diffs[result_name] = result_diff

# save the differences and the results themselves separately on disk
np.save((path + '/dev/_R/results/dev/pip_dev_diffs.npy'), diffs)
np.save((path + '/dev/_R/results/dev/pip_results_all.npy'), pip_results_all)
np.save((path + '/dev/_R/results/dev/dev_results_all.npy'), dev_results_all)

# %% compare if marginal effects lie within the confidence intervals
# of the other implementation
# loop through datasets
for data_idx in data_types:
    # result_name
    effect_name = data_idx + '_margins_effects_I_TRUE_H_TRUE_R_FALSE'
    var_name = data_idx + '_margins_vars_I_TRUE_H_TRUE_R_FALSE'
    # get the effects
    pip_effects = pip_results_all[effect_name]
    dev_effects = dev_results_all[effect_name]
    # get the variances
    pip_vars = pip_results_all[var_name]
    dev_vars = dev_results_all[var_name]
    # get upper confidence intervals
    pip_ci_up = pip_effects + np.abs(stats.norm.ppf(.975)) * np.sqrt(pip_vars)
    dev_ci_up = dev_effects + np.abs(stats.norm.ppf(.975)) * np.sqrt(dev_vars)
    # get lower confidence intervals
    pip_ci_down = pip_effects - np.abs(stats.norm.ppf(.975)) * np.sqrt(pip_vars)
    dev_ci_down = dev_effects - np.abs(stats.norm.ppf(.975)) * np.sqrt(dev_vars)
    # check if results are inbetween
    # R
    if np.all((pip_effects <= dev_ci_up) & (pip_effects >= dev_ci_down)):
        print('Marginal effects from PyPi are covered by the CIs from Dev.')
    else:
        print('Marginal effects from PyPi are NOT covered by the CIs from Dev.')
    # Python
    if np.all((dev_effects <= pip_ci_up) & (dev_effects >= pip_ci_down)):
        print('Marginal effects from Dev are covered by the CIs from PyPi.')
    else:
        print('Marginal effects from Dev are NOT covered by the CIs from PyPi.')
        
# %% End of Comparisons
