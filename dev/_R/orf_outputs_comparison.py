"""
orf: Ordered Random Forests.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Compare Ordered Forest Intermediate Outputs from R and Python.
"""

# import modules
import pandas as pd
import numpy as np
import os
# path="D:\switchdrive\Projects\ORF_Python\ORFpy"
path = "/home/okasag/Documents/HSG/ORF/python/ORFpy"
os.chdir(path)

# load from R package
# weights
weights_R_1 = pd.read_csv('dev/_R/weights_R_1.csv')
weights_R_2 = pd.read_csv('dev/_R/weights_R_2.csv')
# predictions
pred_R_1 = pd.read_csv('dev/_R/pred_R_1.csv')
pred_R_2 = pd.read_csv('dev/_R/pred_R_2.csv')
# train data
data_ind_train_R_1 = pd.read_csv('dev/_R/data_in_train_R_1.csv')
data_ind_train_R_2 = pd.read_csv('dev/_R/data_in_train_R_2.csv')
# honest data
data_ind_honest_R_1 = pd.read_csv('dev/_R/data_in_honest_R_1.csv')
data_ind_honest_R_2 = pd.read_csv('dev/_R/data_in_honest_R_2.csv')
# indices (-1 to comply with Python)
ind_est = pd.read_csv('dev/_R/ind_est_R.csv') - 1
ind_tr = pd.read_csv('dev/_R/ind_tr_R.csv') - 1
# variance
variance_R = pd.read_csv('dev/_R/variance_R.csv')

# load from Python package
# weights
weights_python_1 = pd.read_csv('dev/_R/weights_python_1.csv')
weights_python_2 = pd.read_csv('dev/_R/weights_python_2.csv')
# predictions
pred_python_1 = pd.read_csv('dev/_R/pred_python_1.csv')
pred_python_2 = pd.read_csv('dev/_R/pred_python_2.csv')
# honest data
data_ind_train_python_1 = pd.read_csv('dev/_R/data_in_train_python_1.csv')
data_ind_train_python_2 = pd.read_csv('dev/_R/data_in_train_python_2.csv')
# honest data
data_ind_honest_python_1 = pd.read_csv('dev/_R/data_in_honest_python_1.csv')
data_ind_honest_python_2 = pd.read_csv('dev/_R/data_in_honest_python_2.csv')

# check if predictions correspond to weights * outcomes
# R
# class 1
if (np.round(pred_R_1['x'], 10) == np.round(pd.Series(
        np.dot(weights_R_1, data_ind_honest_R_1['Y']), name='x'), 10)).all():
    print('R: predictions are the same as weights * outcomes.')
else:
    print('R: predictions are NOT the same as weights * outcomes!')
# class 2
if (np.round(pred_R_2['x'], 10) == np.round(pd.Series(
        np.dot(weights_R_2, data_ind_honest_R_2['Y']), name='x'), 10)).all():
    print('R: predictions are the same as weights * outcomes.')
else:
    print('R: predictions are NOT the same as weights * outcomes!')
# Python
# class 1
if (np.round(pred_python_1['0'], 10) == np.round(pd.Series(np.dot(
        weights_python_1, data_ind_honest_python_1['0']), name='x'),
        10)).all():
    print('Python: predictions are the same as weights * outcomes.')
else:
    print('Python: predictions are NOT the same as weights * outcomes!')
# class 2
if (np.round(pred_python_2['0'], 10) == np.round(pd.Series(np.dot(
        weights_python_2, data_ind_honest_python_2['0']), name='x'),
        10)).all():
    print('Python: predictions are the same as weights * outcomes.')
else:
    print('Python: predictions are NOT the same as weights * outcomes!')

# comparison of variance computation given the same input data as in R
X_tr = data_ind_train_R_1.drop('Y', axis=1)
X_est = data_ind_honest_R_1.drop('Y', axis=1)
outcome_binary = {1: data_ind_train_R_1['Y'], 2: data_ind_train_R_2['Y']}
outcome_binary_est = {1: data_ind_honest_R_1['Y'].values,
                      2: data_ind_honest_R_2['Y'].values}
probs = np.hstack((pred_R_1, pred_R_2))
honest_weights = {1: weights_R_1.values, 2: weights_R_2.values}
ind_est = ind_est['x'].values
ind_tr = ind_tr['x'].values


# compute the variances (in-sample)
def get_honest_variance(probs, weights, outcome_binary, nclass,
                        ind_tr, ind_est):
    """Compute the variance of predictions (in-sample)."""
    # get the number of observations in train and honest sample
    n_est = len(ind_est)
    n_tr = len(ind_tr)
    # ### (single class) Variance computation:
    # ## Create storage containers
    # honest sample
    honest_multi_demeaned = {}
    honest_variance = {}
    # train sample
    train_multi_demeaned = {}
    train_variance = {}
    # Loop over classes
    for class_idx in range(1, nclass, 1):
        # divide predictions by N to obtain mean after summing up
        # honest sample
        honest_pred_mean = np.reshape(
            probs[ind_est, (class_idx-1)] / n_est, (-1, 1))
        # train sample
        train_pred_mean = np.reshape(
            probs[ind_tr, (class_idx-1)] / n_tr, (-1, 1))
        # calculate standard multiplication of weights and outcomes
        # outcomes need to be from the honest sample (outcome_binary_est)
        # for both honest and train multi
        # honest sample
        honest_multi = np.multiply(
            weights[class_idx][ind_est, :],  # subset honest weight
            outcome_binary[class_idx].reshape((1, -1)))
        # train sample
        train_multi = np.multiply(
            weights[class_idx][ind_tr, :],  # subset honest weights
            outcome_binary[class_idx].reshape((1, -1)))
        # subtract the mean from each obs i
        # honest sample
        honest_multi_demeaned[class_idx] = honest_multi - honest_pred_mean
        # train sample
        train_multi_demeaned[class_idx] = train_multi - train_pred_mean
        # compute the square
        # honest sample
        honest_multi_demeaned_sq = np.square(
            honest_multi_demeaned[class_idx])
        # train sample
        train_multi_demeaned_sq = np.square(
            train_multi_demeaned[class_idx])
        # sum over all i in the corresponding sample
        # honest sample
        honest_multi_demeaned_sq_sum = np.sum(
            honest_multi_demeaned_sq, axis=1)
        # train sample
        train_multi_demeaned_sq_sum = np.sum(
            train_multi_demeaned_sq, axis=1)
        # multiply by N/N-1 (normalize), N for the corresponding sample
        # honest sample
        honest_variance[class_idx] = (honest_multi_demeaned_sq_sum *
                                      (n_est/(n_est-1)))
        # train sample
        train_variance[class_idx] = (train_multi_demeaned_sq_sum *
                                     (n_tr/(n_tr-1)))

    # ### Covariance computation:
    # Shift categories for computational convenience
    # Postpend matrix of zeros
    # honest sample
    honest_multi_demeaned_0_last = honest_multi_demeaned
    honest_multi_demeaned_0_last[nclass] = np.zeros(
        honest_multi_demeaned_0_last[1].shape)
    # train sample
    train_multi_demeaned_0_last = train_multi_demeaned
    train_multi_demeaned_0_last[nclass] = np.zeros(
        train_multi_demeaned_0_last[1].shape)
    # Prepend matrix of zeros
    # honest sample
    honest_multi_demeaned_0_first = {}
    honest_multi_demeaned_0_first[1] = np.zeros(
        honest_multi_demeaned[1].shape)
    # train sample
    train_multi_demeaned_0_first = {}
    train_multi_demeaned_0_first[1] = np.zeros(
        train_multi_demeaned[1].shape)
    # Shift existing matrices by 1 class
    # honest sample
    for class_idx in range(1, nclass, 1):
        honest_multi_demeaned_0_first[
            class_idx+1] = honest_multi_demeaned[class_idx]
    # train sample
    for class_idx in range(1, nclass, 1):
        train_multi_demeaned_0_first[
            class_idx+1] = train_multi_demeaned[class_idx]
    # Create storage container
    honest_covariance = {}
    train_covariance = {}
    # Loop over classes
    for class_idx in range(1, nclass+1, 1):
        # multiplication of category m with m-1
        # honest sample
        honest_multi_demeaned_cov = np.multiply(
            honest_multi_demeaned_0_first[class_idx],
            honest_multi_demeaned_0_last[class_idx])
        # train sample
        train_multi_demeaned_cov = np.multiply(
            train_multi_demeaned_0_first[class_idx],
            train_multi_demeaned_0_last[class_idx])
        # sum all obs i in honest sample
        honest_multi_demeaned_cov_sum = np.sum(
            honest_multi_demeaned_cov, axis=1)
        # sum all obs i in train sample
        train_multi_demeaned_cov_sum = np.sum(
            train_multi_demeaned_cov, axis=1)
        # multiply by (N/N-1)*2
        # honest sample
        honest_covariance[class_idx] = honest_multi_demeaned_cov_sum*2*(
            n_est/(n_est-1))
        # train sample
        train_covariance[class_idx] = train_multi_demeaned_cov_sum*2*(
            n_tr/(n_tr-1))

    # ### Put everything together
    # Shift categories for computational convenience
    # Postpend matrix of zeros
    # honest sample
    honest_variance_last = honest_variance
    honest_variance_last[nclass] = np.zeros(honest_variance_last[1].shape)
    # train sample
    train_variance_last = train_variance
    train_variance_last[nclass] = np.zeros(train_variance_last[1].shape)
    # Prepend matrix of zeros
    # honest sample
    honest_variance_first = {}
    honest_variance_first[1] = np.zeros(honest_variance[1].shape)
    # train sample
    train_variance_first = {}
    train_variance_first[1] = np.zeros(train_variance[1].shape)
    # Shift existing matrices by 1 class
    for class_idx in range(1, nclass, 1):
        # honest sample
        honest_variance_first[class_idx+1] = honest_variance[class_idx]
        # train sample
        train_variance_first[class_idx+1] = train_variance[class_idx]
    # Create storage container
    honest_variance_final = np.empty((n_est, nclass))
    train_variance_final = np.empty((n_tr, nclass))
    # Compute final variance according to: var_last + var_first - cov
    for class_idx in range(1, nclass+1, 1):
        # honest sample
        honest_variance_final[
            :, (class_idx-1):class_idx] = honest_variance_last[
                class_idx].reshape(-1, 1) + honest_variance_first[
                class_idx].reshape(-1, 1) - honest_covariance[
                    class_idx].reshape(-1, 1)
        # train sample
        train_variance_final[
            :, (class_idx-1):class_idx] = train_variance_last[
                class_idx].reshape(-1, 1) + train_variance_first[
                class_idx].reshape(-1, 1) - train_covariance[
                    class_idx].reshape(-1, 1)
    # put honest and train sample together
    variance_final = np.vstack((honest_variance_final,
                                train_variance_final))
    # Combine indices
    ind_all = np.hstack((ind_est, ind_tr))
    # Sort variance_final according to indices in ind_all
    variance_final = variance_final[ind_all.argsort(), :]
    # retunr final variance
    return variance_final


# compute the variance
variance_python = get_honest_variance(probs=probs,
                                      weights=honest_weights,
                                      outcome_binary=outcome_binary_est,
                                      nclass=3,
                                      ind_est=ind_est['x'].values,
                                      ind_tr=ind_tr['x'].values)

# compare variance
if (np.round(variance_R.values, 10) == np.round(variance_python, 10)).all():
    print('Variances from R and Python are the same.')
else:
    print('Variances from R and Python are NOT the same!')

# comparison of weights computation
