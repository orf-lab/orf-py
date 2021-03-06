"""
orf: Ordered Random Forests.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Comparison of the Variance Computation.
"""

# load libraries
import os
import numpy as np

# set path
path = "/home/okasag/Documents/HSG/ORF/python/ORFpy/dev/_R"
os.chdir(path)

# Hard-coded data to test variance computation in R vs. python
probs1 = np.array((0.569929,0.457652,0.200977,0.163428,0.321162,0.187856,0.557985,0.202383,0.430742,0.24974,0.196137))
probs2 = np.array((0.894521,0.86706,0.609612,0.515869,0.681767,0.511826,0.915146,0.556205,0.793036,0.682363,0.641027))

weights1 = np.array((0.00346975,0.0068159,0.00722998,0.00389689,0.00114338,0,0.000873597,0.000942953,0.00313037,0.0176566,0.0184682,
0.0110608,0.00177689,0.000196078,0.0003125,0.000196078,0.0003125,0.0112818,0,0,0.0011673,0.000636223,
0.00899232,0.00236292,0.000196078,0.0003125,0.00314779,0.0003125,0.0101538,0,0.00124086,0.000185185,0.000322497,
0.0158936,0.00220057,0.000196078,0,0.00110744,0,0.0103377,0.000227273,0.00032002,0.000844722,0.00184519,
0.0301387,0.0065358,0.00196392,0,0.000196078,0,0.010198,0.0003125,0.0034221,0.00157705,0.000322497,
0,0,0.00652776,0.00332004,0.00476587,0.00982205,0.000172414,0.0102191,0.000181818,0,0.00193601,
0,0.0012479,0.000418301,0.00614804,0.000607402,0.00376463,0.00210625,0.0123371,0.00669657,0.00159,0.00856475,
0.0158936,0.00220057,0.000196078,0,0.00110744,0,0.0103377,0.000227273,0.00032002,0.000844722,0.00184519,
0,0,0.00165952,0.00585058,0.00578674,0.0163425,0.000172414,0.0120904,0,0,0.000879279,
0.00390239,0,0,0.00499115,0,0.00327193,0.00181341,0.000753163,0.00109054,0,0,
0.016946,0.00508244,0.00112622,0,0.000196078,0,0.0100045,0,0.00166949,0.0112805,0)).reshape((11,11)).T

weights2 = np.array((0.00710746,0.0077141,0.0074748,0.00421359,0.000340848,0.00028665,0.00112149,0.000988275,0.00404495,0.0231542,0.016871,
0.00642752,0.00340493,0.000358879,0.000699271,0.00186834,0.00143795,0.0114419,0,0,0.000537781,0.00156384,
0.00605589,0.00471814,9.80392e-05,0.000429,0.00531596,0.00143795,0.0102851,0,0.00111091,0.0002152,0.00120933,
0.0138432,0.00320477,0.000358879,0,0.00318137,0,0.0119951,0.000263158,0,0.00130081,0.00342018,
0.0286242,0.0113724,0.00468363,0,0.000448769,0,0.00863355,0,0.0049544,0.00441213,0.00114496,
0,0,0.00625764,0.00248581,0.00456733,0.0104452,0,0.00922252,0.000511762,0,0.00070298,
0.000479303,0.000192308,0,0.00630264,0.00143355,0.00421068,0.00348319,0.0140849,0.00583617,0.000608974,0.00541731,
0.0134236,0.00301246,0.000358879,0,0.00269094,0,0.0109123,0,0,0.00130081,0.00319291,
0,0,0.0019216,0.00484261,0.00578633,0.0168282,0,0.0127809,0,0,0.000591337,
0.00196262,0,0,0.00352792,0,0.00567508,0.0010737,0.000147059,0.000839181,0,0,
0.0173823,0.0101862,0.00262136,0,0.000221496,0,0.0104381,0,0.00159972,0.013501,0.000650806)).reshape((11,11)).T

Y_ind_1 = np.array((0,0,1,0,0,0,0,0,0,0,0))
Y_ind_2 = np.array((1,1,1,0,1,0,0,1,1,1,0))

honest_pred = {1: probs1, 2: probs2}
honest_weights = {1: weights1, 2: weights2}
Y_ind_honest = {1: Y_ind_1, 2: Y_ind_2}

probs = np.array([probs1, probs2]).T
weights = honest_weights
outcome_binary = Y_ind_honest

# number of categories
nclass = 3
n_est = len(outcome_binary[1])
n_samples = len(honest_pred[1])

# train and honest indices
ind_tr = [0,1,2,3,4,5]
ind_est = [6,7,8,9,10]

# honest binary outcome
outcome_binary_est = {1: Y_ind_1[ind_est], 2: Y_ind_2[ind_est]}

# Function to compute variance of predictions.
# -> Does the N in the formula refer to n_samples or to n_est?
# -> Does the N in the formula refer to n_samples or to n_est?
# -> This depends on which data is passed to the function:
# here it should always be n_est if it is out-of-sample, i.e. len(probs)
def honest_variance(probs, weights, outcome_binary, nclass,
                    n_est, n_samples):
    # ### (single class) Variance computation:
    # Create storage containers
    honest_multi_demeaned = {}
    honest_variance = {}
    # Loop over classes
    for class_idx in range(1, nclass, 1):
        # divide predictions by N to obtain mean after summing up
        honest_pred_mean = np.reshape(
            probs[:, (class_idx-1)] / n_est, (-1, 1))
        # calculate standard multiplication of weights and outcomes
        honest_multi = np.multiply(
            weights[class_idx], outcome_binary[class_idx].reshape((1, -1)))
        # subtract the mean from each obs i
        honest_multi_demeaned[class_idx] = honest_multi - honest_pred_mean
        # compute the square
        honest_multi_demeaned_sq = np.square(
            honest_multi_demeaned[class_idx])
        # sum over all i in honest sample
        honest_multi_demeaned_sq_sum = np.sum(
            honest_multi_demeaned_sq, axis=1)
        # multiply by N/N-1 (normalize)
        honest_variance[class_idx] = (honest_multi_demeaned_sq_sum *
                                      (n_est/(n_est-1)))
    # ### Covariance computation:
    # Shift categories for computational convenience
    # Postpend matrix of zeros
    honest_multi_demeaned_0_last = honest_multi_demeaned
    honest_multi_demeaned_0_last[nclass] = np.zeros(
        honest_multi_demeaned_0_last[1].shape)
    # Prepend matrix of zeros
    honest_multi_demeaned_0_first = {}
    honest_multi_demeaned_0_first[1] = np.zeros(
        honest_multi_demeaned[1].shape)
    # Shift existing matrices by 1 class
    for class_idx in range(1, nclass, 1):
        honest_multi_demeaned_0_first[
            class_idx+1] = honest_multi_demeaned[class_idx]
    # Create storage container
    honest_covariance = {}
    # Loop over classes
    for class_idx in range(1, nclass+1, 1):
        # multiplication of category m with m-1
        honest_multi_demeaned_cov = np.multiply(
            honest_multi_demeaned_0_first[class_idx],
            honest_multi_demeaned_0_last[class_idx])
        # sum all obs i in honest sample
        honest_multi_demeaned_cov_sum = np.sum(
            honest_multi_demeaned_cov, axis=1)
        # multiply by (N/N-1)*2
        honest_covariance[class_idx] = honest_multi_demeaned_cov_sum*2*(
            n_est/(n_est-1))
    # ### Put everything together
    # Shift categories for computational convenience
    # Postpend matrix of zeros
    honest_variance_last = honest_variance
    honest_variance_last[nclass] = np.zeros(honest_variance_last[1].shape)
    # Prepend matrix of zeros
    honest_variance_first = {}
    honest_variance_first[1] = np.zeros(honest_variance[1].shape)
    # Shift existing matrices by 1 class
    for class_idx in range(1, nclass, 1):
        honest_variance_first[class_idx+1] = honest_variance[class_idx]
    # Create storage container
    honest_variance_final = np.empty((n_samples, nclass))
    # Compute final variance according to: var_last + var_first - cov
    for class_idx in range(1, nclass+1, 1):
        honest_variance_final[
            :, (class_idx-1):class_idx] = honest_variance_last[
                class_idx].reshape(-1, 1) + honest_variance_first[
                class_idx].reshape(-1, 1) - honest_covariance[
                    class_idx].reshape(-1, 1)
    return honest_variance_final

# Function to compute variance of predictions.
# -> Does the N in the formula refer to n_samples or to n_est?
# -> This depends on which data is passed to the function:
# for train sample N=n_tr and for honest sample N=n_est
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
            weights[class_idx][ind_est, :][:, ind_est],  # subset honest weight
            outcome_binary[class_idx].reshape((1, -1)))
        # train sample
        train_multi = np.multiply(
            weights[class_idx][ind_tr, :][:, ind_est],  # subset honest weights
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


# compute the variances (out-of-sample)
orf_var_python = honest_variance(probs=probs, weights=honest_weights,
                                 outcome_binary=Y_ind_honest, nclass=nclass,
                                 n_est=n_est, n_samples=n_samples)

# load the variances computed in R
orf_var_R = np.genfromtxt('orf_var_R.csv', delimiter=',')
# delete the 0th row
orf_var_R = np.delete(orf_var_R, (0), axis=0)

# check if the two arrays are identical (at the 10th decimal point)
if (np.round(orf_var_python, 10) == np.round(orf_var_R, 10)).all():
    print('Variance computation in Python is the same as in R.')
else:
    print('Variance computation in Python is NOT the same as in R!')

# compute the variances (in-sample)
orf_var_python_in = get_honest_variance(probs=probs, weights=honest_weights,
                                        outcome_binary=outcome_binary_est,
                                        nclass=nclass,
                                        ind_est=ind_est, ind_tr=ind_tr)

# load the variances computed in R
orf_var_R_in = np.genfromtxt('orf_var_R_in.csv', delimiter=',')
# delete the 0th row
orf_var_R_in = np.delete(orf_var_R_in, (0), axis=0)

# check if the two arrays are identical (at the 10th decimal point)
if (np.round(orf_var_python_in, 10) == np.round(orf_var_R_in, 10)).all():
    print('Variance computation in Python is the same as in R in-sample.')
else:
    print('Variance computation in Python is NOT the same as in R in-sample !')

# alternative using the same variance function but twice
n_est = 5
n_tr = 6
# honest sample
orf_var_python_honest = honest_variance(probs=probs[ind_est, :],
   weights=dict([(key, honest_weights[key][ind_est, :][:, ind_est]) for key in range(1, nclass, 1)]),
   outcome_binary=dict([(key, Y_ind_honest[key][ind_est]) for key in range(1, nclass, 1)]),
   nclass=nclass, n_est=n_est, n_samples=n_est)
# train sample
orf_var_python_train = honest_variance(probs=probs[ind_tr, :],
   weights=dict([(key, honest_weights[key][ind_tr, :][:, ind_est]) for key in range(1, nclass, 1)]),
   outcome_binary=dict([(key, Y_ind_honest[key][ind_est]) for key in range(1, nclass, 1)]),
   nclass=nclass, n_est=n_tr, n_samples=n_tr)
# put honest and train sample together
orf_var_python_in_combined = np.vstack((orf_var_python_honest,
                                        orf_var_python_train))
# Combine indices
ind_all = np.hstack((ind_est, ind_tr))
# Sort orf_var_python_in_combined according to indices in ind_all
orf_var_python_in_combined = orf_var_python_in_combined[ind_all.argsort(), :]

# check if the two arrays are identical (at the 10th decimal point)
if (np.round(orf_var_python_in, 10) == np.round(orf_var_python_in_combined, 10)).all():
    print('Variance computation using the long function is the same as using '
          'the short one twice.')
else:
    print('Variance computation using the long function is NOT the same as '
          'using the short one twice!')
