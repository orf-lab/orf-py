"""
orf: Ordered Random Forests.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Definitions of class and functions.

"""
# %% Last changes:
# Set up file according to https://scikit-learn.org/stable/developers/develop.html
# - use subclasses/inheritance
# - no operations in __init__ -> instead new function check_inputs

# ToDo: fitted attributes (ending with a trailing underscore)

# %% Packages

# import modules
import numpy as np
import pandas as pd
import _thread
import sharedmem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from econml.grf import RegressionForest
#import orf.honest_fit as honest_fit
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import Pool, cpu_count, Lock
from mpire import WorkerPool
from functools import partial
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state, check_X_y, check_array
from sklearn.utils.validation import _num_samples, _num_features, check_is_fitted
from scipy import stats
import ray
from plotnine import (ggplot, aes, geom_density, facet_wrap, geom_vline, 
                      ggtitle, xlab, ylab, theme_bw, theme, element_rect)

# %% Class definition
    
# define OrderedForest class (BaseEstimator allows to call get_params and set_params)
class BaseOrderedForest(BaseEstimator):
    """
    Base class for forests of trees.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    # define init function
    def __init__(self, n_estimators=1000,
                 min_samples_leaf=5,
                 max_features=0.3,
                 replace=True,
                 sample_fraction=0.5,
                 honesty=False,
                 honesty_fraction=0.5,
                 inference=False,
                 n_jobs=-1,
                 pred_method='numpy_loop_mpire',
                 weight_method='numpy_loop_shared_mpire',
                 random_state=None):

        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.replace = replace
        self.sample_fraction = sample_fraction
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.inference = inference
        self.n_jobs = n_jobs
        self.pred_method = pred_method
        self.weight_method = weight_method
        self.random_state = random_state
        # initialize performance metrics
        self.confusion = None
        self.measures = None
        
    def input_checks(self):
        # check and define the input parameters
        n_estimators = self.n_estimators
        min_samples_leaf = self.min_samples_leaf
        max_features = self.max_features
        replace = self.replace
        sample_fraction = self.sample_fraction
        honesty = self.honesty
        honesty_fraction = self.honesty_fraction
        inference = self.inference
        n_jobs = self.n_jobs
        pred_method = self.pred_method
        weight_method = self.weight_method
        random_state = self.random_state
        # check the number of trees in the forest
        if isinstance(n_estimators, int):
            # check if its at least 1
            if n_estimators >= 1:
                # assign the input value
                self.n_estimators = n_estimators
            else:
                # raise value error
                raise ValueError("n_estimators must be at least 1"
                                 ", got %s" % n_estimators)
        else:
            # raise value error
            raise ValueError("n_estimators must be an integer"
                             ", got %s" % n_estimators)

        # check if minimum leaf size is integer
        if isinstance(min_samples_leaf, int):
            # check if its at least 1
            if min_samples_leaf >= 1:
                # assign the input value
                self.min_samples_leaf = min_samples_leaf
            else:
                # raise value error
                raise ValueError("min_samples_leaf must be at least 1"
                                 ", got %s" % min_samples_leaf)
        else:
            # raise value error
            raise ValueError("min_samples_leaf must be an integer"
                             ", got %s" % min_samples_leaf)

        # check share of features in splitting
        if isinstance(max_features, float):
            # check if its within (0,1]
            if (max_features > 0 and max_features <= 1):
                # assign the input value
                self.max_features = max_features
            else:
                # raise value error
                raise ValueError("max_features must be within (0,1]"
                                 ", got %s" % max_features)
        else:
            # raise value error
            raise ValueError("max_features must be a float"
                             ", got %s" % max_features)

        # check whether to sample with replacement
        if isinstance(replace, bool):
            # assign the input value
            self.replace = replace
        else:
            # raise value error
            raise ValueError("replace must be of type boolean"
                             ", got %s" % replace)

        # check subsampling fraction
        if isinstance(sample_fraction, float):
            # check if its within (0,1]
            if (sample_fraction > 0 and sample_fraction <= 1):
                # assign the input value
                self.sample_fraction = sample_fraction
            else:
                # raise value error
                raise ValueError("sample_fraction must be within (0,1]"
                                 ", got %s" % sample_fraction)
        else:
            # raise value error
            raise ValueError("sample_fraction must be a float"
                             ", got %s" % sample_fraction)

        # check whether to implement honesty
        if isinstance(honesty, bool):
            # assign the input value
            self.honesty = honesty
        else:
            # raise value error
            raise ValueError("honesty must be of type boolean"
                             ", got %s" % honesty)

        # check honesty fraction
        if isinstance(honesty_fraction, float):
            # check if its within (0,1]
            if (honesty_fraction > 0 and honesty_fraction < 1):
                # assign the input value
                self.honesty_fraction = honesty_fraction
            else:
                # raise value error
                raise ValueError("honesty_fraction must be within (0,1)"
                                 ", got %s" % honesty_fraction)
        else:
            # raise value error
            raise ValueError("honesty_fraction must be a float"
                             ", got %s" % honesty_fraction)

        # Honesty only possible if replace==False
        if (honesty and replace):
            # raise value error
            raise ValueError("Honesty works only when sampling without "
                             "replacement. Set replace=False and run again.")

        # check whether to conduct inference
        if isinstance(inference, bool):
            # assign the input value
            self.inference = inference
        else:
            # raise value error
            raise ValueError("inference must be of type boolean"
                             ", got %s" % inference)

        # Inference only possible if honesty==True
        if (inference and not honesty):
            # raise value error
            raise ValueError("For conducting inference honesty is required. "
                             "Set honesty=True and run again.")

        # Inference only possible if replace==False
        if (inference and replace):
            # raise value error
            raise ValueError("For conducting inference subsampling (without "
                             "replacement) is required. Set replace=False "
                             "and run again.")

        # check whether n_jobs is integer
        if isinstance(n_jobs, int):
            # check max available cores
            max_jobs = cpu_count()
            # check if it is -1
            if (n_jobs == -1):
                # set max - 1 as default
                self.n_jobs = max_jobs - 1
            # check if jobs are admissible for the machine
            elif (n_jobs >= 1 and n_jobs <= max_jobs):
                # assign the input value
                self.n_jobs = n_jobs
            else:
                # throw an error
                raise ValueError("n_jobs must be greater than 0 and less than"
                                 "available cores, got %s" % n_jobs)
        else:
            # raise value error
            raise ValueError("n_jobs must be of type integer"
                             ", got %s" % n_jobs)

        # check whether pred_method is defined correctly
        if (pred_method == 'cython'
                or pred_method == 'loop'
                or pred_method == 'loop_multi'
                or pred_method == 'numpy'
                or pred_method == 'numpy_loop'
                or pred_method == 'numpy_loop_multi'
                or pred_method == 'numpy_loop_mpire'
                or pred_method == 'numpy_sparse'
                or pred_method == 'numpy_loop_ray'
                or pred_method == 'numpy_sparse2'):
            # assign the input value
            self.pred_method = pred_method
        else:
            # raise value error
            raise ValueError("pred_method must be of cython, loop or numpy"
                             ", got %s" % pred_method)
        
        if self.pred_method == 'numpy_loop_ray':
            # Initialize ray
            ray.init(num_cpus=self.n_jobs, ignore_reinit_error=True)
        

        # check whether weight_method is defined correctly
        if (weight_method == 'numpy_loop'
                or weight_method == 'numpy_loop_mpire'
                or weight_method == 'numpy_loop_shared_mpire'
                or weight_method == 'numpy_loop_shared_multi'
                or weight_method == 'numpy_loop_multi'):
            # assign the input value
            self.weight_method = weight_method
        else:
            # raise value error
            raise ValueError("weight_method must be of numpy_loop, "
                             "numpy_loop_mpire, numpy_loop_shared_mpire, "
                             "numpy_loop_shared_multi or numpy_loop_multi"
                             ", got %s" % weight_method)

        # check whether seed is set (using scikitlearn check_random_state)
        self.random_state = check_random_state(random_state)
        # get max np.int32 based on machine limit
        max_int = np.iinfo(np.int32).max
        # use this to initialize seed for honest splitting: this is useful when
        # we want to obtain the same splits later on
        self.subsample_random_seed = self.random_state.randint(max_int)

        # initialize orf
        self.forest = None
        # initialize performance metrics
        self.confusion = None
        self.measures = None
        
    
    # %% Fit function
    # function to estimate ordered forest
    def fit(self, X, y, verbose=False):
        """
        Ordered Forest estimation.

        Parameters
        ----------
        X : TYPE: array-like
            DESCRIPTION: matrix of covariates.
        y : TYPE: array-like
            DESCRIPTION: vector of outcomes.
        verbose : TYPE: bool
            DESCRIPTION: should the results be printed to console?
            Default is False.

        Returns
        -------
        result: ordered probability predictions by Ordered Forest.
        """
        self.input_checks()
        # Use sklearn input checks to allow for multiple types of inputs:
        # - returns numpy arrays for X and y (no matter which input type)
        # - forces y to be numeric
        X,y = check_X_y(X,y, y_numeric=True, estimator="OrderedForest")
        # Get vector of sorted unique values of y
        y_values = np.unique(y)
        # Get the number of outcome classes
        self.n_class = nclass = len(y_values)
        # Next, ensure that y is a vector of contiguous integers starting at 
        # 1 up to nclass
        # Check if y consists of integers
        if not all(isinstance(x, (np.integer)) for x in y_values):
            # Recode y appropriately (keeps order but recodes values as 1,2...)
            y = np.searchsorted(np.unique(y), y)+1
        else:
            # Check if contiguous sequence
            if not ((min(y_values)==1) and (max(y_values)==nclass)):
                # Recode y appropriately
                y = np.searchsorted(np.unique(y), y)+1

# =============================================================================
#         # check if features X are a pandas dataframe
#         self.__xcheck(X)
# 
#         # check if outcome y is a pandas series
#         if isinstance(y, pd.Series):
#             # check if its non-empty
#             if y.empty:
#                 # raise value error
#                 raise ValueError("y Series is empty. Check the input.")
#         else:
#             # raise value error
#             raise ValueError("y is not a Pandas Series. Recode the input.")
# =============================================================================

        # obtain total number of observations
        n_samples = _num_samples(X)
        # obtain total number of observations
        self.n_features = _num_features(X)
        # create an empty dictionary to save the forests
        forests = {}
        # create an empty array to save the predictions
        probs = np.empty((n_samples, nclass-1))
        # create an empty dictionary to save the fitted values
        fitted = {}
        #  create an empty dictionary to save the binarized outcomes
        outcome_binary = {}
        outcome_binary_est = {}
        #  create an empty dictionary to save the weights matrices
        weights = {}
        # generate honest estimation sample
        if self.honesty:
            # initialize random state for sample splitting
            subsample_random_state = check_random_state(
                self.subsample_random_seed)
            # Split the sample
            X_tr, X_est, y_tr, y_est = train_test_split(
                X, y, test_size=self.honesty_fraction,
                random_state=subsample_random_state)
            # Re-initialize random state to obtain indices
            subsample_random_state = check_random_state(
                self.subsample_random_seed)
            # shuffle indices
            ind_tr, ind_est = train_test_split(
                np.arange(n_samples), test_size=self.honesty_fraction,
                random_state=subsample_random_state)
        else:
            X_tr = X
            y_tr = y
            X_est = None
            ind_tr = np.arange(n_samples)
            ind_est = None
        # estimate random forest on each class outcome except the last one
        for class_idx in range(1, nclass, 1):
            # create binary outcome indicator for the outcome in the forest
            outcome_ind = (y_tr <= class_idx) * 1
            outcome_binary[class_idx] = np.array(outcome_ind)
            # check whether to do subsampling or not
            if self.replace:
                # call rf from scikit learn and save it in dictionary
                forests[class_idx] = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    max_samples=self.sample_fraction,
                    oob_score=True,
                    random_state=self.random_state)
                # fit the model with the binary outcome
                forests[class_idx].fit(X=X_tr, y=outcome_ind)
                # get in-sample predictions, i.e. the out-of-bag predictions
                probs[:,class_idx-1] = forests[class_idx].oob_prediction_
            else:
                # call rf from econML and save it in dictionary
                forests[class_idx] = RegressionForest(
                    n_estimators=self.n_estimators,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    max_samples=self.sample_fraction,
                    random_state=self.random_state,
                    honest=False,  # default is True!
                    inference=False,  # default is True!
                    subforest_size=1)
                # fit the model with the binary outcome
                forests[class_idx].fit(X=X_tr, y=outcome_ind)
                # if no honesty, get the oob predictions
                if not self.honesty:
                    # get in-sample predictions, i.e. out-of-bag predictions
                    probs[:,class_idx-1] = forests[class_idx].oob_predict(
                        X_tr).squeeze()
                else:
                    # Get leaf IDs for estimation set
                    forest_apply = forests[class_idx].apply(X_est)
                    # create binary outcome indicator for est sample
                    outcome_ind_est = np.array((y_est <= class_idx) * 1)
                    # save it into a dictionary for later use in variance
                    outcome_binary_est[class_idx] = np.array(outcome_ind_est)
                    # compute maximum leaf id
                    max_id = np.max(forest_apply)+1
                    if self.inference:
                        # Get size of estimation sample
                        n_est = forest_apply.shape[0]
                        # Get leaf IDs for training set
                        forest_apply_tr = forests[class_idx].apply(X_tr)
                        # Combine forest_apply and forest_apply_train
                        forest_apply_all = np.vstack((forest_apply,
                                                      forest_apply_tr))
                        # Combine indices
                        ind_all = np.hstack((ind_est, ind_tr))
                        # Sort forest_apply_all according to indices in ind_all
                        forest_apply_all = forest_apply_all[ind_all.argsort(),
                                                            :]
                        # generate storage matrix for weights
                        # forest_out = np.zeros((n_samples, n_est))

                        # check if parallelization should be used
                        if self.weight_method == 'numpy_loop_mpire':
                            # define partial function by fixing parameters
                            partial_fun = partial(
                                self.honest_weight_numpy,
                                forest_apply=forest_apply,
                                forest_apply_all=forest_apply_all,
                                n_samples=n_samples,
                                n_est=n_est)

                            # set up the worker pool for parallelization
                            pool = WorkerPool(n_jobs=self.n_jobs)
                            # make sure to have enough memory for the outputs
                            trees_out = np.zeros((self.n_estimators,
                                                  n_samples, n_est))
                            forest_out = np.zeros((n_samples, n_est))
                            # loop over trees in parallel
                            trees_out = np.array(pool.map(
                                partial_fun, range(self.n_estimators),
                                progress_bar=False,
                                concatenate_numpy_output=False))
                            # sum the forest
                            forest_out = trees_out.sum(0)
                            # free up the memory
                            del trees_out
                            # stop and join pool
                            pool.stop_and_join()

                        # use shared memory to add matrices using multiprocess
                        if self.weight_method == 'numpy_loop_shared_multi':
                            # define partial function by fixing parameters
                            partial_fun = partial(
                                tree_weights,
                                forest_apply=forest_apply,
                                forest_apply_all=forest_apply_all,
                                n_samples=n_samples,
                                n_est=n_est)
                            # compute the forest weights in parallel
                            forest_out = np.array(forest_weights_multi(
                                partial_fun=partial_fun,
                                n_samples=n_samples,
                                n_est=n_est,
                                n_jobs=self.n_jobs,
                                n_estimators=self.n_estimators))

                        # use shared memory to add matrices using mpire (fast)
                        if self.weight_method == 'numpy_loop_shared_mpire':
                            # define partial function by fixing parameters
                            partial_fun = partial(
                                tree_weights,
                                forest_apply=forest_apply,
                                forest_apply_all=forest_apply_all,
                                n_samples=n_samples,
                                n_est=n_est)
                            # compute the forest weights in parallel
                            forest_out = np.array(forest_weights_mpire(
                                partial_fun=partial_fun,
                                n_samples=n_samples,
                                n_est=n_est,
                                n_jobs=self.n_jobs,
                                n_estimators=self.n_estimators))

                        # use pure multiprocessing 
                        if self.weight_method == 'numpy_loop_multi':
                            # setup the pool for multiprocessing
                            pool = Pool(self.n_jobs)
                            # prepare iterables (need to replicate fixed items)
                            args_iter = []
                            for tree in range(self.n_estimators):
                                args_iter.append((tree, forest_apply,
                                                  forest_apply_all, n_samples,
                                                  n_est))
                            # loop over trees in parallel
                            # tree out saves all n_estimators weight matrices
                            # this is quite memory inefficient!!!
                            tree_out = pool.starmap(honest_weight_numpy_out,
                                                    args_iter)
                            pool.close()  # close parallel
                            pool.join()  # join parallel
                            # sum up all tree weights
                            forest_out = sum(tree_out)

                        if self.weight_method == 'numpy_loop':
                            # generate storage matrix for weights
                            forest_out = np.zeros((n_samples, n_est))
                            # Loop over trees
                            for tree in range(self.n_estimators):
                                # extract vectors of leaf IDs
                                leaf_IDs_honest = forest_apply[:, tree]
                                leaf_IDs_all = forest_apply_all[:, tree]
                                # Take care of cases where not all train leafs
                                # populated by observations from honest sample
                                leaf_IDs_honest_u = np.unique(leaf_IDs_honest)
                                leaf_IDs_all_u = np.unique(leaf_IDs_all)
                                if np.array_equal(leaf_IDs_honest_u, 
                                                  leaf_IDs_all_u):
                                    leaf_IDs_honest_ext = leaf_IDs_honest
                                    leaf_IDs_all_ext = leaf_IDs_all
                                else:
                                    # Find leaf IDs in all that are not in honest
                                    extra_honest = np.setdiff1d(
                                        leaf_IDs_all_u, leaf_IDs_honest_u)
                                    leaf_IDs_honest_ext = np.append(
                                        leaf_IDs_honest, extra_honest)
                                    # Find leaf IDs in honest that are not in all
                                    extra_all = np.setdiff1d(
                                        leaf_IDs_honest_u, leaf_IDs_all_u)
                                    leaf_IDs_all_ext = np.append(
                                        leaf_IDs_all, extra_all)
                                # Generate onehot matrices
                                onehot_honest = OneHotEncoder(
                                    sparse=True).fit_transform(
                                        leaf_IDs_honest_ext.reshape(-1, 1)).T
                                onehot_all = OneHotEncoder(
                                    sparse=True).fit_transform(
                                        leaf_IDs_all_ext.reshape(-1, 1))
                                onehot_all = onehot_all[:n_samples,:]
                                # Multiply matrices
                                # (n, n_leafs)x(n_leafs, n_est)
                                tree_out = onehot_all.dot(onehot_honest).todense()
                                # Get leaf sizes
                                # leaf size only for honest sample !!!
                                leaf_size = tree_out.sum(axis=1)
                                # Delete extra observations for unpopulated
                                # honest leaves
                                if not np.array_equal(
                                        leaf_IDs_honest_u, leaf_IDs_all_u):
                                    tree_out = tree_out[:n_samples, :n_est]
                                # Compute weights
                                tree_out = tree_out/leaf_size
                                # add tree weights to overall forest weights
                                forest_out = forest_out + tree_out
                                
                        
# =============================================================================
#                         # Loop over trees (via loops)
#                         for tree in range(self.n_estimators):
#                             # extract vectors of leaf IDs
#                             leaf_IDs_honest = forest_apply[:, tree]
#                             leaf_IDs_all = forest_apply_all[:, tree]
#                             # Compute leaf sizes in honest sample
#                             unique, counts = np.unique(
#                                 leaf_IDs_honest, return_counts=True)
#                             # generate storage matrices for weights
#                             tree_out = np.empty((n_samples, n_est))
#                             # Loop over sample of evaluation
#                             for i in range(n_samples):
#                                 # Loop over honest sample
#                                 for j in range(n_est):
#                                     # If leaf indices coincide...
#                                     if (leaf_IDs_all[i] ==
#                                             leaf_IDs_honest[j]):
#                                         # ... assign 1 to weight matrix
#                                         tree_out[i, j] = 1
#                                     # else assign 0
#                                     else:
#                                         tree_out[i, j] = 0
#                                 # Compute number of observations in this
#                                 # leaf in the honest sample
#                                 # leaf_size = np.sum(tree_out[i, :])
#                                 leaf_size = counts[np.where(
#                                     unique == leaf_IDs_all[i])]
#                                 # If leaf size > 0 divide by leaf size
#                                 if leaf_size > 0:
#                                     tree_out[i, :] = (
#                                         tree_out[i, :] / leaf_size)
#                             # add tree weights to overall forest weights
#                             forest_out += tree_out
# =============================================================================

# =============================================================================
#                         # generate storage matrix for weights
#                         n_tr = len(ind_tr)
#                         forest_out_train = np.zeros((n_tr, n_est))
#                         forest_out_honest = np.zeros((n_est, n_est))
#
#                         # Loop over trees (via loops)
#                         for tree in range(self.n_estimators):
#                             # extract vectors of leaf IDs
#                             leaf_IDs_honest = forest_apply[:, tree]
#                             leaf_IDs_train = forest_apply_tr[:, tree]
#                             # Compute leaf sizes in honest sample
#                             unique, counts = np.unique(
#                                 leaf_IDs_honest, return_counts=True)
#                             # train sample
#                             # generate storage matrices for weights
#                             tree_out_train = np.empty((n_tr, n_est))
#                             # Loop over train sample
#                             for i in range(n_tr):
#                                 # Loop over honest sample
#                                 for j in range(n_est):
#                                     # If leaf indices coincide...
#                                     if (leaf_IDs_train[i] ==
#                                             leaf_IDs_honest[j]):
#                                         # ... assign 1 to weight matrix
#                                         tree_out_train[i, j] = 1
#                                     # else assign 0
#                                     else:
#                                         tree_out_train[i, j] = 0
#                                 # Compute number of observations in this
#                                 # leaf in the honest sample
#                                 # leaf_size = np.sum(tree_out[i, :])
#                                 leaf_size = counts[np.where(
#                                     unique == leaf_IDs_train[i])]
#                                 # If leaf size > 0 divide by leaf size
#                                 if leaf_size > 0:
#                                     tree_out_train[i, :] = (
#                                         tree_out_train[i, :] / leaf_size)
#                             # add tree weights to overall forest weights
#                             forest_out_train += tree_out_train
#
#                             # honest sample
#                             # generate storage matrices for weights
#                             tree_out_honest = np.empty((n_est, n_est))
#                             # Loop over train sample
#                             for i in range(n_tr):
#                                 # Loop over honest sample
#                                 for j in range(n_est):
#                                     # If leaf indices coincide...
#                                     if (leaf_IDs_honest[i] ==
#                                             leaf_IDs_honest[j]):
#                                         # ... assign 1 to weight matrix
#                                         tree_out_honest[i, j] = 1
#                                     # else assign 0
#                                     else:
#                                         tree_out_honest[i, j] = 0
#                                 # Compute number of observations in this
#                                 # leaf in the honest sample
#                                 # leaf_size = np.sum(tree_out[i, :])
#                                 leaf_size = counts[np.where(
#                                     unique == leaf_IDs_honest[i])]
#                                 # If leaf size > 0 divide by leaf size
#                                 if leaf_size > 0:
#                                     tree_out_honest[i, :] = (
#                                         tree_out_honest[i, :] / leaf_size)
#                             # add tree weights to overall forest weights
#                             forest_out_honest += tree_out_honest
#
#                         # combine train and honest sample
#                         forest_out = np.vstack((forest_out_honest,
#                                                 forest_out_train))
#                         # Combine indices
#                         ind_all = np.hstack((ind_est, ind_tr))
#                         # Sort forest_out according to indices in ind_all
#                         forest_out = forest_out[ind_all.argsort(), :]
# =============================================================================

                        # Divide by the number of trees to obtain final weights
                        forest_out = forest_out / self.n_estimators
                        # Compute predictions and assign to probs vector
                        predictions = np.dot(
                            forest_out, outcome_ind_est)
                        probs[:, class_idx-1] = np.asarray(
                            predictions.T).reshape(-1)
                        # Save weights matrix
                        weights[class_idx] = forest_out
                    else:
                        # Check whether to use cython implementation or not
                        if self.pred_method == 'cython':
                            # Loop over trees
                            leaf_means = Parallel(
                                n_jobs=self.n_jobs, prefer="threads")(
                                    delayed(honest_fit.honest_fit)(
                                        forest_apply=forest_apply,
                                        outcome_ind_est=outcome_ind_est,
                                        trees=tree,
                                        max_id=max_id) for tree in range(
                                            0, self.n_estimators))
                            # assign honest predictions (honest fitted values)
                            fitted[class_idx] = np.vstack(leaf_means).T

                        # Check whether to use loop implementation or not
                        if self.pred_method == 'loop':
                            # Loop over trees
                            leaf_means = Parallel(
                                n_jobs=self.n_jobs,
                                backend="loky")(
                                    delayed(self.honest_fit_func)(
                                        tree=tree,
                                        forest_apply=forest_apply,
                                        outcome_ind_est=outcome_ind_est,
                                        max_id=max_id) for tree in range(
                                            0, self.n_estimators))
                            # assign honest predictions (honest fitted values)
                            fitted[class_idx] = np.vstack(leaf_means).T

                        # Check whether to use multiprocessing or not
                        if self.pred_method == 'loop_multi':
                            # setup the pool for multiprocessing
                            pool = Pool(self.n_jobs)
                            # prepare iterables (need to replicate fixed items)
                            args_iter = []
                            for tree in range(self.n_estimators):
                                args_iter.append((tree, forest_apply,
                                                  outcome_ind_est, max_id))
                            # loop over trees in parallel
                            leaf_means = pool.starmap(honest_fit_func_out,
                                                      args_iter)
                            pool.close()  # close parallel
                            pool.join()  # join parallel
                            # assign honest predictions (honest fitted values)
                            fitted[class_idx] = np.vstack(leaf_means).T

                        # Check whether to use numpy implementation or not
                        if self.pred_method == 'numpy':
                            # https://stackoverflow.com/questions/36960320
                            # Create 3Darray of dim(n_est, n_trees, max_id)
                            onehot = np.zeros(
                                forest_apply.shape + (max_id,),
                                dtype=np.uint8)
                            grid = np.ogrid[tuple(map(
                                slice, forest_apply.shape))]
                            grid.insert(2, forest_apply)
                            onehot[tuple(grid)] = 1
                            # onehot = np.eye(max_id)[forest_apply]
                            # Compute leaf sums for each leaf
                            leaf_sums = np.einsum('kji,k->ij', onehot,
                                                  outcome_ind_est)
                            # convert 0s to nans
                            leaf_sums = leaf_sums.astype(float)
                            leaf_sums[leaf_sums == 0] = np.nan
                            # Determine number of observations per leaf
                            leaf_n = sum(onehot).T
                            # convert 0s to nans
                            leaf_n = leaf_n.astype(float)
                            leaf_n[leaf_n == 0] = np.nan
                            # Compute leaf means for each leaf
                            leaf_means = leaf_sums/leaf_n
                            # convert nans back to 0s
                            leaf_means = np.nan_to_num(leaf_means)
                            # assign the honest predictions, i.e. fitted values
                            fitted[class_idx] = leaf_means

                        if self.pred_method == 'numpy_sparse':
                            # Create 3Darray of dim(n_est, n_trees, max_id)
                            onehot = OneHotEncoder(sparse=True).fit(
                                forest_apply)
                            names = onehot.get_feature_names(
                                input_features=np.arange(
                                    self.n_estimators).astype('str'))
                            onehot = onehot.transform(forest_apply)
                            # Compute leaf sums for each leaf
                            leaf_sums = onehot.T.dot(outcome_ind_est)
                            # Determine number of observations per leaf
                            leaf_n = onehot.sum(axis=0)
                            # Compute leaf means for each leaf
                            leaf_means_vec = (leaf_sums/leaf_n).T
                            # get tree and leaf IDs from names
                            ID = np.char.split(names.astype('str_'), sep='_')
                            ID = np.stack(ID, axis=0).astype('int')
                            # Generate container matrix to store leaf means
                            leaf_means = np.zeros((max_id, self.n_estimators))
                            # Assign leaf means to matrix according to IDs
                            leaf_means[ID[:, 1], ID[:, 0]] = np.squeeze(
                                leaf_means_vec)
                            # assign the honest predictions, i.e. fitted values
                            fitted[class_idx] = leaf_means

                        if self.pred_method == 'numpy_sparse2':
                            # Create 3D array of dim(n_est, n_trees, max_id)
                            onehot = OneHotEncoder(
                                sparse=True,
                                categories=([range(max_id)] *
                                            self.n_estimators)).fit(
                                forest_apply).transform(forest_apply)
                            # Compute leaf sums for each leaf
                            leaf_sums = onehot.T.dot(outcome_ind_est)
                            # Determine number of observations per leaf
                            leaf_n = np.asarray(onehot.sum(axis=0))
                            # convert 0s to nans to avoid division by 0
                            leaf_n[leaf_n == 0] = np.nan
                            # Compute leaf means for each leaf
                            leaf_means_vec = (leaf_sums/leaf_n).T
                            # convert nans back to 0s
                            leaf_means_vec = np.nan_to_num(leaf_means_vec)
                            # reshape to array of dim(max_id, n_estimators)
                            leaf_means = np.reshape(
                                leaf_means_vec, (-1, max_id)).T
                            # assign the honest predictions, i.e. fitted values
                            fitted[class_idx] = leaf_means

                        # if self.pred_method == 'numpy_loop':
                        #     # Loop over trees
                        #     leaf_means = Parallel(n_jobs=self.n_jobs,
                        #                           backend="loky")(
                        #         delayed(self.honest_fit_numpy_func)(
                        #             tree=tree,
                        #             forest_apply=forest_apply,
                        #             outcome_ind_est=outcome_ind_est,
                        #             max_id=max_id) for tree in range(
                        #                 0, self.n_estimators))
                        #     # assign honest predictions, i.e. fitted values
                        #     fitted[class_idx] = np.vstack(leaf_means).T
                            
                        if self.pred_method == 'numpy_loop':
                            # Loop over trees
                            with parallel_backend('threading', n_jobs=self.n_jobs):
                                leaf_means = Parallel()(
                                    delayed(self.honest_fit_numpy_func)(
                                    tree=tree,
                                    forest_apply=forest_apply,
                                    outcome_ind_est=outcome_ind_est,
                                    max_id=max_id) for tree in range(
                                        0, self.n_estimators))
                            # assign honest predictions, i.e. fitted values
                            fitted[class_idx] = np.vstack(leaf_means).T
                            
                        if self.pred_method == 'numpy_loop_ray':
                            # Loop over trees
                            leaf_means = (ray.get(
                                [honest_fit_numpy_func_out.remote(
                                    tree=tree,
                                    forest_apply=forest_apply,
                                    outcome_ind_est=outcome_ind_est,
                                    max_id=max_id) for tree in range(
                                        0, self.n_estimators)]))
                            # assign honest predictions, i.e. fitted values
                            fitted[class_idx] = np.vstack(leaf_means).T


                        # Check whether to use multiprocessing or not
                        if self.pred_method == 'numpy_loop_multi':
                            # setup the pool for multiprocessing
                            pool = Pool(self.n_jobs)
                            # prepare iterables (need to replicate fixed items)
                            args_iter = []
                            for tree in range(self.n_estimators):
                                args_iter.append((tree, forest_apply,
                                                  outcome_ind_est, max_id))
                            # loop over trees in parallel
                            leaf_means = pool.starmap(
                                honest_fit_numpy_func_out, args_iter)
                            pool.close()  # close parallel
                            pool.join()  # join parallel
                            # assign honest predictions (honest fitted values)
                            fitted[class_idx] = np.vstack(leaf_means).T


                        if self.pred_method == 'numpy_loop_mpire':
                            # define partial function by fixing parameters
                            partial_fun = partial(
                                self.honest_fit_numpy_func,
                                forest_apply=forest_apply,
                                outcome_ind_est=outcome_ind_est,
                                max_id=max_id)
                            # set up the worker pool for parallelization
                            pool = WorkerPool(n_jobs=self.n_jobs)
                            # setup the pool for multiprocessing
                            # pool = Pool(self.n_jobs)
                            # loop over trees in parallel
                            leaf_means = pool.map(
                                partial_fun, range(self.n_estimators),
                                progress_bar=False,
                                concatenate_numpy_output=False)
                            # stop and join pool
                            pool.stop_and_join()
                            # pool.close()  # close parallel
                            # pool.join()  # join parallel
                            # assign honest predictions (honest fitted values)
                            fitted[class_idx] = np.vstack(leaf_means).T

                        # Compute predictions for whole sample: both tr and est
                        # Get leaf IDs for the whole set of observations
                        forest_apply = forests[class_idx].apply(X)
                        # generate grid to read out indices column by column
                        grid = np.meshgrid(np.arange(0, self.n_estimators),
                                           np.arange(0, X.shape[0]))[0]
                        # assign leaf means to indices
                        y_hat = fitted[class_idx][forest_apply, grid]
                        # Average over trees
                        probs[:, class_idx-1] = np.mean(y_hat, axis=1)
        # create 2 distinct matrices with zeros and ones for easy subtraction
        # prepend vector of zeros
        probs_0 = np.hstack((np.zeros((n_samples, 1)), probs))
        # postpend vector of ones
        probs_1 = np.hstack((probs, np.ones((n_samples, 1))))
        # difference out the adjacent categories to singleout the class probs
        class_probs = probs_1 - probs_0
        # check if some probabilities become negative and set them to zero
        class_probs[class_probs < 0] = 0
        # normalize predictions to sum up to 1 after non-negativity correction
        class_probs = class_probs / class_probs.sum(axis=1).reshape(-1, 1)
        
        # Don't transform to pandas df!!! Uncomment next two rows to use "old"
        # predict function
        # labels = ['Class ' + str(c_idx) for c_idx in range(1, nclass + 1)]
        # class_probs = pd.DataFrame(class_probs, columns=labels)
        
        # Compute variance of predicitons if inference = True
        # outcome need to come from the honest sample here, outcome_binary_est
        if self.inference:
            # prepare honest sample
            probs_honest = probs[ind_est, :]
            weights_honest = dict([(key, weights[key][ind_est, :])
                                   for key in range(1, nclass, 1)])
            # compute variance
            variance_honest = self.honest_variance(
                probs=probs_honest, weights=weights_honest,
                outcome_binary=outcome_binary_est, nclass=nclass, n_est=n_est)
            # prepare train sample
            n_tr = len(ind_tr)
            probs_train = probs[ind_tr, :]
            weights_train = dict([(key, weights[key][ind_tr, :])
                                  for key in range(1, nclass, 1)])
            # compute variance
            variance_train = self.honest_variance(
                probs=probs_train, weights=weights_train,
                outcome_binary=outcome_binary_est, nclass=nclass, n_est=n_tr)
            # put honest and train variance together
            variance = np.vstack((variance_honest, variance_train))
            # Combine indices
            ind_all = np.hstack((ind_est, ind_tr))
            # Sort variance according to indices in ind_all
            variance = variance[ind_all.argsort(), :]

# =============================================================================
#             variance = self.get_honest_variance(
#                 probs=probs, weights=weights,
#                 outcome_binary=outcome_binary_est,
#                 nclass=nclass, ind_tr=ind_tr, ind_est=ind_est)
# =============================================================================
        else:
            variance = {}
        # pack estimated forest and class predictions into output dictionary
        self.forest = {'forests': forests,
                       'probs': class_probs,
                       'fitted': fitted,
                       'outcome_binary_est': outcome_binary_est,
                       'variance': variance,
                       'X_fit': X,
                       'y_fit': y,
                       'ind_tr': ind_tr,
                       'ind_est': ind_est,
                       'weights': weights}
        # compute prediction performance
        self.__performance(y, y_values)
        # check if performance metrics should be printed
        if verbose:
            self.performance()

        # return the output
        return self
    
    
    # %% Performance functions
    # performance measures (private method, not available to user)
    def __performance(self, y, y_values):
        """
        Evaluate the prediction performance using MSE and CA.

        Parameters
        ----------
        y : TYPE: pd.Series
            DESCRIPTION: vector of outcomes.

        Returns
        -------
        None. Calculates MSE, Classification accuracy and confusion matrix.
        """
        # take over needed values
        predictions = self.forest['probs']

        # compute the mse: version 1
        # create storage empty dataframe
        mse_matrix = np.zeros(predictions.shape)
        # allocate indicators for true outcome and leave zeros for the others
        # minus 1 for the column index as indices start with 0, outcomes with 1
        mse_matrix[np.arange(y.shape[0]),y-1] = 1
        # compute mse directly now by substracting two dataframes and rowsums
        mse_1 = np.mean(((mse_matrix - predictions) ** 2).sum(axis=1))

        # compute the mse: version 2
        # create storage for modified predictions
        modified_pred = np.zeros(y.shape[0])
        # modify the predictions with 1*P(1)+2*P(2)+3*P(3) as an alternative
        modified_pred = np.dot(predictions, np.arange(
            start=1, stop=predictions.shape[1]+1))
        # Compute MSE
        mse_2 = np.mean((y - modified_pred) ** 2)

        # compute classification accuracy
        # define classes with highest probability (+1 as index starts with 0)
        class_pred = predictions.argmax(axis=1) + 1
        # the accuracy directly now by mean of matching classes
        acc = np.mean(y == class_pred)

        # create te confusion matrix
        # First generate onehot matrices of y and class_pred        
        y_onehot = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
        class_pred_onehot = OneHotEncoder(sparse=False).fit_transform(
            class_pred.reshape(-1, 1))
        # Compute dot product of these matrices to obtain confusion matrix
        confusion_mat = np.dot(np.transpose(y_onehot), class_pred_onehot)
        labels = ['Class ' + str(c_idx) for c_idx in y_values]
        self.confusion = pd.DataFrame(confusion_mat, 
                                      index=labels, columns=labels)

        # wrap the results into a dataframe
        self.measures = pd.DataFrame({'mse 1': mse_1, 'mse 2': mse_2,
                                      'accuracy': acc}, index=['value'])

        # empty return
        return None

    # performance measures (public method, available to user)
    def performance(self):
        """
        Print the prediction performance based on MSE and CA.

        Parameters
        ----------
        None.

        Returns
        -------
        None. Prints MSE, Classification accuracy and confusion matrix.
        """
        # print the result
        print('Prediction Performance of Ordered Forest', '-' * 80,
              self.measures, '-' * 80, '\n\n', sep='\n')

        # print the confusion matrix
        print('Confusion Matrix for Ordered Forest', '-' * 80,
              '                         Predictions ', '-' * 80,
              self.confusion, '-' * 80, '\n\n', sep='\n')

        # empty return
        return None

# Not needed anymore, use sklearn check_X_y instead
# =============================================================================
#     # check user input for covariates (private method, not available to user)
#     def __xcheck(self, X):
#         """
#         Check the user input for the pandas dataframe of covariates.
# 
#         Parameters
#         ----------
#         X : TYPE: pd.DataFrame
#             DESCRIPTION: matrix of covariates.
# 
#         Returns
#         -------
#         None. Checks for the correct user input.
#         """
#         # check if features X are a pandas dataframe
#         if isinstance(X, pd.DataFrame):
#             # check if its non-empty
#             if X.empty:
#                 # raise value error
#                 raise ValueError("X DataFrame is empty. Check the input.")
#         else:
#             # raise value error
#             raise ValueError("X is not a Pandas DataFrame. Recode the input.")
# 
#         # empty return
#         return None
# =============================================================================

    # %% In-class honesty and weight functions
    def honest_fit_func(self, tree, forest_apply, outcome_ind_est, max_id):
        """Compute the honest leaf means using loop."""
        # create an empty array to save the leaf means
        leaf_means = np.empty(max_id)
        # loop over leaf indices
        for idx in range(0, max_id):
            # get row numbers of obs with this leaf index
            row_idx = np.where(forest_apply[:, tree] == idx)
            # Compute mean of outcome of these obs
            if row_idx[0].size == 0:
                leaf_means[idx] = 0
            else:
                leaf_means[idx] = np.mean(outcome_ind_est[row_idx])
        return leaf_means

    def honest_fit_numpy_func(self, tree, forest_apply, outcome_ind_est,
                              max_id):
        """Compute the honest leaf means using numpy."""
        # create an empty array to save the leaf means
        leaf_means = np.zeros(max_id)
        # Create dummy matrix dim(n_est, max_id)
        onehot = OneHotEncoder(sparse=True).fit_transform(
            forest_apply[:, tree].reshape(-1, 1))
        # Compute leaf sums for each leaf
        leaf_sums = onehot.T.dot(outcome_ind_est)
        # Determine number of observations per leaf
        leaf_n = onehot.sum(axis=0)
        # Compute leaf means for each leaf
        leaf_means[np.unique(forest_apply[:, tree])] = leaf_sums/leaf_n
        return leaf_means

    def honest_weight_numpy(self, tree, forest_apply, forest_apply_all,
                            n_samples, n_est):
        """Compute the honest weights using numpy."""
        # extract vectors of leaf IDs
        leaf_IDs_honest = forest_apply[:, tree]
        leaf_IDs_all = forest_apply_all[:, tree]
        # Take care of cases where not all train leafs
        # populated by observations from honest sample
        leaf_IDs_honest_u = np.unique(leaf_IDs_honest)
        leaf_IDs_all_u = np.unique(leaf_IDs_all)
        if np.array_equal(leaf_IDs_honest_u, 
                          leaf_IDs_all_u):
            leaf_IDs_honest_ext = leaf_IDs_honest
            leaf_IDs_all_ext = leaf_IDs_all
        else:
            # Find leaf IDs in all that are not in honest
            extra_honest = np.setdiff1d(
                leaf_IDs_all_u, leaf_IDs_honest_u)
            leaf_IDs_honest_ext = np.append(
                leaf_IDs_honest, extra_honest)
            # Find leaf IDs in honest that are not in all
            extra_all = np.setdiff1d(
                leaf_IDs_honest_u, leaf_IDs_all_u)
            leaf_IDs_all_ext = np.append(
                leaf_IDs_all, extra_all)
        # Generate onehot matrices
        onehot_honest = OneHotEncoder(
            sparse=True).fit_transform(
                leaf_IDs_honest_ext.reshape(-1, 1)).T
        onehot_all = OneHotEncoder(
            sparse=True).fit_transform(
                leaf_IDs_all_ext.reshape(-1, 1))
        onehot_all = onehot_all[:n_samples,:]
        # Multiply matrices
        # (n, n_leafs)x(n_leafs, n_est)
        tree_out = onehot_all.dot(onehot_honest).todense()
        # Get leaf sizes
        # leaf size only for honest sample !!!
        leaf_size = tree_out.sum(axis=1)
        # Delete extra observations for unpopulated
        # honest leaves
        if not np.array_equal(
                leaf_IDs_honest_u, leaf_IDs_all_u):
            tree_out = tree_out[:n_samples, :n_est]
        # Compute weights
        tree_out = tree_out/leaf_size
        return tree_out



    




# =============================================================================
#     def honest_weight_numpy(self, n_tree, forest_out, forest_apply, forest_apply_all,
#                             n_samples, n_est):
#         """Compute the honest weights using numpy."""
#         # generate storage matrix for weights
#         forest_out = np.zeros((n_samples, n_est))
#         # Loop over trees
#         for tree in range(n_tree):
#             # extract vectors of leaf IDs
#             leaf_IDs_honest = forest_apply[:, tree]
#             leaf_IDs_all = forest_apply_all[:, tree]
#             # Take care of cases where not all training leafs
#             # populated by observations from honest sample
#             leaf_IDs_honest_u = np.unique(leaf_IDs_honest)
#             leaf_IDs_all_u = np.unique(leaf_IDs_all)
#             if (leaf_IDs_honest_u.size == leaf_IDs_all_u.size):
#                 leaf_IDs_honest_ext = leaf_IDs_honest
#             else:
#                 extra = np.setxor1d(leaf_IDs_all_u,
#                                     leaf_IDs_honest_u)
#                 leaf_IDs_honest_ext = np.append(
#                     leaf_IDs_honest, extra)
#             # Generate onehot matrices
#             onehot_honest = OneHotEncoder(
#                 sparse=True).fit_transform(
#                     leaf_IDs_honest_ext.reshape(-1, 1)).T
#             onehot_all = OneHotEncoder(
#                 sparse=True).fit_transform(
#                     leaf_IDs_all.reshape(-1, 1))
#             # Multiply matrices (n, n_leafs)x(n_leafs, n_est)
#             tree_out = onehot_all.dot(onehot_honest).todense()
#             # Get leaf sizes
#             # leaf size only for honest sample !!!
#             leaf_size = tree_out.sum(axis=1)
#             # Delete extra observations for unpopulated honest
#             # leaves
#             if not leaf_IDs_honest_u.size == leaf_IDs_all_u.size:
#                 tree_out = tree_out[:n_samples, :n_est]
#             # Compute weights
#             tree_out = tree_out/leaf_size
#             # add tree weights to overall forest weights
#             forest_out = forest_out + tree_out
#             return forest_out
# =============================================================================


    # Function to compute variance of predictions.
    # -> Does the N in the formula refer to n_samples or to n_est?
    def honest_variance(self, probs, weights, outcome_binary, nclass, n_est):
        """Compute the variance of predictions (out-of-sample)."""
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
        honest_variance_final = np.empty((probs.shape[0], nclass))
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
    def get_honest_variance(self, probs, weights, outcome_binary, nclass,
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
                weights[class_idx][ind_est, :],
                outcome_binary[class_idx].reshape((1, -1)))
            # train sample
            train_multi = np.multiply(
                weights[class_idx][ind_tr, :],
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

# %% Out-of-class honesty and weight functions (for parallelization)
# define function outside of the class for speedup of multiprocessing
def honest_fit_func_out(tree, forest_apply, outcome_ind_est, max_id):
    """Compute the honest leaf means using loop."""
    # create an empty array to save the leaf means
    leaf_means = np.empty(max_id)
    # loop over leaf indices
    for idx in range(0, max_id):
        # get row numbers of obs with this leaf index
        row_idx = np.where(forest_apply[:, tree] == idx)
        # Compute mean of outcome of these obs
        if row_idx[0].size == 0:
            leaf_means[idx] = 0
        else:
            leaf_means[idx] = np.mean(outcome_ind_est[row_idx])
    return leaf_means

@ray.remote
def honest_fit_numpy_func_out(tree, forest_apply, outcome_ind_est, max_id):
    """Compute the honest leaf means using numpy."""
    # create an empty array to save the leaf means
    leaf_means = np.zeros(max_id)
    # Create dummy matrix dim(n_est, max_id)
    onehot = OneHotEncoder(sparse=True).fit_transform(
        forest_apply[:, tree].reshape(-1, 1))
    # Compute leaf sums for each leaf
    leaf_sums = onehot.T.dot(outcome_ind_est)
    # Determine number of observations per leaf
    leaf_n = onehot.sum(axis=0)
    # Compute leaf means for each leaf
    leaf_means[np.unique(forest_apply[:, tree])] = leaf_sums/leaf_n
    return leaf_means


def honest_weight_numpy_out(tree, forest_apply, forest_apply_all, n_samples,
                            n_est):
    """Compute the honest weights using numpy."""
    # extract vectors of leaf IDs
    leaf_IDs_honest = forest_apply[:, tree]
    leaf_IDs_all = forest_apply_all[:, tree]
    # Take care of cases where not all train leafs
    # populated by observations from honest sample
    leaf_IDs_honest_u = np.unique(leaf_IDs_honest)
    leaf_IDs_all_u = np.unique(leaf_IDs_all)
    if np.array_equal(leaf_IDs_honest_u, 
                      leaf_IDs_all_u):
        leaf_IDs_honest_ext = leaf_IDs_honest
        leaf_IDs_all_ext = leaf_IDs_all
    else:
        # Find leaf IDs in all that are not in honest
        extra_honest = np.setdiff1d(
            leaf_IDs_all_u, leaf_IDs_honest_u)
        leaf_IDs_honest_ext = np.append(
            leaf_IDs_honest, extra_honest)
        # Find leaf IDs in honest that are not in all
        extra_all = np.setdiff1d(
            leaf_IDs_honest_u, leaf_IDs_all_u)
        leaf_IDs_all_ext = np.append(
            leaf_IDs_all, extra_all)
    # Generate onehot matrices
    onehot_honest = OneHotEncoder(
        sparse=True).fit_transform(
            leaf_IDs_honest_ext.reshape(-1, 1)).T
    onehot_all = OneHotEncoder(
        sparse=True).fit_transform(
            leaf_IDs_all_ext.reshape(-1, 1))
    onehot_all = onehot_all[:n_samples,:]
    # Multiply matrices
    # (n, n_leafs)x(n_leafs, n_est)
    tree_out = onehot_all.dot(onehot_honest).todense()
    # Get leaf sizes
    # leaf size only for honest sample !!!
    leaf_size = tree_out.sum(axis=1)
    # Delete extra observations for unpopulated
    # honest leaves
    if not np.array_equal(
            leaf_IDs_honest_u, leaf_IDs_all_u):
        tree_out = tree_out[:n_samples, :n_est]
    # Compute weights
    tree_out = tree_out/leaf_size
    return tree_out


# multiprocessing with shared memory
_lock = Lock()  # initiate lock


# define tree weight function in shared memory
def tree_weights(_shared_buffer, tree, forest_apply, forest_apply_all,
                 n_samples, n_est):
    # get the tree weights
    tree_out = honest_weight_numpy_out(tree, forest_apply, forest_apply_all,
                                       n_samples, n_est)
    _lock.acquire()
    _shared_buffer += tree_out  # update the buffer with tree weights
    _lock.release()


# define forest weights function in shared memory using multiprocessing
def forest_weights_multi(partial_fun, n_samples, n_est, n_jobs, n_estimators):
    # initiate output in shared memory
    forest_out = sharedmem.empty((n_samples, n_est), dtype=np.float64)
    pool = Pool(n_jobs)  # start the multiprocessing pool
    pool.starmap(partial_fun, [(forest_out, _) for _ in range(n_estimators)])
    pool.close()  # close parallel
    pool.join()  # join parallel
    return forest_out


# # define forest weights function in shared memory using mpire (faster)
def forest_weights_mpire(partial_fun, n_samples, n_est, n_jobs, n_estimators):
    # initiate output in shared memory
    forest_out = sharedmem.empty((n_samples, n_est), dtype=np.float64)
    pool = WorkerPool(n_jobs)  # start the mpire pool
    pool.map(partial_fun, [(forest_out, _) for _ in range(n_estimators)])
    pool.stop_and_join()  # stop and join pool
    return forest_out


class OrderedForest(BaseOrderedForest):
    """
    Base class for forests of trees.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    # define init function
    def __init__(self, n_estimators=1000,
                 min_samples_leaf=5,
                 max_features=0.3,
                 replace=True,
                 sample_fraction=0.5,
                 honesty=False,
                 honesty_fraction=0.5,
                 inference=False,
                 n_jobs=-1,
                 pred_method='numpy_loop_mpire',
                 weight_method='numpy_loop_shared_mpire',
                 random_state=None):
        # access inherited methods
        super().__init__(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            replace=replace,
            sample_fraction=sample_fraction,
            honesty=honesty,
            honesty_fraction=honesty_fraction,
            inference=inference,
            n_jobs=n_jobs,
            pred_method=pred_method,
            weight_method=weight_method,
            random_state=random_state
        )
    
    # %% Predict function
    # function to predict with estimated ordered forest
    def predict(self, X=None, prob=True):
        """
        Ordered Forest prediction.

        Parameters
        ----------
        X : TYPE: array-like or NoneType
            DESCRIPTION: Matrix of new covariates or None if covariates from
            fit function should be used. If new data provided it must have
            the same number of features as the X in the fit function.
        prob : TYPE: bool
            DESCRIPTION: Should the ordered probabilities be predicted?
            If False, ordered classes will be predicted instead.
            Default is True.

        Returns
        -------
        result: Tuple of ordered probability predictions by Ordered Forest
        and respective variances.
        """
        # Input checks
        # check if input has been fitted (sklearn function)
        check_is_fitted(self, attributes=["forest"])
        
        # Check if X defined properly (sklearn function)
        if not X is None:
            X = check_array(X)
            # Check if number of variables matches with input in fit
            if not X.shape[1]==self.n_features:
                raise ValueError("Number of features (covariates) should be "
                                 "%s but got %s. Provide \narray with the same"
                                 " number of features as the X in the fit "
                                 "function." % (self.n_features,X.shape[1]))
            # get the number of observations in X
            n_samples = _num_samples(X)
            # Check if provided X exactly matches X used in fit function
            if np.array_equal(X, self.forest['X_fit']):
                X = None
        else:
            n_samples = _num_samples(self.forest['X_fit'])
        
        # check whether to predict probabilities or classes
        if isinstance(prob, bool):
            # assign the input value
            self.prob = prob
        else:
            # raise value error
            raise ValueError("prob must be of type boolean"
                             ", got %s" % prob)
        
        # get the forest inputs
        outcome_binary_est = self.forest['outcome_binary_est']
        probs = self.forest['probs']
        variance = self.forest['variance']
        # get the number of outcome classes
        nclass = self.n_class
        # get inference argument
        inference = self.inference
        
        # Check if prob allows to do inference
        if ((not prob) and (inference) and (X is not None)):
            print('-' * 70, 
                  'WARNING: Inference is not possible if prob=False.' 
                  '\nClass predictions for large samples might be obtained faster '
                  '\nwhen re-estimating OrderedForest with option inference=False.', 
                  '-' * 70, sep='\n')

        # Initialize final variance output
        var_final = None
        # Initialize storage dictionary for weights
        weights = {}
        
        # Get fitted values if X = None
        if X is None:
            # Check desired type of predictions
            if prob:
                # Take in-sample predictions and variance
                pred_final = probs
                var_final = variance
            else:
                # convert in-sample probabilities into class predictions 
                # ("ordered classification")
                pred_final = probs.argmax(axis=1) + 1
        # Remaining case: X is not None
        else:    
            # If honesty has not been used, used standard predict function
            # from sklearn or econML
            if not self.honesty:
                probs = self.predict_default(X=X, n_samples=n_samples)
            # If honesty True, inference argument decides how to compute
            # predicions
            elif self.honesty and not inference:
                probs = self.predict_leafmeans(X=X, n_samples=n_samples)
            # Remaining case refers to honesty=True and inference=True
            else:
                probs, weights = self.predict_weights(
                    X=X, n_samples=n_samples)
            # create 2 distinct matrices with zeros and ones for easy subtraction
            # prepend vector of zeros
            probs_0 = np.hstack((np.zeros((n_samples, 1)), probs))
            # postpend vector of ones
            probs_1 = np.hstack((probs, np.ones((n_samples, 1))))
            # difference out the adjacent categories to singleout the class probs
            class_probs = probs_1 - probs_0
            # check if some probabilities become negative and set them to zero
            class_probs[class_probs < 0] = 0
            # normalize predictions to sum up to 1 after non-negativity correction
            class_probs = class_probs / class_probs.sum(axis=1).reshape(-1, 1)
            
            # Check desired type of predictions (applies only to cases where
            # inference = false)
            if prob:
                # Take in-sample predictions and variance
                pred_final = class_probs
            else:
                # convert in-sample probabilities into class predictions 
                # ("ordered classification")
                pred_final = class_probs.argmax(axis=1) + 1
            
            # Last step: Compute variance of predicitons 
            # If flag_newdata = True, variance can be computed in one step.
            # Otherwise use same variance method as in fit function which 
            # accounts for splitting in training and honest sample
            if inference and prob:
                # compute variance
                var_final = self.honest_variance(
                    probs=probs, weights=weights,
                    outcome_binary=outcome_binary_est, nclass=nclass,
                    n_est=len(self.forest['ind_est']))

        # return the class predictions
        result = {'output': 'predict',
                  'prob': prob,
                  'predictions': pred_final,
                  'variances': var_final}
        return result
    
    
    
    
    
    # %% Margin function
    # function to evaluate marginal effects with estimated ordered forest
    def margin(self, X, eval_point="mean", window=0.1, verbose=True):
        """
        Ordered Forest prediction.

        Parameters
        ----------
        X : TYPE: array-like or NoneType
            DESCRIPTION: Matrix of new covariates or None if covariates from
            fit function should be used. If new data provided it must have
            the same number of features as the X in the fit function.
        eval_point: TYPE: string
            DESCRIPTION: defining evaluation point for marginal effects. These
            can be one of "mean", "atmean", or "atmedian". (Default is "mean")
        window : TYPE: float
            DESCRIPTION: share of standard deviation of X to be used for
            evaluation of the marginal effect. Default is 0.1.
        verbose : TYPE: bool
            DESCRIPTION: should be the results printed to console?
            Default is False.
            

        Returns
        -------
        result: Mean marginal effects by Ordered Forest.
        """
        
        # Input checks
        # check if input has been fitted (sklearn function)
        check_is_fitted(self, attributes=["forest"])
        
        # Check if X defined properly (sklearn function)
        if not X is None:
            X = check_array(X)
            # Check if number of variables matches with input in fit
            if not X.shape[1]==self.n_features:
                raise ValueError("Number of features (covariates) should be "
                                 "%s but got %s. Provide \narray with the same"
                                 " number of features as the X in the fit "
                                 "function." % (self.n_features,X.shape[1]))
            # Check if provided X exactly matches X used in fit function
            if np.array_equal(X, self.forest['X_fit']):
                X = None

        # check whether to predict probabilities or classes
        if not isinstance(verbose, bool):
            # raise value error
            raise ValueError("verbose must be of type boolean"
                             ", got %s" % verbose)

        # check the window argument
        if isinstance(window, float):
            # check if its within (0,1]
            if not (window > 0 and window <= 1):
                # raise value error
                raise ValueError("window must be within (0,1]"
                                 ", got %s" % window)
        else:
            # raise value error
            raise ValueError("window must be a float"
                             ", got %s" % window)

        # check whether eval_point is defined correctly
        if isinstance(eval_point, str):
            if not (eval_point == 'mean' or eval_point == 'atmean' 
                or eval_point == 'atmedian'):
                # raise value error
                raise ValueError("eval_point must be one of 'mean', 'atmean' " 
                                 "or 'atmedian', got '%s'" % eval_point)
        else:
            # raise value error
            raise ValueError("eval_point must be of type string"
                             ", got %s" % eval_point)
        
        # get the indices of the honest sample
        ind_est = self.forest['ind_est']
        # get inference argument
        inference = self.inference
        

        ## Prepare data sets
        # check if new data provided or not
        if X is None:
            # if no new data supplied, estimate in-sample marginal effects
            if self.honesty:
                # if using honesty, data refers to the honest sample
                X_eval = self.forest['X_fit'][ind_est,:]
                X_est = self.forest['X_fit'][ind_est,:]
            else:
                # if not using honesty, data refers to the full sample
                X_eval = self.forest['X_fit']
                X_est = self.forest['X_fit']
        else:
            # if new data supplied, need to use this for prediction
            if self.honesty:
                # if using honesty, need to consider new and honest sample
                X_eval = X
                X_est = self.forest['X_fit'][ind_est,:]
            else:
                # if not using honesty, data refers to the new sample
                X_eval = X
                X_est = self.forest['X_fit']
        # get the number of observations in X
        n_samples = _num_samples(X_eval)
        # define the window size share for evaluating the effect
        h_std = window    
        
        # check if X is continuous, dummy or categorical
        # first find number of unique values per column
        X_eval_sort = np.sort(X_eval,axis=0)
        n_unique = (X_eval_sort[1:,:] != X_eval_sort[:-1,:]).sum(axis=0)+1
        # get indices of respective columns
        X_dummy = (n_unique == 2).nonzero()
        X_categorical = ((n_unique > 2) & (n_unique <= 10)).nonzero()
        if np.any(n_unique<= 1):
            # raise value error
            raise ValueError("Some of the covariates are constant. This is "
                             "not allowed for evaluation of marginal effects. "
                             "Programme terminated.")
        
        ## Get the evaluation point(s)
        # Save evaluation point(s) in X_mean
        if eval_point == "atmean":
            X_mean = np.mean(X_eval, axis=0).reshape(1,-1)
        elif eval_point == "atmedian":
            X_mean = np.median(X_eval, axis=0).reshape(1,-1)
        else:
            X_mean = X_eval.copy()
        # Get dimension of evaluation points
        X_rows = np.shape(X_mean)[0]
        X_cols = np.shape(X_mean)[1]
        # Get standard deviation of X_est in the same shape as X_mean
        X_sd = np.repeat(np.std(X_est, axis=0, ddof=1).reshape(1,-1
                                                               ),X_rows, axis=0)
        # create X_up (X_mean + h_std * X_sd)
        X_up = X_mean + h_std*X_sd
        # create X_down (X_mean - h_std * X_sd)
        X_down = X_mean - h_std*X_sd
            
        ## now check if support of X_eval is within X_est
        # check X_max
        X_max = np.repeat(np.max(X_est, axis=0).reshape(1,-1),X_rows, axis=0)
        # check X_min
        X_min = np.repeat(np.min(X_est, axis=0).reshape(1,-1),X_rows, axis=0)
        # check if X_up is within the range X_min and X_max
        # If some X_up is larger than the max in X_est, replace entry in X_up 
        # by this max value of X_est. If some X_up is smaller than the min in
        # X_est, replace entry in X_up by this min value + h_std * X_sd
        X_up = (X_up < X_max) * X_up + (X_up >= X_max) * X_max
        X_up = (X_up > X_min) * X_up + (X_up <= X_min) * (X_min + h_std * X_sd)
        # check if X_down is within the range X_min and X_max
        X_down = (X_down > X_min) * X_down + (X_down <= X_min) * X_min
        X_down = (X_down < X_max) * X_down + (X_down >= X_max) *(
            X_max - h_std * X_sd)
        # check if X_up and X_down are same
        if (np.any(X_up == X_down)):
            # adjust to higher share of SD
            X_up = (X_up > X_down) * X_up + (X_up == X_down) * (
                X_up + 0.5 * h_std * X_sd)
            X_down = (X_up > X_down) * X_down + (X_up == X_down) * (
                X_down - 0.5 * h_std * X_sd)
            # check the min max range again
            X_up = (X_up < X_max) * X_up + (X_up >= X_max) * X_max
            X_down = (X_down > X_min) * X_down + (X_down <= X_min) * X_min
        
        # Adjust for dummies
        X_up[:, X_dummy] = np.max(X_eval[:, X_dummy], axis=0)
        X_down[:, X_dummy] = np.min(X_eval[:, X_dummy], axis=0)
        
        # Adjust for categorical variables
        X_up[:, X_categorical] = np.ceil(X_up[:, X_categorical])
        X_down[:, X_categorical] = X_up[:, X_categorical]-1
        
        ## Compute predictions
        # Create storage arrays to save predictions
        forest_pred_up = np.empty((X_cols, self.n_class-1))
        forest_pred_down = np.empty((X_cols, self.n_class-1))
        # Case 1: No honesty (= no inference)
        if not self.honesty:
            # loop over all covariates
            for x_id in range(X_cols):
                # Prepare input matrix where column x_id is adjusted upwards
                X_mean_up = X_eval.copy()
                X_mean_up[:,x_id] = X_up[:,x_id]
                # Compute mean predictions (only needed for eval_point=mean
                # but no change im atmean or atmedian)
                forest_pred_up[x_id,:] = np.mean(self.predict_default(
                    X=X_mean_up, n_samples=n_samples), axis=0)
                # Prepare input matrix where column x_id is adjusted downwards
                X_mean_down = X_eval.copy()
                X_mean_down[:,x_id] = X_down[:,x_id]
                # Compute mean predictions
                forest_pred_down[x_id,:] = np.mean(self.predict_default(
                    X=X_mean_down, n_samples=n_samples), axis=0)
        if self.honesty and not inference:
            # loop over all covariates
            for x_id in range(X_cols):
                # Prepare input matrix where column x_id is adjusted upwards
                X_mean_up = X_eval.copy()
                X_mean_up[:,x_id] = X_up[:,x_id]
                # Compute mean predictions (only needed for eval_point=mean
                # but no change im atmean or atmedian)
                forest_pred_up[x_id,:] = np.mean(self.predict_leafmeans(
                    X=X_mean_up, n_samples=n_samples), axis=0)
                # Prepare input matrix where column x_id is adjusted downwards
                X_mean_down = X_eval.copy()
                X_mean_down[:,x_id] = X_down[:,x_id]
                # Compute mean predictions
                forest_pred_down[x_id,:] = np.mean(self.predict_leafmeans(
                    X=X_mean_down, n_samples=n_samples), axis=0)
        if self.honesty and inference:
            # storage container for weight matrices
            forest_weights_up={}
            forest_weights_down={}
            # loop over all covariates
            for x_id in range(X_cols):
                # Prepare input matrix where column x_id is adjusted upwards
                X_mean_up = X_eval.copy()
                X_mean_up[:,x_id] = X_up[:,x_id]
                # Compute predictions and weights matrix
                forest_pred_up_x_id, forest_weights_up[x_id] = (
                    self.predict_weights(X=X_mean_up, n_samples=n_samples))
                # Compute mean predictions (only needed for eval_point=mean
                # but no change im atmean or atmedian)
                forest_pred_up[x_id,:] = np.mean(forest_pred_up_x_id, axis=0)
                # Prepare input matrix where column x_id is adjusted downwards
                X_mean_down = X_eval.copy()
                X_mean_down[:,x_id] = X_down[:,x_id]
                # Compute predictions and weights matrix
                forest_pred_down_x_id, forest_weights_down[x_id] = (
                    self.predict_weights(X=X_mean_down, n_samples=n_samples))
                # Compute mean predictions (only needed for eval_point=mean
                # but no change im atmean or atmedian)
                forest_pred_down[x_id,:] = np.mean(
                    forest_pred_down_x_id, axis=0)
            # Compute means of weights
            forest_weights_up = {r: {k: np.mean(v, axis=0) for k,v in 
                                     forest_weights_up[r].items()} for r in 
                                 forest_weights_up.keys()}
            forest_weights_down = {r: {k: np.mean(v, axis=0) for k,v in 
                                     forest_weights_down[r].items()} for r in 
                                 forest_weights_down.keys()}
        # ORF predictions for forest_pred_up
        # create 2 distinct matrices with zeros and ones for easy subtraction
        # prepend vector of zeros
        forest_pred_up_0 = np.hstack((np.zeros((X_cols, 1)), forest_pred_up))
        # postpend vector of ones
        forest_pred_up_1 = np.hstack((forest_pred_up, np.ones((X_cols, 1))))
        # difference out the adjacent categories to singleout the class probs
        forest_pred_up = forest_pred_up_1 - forest_pred_up_0
        # check if some probabilities become negative and set them to zero
        forest_pred_up[forest_pred_up < 0] = 0
        # normalize predictions to sum up to 1 after non-negativity correction
        forest_pred_up = forest_pred_up / forest_pred_up.sum(
            axis=1).reshape(-1, 1)
        # ORF predictions for forest_pred_down
        # create 2 distinct matrices with zeros and ones for easy subtraction
        # prepend vector of zeros
        forest_pred_down_0 = np.hstack((np.zeros((X_cols, 1)), forest_pred_down))
        # postpend vector of ones
        forest_pred_down_1 = np.hstack((forest_pred_down, np.ones((X_cols, 1))))
        # difference out the adjacent categories to singleout the class probs
        forest_pred_down = forest_pred_down_1 - forest_pred_down_0
        # check if some probabilities become negative and set them to zero
        forest_pred_down[forest_pred_down < 0] = 0
        # normalize predictions to sum up to 1 after non-negativity correction
        forest_pred_down = forest_pred_down / forest_pred_down.sum(
            axis=1).reshape(-1, 1)
        
        ## Compute marginal effects from predictions
        # compute difference between up and down (numerator)
        forest_pred_diff_up_down = forest_pred_up - forest_pred_down
        # compute scaling factor (denominator)
        scaling_factor = np.mean(X_up - X_down, axis=0).reshape(-1,1)
        # Set scaling factor to 1 for categorical and dummy variables
        scaling_factor[X_dummy,:] = 1
        scaling_factor[X_categorical,:] = 1
        # Scale the differences to get the marginal effects
        marginal_effects_scaled = forest_pred_diff_up_down / scaling_factor
        
        # redefine all effect results as floats
        margins = marginal_effects_scaled.astype(float)
        
        if inference:
            ## variance for the marginal effects
            # compute prerequisities for variance of honest marginal effects
            # squared scaling factor
            scaling_factor_squared = np.square(scaling_factor)
            # Get the size of the honest sample
            n_est = len(ind_est)
            # Create storage container for variance
            variance_me = np.empty((X_cols, self.n_class))
            # loop over all covariates
            for x_id in range(X_cols):
                # Generate sub-dictionary
                # Create storage containers
                forest_multi_demeaned = {}
                variance = {}
                covariance = {}
                # Loop over classes
                for class_idx in range(1, self.n_class, 1):
                    #subtract the weights according to the ME formula:
                    forest_weights_diff_up_down = (
                        forest_weights_up[x_id][class_idx] - 
                        forest_weights_down[x_id][class_idx])
                    # Get binary outcoms of honest sample
                    outcome_binary_est = self.forest['outcome_binary_est'][
                        class_idx].reshape(-1,1)
                    # compute the conditional means: 1/N(weights%*%y) (predictions are based on honest sample)
                    forest_cond_means = np.multiply(
                        (1/len(self.forest['ind_est'])), np.dot(
                            forest_weights_diff_up_down, outcome_binary_est))
                    # calculate standard multiplication of weights and outcomes
                    forest_multi = np.multiply(
                        forest_weights_diff_up_down, outcome_binary_est.reshape((1, -1)))
                    # subtract the mean from each obs i
                    forest_multi_demeaned[class_idx] = forest_multi - forest_cond_means
                    # compute the square
                    forest_multi_demeaned_sq = np.square(
                        forest_multi_demeaned[class_idx])
                    # sum over all i in honest sample
                    forest_multi_demeaned_sq_sum = np.sum(
                        forest_multi_demeaned_sq, axis=1)
                    # multiply by N/N-1 (normalize)
                    forest_multi_demeaned_sq_sum_norm = (
                        forest_multi_demeaned_sq_sum * (n_est/(n_est-1)))
                    # divide by scaling factor to get the variance
                    variance[class_idx] = (
                        forest_multi_demeaned_sq_sum_norm/
                                scaling_factor_squared[x_id])
                # ### Covariance computation:
                # Shift categories for computational convenience
                # Postpend matrix of zeros
                forest_multi_demeaned_0_last = forest_multi_demeaned
                forest_multi_demeaned_0_last[self.n_class] = np.zeros(
                    forest_multi_demeaned_0_last[1].shape)
                # Prepend matrix of zeros
                forest_multi_demeaned_0_first = {}
                forest_multi_demeaned_0_first[1] = np.zeros(
                    forest_multi_demeaned[1].shape)
                # Shift existing matrices by 1 class
                for class_idx in range(1, self.n_class, 1):
                    forest_multi_demeaned_0_first[
                        class_idx+1] = forest_multi_demeaned[class_idx]
                # Loop over classes
                for class_idx in range(1, self.n_class+1, 1):
                    # multiplication of category m with m-1
                    forest_multi_demeaned_cov = np.multiply(
                        forest_multi_demeaned_0_first[class_idx],
                        forest_multi_demeaned_0_last[class_idx])
                    # sum all obs i in honest sample
                    forest_multi_demeaned_cov_sum = np.sum(
                        forest_multi_demeaned_cov, axis=1)
                    # multiply by (N/N-1)*2
                    forest_multi_demeaned_cov_sum_norm_mult2 = (
                        forest_multi_demeaned_cov_sum*2*(
                        n_est/(n_est-1)))
                    # divide by scaling factor to get the covariance
                    covariance[class_idx] = (
                        forest_multi_demeaned_cov_sum_norm_mult2/
                        scaling_factor_squared[x_id])
                # ### Put everything together
                # Shift categories for computational convenience
                # Postpend matrix of zeros
                variance_last = variance.copy()
                variance_last[self.n_class] = np.zeros(variance_last[1].shape)
                # Prepend matrix of zeros
                variance_first = {}
                variance_first[1] = np.zeros(variance[1].shape)
                # Shift existing matrices by 1 class
                for class_idx in range(1, self.n_class, 1):
                    variance_first[class_idx+1] = variance[class_idx]
                # Compute final variance according to: var_last + var_first - cov
                for class_idx in range(1, self.n_class+1, 1):
                    variance_me[x_id,class_idx-1]  = variance_last[
                            class_idx].reshape(-1, 1) + variance_first[
                            class_idx].reshape(-1, 1) - covariance[
                                class_idx].reshape(-1, 1)    
            # standard deviations
            sd_me = np.sqrt(variance_me)     
            # t values and p values (control for division by zero)
            t_values = np.divide(margins, sd_me, out=np.zeros_like(margins), 
                                where=sd_me!=0)
            # p values
            p_values = 2*stats.norm.sf(np.abs(t_values))
        else:
            # no values for the other parameters if inference is not desired
            variance_me = None
            sd_me = None
            t_values = None
            p_values = None
        # put everything into a list of results
        results = {'output': 'margin',
                   'eval_point': eval_point,
                   'window': h_std,
                   'effects': margins,
                   'variances': variance_me,
                   'std_errors': sd_me,
                   't-values': t_values,
                   'p-values': p_values}
        # check if marginal effects should be printed
        if verbose:
            string_seq_X = [str(x) for x in np.arange(1,X_cols+1)]
            string_seq_cat = [str(x) for x in np.arange(1,self.n_class+1)]
            # print marginal effects nicely
            if not inference:
                print('-' * 70,
                      'Marginal Effects of Ordered Forest, evaluation point: '+ 
                      eval_point, '-' * 70, 'Effects:', '-' * 70,
                      pd.DataFrame(data=margins, 
                                   index=['X' + sub for sub in string_seq_X], 
                                   columns=['Cat' + sub for sub in string_seq_cat]),
                      '-' * 70, sep='\n')
            else:
                print('-' * 70,
                      'Marginal Effects of Ordered Forest, evaluation point: '+ 
                      eval_point, '-' * 70, 'Effects:', '-' * 70,
                      pd.DataFrame(data=margins, 
                                   index=['X' + sub for sub in string_seq_X], 
                                   columns=['Cat' + sub for sub in string_seq_cat]),
                      '-' * 70,'Standard errors:', '-' * 70,
                      pd.DataFrame(data=sd_me, 
                                   index=['X' + sub for sub in string_seq_X], 
                                   columns=['Cat' + sub for sub in string_seq_cat]),
                      '-' * 70, sep='\n')
        return results
    
    #Function to predict via sklearn
    def predict_default(self, X, n_samples):
        # create an empty array to save the predictions
        probs = np.empty((n_samples, self.n_class-1))
        for class_idx in range(1, self.n_class, 1):
            # get in-sample predictions, i.e. out-of-bag predictions
            probs[:,class_idx-1] = self.forest['forests'][class_idx].predict(
                X=X).squeeze() 
        return probs
        

    #Function to predict via leaf means
    def predict_leafmeans(self, X, n_samples):
        # create an empty array to save the predictions
        probs = np.empty((n_samples, self.n_class-1))
        # Run new Xs through estimated train forest and compute 
        # predictions based on honest sample. No need to predict
        # weights, get predictions directly through leaf means.
        # Loop over classes
        for class_idx in range(1, self.n_class, 1):
            # Get leaf IDs for new data set
            forest_apply = self.forest['forests'][class_idx].apply(X)
            # generate grid to read out indices column by column
            grid = np.meshgrid(np.arange(0, self.n_estimators), 
                               np.arange(0, n_samples))[0]
            # assign leaf means to indices
            y_hat = self.forest['fitted'][class_idx][forest_apply, grid]
            # Average over trees
            probs[:, class_idx-1] = np.mean(y_hat, axis=1)  
        return probs

        
    #Function to predict via weights
    def predict_weights(self, X, n_samples):
        # create an empty array to save the predictions
        probs = np.empty((n_samples, self.n_class-1))
        # create empty dict to save weights
        weights = {}
        # Step 1: Predict weights by using honest data from fit and
        # newdata (for each category except one)
        # First extract honest data from fit output
        X_est = self.forest['X_fit'][self.forest['ind_est'],:]
        # Loop over classes
        for class_idx in range(1, self.n_class, 1):
            # Get leaf IDs for estimation set
            forest_apply = self.forest['forests'][class_idx].apply(X_est)
            # create binary outcome indicator for est sample
            outcome_ind_est = self.forest['outcome_binary_est'][class_idx]
            # Get size of estimation sample
            n_est = forest_apply.shape[0]
            # Get leaf IDs for newdata
            forest_apply_all = self.forest['forests'][class_idx].apply(X)
# =============================================================================
# In the end: insert here weight.method which works best. For now numpy_loop
# =============================================================================
            # self.weight_method == 'numpy_loop':
            # generate storage matrix for weights
            forest_out = np.zeros((n_samples, n_est))
            # Loop over trees
            for tree in range(self.n_estimators):
                tree_out = self.honest_weight_numpy(
                    tree=tree, forest_apply=forest_apply, 
                    forest_apply_all=forest_apply_all,
                    n_samples=n_samples, n_est=n_est)
                # add tree weights to overall forest weights
                forest_out = forest_out + tree_out
            # Divide by the number of trees to obtain final weights
            forest_out = forest_out / self.n_estimators
            # Compute predictions and assign to probs vector
            predictions = np.dot(forest_out, outcome_ind_est)
            probs[:, class_idx-1] = np.asarray(predictions.T).reshape(-1)
            # Save weights matrix
            weights[class_idx] = forest_out
# =============================================================================
# End of numpy_loop
# =============================================================================
        return probs, weights
    
    def summary(self, item=None):
        """
        Print forest information and prediction performance.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # Input checks
        # check if input has been fitted (sklearn function)
        check_is_fitted(self, attributes=["forest"])
        # Check if outout item properly (sklearn function)
        if item is not None:
            if not (item['output']=='predict'
                    or item['output']=='margin'):
                # raise value error
                raise ValueError("item needs to be prediction or margin "
                                 "output or Nonetype")
        if item is None:
            # print the result
            print('-' * 50,'Summary of the OrderedRandomForest estimation', 
                  '-' * 50, 
                  sep='\n')
            print('%-18s%-15s' % ('type:', 'OrderedForestRegressor'))
            print('%-18s%-15s' % ('categories:', self.n_class))
            print('%-18s%-15s' % ('build:', 'Subsampling' if not
                                  self.replace else 'Bootstrap'))
            print('%-18s%-15s' % ('n_estimators:', self.n_estimators))
            print('%-18s%-15s' % ('max_features:', self.max_features))
            print('%-18s%-15s' % ('min_samples_leaf:', self.min_samples_leaf))
            print('%-18s%-15s' % ('replace:', self.replace))
            print('%-18s%-15s' % ('sample_fraction:', self.sample_fraction))
            print('%-18s%-15s' % ('honesty:', self.honesty))
            print('%-18s%-15s' % ('honesty_fraction:', self.honesty_fraction))
            print('%-18s%-15s' % ('inference:', self.inference))
            print('%-18s%-15s' % ('trainsize:', len(self.forest['ind_tr'])))
            print('%-18s%-15s' % ('honestsize:', len(self.forest['ind_est'])))
            print('%-18s%-15s' % ('features:', self.n_features))
            print('%-18s%-15s' % ('mse1:',np.round(
                float(self.measures['mse 1']),3)))
            print('%-18s%-15s' % ('mse2:',np.round(
                float(self.measures['mse 2']),3)))
            print('%-18s%-15s' % ('accuracy:',np.round(
                float(self.measures['accuracy']),3)))
            print('-' * 50)
        
        elif item['output']=='predict':
            print('-' * 60, 'Summary of the OrderedRandomForest predictions', '-' * 60, 
                  sep='\n')
            print('%-18s%-15s' % ('type:', 
                                  'OrderedForestRegressor predictions'))
            print('%-18s%-15s' % ('prediction_type:', 'Probability' if 
                                  item['prob'] else 'Class'))
            print('%-18s%-15s' % ('categories:', self.n_class))
            print('%-18s%-15s' % ('build:', 'Subsampling' if not
                                  self.replace else 'Bootstrap'))
            print('%-18s%-15s' % ('n_estimators:', self.n_estimators))
            print('%-18s%-15s' % ('max_features:', self.max_features))
            print('%-18s%-15s' % ('min_samples_leaf:', self.min_samples_leaf))
            print('%-18s%-15s' % ('replace:', self.replace))
            print('%-18s%-15s' % ('sample_fraction:', self.sample_fraction))
            print('%-18s%-15s' % ('honesty:', self.honesty))
            print('%-18s%-15s' % ('honesty_fraction:', self.honesty_fraction))
            print('%-18s%-15s' % ('inference:', self.inference))
            print('%-18s%-15s' % ('sample_size:', np.shape(
                item['predictions'])[0]))
            print('-' * 60)
        
        elif item['output']=='margin':
            string_seq_X = [str(x) for x in np.arange(1,self.n_features+1)]
            string_seq_cat = [str(x) for x in np.arange(1,self.n_class+1)]
            print('-' * 60, 
                  'Summary of the OrderedRandomForest marginal effects',
                  '-' * 60, sep='\n')
            print('%-18s%-15s' % ('type:', 
                                  'OrderedForestRegressor marginal effects'))
            print('%-18s%-15s' % ('eval_point:', item['eval_point']))
            print('%-18s%-15s' % ('window:', item['window']))
            print('%-18s%-15s' % ('categories:', self.n_class))
            print('%-18s%-15s' % ('build:', 'Subsampling' if not
                                  self.replace else 'Bootstrap'))
            print('%-18s%-15s' % ('n_estimators:', self.n_estimators))
            print('%-18s%-15s' % ('max_features:', self.max_features))
            print('%-18s%-15s' % ('min_samples_leaf:', self.min_samples_leaf))
            print('%-18s%-15s' % ('replace:', self.replace))
            print('%-18s%-15s' % ('sample_fraction:', self.sample_fraction))
            print('%-18s%-15s' % ('honesty:', self.honesty))
            print('%-18s%-15s' % ('honesty_fraction:', self.honesty_fraction))
            print('%-18s%-15s' % ('inference:', self.inference))
            print('-' * 60,'Effects:', 
                  pd.DataFrame(data=item['effects'], 
                               index=['X' + sub for sub in string_seq_X], 
                               columns=['Cat' + sub for sub in string_seq_cat]),
                  '-' * 60, sep='\n')
            if item['std_errors'] is not None:
                print('Standard errors:', 
                      pd.DataFrame(data=item['std_errors'], 
                                   index=['X' + sub for sub in string_seq_X], 
                                   columns=['Cat' + sub for sub in string_seq_cat]),
                      '-' * 60, sep='\n')
        # empty return
        return None
    
    def plot(self):
        """
        Plot the probability distributions fitted by the OrderedRandomForest

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        # check if input has been fitted (sklearn function)
        check_is_fitted(self, attributes=["forest"])
        # Stack true outcomes and predictions and convert to pandas df
        df_plot = pd.DataFrame(
            np.concatenate((self.forest['y_fit'].reshape(-1,1),
                            self.forest['probs']), axis=1))
        # Convert to wide format
        # New columns: 
        #   0 = true outcome
        #   variable = category where prob is analysed
        #   value = probability of specific category
        df_plot_long = pd.melt(df_plot, id_vars=0)
        # Rename columns
        df_plot_long = df_plot_long.rename(columns={0: "Outcome",
                                                    "variable": "Density",
                                                    "value": "Probability"})
        # Add strings to columns for nice printing in plot
        df_plot_long['Outcome'] = (
            'Class ' + df_plot_long['Outcome'].astype(int).astype(str))
        df_plot_long['Density'] = (
            'P(Y=' + df_plot_long['Density'].astype(int).astype(str) + ')')
        # Compute average prediction per Density-Outcome combination
        df_plot_mean = df_plot_long.copy()
        df_plot_mean['Probability'] = df_plot_mean.groupby(
            ['Density','Outcome'])['Probability'].transform('mean') 
        # Plot using plotnine package
        fig = (ggplot(df_plot_long, aes(x = 'Probability', fill = 'Density'))
         + geom_density(alpha = 0.4)
         + aes(y = "..scaled..")
         + facet_wrap("Outcome", ncol = 1)
         + geom_vline(df_plot_mean, aes(xintercept = "Probability", color = "Density"), linetype="dashed")         
         + xlab("Predicted Probability")
         + ylab("Probability Mass")
         + theme_bw()
         + theme(strip_background = element_rect(fill = "#EBEBEB"))
         + theme(legend_direction = "horizontal", 
                 legend_position = (0.5, -0.03))
         + ggtitle("Distribution of Ordered Forest Probability Predictions")
         )
        # empty return
        return fig


class OrderedRandomForest(OrderedForest):
    """
    Ordered Random Forests class labeled 'OrderedForest'.

    includes methods to fit the model, predict and estimate marginal effects.

    Parameters
    ----------
    n_estimators : TYPE: integer
        DESCRIPTION: Number of trees in the forest. The default is 1000.
    min_samples_leaf : TYPE: integer
        DESCRIPTION: Minimum leaf size in the forest. The default is 5.
    max_features : TYPE: float
        DESCRIPTION: Share of random covariates (0,1). The default is 0.3.
    replace : TYPE: bool
        DESCRIPTION: If True sampling with replacement, i.e. bootstrap is used
        to grow the trees, otherwise subsampling without replacement is used.
        The default is False.
    sample_fraction : TYPE: float
        DESCRIPTION: Subsampling rate, i.e. the share of samples to draw from
        X to train each tree. The default is 0.5.
    honesty : TYPE: bool
        DESCRIPTION: If True honest forest is built using sample splitting.
        The default is False.
    honesty_fraction : TYPE: float
        DESCRIPTION: Share of observations belonging to honest sample not used
        for growing the forest. The default is 0.5.
    inference : TYPE: bool
        DESCRIPTION: If True the weight based inference is conducted. The
        default is False.
    n_jobs : TYPE: int or None
        DESCRIPTION: The number of parallel jobs to be used for parallelism;
        follows joblib semantics. n_jobs=-1 means all - 1 available cpu cores.
        n_jobs=None means no parallelism. There is no parallelism implemented
        for pred_method='numpy'. The default is -1.
    pred_method : TYPE str, one of 'cython', 'loop', 'numpy', 'numpy_loop'
        'numpy_loop_multi', 'numpy_loop_mpire' or 'numpy_sparse'.
        DESCRIPTION: Which method to use to compute honest predictions. The
        default is 'numpy_loop_mpire'.
    weight_method : TYPE str, one of 'numpy_loop', 'numpy_loop_mpire',
        numpy_loop_multi', numpy_loop_shared_multi or numpy_loop_shared_mpire.
        DESCRIPTION: Which method to use to compute honest weights. The
        default is 'numpy_loop_shared_mpire'.
    random_state : TYPE: int, None or numpy.random.RandomState object
        DESCRIPTION: Random seed used to initialize the pseudo-random number
        generator. The default is None. See numpy documentation for details.

    Returns
    -------
    None. Initializes parameters for Ordered Forest.
    """
    # define init function
    def __init__(self, n_estimators=1000,
                 min_samples_leaf=5,
                 max_features=0.3,
                 replace=True,
                 sample_fraction=0.5,
                 honesty=False,
                 honesty_fraction=0.5,
                 inference=False,
                 n_jobs=-1,
                 pred_method='numpy_loop_mpire',
                 weight_method='numpy_loop_shared_mpire',
                 random_state=None):
        # access inherited methods
        super().__init__(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            replace=replace,
            sample_fraction=sample_fraction,
            honesty=honesty,
            honesty_fraction=honesty_fraction,
            inference=inference,
            n_jobs=n_jobs,
            pred_method=pred_method,
            weight_method=weight_method,
            random_state=random_state
        )
 


