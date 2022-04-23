# -*- coding: utf-8 -*-
"""
orf: Ordered Random Forest.

Python implementation of the Ordered Random Forest as in Lechner & Okasa (2019).

Definition of base ordered forest estimator and fit function.

"""

# import modules
import sharedmem

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state, check_X_y
from sklearn.utils.validation import _num_samples, _num_features
from econml.grf import RegressionForest
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import cpu_count, Lock


# %% Class definition
    
# define BaseOrderedForest class (BaseEstimator allows to call get_params and set_params)
class BaseOrderedForest(BaseEstimator):
    """
    Base class for OrderedForest.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    # define init function
    def __init__(self, n_estimators=1000,
                 min_samples_leaf=5,
                 max_features=None,
                 replace=False,
                 sample_fraction=0.5,
                 honesty=True,
                 honesty_fraction=0.5,
                 inference=False,
                 n_jobs=-1,
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
        self.random_state = random_state

        # initialize performance metrics
        self.confusion = None
        self.measures = None

    def _input_checks(self):
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
        random_state = self.random_state

        # define number of features
        n_features = self.n_features

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
                raise ValueError("min_samples_leaf must be an integer >= 1 or "
                                 "a float within (0,1), got %s" 
                                 % min_samples_leaf)
        elif isinstance(min_samples_leaf, float):
            # check if its at least 1
            if (min_samples_leaf > 0 and min_samples_leaf < 1):
                # assign the input value
                self.min_samples_leaf = min_samples_leaf
            else:
                # raise value error
                raise ValueError("min_samples_leaf must be an integer >= 1 or "
                                 "a float within (0,1), got %s" 
                                 % min_samples_leaf)
        else:
            # raise value error
            raise ValueError("min_samples_leaf must be an integer >= 1 or "
                             "a float within (0,1), got %s" 
                             % min_samples_leaf)

        # check share of features in splitting
        if max_features is None:
            # then set as default np.ceil(np.sqrt(n_features)) for X supplied
            self.max_features = int(np.ceil(np.sqrt(n_features)))
        # otherwise if share is supplied check if its admissibe
        elif isinstance(max_features, float):
            # check if its within (0,1]
            if (max_features > 0 and max_features <= 1):
                # assign the input value
                self.max_features = max_features
            else:
                # raise value error
                raise ValueError("max_features must be within (0,1]"
                                 ", got %s" % max_features)
        # otherwise check if number supplied if its admissible
        elif isinstance(max_features, int):
            # check if its positive and less or equal max n_features
            if (max_features > 0 and max_features <= n_features):
                # assign the input value
                self.max_features = max_features
            else:
                # raise value error
                raise ValueError("max_features must be greater than 0 and less"
                                 " or equal to total number of features"
                                 ", got %s" % max_features)
        else:
            # raise value error
            raise ValueError("max_features must be a float, int or None"
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
        if self.replace:
            # if bootstrapping, set the subsampling share to take all obs
            self.sample_fraction = None
        else:
            # else if subsampling is being used
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
        if self.honesty:
            # only relevant if honesty is set, otherwise ignored
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
        # check whether n_jobs is set to None
        elif n_jobs is None:
            # set n_jobs to 1 - sequential execution
            self.n_jobs = 1
        else:
            # raise value error
            raise ValueError("n_jobs must be of type integer or None"
                             ", got %s" % n_jobs)

        # check whether seed is set (using scikitlearn check_random_state)
        self.random_state = check_random_state(random_state)
        # get max np.int32 based on machine limit
        max_int = np.iinfo(np.int32).max
        # use this to initialize seed for honest splitting: this is useful when
        # we want to obtain the same splits later on
        self.subsample_random_seed = self.random_state.randint(max_int)
        
    
    # %% Fit function
    # function to estimate OrderedForest
    def fit(self, X, y):
        """
        OrderedForest estimation.
        """
        # Check if input is pandas data frame. If yes, save feature names
        self.feature_names=None
        if isinstance(X, pd.DataFrame):
            self.feature_names=X.columns.values
        # Use sklearn input checks to allow for multiple types of inputs:
        # - returns numpy arrays for X and y (no matter which input type)
        # - forces y to be numeric
        X,y = check_X_y(X, y, y_numeric=True, estimator="OrderedForest")
        # obtain total number of observations
        self.n_features = _num_features(X)
        # run input checks for arguments
        self._input_checks()
        # Get vector of sorted unique values of y
        y_values = np.unique(y)
        # obtain total number of observations
        n_samples = _num_samples(X)

        # Get the number of outcome classes
        self.n_class = nclass = len(y_values)
        # Next, ensure that y is a vector of continuous integers starting at 
        # 1 up to nclass
        # Check if y consists of integers
        if all(isinstance(x, (np.integer)) for x in y_values):
            # Recode y appropriately (keeps order but recodes values as 1,2...)
            y = np.searchsorted(np.unique(y), y)+1
            y_values = np.unique(y)
        else:
            # Check if continuous sequence
            if ((min(y_values)==1) and (max(y_values)==nclass)):
                # Recode y appropriately
                y = (np.searchsorted(np.unique(y), y)+1).astype(int)
                y_values = np.unique(y)
            else:
                # raise value error
                raise ValueError("n_jobs must be of type integer"
                                 ", got %s" % type(y_values[0]))

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

        # %% generate honest estimation sample
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

        # %% estimate random forest on each class outcome except the last one
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
                    n_jobs=self.n_jobs,
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
                    n_jobs=self.n_jobs,
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
                        forest_out = np.zeros((n_samples, n_est))

                        # check if parallelization should be used
                        if self.n_jobs > 1:
                            # initiate lock
                            _lock = Lock()
                            # parallel in shared memory
                            Parallel(
                                n_jobs=self.n_jobs,
                                backend="threading"
                                )(delayed(
                                    self._forest_weights_shared)(
                                        tree=tree,
                                        forest_apply=forest_apply,
                                        forest_apply_all=forest_apply_all,
                                        n_samples=n_samples,
                                        n_est=n_est,
                                        shared_object=forest_out,
                                        lock=_lock)
                                        for tree in range(self.n_estimators))

                        else:
                            # Loop over trees sequentially
                            for tree in range(self.n_estimators):
                                # get honest tree weights
                                tree_out = self._honest_weight_numpy(
                                    tree=tree,
                                    forest_apply=forest_apply,
                                    forest_apply_all=forest_apply_all,
                                    n_samples=n_samples,
                                    n_est=n_est)
                                # add tree weights to overall forest weights
                                forest_out += tree_out

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
                        # predict via leaf means
                        # check if multithreading should be done
                        if self.n_jobs > 1:
                            # Loop over trees in parallel via 'numpy_joblib'
                            with parallel_backend('threading',
                                                  n_jobs=self.n_jobs):
                                leaf_means = Parallel()(
                                    delayed(self._honest_fit_numpy_func)(
                                    tree=tree,
                                    forest_apply=forest_apply,
                                    outcome_ind_est=outcome_ind_est,
                                    max_id=max_id) for tree in range(
                                        0, self.n_estimators))
                            # assign honest predictions, i.e. fitted values
                            fitted[class_idx] = np.vstack(leaf_means).T

# =============================================================================
#                             # Loop over trees in parallel via 'numpy_joblib_shared'
#                             # storage as numpy array
#                             leaf_means = np.zeros((max_id, self.n_estimators))
#                             _lock = Lock()  # initiate lock
#                             # parallel in shared memory
#                             Parallel(
#                                 n_jobs=self.n_jobs,
#                                 backend="threading"
#                                 )(delayed(
#                                     self._honest_fit_numpy_shared)(
#                                         tree=tree,
#                                         forest_apply=forest_apply,
#                                         outcome_ind_est=outcome_ind_est,
#                                         max_id=max_id,
#                                         shared_object=leaf_means,
#                                         lock=_lock)
#                                         for tree in range(self.n_estimators))
#                             # assign honest predictions (honest fitted values)
#                             fitted[class_idx] = leaf_means
# =============================================================================

                        else:
                            # storage as numpy array
                            leaf_means = np.zeros((max_id, self.n_estimators))
                            # Loop over trees sequentially
                            for tree in range(self.n_estimators):
                                # Create dummy matrix dim(n_est, max_id)
                                onehot = OneHotEncoder(sparse=True).fit_transform(
                                    forest_apply[:, tree].reshape(-1, 1))
                                # Compute leaf sums for each leaf
                                leaf_sums = onehot.T.dot(outcome_ind_est)
                                # Determine number of observations per leaf
                                leaf_n = onehot.sum(axis=0)
                                # Compute leaf means for each leaf
                                leaf_means[np.unique(forest_apply[:, tree]),
                                           tree] = leaf_sums/leaf_n
                            # assign honest predictions (honest fitted values)
                            fitted[class_idx] = leaf_means

# =============================================================================
#                         if self.pred_method == 'numpy_loop':
#                             # storage as list
#                             leaf_means = [np.nan for _ in range(self.n_estimators)]
#                             # Loop over trees
#                             for tree in range(self.n_estimators):
#                                 # compute preds
#                                 leaf_means[tree] = self._honest_fit_numpy_func(
#                                     tree=tree,
#                                     forest_apply=forest_apply,
#                                     outcome_ind_est=outcome_ind_est,
#                                     max_id=max_id)
#                             # assign honest predictions (honest fitted values)
#                             fitted[class_idx] = np.vstack(leaf_means).T
# =============================================================================

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
        
        # %% Compute variance of predicitons if inference = True
        # outcome need to come from the honest sample here, outcome_binary_est
        if self.inference:
            # prepare honest sample
            probs_honest = probs[ind_est, :]
            weights_honest = dict([(key, weights[key][ind_est, :])
                                   for key in range(1, nclass, 1)])
            # compute variance
            variance_honest = self._honest_variance(
                probs=probs_honest, weights=weights_honest,
                outcome_binary=outcome_binary_est, nclass=nclass, n_est=n_est)
            # prepare train sample
            n_tr = len(ind_tr)
            probs_train = probs[ind_tr, :]
            weights_train = dict([(key, weights[key][ind_tr, :])
                                  for key in range(1, nclass, 1)])
            # compute variance
            variance_train = self._honest_variance(
                probs=probs_train, weights=weights_train,
                outcome_binary=outcome_binary_est, nclass=nclass, n_est=n_tr)
            # put honest and train variance together
            variance = np.vstack((variance_honest, variance_train))
            # Combine indices
            ind_all = np.hstack((ind_est, ind_tr))
            # Sort variance according to indices in ind_all
            variance = variance[ind_all.argsort(), :]
        else:
            variance = None

        # %% pack estimated forest and class predictions into output dictionary
        self.forest_ = {'forests': forests,
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
        self._performance(y, y_values)

        # return the output
        return self


    # %% Performance functions
    # performance measures (private method, not available to user)
    def _performance(self, y, y_values):
        """
        Evaluate the prediction performance using MSE, RPS and CA.

        Parameters
        ----------
        y : ndarray
            Vector of outcomes.

        Returns
        -------
        None. Calculates MSE, RPS, Classification accuracy, confusion matrix.
        """

        # take over needed values
        predictions = self.forest_['probs']

        # compute the mse
        # create storage empty dataframe
        mse_matrix = np.zeros(predictions.shape)
        # allocate indicators for true outcome and leave zeros for the others
        # minus 1 for the column index as indices start with 0, outcomes with 1
        mse_matrix[np.arange(y.shape[0]), y-1] = 1

        # compute mse directly now by substracting two dataframes and rowsums
        mse = np.mean(((mse_matrix - predictions) ** 2).sum(axis=1))

        # compute rps (ranked probabilty score)
        # get the indicator matrix (same as for mse)
        rps_matrix = mse_matrix.copy()
        # prepare storage for cumulative scores
        cum = np.zeros(len(y))
        # loop over the categories (inspired by and thanks to:
        # https://opisthokonta.net/?p=1333)
        for i in y_values:
            # update the cumulative score
            cum = (cum + (np.sum(predictions[:, 0:i], axis=1) -
                         np.sum(rps_matrix[:, 0:i], axis=1))**2
                   )
        # compute the RPS
        rps = np.mean((1/(len(y_values)-1))*cum)

        # compute classification accuracy
        # define classes with highest probability (+1 as index starts with 0)
        class_pred = predictions.argmax(axis=1) + 1
        # the accuracy directly now by mean of matching classes
        acc = np.mean(y == class_pred)

        # create te confusion matrix
        # First generate onehot matrice of y      
        y_onehot = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
        # Find unique predicted classes
        class_pred_u = np.unique(class_pred)
        # Check if there exists a class for which no predictions exist
        if not np.array_equal(class_pred_u, y_values):
            # Find class(es) which have no predictions
            extra = np.setdiff1d(y_values, class_pred_u)
            # Add them to the predictions
            class_pred_ext = np.append(class_pred, extra)
            # Generate onehot matrix of predictions and drop extra classes
            class_pred_onehot = OneHotEncoder(sparse=False).fit_transform(
                class_pred_ext.reshape(-1, 1))[:len(y),:]
        else:
            # Generate onehot matrix of predictions
            class_pred_onehot = OneHotEncoder(sparse=False).fit_transform(
                class_pred.reshape(-1, 1))
        # Compute dot product of these matrices to obtain confusion matrix
        confusion_mat = np.dot(np.transpose(y_onehot), class_pred_onehot)
        labels = ['Class ' + str(c_idx) for c_idx in y_values]
        self.confusion = pd.DataFrame(confusion_mat, 
                                      index=labels, columns=labels)

        # wrap the results into a dataframe
        self.measures = pd.DataFrame({'mse': mse, 'rps': rps, 'accuracy': acc},
                                     index=['value'])

        # empty return
        return None


    # %% In-class honesty and weight functions
    # Function to compute honest predictions.
    def _honest_fit_numpy_func(self, tree, forest_apply, outcome_ind_est,
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


    # Function to compute honest predictions in shared memory.
    def _honest_fit_numpy_shared(self, tree, forest_apply, outcome_ind_est,
                                 max_id, shared_object, lock):
        """Compute the honest leaf means using numpy with shared memory."""

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

        # write into shared memory under lock
        lock.acquire()
        shared_object[:, tree] = leaf_means
        lock.release()

        return


    # Function to compute honest weights using numpy.
    def _honest_weight_numpy(self, tree, forest_apply, forest_apply_all,
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


    # Function to compute variance of predictions.
    def _honest_variance(self, probs, weights, outcome_binary, nclass, n_est):
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


    # using shared memory for weight computation in multithreading
    def _forest_weights_shared(self, tree, forest_apply, forest_apply_all,
                               n_samples, n_est, shared_object, lock):
        # compute the tree weights first in threads
        tree_weights = self._honest_weight_numpy(tree, forest_apply,
                                                   forest_apply_all, n_samples,
                                                   n_est)
        # write into shared memory under lock
        lock.acquire()
        shared_object += tree_weights
        lock.release()

        return
