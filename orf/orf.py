"""
orf: Ordered Random Forests.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Definitions of class and functions.

"""

# import modules
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from econml.grf import RegressionForest
import orf.honest_fit as honest_fit
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples
# from numba import jit, njit, types, vectorize


# define OrderedForest class
class OrderedForest:
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
        follows joblib semantics. n_jobs=-1 means all available cpu cores.
        n_jobs=None means no parallelism. There is no parallelism implemented
        for pred_method='numpy'. The default is -1.
    pred_method : TYPE str, one of 'cython', 'loop', 'numpy', 'numpy_loop'
        or 'numpy_sparse'.
        DESCRIPTION: Which method to use to compute honest predictions. The
        default is 'cython'.
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
                 pred_method='cython',
                 random_state=None):

        # check and define the input parameters
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
            # assign the input value
            self.n_jobs = n_jobs
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
                or pred_method == 'numpy_sparse'
                or pred_method == 'numpy_sparse2'):
            # assign the input value
            self.pred_method = pred_method
        else:
            # raise value error
            raise ValueError("pred_method must be of cython, loop or numpy"
                             ", got %s" % pred_method)
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

    # function to estimate ordered forest
    def fit(self, X, y, verbose=False):
        """
        Ordered Forest estimation.

        Parameters
        ----------
        X : TYPE: pd.DataFrame
            DESCRIPTION: matrix of covariates.
        y : TYPE: pd.Series
            DESCRIPTION: vector of outcomes.
        verbose : TYPE: bool
            DESCRIPTION: should be the results printed to console?
            Default is False.

        Returns
        -------
        result: ordered probability predictions by Ordered Forest.
        """
        # check if features X are a pandas dataframe
        self.__xcheck(X)

        # check if outcome y is a pandas series
        if isinstance(y, pd.Series):
            # check if its non-empty
            if y.empty:
                # raise value error
                raise ValueError("y Series is empty. Check the input.")
        else:
            # raise value error
            raise ValueError("y is not a Pandas Series. Recode the input.")

        # get the number of outcome classes
        nclass = len(y.unique())
        # obtain total number of observations
        n_samples = _num_samples(X)
        # define the labels if not supplied using list comprehension
        labels = ['Class ' + str(c_idx) for c_idx in range(1, nclass + 1)]
        # create an empty dictionary to save the forests
        forests = {}
        # create an empty dictionary to save the predictions
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
                probs[class_idx] = pd.Series(
                    forests[class_idx].oob_prediction_,
                    name=labels[class_idx - 1],
                    index=X_tr.index)
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
                    probs[class_idx] = pd.Series(
                        forests[class_idx].oob_predict(X_tr).squeeze(),
                        name=labels[class_idx - 1],
                        index=X_tr.index)
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
                        
                        # Loop over trees
                        for tree in range(self.n_estimators):
                            # extract vectors of leaf IDs
                            leaf_IDs_honest = forest_apply[:, tree]
                            leaf_IDs_all = forest_apply_all[:, tree]
                            # Take care of cases where not all training leafs
                            # populated by observations from honest sample
                            leaf_IDs_honest_u = np.unique(leaf_IDs_honest)
                            leaf_IDs_all_u = np.unique(leaf_IDs_all)
                            if (leaf_IDs_honest_u.size == leaf_IDs_all_u.size):
                                leaf_IDs_honest_ext = leaf_IDs_honest
                            else:
                                extra = np.setxor1d(leaf_IDs_all_u,
                                                    leaf_IDs_honest_u)
                                leaf_IDs_honest_ext = np.append(
                                    leaf_IDs_honest, extra)
                            # Generate onehot matrices
                            onehot_honest = OneHotEncoder(
                                sparse=True).fit_transform(
                                    leaf_IDs_honest_ext.reshape(-1, 1)).T
                            onehot_all = OneHotEncoder(
                                sparse=True).fit_transform(
                                    leaf_IDs_all.reshape(-1, 1))
                            # Multiply matrices (n, n_leafs)x(n_leafs, n_est)
                            tree_out = onehot_all.dot(onehot_honest).todense()
                            # Get leaf sizes
                            # leaf size only for honest sample !!!
                            leaf_size = tree_out.sum(axis=1)
                            # Delete extra observations for unpopulated honest
                            # leaves
                            if not leaf_IDs_honest_u.size == leaf_IDs_all_u.size:
                                tree_out = tree_out[:n_samples,:n_est]
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
                                prefer="threads")(
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

                        if self.pred_method == 'numpy_loop':
                            # Loop over trees
                            leaf_means = Parallel(n_jobs=self.n_jobs,
                                                  prefer="threads")(
                                delayed(self.honest_fit_numpy_func)(
                                    tree=tree,
                                    forest_apply=forest_apply,
                                    outcome_ind_est=outcome_ind_est,
                                    max_id=max_id) for tree in range(
                                        0, self.n_estimators))
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
        # set the new column names according to specified class labels
        class_probs = pd.DataFrame(class_probs, columns=labels)
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
        self.forest = {'forests': forests, 'probs': class_probs,
                       'fitted': fitted, 'outcome_binary': outcome_binary,
                       'weights': weights, 'variance': variance}
        # compute prediction performance
        self.__performance(y)
        # check if performance metrics should be printed
        if verbose:
            self.performance()

        # return the output
        return self

    # function to predict with estimated ordered forest
    def predict(self, X, prob=True):
        """
        Ordered Forest prediction.

        Parameters
        ----------
        X : TYPE: pd.DataFrame
            DESCRIPTION: matrix of covariates.
        prob : TYPE: bool
            DESCRIPTION: should the ordered probabilities be predicted?
            If False, ordered classes will be predicted instead.
            Default is True.

        Returns
        -------
        result: ordered probability predictions by Ordered Forest.
        """
        # check if features X are a pandas dataframe
        self.__xcheck(X)

        # get the estimated forests
        forests = self.forest['forests']
        # get the class labels
        labels = list(self.forest['probs'].columns)
        # get the number of outcome classes
        nclass = len(labels)
        # get the number of trees
        ntrees = self.n_estimators
        # get the number of observations in X
        nobs = X.shape[0]
        # create an empty dictionary to save the predictions
        probs = {}

        # estimate random forest on each class outcome except the last one
        for class_idx in range(1, nclass, 1):
            # if no honesty used, predict the standard way
            if not self.honesty:
                # predict with the estimated forests out-of-sample
                probs[class_idx] = pd.Series(
                    forests[class_idx].predict(X=X).squeeze(),
                    name=labels[class_idx - 1],
                    index=X.index)
            else:
                # Get leaf means
                leaf_means = self.forest['fitted'][class_idx]
                # Get leaf IDs for test set
                forest_apply = forests[class_idx].apply(X)
                # generate grid to read out indices column by column
                grid = np.meshgrid(np.arange(0, ntrees), np.arange(0, nobs))[0]
                # assign leaf means to indices
                y_hat = leaf_means[forest_apply, grid]
                # Average over trees
                probs[class_idx] = pd.Series(np.mean(y_hat, axis=1),
                                             name=labels[class_idx - 1],
                                             index=X.index)
        # collect predictions into a dataframe
        probs = pd.DataFrame(probs)
        # create 2 distinct matrices with zeros and ones for easy subtraction
        probs_0 = pd.concat([pd.Series(np.zeros(probs.shape[0]),
                                       index=probs.index,
                                       name=0), probs], axis=1)
        probs_1 = pd.concat([probs, pd.Series(np.ones(probs.shape[0]),
                                              index=probs.index, name=nclass)],
                            axis=1)
        # difference out the adjacent categories to singleout the class probs
        class_probs = probs_1 - probs_0.values
        # check if some probabilities become negative and set them to zero
        class_probs[class_probs < 0] = 0
        # normalize predictions to sum up to 1 after non-negativity correction
        class_probs = class_probs.divide(class_probs.sum(axis=1), axis=0)
        # check if ordered classes instead of ordered probabilities are desired
        if not prob:
            # predict classes with highest probability (+1 as idx starts at 0)
            class_probs = pd.Series((class_probs.values.argmax(axis=1) + 1),
                                    index=X.index)
        # set the new column names according to specified class labels
        class_probs.columns = labels

        # return the class predictions
        return class_probs

    # function to evaluate marginal effects with estimated ordered forest
    def margin(self, X, window=0.1, verbose=False):
        """
        Ordered Forest prediction.

        Parameters
        ----------
        X : TYPE: pd.DataFrame
            DESCRIPTION: matrix of covariates.
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
        # check if features X are a pandas dataframe
        self.__xcheck(X)
        # get a copy to avoid SettingWithCopyWarning
        X_copy = X.copy()

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

        # get the class labels
        labels = list(self.forest['probs'].columns)
        # define the window size share for evaluating the effect
        h_std = window
        # create empty dataframe to store marginal effects
        margins = pd.DataFrame(index=X_copy.columns, columns=labels)

        # loop over all covariates
        for x_id in list(X_copy.columns):
            # first check if its dummy, categorical or continuous
            if list(np.sort(X_copy[x_id].unique())) == [0, 1]:
                # compute the marginal effect as a discrete change in probs
                # save original values of the dummy variable
                dummy = np.array(X_copy[x_id])
                # set x=1
                X_copy[x_id] = 1
                prob_x1 = self.predict(X=X_copy)
                # set x=0
                X_copy[x_id] = 0
                prob_x0 = self.predict(X=X_copy)
                # take the differences and columns means
                effect = (prob_x1 - prob_x0).mean(axis=0)
                # reset the dummy into the original values
                X_copy[x_id] = dummy
            else:
                # compute the marginal effect as continuous change in probs
                # save original values of the continous variable
                original = np.array(X_copy[x_id])
                # get the min and max of x for the support check
                x_min = original.min()
                x_max = original.max()
                # get the standard deviation of x for marginal effect
                x_std = original.std()
                # set x_up=x+h_std*x_std
                x_up = original + (h_std * x_std)
                # check if x_up is within the support of x
                x_up = ((x_up < x_max) * x_up + (x_up >= x_max) * x_max)
                x_up = ((x_up > x_min) * x_up + (x_up <= x_min) *
                        (x_min + h_std * x_std))
                # check if x is categorical and adjust to integers accordingly
                if len(X_copy[x_id].unique()) <= 10:
                    # set x_up=ceiling(x_up)
                    x_up = np.ceil(x_up)
                # replace the x with x_up
                X_copy[x_id] = x_up
                # get orf predictions
                prob_x1 = self.predict(X=X_copy)
                # set x_down=x-h_std*x_std
                x_down = original - (h_std * x_std)
                # check if x_down is within the support of x
                x_down = ((x_down > x_min) * x_down + (x_down <= x_min) *
                          x_min)
                x_down = ((x_down < x_max) * x_down + (x_down >= x_max) *
                          (x_max - h_std * x_std))
                # check if x is categorical and adjust to integers accordingly
                if len(X_copy[x_id].unique()) <= 10:
                    # set x_down=ceiling(x_down) or x_down=floor(x_down)
                    # adjustment such that the difference is always by 1 value
                    x_down[np.ceil(x_down) == np.ceil(x_up)] = np.floor(
                        x_down[np.ceil(x_down) == np.ceil(x_up)])
                    x_down[np.ceil(x_down) != np.ceil(x_up)] = np.ceil(
                        x_down[np.ceil(x_down) != np.ceil(x_up)])
                # replace the x with x_down
                X_copy[x_id] = x_down
                # get orf predictions
                prob_x0 = self.predict(X=X_copy)
                # take the differences, scale them and take columns means
                diff = prob_x1 - prob_x0
                # define scaling parameter
                scale = pd.Series((x_up - x_down), index=X_copy.index)
                # rescale the differences and take the column means
                effect = diff.divide(scale, axis=0).mean(axis=0)
                # reset x into the original values
                X_copy[x_id] = original
            # assign the effects into the output dataframe
            margins.loc[x_id, :] = effect

        # redefine all effect results as floats
        margins = margins.astype(float)

        # check if marginal effects should be printed
        if verbose:
            # print marginal effects nicely
            print('Ordered Forest: Mean Marginal Effects', '-' * 80,
                  margins, '-' * 80, '\n\n', sep='\n')

        # return marginal effects
        return margins

    # performance measures (private method, not available to user)
    def __performance(self, y):
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
        mse_matrix = pd.DataFrame(0, index=y.index,
                                  columns=predictions.columns)
        # allocate indicators for true outcome and leave zeros for the others
        # minus 1 for the column index as indices start with 0, outcomes with 1
        for obs_idx in range(len(y)):
            mse_matrix.iloc[obs_idx, y.iloc[obs_idx] - 1] = 1
        # compute mse directly now by substracting two dataframes and rowsums
        mse_1 = np.mean(((mse_matrix - predictions) ** 2).sum(axis=1))

        # compute the mse: version 2
        # create storage for modified predictions
        modified_pred = pd.Series(0, index=y.index)
        # modify the predictions with 1*P(1)+2*P(2)+3*P(3) as an alternative
        for class_idx in range(len(predictions.columns)):
            # add modified predictions together for all class values
            modified_pred = (modified_pred +
                             (class_idx + 1) * predictions.iloc[:, class_idx])
        # compute the mse directly now by substracting two series and mean
        mse_2 = np.mean((y - modified_pred) ** 2)

        # compute classification accuracy
        # define classes with highest probability (+1 as index starts with 0)
        class_pred = pd.Series((predictions.values.argmax(axis=1) + 1),
                               index=y.index)
        # the accuracy directly now by mean of matching classes
        acc = np.mean(y == class_pred)

        # create te confusion matrix
        self.confusion = pd.DataFrame(index=predictions.columns,
                                      columns=predictions.columns)
        # fill in the matrix by comparisons
        # loop over the actual outcomes
        for actual in range(len(self.confusion)):
            # loop over the predicted outcomes
            for predicted in range(len(self.confusion)):
                # compare the actual with predicted and sum it up
                self.confusion.iloc[actual, predicted] = sum(
                    (y == actual + 1) & (class_pred == predicted + 1))

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

    # check user input for covariates (private method, not available to user)
    def __xcheck(self, X):
        """
        Check the user input for the pandas dataframe of covariates.

        Parameters
        ----------
        X : TYPE: pd.DataFrame
            DESCRIPTION: matrix of covariates.

        Returns
        -------
        None. Checks for the correct user input.
        """
        # check if features X are a pandas dataframe
        if isinstance(X, pd.DataFrame):
            # check if its non-empty
            if X.empty:
                # raise value error
                raise ValueError("X DataFrame is empty. Check the input.")
        else:
            # raise value error
            raise ValueError("X is not a Pandas DataFrame. Recode the input.")

        # empty return
        return None

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
        honest_variance_final = np.empty((n_est, nclass))
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
