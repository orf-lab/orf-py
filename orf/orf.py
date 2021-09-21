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
    n_jobs : TYPE: int or None
        DESCRIPTION: The number of parallel jobs to be used for parallelism;
        follows joblib semantics. n_jobs=-1 means all available cpu cores.
        n_jobs=None means no parallelism. There is no parallelism implemented
        for pred_method='numpy'. The default is -1.
    pred_method : TYPE str, one of 'cython', 'loop' or 'numpy'
        DESCRIPTION: Which method to use to compute honest predictions. The
        default is 'cython'.


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
                 n_jobs=-1,
                 pred_method='cython'):

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

        # check whether n_jobs is integer
        if isinstance(n_jobs, int):
            # assign the input value
            self.n_jobs = n_jobs
        else:
            # raise value error
            raise ValueError("n_jobs must be of type integer"
                             ", got %s" % n_jobs)

        # check whether pred_method is defined correctly
        if (pred_method == 'cython' or pred_method == 'loop' or
                pred_method == 'numpy'):
            # assign the input value
            self.pred_method = pred_method
        else:
            # raise value error
            raise ValueError("pred_method must be of cython, loop or numpy"
                             ", got %s" % pred_method)
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
        # define the labels if not supplied using list comprehension
        labels = ['Class ' + str(c_idx) for c_idx in range(1, nclass + 1)]
        # create an empty dictionary to save the forests
        forests = {}
        # create an empty dictionary to save the predictions
        probs = {}
        # create an empty dictionary to save the fitted values
        fitted = {}
        # generate honest estimation sample
        if self.honesty:
            X_tr, X_est, y_tr, y_est = train_test_split(
                X, y, test_size=self.honesty_fraction)
        else:
            X_tr = X
            y_tr = y
        # estimate random forest on each class outcome except the last one
        for class_idx in range(1, nclass, 1):
            # create binary outcome indicator for the outcome in the forest
            outcome_ind = (y_tr <= class_idx) * 1
            # check whether to do subsampling or not
            if self.replace:
                # call rf from scikit learn and save it in dictionary
                forests[class_idx] = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    max_samples=self.sample_fraction,
                    oob_score=True)
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
                    max_samples=self.sample_fraction)
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
                    # create binary outcome indicator for the estimation sample
                    outcome_ind_est = np.array((y_est <= class_idx) * 1)
                    # compute maximum leaf id
                    max_id = np.max(forest_apply)+1
                    # Check whether to use cython implementation or not
                    if self.pred_method == 'cython':
                        # Loop over trees
                        leaf_means = Parallel(n_jobs=self.n_jobs,
                                              prefer="threads")(
                            delayed(honest_fit.honest_fit)(
                                forest_apply=forest_apply,
                                outcome_ind_est=outcome_ind_est,
                                trees=tree,
                                max_id=max_id) for tree in range(
                                    0, self.n_estimators))
                        # assign honest predictions, i.e. honest fitted values
                        fitted[class_idx] = np.vstack(leaf_means).T
                    # Check whether to use loop implementation or not
                    if self.pred_method == 'loop':
                        # Loop over trees
                        leaf_means = Parallel(n_jobs=self.n_jobs,
                                              prefer="threads")(
                            delayed(self.__honest_fit_func)(
                                forest_apply=forest_apply,
                                outcome_ind_est=outcome_ind_est,
                                tree=tree,
                                max_id=max_id) for tree in range(
                                    0, self.n_estimators))
                        # assign honest predictions, i.e. honest fitted values
                        fitted[class_idx] = np.vstack(leaf_means).T
                    # Check whether to use numpy implementation or not
                    if self.pred_method == 'numpy':
                        # https://stackoverflow.com/questions/36960320
                        onehot = np.zeros(forest_apply.shape + (max_id,),
                                          dtype=np.uint8)
                        grid = np.ogrid[tuple(map(slice, forest_apply.shape))]
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
                    # Compute predictions for whole sample: both tr and est
                    # Get leaf IDs for the whole set of observations
                    forest_apply = forests[class_idx].apply(X)
                    # generate grid to read out indices column by column
                    grid = np.meshgrid(np.arange(0, self.n_estimators),
                                       np.arange(0, X.shape[0]))[0]
                    # assign leaf means to indices
                    y_hat = fitted[class_idx][forest_apply, grid]
                    # Average over trees
                    probs[class_idx] = np.mean(y_hat, axis=1)
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
        # set the new column names according to specified class labels
        class_probs.columns = labels

        # pack estimated forest and class predictions into output dictionary
        self.forest = {'forests': forests, 'probs': class_probs,
                       'fitted': fitted}
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
    def margin(self, X, verbose=False):
        """
        Ordered Forest prediction.

        Parameters
        ----------
        X : TYPE: pd.DataFrame
            DESCRIPTION: matrix of covariates.
        verbose : TYPE: bool
            DESCRIPTION: should be the results printed to console?
            Default is False.

        Returns
        -------
        result: Mean marginal effects by Ordered Forest.
        """
        # check if features X are a pandas dataframe
        self.__xcheck(X)

        # get the class labels
        labels = list(self.forest['probs'].columns)
        # define the window size share for evaluating the effect
        h_std = 0.1
        # create empty dataframe to store marginal effects
        margins = pd.DataFrame(index=X.columns, columns=labels)

        # loop over all covariates
        for x_id in list(X.columns):
            # first check if its dummy, categorical or continuous
            if list(np.sort(X[x_id].unique())) == [0, 1]:
                # compute the marginal effect as a discrete change in probs
                # save original values of the dummy variable
                dummy = np.array(X[x_id])
                # set x=1
                X[x_id] = 1
                prob_x1 = self.predict(X=X)
                # set x=0
                X[x_id] = 0
                prob_x0 = self.predict(X=X)
                # take the differences and columns means
                effect = (prob_x1 - prob_x0).mean(axis=0)
                # reset the dummy into the original values
                X[x_id] = dummy
            else:
                # compute the marginal effect as continuous change in probs
                # save original values of the continous variable
                original = np.array(X[x_id])
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
                if len(X[x_id].unique()) <= 10:
                    # set x_up=ceiling(x_up)
                    x_up = np.ceil(x_up)
                # replace the x with x_up
                X[x_id] = x_up
                # get orf predictions
                prob_x1 = self.predict(X=X)
                # set x_down=x-h_std*x_std
                x_down = original - (h_std * x_std)
                # check if x_down is within the support of x
                x_down = ((x_down > x_min) * x_down + (x_down <= x_min) *
                          x_min)
                x_down = ((x_down < x_max) * x_down + (x_down >= x_max) *
                          (x_max - h_std * x_std))
                # check if x is categorical and adjust to integers accordingly
                if len(X[x_id].unique()) <= 10:
                    # set x_down=floor(x_down)
                    x_down = np.floor(x_down)
                # replace the x with x_down
                X[x_id] = x_down
                # get orf predictions
                prob_x0 = self.predict(X=X)
                # take the differences, scale them and take columns means
                diff = prob_x1 - prob_x0
                # define scaling parameter
                scale = pd.Series((x_up - x_down), index=X.index)
                # rescale the differences and take the column means
                effect = diff.divide(scale, axis=0).mean(axis=0)
                # reset x into the original values
                X[x_id] = original
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

    def __honest_fit_func(self, forest_apply, outcome_ind_est, tree, max_id):
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
