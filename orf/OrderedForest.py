# -*- coding: utf-8 -*-
"""
orf: Ordered Random Forest.

Python implementation of the Ordered Random Forest as in Lechner & Okasa (2019).

Definition of post-fitting methods.

"""

# import modules
import numpy as np
import pandas as pd
from orf.BaseOrderedForest import BaseOrderedForest
from sklearn.utils import check_array
from sklearn.utils.validation import _num_samples, check_is_fitted
from scipy import stats
from plotnine import (ggplot, aes, geom_density, facet_wrap, geom_vline, 
                      ggtitle, xlab, ylab, theme_bw, theme, element_rect)


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
        OrderedRandomForest prediction.

        Parameters
        ----------
        X : array-like or NoneType
            Matrix of new covariates or None if covariates from
            fit function should be used. If new data provided it must have
            the same number of features as the X in the fit function.
        prob : bool
            Should the ordered probabilities be predicted? If False, ordered 
            classes will be predicted instead. Default is True.

        Returns
        -------
        result : dict
                 Dictionary containing prediction results. Use 
                 result.get("predictions") to extract array of predictions and
                 result.get("variances") to extract array of variances.
        """

        # Input checks
        # check if input has been fitted (sklearn function)
        check_is_fitted(self, attributes=["forest_"])
        
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
            if np.array_equal(X, self.forest_['X_fit']):
                X = None
        else:
            n_samples = _num_samples(self.forest_['X_fit'])
        
        # check whether to predict probabilities or classes
        if isinstance(prob, bool):
            # assign the input value
            self.prob = prob
        else:
            # raise value error
            raise ValueError("prob must be of type boolean"
                             ", got %s" % prob)
        
        # get the forest inputs
        outcome_binary_est = self.forest_['outcome_binary_est']
        probs = self.forest_['probs']
        variance = self.forest_['variance']
        # get the number of outcome classes
        nclass = self.n_class
        # get inference argument
        inference = self.inference
        
        # Check if prob allows to do inference
        if ((not prob) and (inference) and (X is not None)):
            print('-' * 70, 
                  'WARNING: Inference is not possible if prob=False.' 
                  '\nClass predictions for large samples might be obtained'
                  '\nfaster when re-estimating OrderedRandomForest with option'
                  '\ninference=False.', 
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
                probs = self._predict_default(X=X, n_samples=n_samples)
            # If honesty True, inference argument decides how to compute
            # predicions
            elif self.honesty and not inference:
                probs = self._predict_leafmeans(X=X, n_samples=n_samples)
            # Remaining case refers to honesty=True and inference=True
            else:
                probs, weights = self._predict_weights(
                    X=X, n_samples=n_samples)

            # create 2 distinct matrices with zeros and ones for subtraction
            # prepend vector of zeros
            probs_0 = np.hstack((np.zeros((n_samples, 1)), probs))
            # postpend vector of ones
            probs_1 = np.hstack((probs, np.ones((n_samples, 1))))
            # difference the adjacent categories to singleout the class probs
            class_probs = probs_1 - probs_0
            # check if some probabilities become negative and set them to zero
            class_probs[class_probs < 0] = 0
            # normalize predictions to sum to 1 after non-negativity correction
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
                var_final = self._honest_variance(
                    probs=probs, weights=weights,
                    outcome_binary=outcome_binary_est, nclass=nclass,
                    n_est=len(self.forest_['ind_est']))

        # return the class predictions
        result = {'output': 'predict',
                  'prob': prob,
                  'predictions': pred_final,
                  'variances': var_final}

        return result


    # %% Margin function
    # function to evaluate marginal effects with estimated ordered forest
    def margin(self, X=None, X_cat=None, X_eval=None, eval_point="mean",
               window=0.1, verbose=True):
        """
        OrderedRandomForest marginal effects.

        Parameters
        ----------
        X : array-like or NoneType
            Matrix of new covariates or None if covariates from
            fit function should be used. If new data provided it must have
            the same number of features as the X in the fit function.
        X_cat : list or tuple or NoneType
            List or tuple indicating the columns with categorical covariates,
            i.e. X_cat=(1,) if the second column includes categorical values.
            If not defined, covariates with integer values and less than 10
            unique values are considered to be categorical as default.
        X_eval : list or tuple or NoneType
            List or tuple indicating the columns with covariates for which the,
            marginal effect should be evaluated, i.e. X_eval=(1,) if the effect
            for the covariate in the column should be evaluated. This can
            significantly speed up the computations. If not defined,
            all covariates are considered as default.
        eval_point: string
            Defining evaluation point for marginal effects. These
            can be one of "mean", "atmean", or "atmedian". Default is "mean".
        window : float
            Share of standard deviation of X to be used for
            evaluation of the marginal effect. Default is 0.1.
        verbose : bool
            Should the results printed to console? Default is True.
            

        Returns
        -------
        result : dict
                 Dictionary containing results of marginal effects estimation.
                 Use result.get("...") with "effects", "variances",
                 "std_errors", "t-values" or "p-values" to extract arrays of
                 marginal effects, variances, standard errors, t-values or 
                 p-values, respectively.
        """

        # %% Input checks
        # check if input has been fitted (sklearn function)
        check_is_fitted(self, attributes=["forest_"])

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
            if np.array_equal(X, self.forest_['X_fit']):
                X = None

        # Check if X_cat defined properly
        if not X_cat is None:
            if isinstance(X_cat, (list, tuple)):
                # Check if number of indices is admissible
                if not len(X_cat) <= self.n_features:
                    # raise value error
                    raise ValueError("Number of indices for categorical "
                                     "covariates must be less or equal "
                                     "than the overall number of covariates. "
                                     ", got %s" % len(X_cat))
                # Check if max index is admissible
                if not ((np.max(X_cat) <= (self.n_features - 1)) &
                        (np.min(X_cat) >= 0)):
                    # raise value error
                    raise ValueError("Indices for categorical covariates "
                                     "must be between 0 and the overall number"
                                     " of covariates - 1, got %s" % len(X_cat))
            else:
                # raise value error
                raise ValueError("X_cat must be of type tuple, list or None"
                                 ", got %s" % X_cat)

        # Check if X_eval defined properly
        if not X_eval is None:
            if isinstance(X_eval, (list, tuple)):
                # Check if number of indices is admissible
                if not len(X_eval) <= self.n_features:
                    # raise value error
                    raise ValueError("Number of indices for covariates "
                                     "for which the marginal effects are "
                                     "evaluated must be less or equal "
                                     "than the overall number of covariates. "
                                     ", got %s" % len(X_eval))
                # Check if max index is admissible
                if not ((np.max(X_eval) <= (self.n_features - 1)) &
                        (np.min(X_eval) >= 0)):
                    # raise value error
                    raise ValueError("Indices for effect covariates must "
                                     "be between 0 and the overall number of "
                                     "covariates - 1, got %s" % len(X_eval))
                # assign the indices
                X_eval_ind = X_eval
            else:
                # raise value error
                raise ValueError("X_eval must be of type tuple, list or None"
                                 ", got %s" % X_eval)
        else:
            # select all covariates for margins evaluation
            X_eval_ind = [*range(self.n_features)]

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
        ind_est = self.forest_['ind_est']

        # %% Prepare data sets
        # check if new data provided or not
        if X is None:
            # if no new data supplied, estimate in-sample marginal effects
            if self.honesty:
                # if using honesty, data refers to the honest sample
                X_eval = self.forest_['X_fit'][ind_est,:]
                X_est = self.forest_['X_fit'][ind_est,:]
            else:
                # if not using honesty, data refers to the full sample
                X_eval = self.forest_['X_fit']
                X_est = self.forest_['X_fit']
        else:
            # if new data supplied, need to use this for prediction
            if self.honesty:
                # if using honesty, need to consider new and honest sample
                X_eval = X
                X_est = self.forest_['X_fit'][ind_est,:]
            else:
                # if not using honesty, data refers to the new sample
                X_eval = X
                X_est = self.forest_['X_fit']

        # get the number of observations in X
        n_samples = _num_samples(X_eval)
        
        # check if X is continuous, dummy or categorical
        # first find number of unique values per column
        X_eval_sort = np.sort(X_eval, axis=0)
        n_unique = (X_eval_sort[1:, :] != X_eval_sort[:-1, :]).sum(axis=0) + 1

        # get indices of respective columns
        # dummies with 2 values
        X_dummy = (n_unique == 2).nonzero()

        # categorical if int and less than 10 values if not supplied by user
        if X_cat is None:
            X_cat = (np.all(np.mod(X_eval, 1) == 0, axis=0) &
                     (n_unique > 2) & (n_unique <= 10)).nonzero()

        # check if there are covariates without variation
        if np.any(n_unique<= 1):
            # raise value error
            raise ValueError("Some of the covariates are constant. This is "
                             "not allowed for evaluation of marginal effects. "
                             "Programme terminated.")

        ## Get the evaluation point(s)
        # Save evaluation point(s) in X_mean
        if eval_point == "atmean":
            X_mean = np.mean(X_eval, axis=0).reshape(1,-1)
            # adjust for dummies and categorical covariates
            X_mean[:, X_dummy] = np.median(X_eval[:, X_dummy],
                                           axis=0).reshape(1,-1)
            X_mean[:, X_cat] = np.median(X_eval[:, X_cat],
                                         axis=0).reshape(1,-1)
        elif eval_point == "atmedian":
            X_mean = np.median(X_eval, axis=0).reshape(1,-1)
        else:
            X_mean = X_eval.copy()

        # Get dimension of evaluation points
        X_rows = np.shape(X_mean)[0]
        X_cols = np.shape(X_mean)[1]
        # Get standard deviation of X_est in the same shape as X_mean
        # X_est here as for X_eval it is not ensured that the std is not zero
        # if the X_eval includes only 1 row
        X_sd = np.repeat(np.std(
            X_est, axis=0, ddof=1).reshape(1,-1), X_rows, axis=0)
        # create X_up (X_mean + window * X_sd) only for selected covariates
        X_up = X_mean.copy()
        X_up[:, X_eval_ind] = X_mean[:, X_eval_ind]+window*X_sd[:, X_eval_ind]
        # create X_down (X_mean - window * X_sd)
        X_down = X_mean.copy()
        X_down[:, X_eval_ind] = X_mean[:, X_eval_ind]-window*X_sd[:, X_eval_ind]

        ## now check if support of X_eval is within X_est
        # check X_max
        X_max = np.repeat(np.max(X_est, axis=0).reshape(1,-1), X_rows, axis=0)
        # check X_min
        X_min = np.repeat(np.min(X_est, axis=0).reshape(1,-1), X_rows, axis=0)
        # check if X_up is within the range X_min and X_max
        # If some X_up is larger than the max in X_est, replace entry in X_up 
        # by this max value of X_est. If some X_up is smaller than the min in
        # X_est, replace entry in X_up by this min value + window * X_sd
        X_up[:, X_eval_ind] = (
            (X_up[:, X_eval_ind] < X_max[:, X_eval_ind])*X_up[:, X_eval_ind]+ 
            (X_up[:, X_eval_ind] >= X_max[:, X_eval_ind])*X_max[:, X_eval_ind]
            )
        X_up[:, X_eval_ind] = (
            (X_up[:, X_eval_ind] > X_min[:, X_eval_ind])*X_up[:, X_eval_ind]+
            (X_up[:, X_eval_ind] <= X_min[:, X_eval_ind])*
            (X_min[:, X_eval_ind] + window * X_sd[:, X_eval_ind])
            )
        # check if X_down is within the range X_min and X_max
        X_down[:, X_eval_ind] = (
            (X_down[:, X_eval_ind]>X_min[:, X_eval_ind])*X_down[:, X_eval_ind]+
            (X_down[:, X_eval_ind]<=X_min[:, X_eval_ind])*X_min[:, X_eval_ind]
            )
        X_down[:, X_eval_ind] = (
            (X_down[:, X_eval_ind]<X_max[:, X_eval_ind])*X_down[:, X_eval_ind]+
            (X_down[:, X_eval_ind]>=X_max[:, X_eval_ind])*
            (X_max[:, X_eval_ind] - window * X_sd[:, X_eval_ind])
            )
        
        # Adjust for dummies
        X_up[:, np.intersect1d(X_dummy, X_eval_ind)] = np.max(
            X_eval[:, np.intersect1d(X_dummy, X_eval_ind)], axis=0)
        X_down[:, np.intersect1d(X_dummy, X_eval_ind)] = np.min(
            X_eval[:, np.intersect1d(X_dummy, X_eval_ind)], axis=0)
        
        # Adjust for categorical variables
        X_up[:, np.intersect1d(X_cat, X_eval_ind)] = np.ceil(
            X_up[:, np.intersect1d(X_cat, X_eval_ind)])
        X_down[:, np.intersect1d(X_cat, X_eval_ind)] = (
            X_up[:, np.intersect1d(X_cat, X_eval_ind)]-1)

        # check if X_up and X_down are same this should not happen at all
        # increase the window size by 1% iteratively until no X_up==X_down
        wider_window = window + 0.01
        # start the while loop
        while (np.any(X_up[:, X_eval_ind] == X_down[:, X_eval_ind])):
            # adjust to higher share of SD
            X_up[:, X_eval_ind] = (
                (X_up[:, X_eval_ind]>X_down[:, X_eval_ind])*X_up[:, X_eval_ind]+
                (X_up[:, X_eval_ind]==X_down[:, X_eval_ind])*
                (X_up[:, X_eval_ind] + wider_window * X_sd[:, X_eval_ind])
                )
            # check the support (must be before X_down will be adjusted)
            X_up[:, X_eval_ind] = (
                (X_up[:, X_eval_ind]<X_max[:, X_eval_ind])*X_up[:, X_eval_ind]+
                (X_up[:, X_eval_ind]>=X_max[:, X_eval_ind])*X_max[:, X_eval_ind]
                )
            # adjust to higher share of SD
            X_down[:, X_eval_ind] = (
                (X_up[:, X_eval_ind] > X_down[:, X_eval_ind]) *
                X_down[:, X_eval_ind] +
                (X_up[:, X_eval_ind] == X_down[:, X_eval_ind]) *
                (X_down[:, X_eval_ind] - wider_window * X_sd[:, X_eval_ind])
                )
            # check the support
            X_down[:, X_eval_ind] = (
                (X_down[:, X_eval_ind] > X_min[:, X_eval_ind]) *
                X_down[:, X_eval_ind] +
                (X_down[:, X_eval_ind] <= X_min[:, X_eval_ind]) *
                X_min[:, X_eval_ind]
                )
            # increase window size by 1%
            wider_window = wider_window + 0.01
            # check if the new window size is admissible
            if wider_window > 1:
                # break here
                break
        
        ## Compute predictions
        # Create storage arrays to save predictions
        forest_pred_up = np.empty((len(X_eval_ind), self.n_class-1))
        forest_pred_down = np.empty((len(X_eval_ind), self.n_class-1))

        # Case 1: No honesty (= no inference)
        if not self.honesty:
            # loop over evaluation covariates
            for x_id, x_pos in zip(X_eval_ind, range(len(X_eval_ind))):
                # Prepare input matrix where column x_id is adjusted upwards
                X_mean_up = X_mean.copy()
                X_mean_up[:,x_id] = X_up[:,x_id]
                # Compute mean predictions (only needed for eval_point=mean
                # but no change im atmean or atmedian)
                forest_pred_up[x_pos,:] = np.mean(self._predict_default(
                    X=X_mean_up, n_samples=n_samples), axis=0)
                # Prepare input matrix where column x_id is adjusted downwards
                X_mean_down = X_mean.copy()
                X_mean_down[:,x_id] = X_down[:,x_id]
                # Compute mean predictions
                forest_pred_down[x_pos,:] = np.mean(self._predict_default(
                    X=X_mean_down, n_samples=n_samples), axis=0)

        # Case 2: honesty but no inference
        if self.honesty and not self.inference:
            # loop over evaluation covariates
            for x_id, x_pos in zip(X_eval_ind, range(len(X_eval_ind))):
                # Prepare input matrix where column x_id is adjusted upwards
                X_mean_up = X_mean.copy()
                X_mean_up[:,x_id] = X_up[:,x_id]
                # Compute mean predictions (only needed for eval_point=mean
                # but no change im atmean or atmedian)
                forest_pred_up[x_pos,:] = np.mean(self._predict_leafmeans(
                    X=X_mean_up, n_samples=n_samples), axis=0)
                # Prepare input matrix where column x_id is adjusted downwards
                X_mean_down = X_mean.copy()
                X_mean_down[:,x_id] = X_down[:,x_id]
                # Compute mean predictions
                forest_pred_down[x_pos,:] = np.mean(self._predict_leafmeans(
                    X=X_mean_down, n_samples=n_samples), axis=0)

        # Case 3: honesty and inference
        if self.honesty and self.inference:
            # storage container for weight matrices
            forest_weights_up={}
            forest_weights_down={}
            # loop over evaluation covariates
            for x_id, x_pos in zip(X_eval_ind, range(len(X_eval_ind))):
                # Prepare input matrix where column x_id is adjusted upwards
                X_mean_up = X_mean.copy()
                X_mean_up[:,x_id] = X_up[:,x_id]
                # Compute predictions and weights matrix
                forest_pred_up_x_id, forest_weights_up[x_id] = (
                    self._predict_weights(X=X_mean_up, n_samples=n_samples))
                # Compute mean predictions (only needed for eval_point=mean
                # but no change im atmean or atmedian)
                forest_pred_up[x_pos,:] = np.mean(forest_pred_up_x_id, axis=0)
                # Prepare input matrix where column x_id is adjusted downwards
                X_mean_down = X_mean.copy()
                X_mean_down[:,x_id] = X_down[:,x_id]
                # Compute predictions and weights matrix
                forest_pred_down_x_id, forest_weights_down[x_id] = (
                    self._predict_weights(X=X_mean_down, n_samples=n_samples))
                # Compute mean predictions (only needed for eval_point=mean
                # but no change im atmean or atmedian)
                forest_pred_down[x_pos,:] = np.mean(
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
        forest_pred_up_0 = np.hstack((np.zeros((len(X_eval_ind), 1)),
                                      forest_pred_up))
        # postpend vector of ones
        forest_pred_up_1 = np.hstack((forest_pred_up,
                                      np.ones((len(X_eval_ind), 1))))
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
        forest_pred_down_0 = np.hstack((np.zeros((len(X_eval_ind), 1)),
                                        forest_pred_down))
        # postpend vector of ones
        forest_pred_down_1 = np.hstack((forest_pred_down,
                                        np.ones((len(X_eval_ind), 1))))
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
        scaling_factor = np.mean(X_up[:, X_eval_ind] - X_down[:, X_eval_ind],
                                 axis=0).reshape(-1,1)
        # Set scaling factor to 1 for categorical and dummy variables
        # this should be either way the case
        # scaling_factor[np.intersect1d(X_dummy, X_eval_ind), :] = 1
        # scaling_factor[np.intersect1d(X_cat, X_eval_ind), :] = 1
        # Scale the differences to get the marginal effects
        marginal_effects_scaled = forest_pred_diff_up_down / scaling_factor
        
        # redefine all effect results as floats
        margins = marginal_effects_scaled.astype(float)
        
        if self.inference:
            ## variance for the marginal effects
            # compute prerequisities for variance of honest marginal effects
            # squared scaling factor
            scaling_factor_squared = np.square(scaling_factor)
            # Get the size of the honest sample
            n_est = len(ind_est)
            # Create storage container for variance
            variance_me = np.empty((len(X_eval_ind), self.n_class))

            # loop over selected covariates
            for x_id, x_pos in zip(X_eval_ind, range(len(X_eval_ind))):
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
                    outcome_binary_est = self.forest_['outcome_binary_est'][
                        class_idx].reshape(-1,1)
                    # compute the conditional means: 1/N(weights%*%y)
                    # (predictions are based on honest sample)
                    forest_cond_means = np.multiply(
                        (1/len(self.forest_['ind_est'])), np.dot(
                            forest_weights_diff_up_down, outcome_binary_est))

                    # calculate standard multiplication of weights and outcomes
                    forest_multi = np.multiply(
                        forest_weights_diff_up_down,
                        outcome_binary_est.reshape((1, -1)))
                    # subtract the mean from each obs i
                    forest_multi_demeaned[class_idx] = (
                        forest_multi - forest_cond_means)
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
                                scaling_factor_squared[x_pos])

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
                        scaling_factor_squared[x_pos])

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
                # Compute final variance according to: var_last+var_first-cov
                for class_idx in range(1, self.n_class+1, 1):
                    variance_me[x_pos, class_idx-1]  = variance_last[
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

        # put everything into a dict of results
        results = {'output': 'margin',
                   'eval_point': eval_point,
                   'window': window,
                   'effects': margins,
                   'variances': variance_me,
                   'std_errors': sd_me,
                   't-values': t_values,
                   'p-values': p_values}

        # check if marginal effects should be printed
        if verbose:
            string_seq_X = [str(x) for x in X_eval_ind]
            string_seq_cat = [str(x) for x in np.arange(1,self.n_class+1)]

            # print marginal effects nicely
            if not self.inference:
                print('-' * 70,
                      'Marginal Effects of OrderedRandomForest, evaluation point: '+ 
                      eval_point, '-' * 70, 'Effects:', '-' * 70,
                      pd.DataFrame(data=margins, 
                                   index=['X' + sub for sub in string_seq_X], 
                                   columns=['Cat' + sub for sub in string_seq_cat]),
                      '-' * 70, sep='\n')
            else:
                print('-' * 70,
                      'Marginal Effects of OrderedRandomForest, evaluation point: '+ 
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
    def _predict_default(self, X, n_samples):
        # create an empty array to save the predictions
        probs = np.empty((n_samples, self.n_class-1))
        for class_idx in range(1, self.n_class, 1):
            # get in-sample predictions, i.e. out-of-bag predictions
            probs[:,class_idx-1] = self.forest_['forests'][class_idx].predict(
                X=X).squeeze() 
        return probs
        

    #Function to predict via leaf means
    def _predict_leafmeans(self, X, n_samples):
        # create an empty array to save the predictions
        probs = np.empty((n_samples, self.n_class-1))
        # Run new Xs through estimated train forest and compute 
        # predictions based on honest sample. No need to predict
        # weights, get predictions directly through leaf means.
        # Loop over classes
        for class_idx in range(1, self.n_class, 1):
            # Get leaf IDs for new data set
            forest_apply = self.forest_['forests'][class_idx].apply(X)
            # generate grid to read out indices column by column
            grid = np.meshgrid(np.arange(0, self.n_estimators), 
                               np.arange(0, n_samples))[0]
            # assign leaf means to indices
            y_hat = self.forest_['fitted'][class_idx][forest_apply, grid]
            # Average over trees
            probs[:, class_idx-1] = np.mean(y_hat, axis=1)  
        return probs

        
    #Function to predict via weights
    def _predict_weights(self, X, n_samples):
        # create an empty array to save the predictions
        probs = np.empty((n_samples, self.n_class-1))
        # create empty dict to save weights
        weights = {}
        # Step 1: Predict weights by using honest data from fit and
        # newdata (for each category except one)
        # First extract honest data from fit output
        X_est = self.forest_['X_fit'][self.forest_['ind_est'],:]
        # Loop over classes
        for class_idx in range(1, self.n_class, 1):
            # Get leaf IDs for estimation set
            forest_apply = self.forest_['forests'][class_idx].apply(X_est)
            # create binary outcome indicator for est sample
            outcome_ind_est = self.forest_['outcome_binary_est'][class_idx]
            # Get size of estimation sample
            n_est = forest_apply.shape[0]
            # Get leaf IDs for newdata
            forest_apply_all = self.forest_['forests'][class_idx].apply(X)
# =============================================================================
# In the end: insert here weight.method which works best. For now numpy_loop
# =============================================================================
            # self.weight_method == 'numpy_loop':
            # generate storage matrix for weights
            forest_out = np.zeros((n_samples, n_est))
            # Loop over trees
            for tree in range(self.n_estimators):
                tree_out = self._honest_weight_numpy(
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
        item : Nonetype or dict
               Object that should be summarized: Either prediction or margin 
               output or None. If None then forest parameters will be printed.
               
        Returns
        -------
        None.
        """

        # Input checks
        # check if input has been fitted (sklearn function)
        check_is_fitted(self, attributes=["forest_"])
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
            print('%-18s%-15s' % ('type:', 'OrderedRandomForest'))
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
            print('%-18s%-15s' % ('trainsize:', len(self.forest_['ind_tr'])))
            print('%-18s%-15s' % ('honestsize:', len(self.forest_['ind_est'])))
            print('%-18s%-15s' % ('features:', self.n_features))
            print('%-18s%-15s' % ('mse:',np.round(
                float(self.measures['mse']),3)))
            print('%-18s%-15s' % ('rps:',np.round(
                float(self.measures['rps']),3)))
            print('%-18s%-15s' % ('accuracy:',np.round(
                float(self.measures['accuracy']),3)))
            print('-' * 50)
        
        elif item['output']=='predict':
            print('-' * 60, 'Summary of the OrderedRandomForest predictions',
                  '-' * 60, sep='\n')
            print('%-18s%-15s' % ('type:', 
                                  'OrderedRandomForest predictions'))
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
                                  'OrderedRandomForest marginal effects'))
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
            print('-' * 60,'Marginal effects:', 
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
            if item['t-values'] is not None:
                print('t-values:', 
                      pd.DataFrame(data=item['t-values'], 
                                   index=['X' + sub for sub in string_seq_X], 
                                   columns=['Cat' + sub for sub in string_seq_cat]),
                      '-' * 60, sep='\n')
            if item['p-values'] is not None:
                print('p-values:', 
                      pd.DataFrame(data=item['p-values'], 
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
        check_is_fitted(self, attributes=["forest_"])
        # Stack true outcomes and predictions and convert to pandas df
        df_plot = pd.DataFrame(
            np.concatenate((self.forest_['y_fit'].reshape(-1,1),
                            self.forest_['probs']), axis=1))

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
         + ggtitle("Distribution of OrderedRandomForest Probability Predictions")
         )

        # empty return
        return fig


    # performance measures (public method, available to user)
    def performance(self):
        """
        Print the prediction performance based on MSE, RPS and CA.

        Parameters
        ----------
        None.

        Returns
        -------
        None. Prints MSE, RPS, Classification accuracy and confusion matrix.
        """

        # Input checks
        # check if input has been fitted (sklearn function)
        check_is_fitted(self, attributes=["forest_"])
        # print the result
        print('Prediction Performance of Ordered Forest', '-' * 80,
              self.measures, '-' * 80, '\n\n', sep='\n')

        # print the confusion matrix
        print('Confusion Matrix for Ordered Forest', '-' * 80,
              '                         Predictions ', '-' * 80,
              self.confusion, '-' * 80, '\n\n', sep='\n')

        # empty return
        return None
