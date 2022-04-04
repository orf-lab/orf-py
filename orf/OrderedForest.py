# -*- coding: utf-8 -*-
"""
orf: Ordered Random Forest.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Definition of main user class.

"""

from orf._OrderedRandomForest import OrderedRandomForest

class OrderedForest(OrderedRandomForest):
    """
    Ordered Random Forests class labeled 'OrderedForest'.

    includes methods to fit the model, predict and estimate marginal effects.

    Parameters
    ----------
    n_estimators : integer
        Number of trees in the forest. The default is 1000.
    min_samples_leaf : integer
        Minimum leaf size in the forest. The default is 5.
    max_features : float
        Share of random covariates (0,1). The default is 0.3.
    replace : bool
        If True sampling with replacement, i.e. bootstrap is used
        to grow the trees, otherwise subsampling without replacement is used.
        The default is False.
    sample_fraction : float
        Subsampling rate, i.e. the share of samples to draw from
        X to train each tree. The default is 0.5.
    honesty : bool
        If True honest forest is built using sample splitting.
        The default is False.
    honesty_fraction : float
        Share of observations belonging to honest sample not used
        for growing the forest. The default is 0.5.
    inference : bool
        If True the weight based inference is conducted. The
        default is False.
    n_jobs : int or None
        The number of parallel jobs to be used for parallelism;
        follows joblib semantics. `n_jobs=-1` means all - 1 available cpu cores.
        `n_jobs=None` means no parallelism. There is no parallelism implemented
        for `pred_method='numpy'`. The default is -1.
    pred_method : str
        Which method to use to compute honest predictions, one of `'cython'`, 
        `'loop'`, `'numpy'`, `'numpy_loop'`, `'numpy_loop_multi'`, 
        `'numpy_loop_mpire'` or `'numpy_sparse'`. The
        default is `'numpy_loop_mpire'`.
    weight_method : str
        Which method to use to compute honest weights, one of `'numpy_loop'`, 
        `'numpy_loop_mpire'`, `numpy_loop_multi'`, `numpy_loop_shared_multi`
        or `numpy_loop_shared_mpire`. The default is 
        `'numpy_loop_shared_mpire'`.
    random_state : int, None or numpy.random.RandomState object
        Random seed used to initialize the pseudo-random number
        generator. The default is None. See **[numpy documentation](https://numpy.org/doc/stable/reference/random/legacy.html){:target="_blank"}**
        for details.

    Returns
    -------
    None. Initializes parameters for OrderedForest.
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
                 pred_method='numpy_mpire',
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
        
    
    def fit(self, X, y):
        """
        OrderedForest estimation.

        Parameters
        ----------
        X : array-like
            The matrix of covariates.
        y : ndarray
            Vector of outcomes.

        Returns
        -------
        self : object
               The fitted estimator.
               
        Example
        -------       
        ```
        # import modules
        import pandas as pd
        import os
        import numpy as np
        
        # load the ordered forest
        from orf.OrderedForest import OrderedForest
        
        # read in example data from the orf package in R
        odata = pd.read_csv('orf/odata.csv')
        
        # define outcome and features
        outcome = odata['Y']
        features = odata.drop('Y', axis=1)
            
        # Initiate the class with tuning parameters
        oforest = OrderedRandomForest(n_estimators=1000, min_samples_leaf=5,
                                      max_features=0.3, replace=False,
                                      sample_fraction=0.5, honesty=True,
                                      n_jobs=-1, pred_method='numpy_loop',
                                      weight_method='numpy_loop',
                                      inference=True)
        
        # OrderedForest estimation
        oforest.fit(X=features, y=outcome)
        ```
        """
        return super().fit(X=X, y=y)
    
    
    def predict(self, X=None, prob=True):
        """
        OrderedForest prediction.

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
        return super().predict(X=X, prob=prob)
    
    
    def margin(self, X=None, X_cat=None, X_eval=None, eval_point="mean",
               window=0.1, verbose=True):
        """
        OrderedForest marginal effects.

        Parameters
        ----------
        X : array-like or NoneType
            Matrix of new covariates or None if covariates from
            fit function should be used. If new data provided it must have
            the same number of features as the `X` in the fit function.
        X_cat : list or tuple or NoneType
            List or tuple indicating the columns with categorical covariates,
            i.e. `X_cat=(1,)` or `X_cat=[1]` if the second column includes
            categorical values. If not defined, covariates with integer values
            and less than 10 unique values are considered to be categorical as
            default.
        X_eval : list or tuple or NoneType
            List or tuple indicating the columns with covariates for which the,
            marginal effect should be evaluated, i.e. `X_eval=(1,)` or `X_eval=[1]`
            if the effect for the covariate in the column should be evaluated.
            This can significantly speed up the computations. If not defined,
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
                 Use `result.get("...")` with `"effects"`, `"variances"`,
                 `"std_errors"`, `"t-values"` or `"p-values"` to extract arrays
                 of marginal effects, variances, standard errors, t-values or 
                 p-values, respectively.
        """
        return super().margin(X=X, X_cat=X_cat, X_eval=X_eval, 
                              eval_point=eval_point, window=window,
                              verbose=verbose)
    
    
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
        return super().summary(item=item)
    
    
    def plot(self):
        """
        Plot the probability distributions fitted by the OrderedForest

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        return super().plot()
    
    
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
        return super().performance()
