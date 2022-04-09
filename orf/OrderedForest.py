# -*- coding: utf-8 -*-
"""
orf: Ordered Random Forest.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Definition of main user classes.

"""

from orf._OrderedRandomForest import OrderedRandomForest

class OrderedForest(OrderedRandomForest):
    """
    Ordered Random Forests class labeled `OrderedForest()`.

    `OrderedForest()` includes methods to `.fit()` the model, `.predict()` the
    probabilities and evaluate marginal effects via `.margins()`. Furthermore, 
    it provides function to interpret the estimation outputs such as 
    `.summary()`, `.plot()` and `.performance()`.
    
    The Ordered Forest estimates the conditional ordered choice probabilities, 
    i.e. P[Y=m|X=x]. Additionally, weight-based inference for the probability 
    predictions can be conducted as well. If inference is desired, the Ordered 
    Forest must be estimated with honesty and subsampling. If prediction only 
    is desired, estimation without honesty and with bootstrapping is 
    recommended for optimal prediction performance.

    In order to estimate the Ordered Forest user must supply the data in form 
    of array-like matrix of features `X` and array-like vector of outcomes `y` 
    to the `.fit()` function. These data inputs are also the only inputs that 
    must be specified by the user without any defaults. Further optional 
    arguments for the `OrderedForest()` class include the classical forest 
    hyperparameters such as number of trees, `n_estimators`, number of randomly
    selected features at split, `max_features`, and the minimum leaf size, 
    `min_samples_leaf`. The forest building scheme is regulated by the 
    `replace` argument, meaning bootstrapping if `replace=True` or subsampling 
    if `replace=False`. For the case of subsampling, `sample_fraction` argument
    regulates the subsampling rate. Further, honest forest is estimated if the 
    `honesty` argument is set to `True`, which is also the default. Similarly, 
    the fraction of the sample used for the honest estimation is regulated by 
    the `honesty_fraction` argument. The default setting conducts a 50:50 
    sample split, which is also generally advised to follow for optimal 
    performance. Inference procedure of the Ordered Forest is based on the 
    forest weights and is controlled by the `inference` argument. Note, that 
    such weight-based inference is computationally demanding exercise due to 
    the estimation of the forest weights and as such longer computation time is
    to be expected. To speed up the estimations `n_jobs` provides option for
    multithreading from the `joblib` library. Lastly, the `random_state` 
    argument allows to set the seed for replicability.

    For further details, see examples below.

    Parameters
    ----------
    n_estimators : integer
        Number of trees in the forest. The default is 1000.
    min_samples_leaf : integer
        Minimum leaf size in the forest. The default is 5.
    max_features : float, int or None
        Share (0,1) or number of randomly chosen covariates at each split.
        The default (None) is (rounded up) square root of number of covariates.
    replace : bool
        If True sampling with replacement, i.e. bootstrap is used
        to grow the trees, otherwise subsampling without replacement is used.
        The default is False.
    sample_fraction : float
        Subsampling rate, i.e. the share of samples to draw from
        the training data to build each tree. The default is 0.5.
    honesty : bool
        If True honest forest is built using sample splitting.
        The default is True.
    honesty_fraction : float
        Share of observations belonging to honest sample not used
        for growing the forest. The default is 0.5.
    inference : bool
        If True the weight based inference is conducted. The
        default is False.
    n_jobs : int or None
        The number of parallel jobs to be used for multithreading;
        follows joblib semantics. `n_jobs=-1` means all - 1 available cpu cores.
        `n_jobs=None` and `n_jobs=1` means no parallelism. The default is -1.
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
                 max_features=None,
                 replace=False,
                 sample_fraction=0.5,
                 honesty=True,
                 honesty_fraction=0.5,
                 inference=False,
                 n_jobs=-1,
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
            random_state=random_state
        )
        
    
    def fit(self, X, y):
        """
        Estimation of the ordered choice model via the Ordered Forest
        estimator of class `OrderedForest()`.
        
        `.fit()` estimates the ordered choice model via the Ordered Forest 
        estimator and outputs the conditional ordered choice probabilities, i.e.
        P[Y=m|X=x]. The user must supply the data in form of array-like matrix
        of features `X` and array-like vector of outcomes `y` of ordered
        classes.

        Parameters
        ----------
        X : array-like
            The matrix of covariates.
        y : ndarray
            Vector of outcomes.

        Returns
        -------
        self : object
               The fitted model.
               
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
                                      n_jobs=-1, inference=True)
        
        # OrderedForest estimation
        oforest.fit(X=features, y=outcome)
        ```
        """
        return super().fit(X=X, y=y)
    
    
    def predict(self, X=None, prob=True):
        """
        Prediction for new observations based on the estimated Ordered Forest
        of class `OrderedForest()`.
        
        `.predict()` estimates the conditional ordered choice probabilities,
        i.e. P[Y=m|X=x] for new data points (array-like matrix of features X 
        containing new test observations) based on the estimated Ordered Forest
        object of class `OrderedForest()`. Furthermore, weight-based inference
        for the probability predictions can be conducted as well, this is 
        inherited from the `OrderedForest()` class arguments. If inference is 
        desired, the supplied Ordered Forest must be estimated with honesty and
        subsampling. If prediction only is desired, estimation without honesty 
        and with bootstrapping is recommended for optimal prediction 
        performance. Additionally to the probability predictions, class 
        predictions can be estimated as well setting `prob=False`. In this
        case, the predicted classes are obtained as classes with the highest 
        predicted probability.

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
    
    
    def margins(self, X=None, X_cat=None, X_eval=None, eval_point="mean",
                window=0.1, verbose=True):
        """
        Evaluation of marginal effects based on the estimated Ordered Forest
        of class `OrderedForest()`.
        
        `.margins()` evaluates marginal effects at the mean, at the median, or 
        the mean marginal effects, depending on the `eval_point` argument. For 
        a greater flexibility in the marginal effects comptation, the argument 
        `X_eval` controls for which features the marginal effects should be 
        evaluated. Additionally, the evaluation window for the marginal effects
        can be regulated through the `window` argument. Furthermore, the user
        might specify which features should be handled as categorical ones
        explicitly via the `X_cat` argument. Moreover, new test data for which 
        marginal effects should be evaluated can be supplied as well via `X`
        argument as long as it lies within the support of the training X data.
        Additionally to the estimation of the marginal effects, the 
        weight-based inference for the effects is supported as well, this is 
        inherited from the `OrderedForest()` class arguments. Note, that the 
        inference procedure is much more computationally exhausting exercise 
        due to the computation of the forest weights. It is advised to increase
        the number of subsampling replications in the supplied `OrderedForest()`
        object as the estimation of the marginal effects is a more demanding 
        exercise than a simple Ordered Forest estimation/prediction.
        

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
                 Use result.get("...") with "effects", "variances",
                 "std_errors", "t-values", "p-values", "ci-up" or "ci-down" to 
                 extract arrays of marginal effects, variances, standard
                 errors, t-values, p-values or upper and lower confidence
                 intervals, respectively.
        """
        return super().margins(X=X, X_cat=X_cat, X_eval=X_eval, 
                               eval_point=eval_point, window=window,
                               verbose=verbose)
    
    
    def summary(self, item=None):
        """
        Summary of estimated Ordered Forest object of class `OrderedForest()`.
        
        `.summary()` provides a short summary of the Ordered Forest estimation,
        including the input information regarding the values of hyperparameters
        as well as the output information regarding the prediction accuracy.

        Parameters
        ----------
        item : Nonetype or dict
               Object that should be summarized: Either prediction or margins 
               output or None. If None then forest parameters will be printed.
               
        Returns
        -------
        None.
        """
        return super().summary(item=item)
    
    
    def plot(self):
        """
        Plot the probability distributions estimated by the Ordered Forest 
        object of class `OrderedForest()`.
        
        `.plot()` generates probability distributions, i.e. density plots of 
        estimated ordered probabilities by the Ordered Forest for each outcome 
        class considered. The plots effectively visualize the estimated 
        probability density in contrast to a real observed ordered outcome 
        class and as such provide a visual inspection of the overall in-sample 
        estimation accuracy. The dashed lines locate the means of the 
        respective probability distributions.

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
        Print the prediction performance of the Ordered Forest object of class
        `OrderedForest()`.
        
        .performance()` evaluates the probability and class predictions in
        terms of Ranked Probability Score (RPS), Mean Squared Error (MSE) and
        Classification Accuracy (CA).

        Parameters
        ----------
        None.

        Returns
        -------
        None. Prints MSE, RPS, Classification accuracy and confusion matrix.
        """
        return super().performance()
