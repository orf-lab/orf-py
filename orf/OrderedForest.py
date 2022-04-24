"""
orf: Ordered Random Forest.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Definition of main user classes.

"""

from orf._OrderedRandomForest import OrderedRandomForest

class OrderedForest(OrderedRandomForest):
    """
    Ordered Random Forests class labeled `OrderedForest()`. Initializes
    parameters for estimation.

    Parameters
    ----------
    n_estimators : int
        The number of trees in the forest. The default is 1000.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node. A split
        point at any depth will only be considered if it leaves at least
        `min_samples_leaf` training samples in each of the left and right
        branches. This may have the effect of smoothing the model.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum number of
          samples for each node.

        The default is 5.
    max_features : float, int or NoneType
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `round(max_features * n_features)` features are considered at each
          split.
        - If None, then `max_features=ceil(sqrt(n_features))`.

        Note: the search for a split does not stop until at least one valid
        partition of the node samples is found, even if it requires to
        effectively inspect more than `max_features` features.

        The default is None.
    replace : bool
        If True, sampling with replacement (i.e. bootstrap) is used
        to grow the trees, otherwise subsampling without replacement is used.
        For bootstrap the core forest algorithm is based on
        [`scikit-learn`](https://scikit-learn.org/stable/){:target="_blank"}
        while [`EconML`](https://econml.azurewebsites.net/){:target="_blank"}
        is used for subsampling without replacement. The default is False.
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
        If True, weight-based inference (i.e. variance estimation and
        uncertainty quantification of the estimates) is conducted. Note, that
        this is a computationally intensive procedure and slows down the
        program. The default is False.
    n_jobs : int or None
        The number of parallel jobs to be used for multithreading in
        [`.fit()`](#orf.OrderedForest.fit),
        [`.predict()`](#orf.OrderedForest.predict) and
        [`.margins()`](#orf.OrderedForest.margins).
        Follows
        [`joblib`](https://joblib.readthedocs.io){:target="_blank"} semantics:

        - `n_jobs=-1` means all - 1 available cpu cores.
        - `n_jobs=None` and `n_jobs=1` means no parallelism.

        The default is -1.
    random_state : int, None or numpy.random.RandomState object
        Random seed used to initialize the pseudo-random number
        generator. See
        [`numpy` documentation](https://numpy.org/doc/stable/reference/random/legacy.html){:target="_blank"}
        for details. The default is None.

    Returns
    -------
    None. Initializes parameters for OrderedForest.


    Notes
    -----
    `OrderedForest()` includes methods to [`.fit()`](#orf.OrderedForest.fit)
    the model, [`.predict()`](#orf.OrderedForest.predict) the probabilities
    and evaluate marginal effects via
    [`.margins()`](#orf.OrderedForest.margins). Furthermore,
    it provides functions to interpret the estimation outputs such as
    [`.summary()`](#orf.OrderedForest.summary),
    [`.plot()`](#orf.OrderedForest.plot) and
    [`.performance()`](#orf.OrderedForest.performance).

    The Ordered Forest estimates the conditional ordered choice probabilities,
    i.e. `P[Y=m|X=x]`. Additionally, weight-based inference for the probability
    predictions can be conducted as well. If inference is desired, the Ordered
    Forest must be estimated with honesty and subsampling. If prediction only
    is desired, estimation without honesty and with bootstrapping is
    recommended for optimal prediction performance.

    In order to estimate the Ordered Forest users must supply the data in form
    of array-like matrix of features `X` and array-like vector of outcomes `y`
    to the [`.fit()`](#orf.OrderedForest.fit)
    function. These data inputs are also the only inputs that
    must be specified by the user without any defaults. Further optional
    arguments for the `OrderedForest()` class include the classical forest
    hyperparameters such as number of trees, `n_estimators`, number of randomly
    selected features at split, `max_features`, and the minimum leaf size,
    `min_samples_leaf`. The forest building scheme is regulated by the
    `replace` argument, meaning bootstrapping if `replace=True` or subsampling
    if `replace=False`. For the case of subsampling, the `sample_fraction`
    argument regulates the subsampling rate. Further, an honest forest is
    estimated if the `honesty` argument is set to `True`, which is also the
    default. Similarly, the fraction of the sample used for the honest
    estimation is regulated by the `honesty_fraction` argument. The default
    setting conducts a 50:50 sample split, which is also generally advised to
    follow for optimal performance. The inference procedure of the Ordered
    Forest is based on the forest weights and is controlled by the `inference`
    argument. Note, that such weight-based inference is a computationally
    demanding exercise due to the estimation of the forest weights and as such
    longer computation time is to be expected. To speed up the estimations
    `n_jobs` provides option for multithreading from the
    [`joblib`](https://joblib.readthedocs.io){:target="_blank"} library.
    Lastly, the `random_state` argument allows to set the seed for
    replicability.

    For further details, see examples below.

    Examples
    --------
    ```py
    # load orf package
    import orf

    # initialize Ordered Forest with default parameters
    oforest = orf.OrderedForest()

    # initialize Ordered Forest with own tuning parameters
    oforest = orf.OrderedForest(n_estimators = 2000, min_samples_leaf = 10,
                                max_features = 3)

    # initialize Ordered Forest with bootstrapping and without honesty
    oforest = orf.OrderedForest(replace = True, honesty = False)

    # initialize Ordered Forest with subsampling and with honesty
    oforest = orf.OrderedForest(replace = False, honesty = True)

    # initialize Ordered Forest with subsampling and with honesty
    # with own tuning for subsample fraction and honesty fraction
    oforest = orf.OrderedForest(replace = False, sample_fraction = 0.5,
                                honesty = True, honesty_fraction = 0.5)

    # initialize Ordered Forest with subsampling, honesty and
    # inference (for inference, subsampling and honesty are required)
    oforest = orf.OrderedForest(replace = False, honesty = True,
                                inference = True)

    # initialize Ordered Forest with all custom settings
    oforest = orf.OrderedForest(n_estimators = 2000, min_samples_leaf = 10,
                                max_features = 3, replace = True,
                                sample_fraction = 1, honesty = False,
                                honesty_fraction = 0, inference = False)
    ```
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

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples (i.e. the matrix of covariates).
            Internally, its dtype will be converted to `dtype=np.float32`.
        y : array-like of shape (n_samples,)
            The ordinal outcome values as integers ranging from `1` up to
            `nclass`

        Returns
        -------
        self : object
               The fitted estimator.

        Notes
        -----
        [`.fit()`](#orf.OrderedForest.fit) estimates the ordered choice model
        via the Ordered Forest estimator and outputs the conditional ordered
        choice probabilities, i.e. `P[Y=m|X=x]`. The user must supply the data
        in form of array-like matrix of features `X` and array-like vector of
        outcomes `y` of ordered classes.

        Examples
        --------
        ```py
        # load orf package
        import orf

        # get example data
        features, outcome  = orf.make_ordered_regression(seed=123)

        # initialize Ordered Forest with default parameters
        oforest = orf.OrderedForest()
        # estimate Ordered Forest
        oforest.fit(X=features, y=outcome)

        # initialize Ordered Forest with own tuning parameters
        oforest = orf.OrderedForest(n_estimators = 2000,
                                    min_samples_leaf = 10,
                                    max_features = 3)
        # estimate Ordered Forest
        oforest.fit(X=features, y=outcome)

        # initialize Ordered Forest with bootstrapping and without honesty
        oforest = orf.OrderedForest(replace = True, honesty = False)
        # estimate Ordered Forest
        oforest.fit(X=features, y=outcome)

        # initialize Ordered Forest with subsampling and with honesty
        oforest = orf.OrderedForest(replace = False, honesty = True)
        # estimate Ordered Forest
        oforest.fit(X=features, y=outcome)

        # initialize Ordered Forest with subsampling and with honesty
        # with own tuning for subsample fraction and honesty fraction
        oforest = orf.OrderedForest(replace = False, sample_fraction = 0.5,
                                    honesty = True, honesty_fraction = 0.5)
        # estimate Ordered Forest
        oforest.fit(X=features, y=outcome)

        # initialize Ordered Forest with subsampling, honesty and
        # inference (for inference, subsampling and honesty are required)
        oforest = orf.OrderedForest(replace = False, honesty = True,
                                    inference = True)
        # estimate Ordered Forest
        oforest.fit(X=features, y=outcome)

        # initialize Ordered Forest with all custom settings
        oforest = orf.OrderedForest(n_estimators = 2000,
                                    min_samples_leaf = 10,
                                    max_features = 3, replace = True,
                                    sample_fraction = 1, honesty = False,
                                    honesty_fraction = 0,
                                    inference = False)
        # estimate Ordered Forest
        oforest.fit(X=features, y=outcome)
        ```
        """
        return super().fit(X=X, y=y)


    def predict(self, X=None, prob=True):
        """
        Prediction for new observations based on the estimated Ordered Forest
        of class `OrderedForest()`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or NoneType
            Matrix of new features/covariates or `None` if covariates from
            fit function should be used. If new data provided, it must have
            the same number of features as the `X` in the
            [`.fit()`](#orf.OrderedForest.fit) function.
        prob : bool
            If True, ordered probabilities are predicted. Otherwise, ordered
            classes are predicted instead. Note that inference is only
            available for probability predictions. The default is True.

        Returns
        -------
        result : dict
                 Dictionary containing prediction results. Use
                 `result.get("predictions")` to extract array of predictions
                 and `result.get("variances")` to extract array of variances.
                 Both of these arrays are of shape `(n_samples, nclass)`.

        Notes
        -----
        [`.predict()`](#orf.OrderedForest.predict) estimates the conditional
        ordered choice probabilities, i.e. `P[Y=m|X=x]` for new data points
        (array-like matrix of features `X` containing new test observations)
        based on the estimated Ordered Forest object of class
        `OrderedForest()`. Furthermore, weight-based inference for the
        probability predictions can be conducted as well, this is
        inherited from the `OrderedForest()` class arguments. If inference is
        desired, the supplied Ordered Forest must be estimated with honesty and
        subsampling. If only prediction is desired, estimation without honesty
        and with bootstrapping is recommended for optimal predictive
        performance. In addition to the probability predictions, class
        predictions can be estimated as well setting `prob=False`. In this
        case, for each observation the class with the highest predicted
        probability is returned.

        Examples
        --------
        ```py
        # load packages
        import orf
        from sklearn.model_selection import train_test_split

        # get example data
        features, outcome  = orf.make_ordered_regression(seed=123)

        # generate train and test set
        X_train, X_test, y_train, y_test = train_test_split(
            features, outcome, test_size=0.2, random_state=123)

        # estimate Ordered Forest
        oforest = orf.OrderedForest().fit(X=X_train, y=y_train)

        # predict the probabilities with the estimated Ordered Forest
        pred = oforest.predict(X=X_test)
        # return predictions as array
        print(pred.get("predictions"))

        # predict the classes with estimated Ordered Forest
        pred_class = oforest.predict(X=X_test, prob=False)
        # return predictions as array
        pred_class.get("predictions")

        # estimate Ordered Forest using inference
        oforest = orf.OrderedForest(inference=True).fit(X=X_train, y=y_train)

        # predict the probabilities together with variances
        pred_inf = oforest.predict(X=X_test)
        # return predictions as array
        print(pred_inf.get("predictions"))
        # return variances as array
        print(pred_inf.get("variances"))
        ```
        """
        return super().predict(X=X, prob=prob)


    def margins(self, X=None, X_cat=None, X_eval=None, eval_point="mean",
                window=0.1, verbose=True):
        """
        Evaluation of marginal effects based on the estimated Ordered Forest
        of class `OrderedForest()`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or NoneType
            Matrix of new features/covariates or `None` if covariates from
            fit function should be used. If new data provided, it must have
            the same number of features as the `X` in the
            [`.fit()`](#orf.OrderedForest.fit) function.
        X_cat : list or tuple or NoneType
            List or tuple indicating the columns with categorical covariates,
            i.e. `X_cat=(1,)` or `X_cat=[1]` if the second column includes
            categorical values. If not defined, covariates with integer values
            and less than 10 unique values are considered to be categorical as
            default.
        X_eval : list or tuple or NoneType
            List or tuple indicating the columns with covariates for which the,
            marginal effect should be evaluated, i.e. `X_eval=(0,)` or
            `X_eval=[0]` if the effect for the covariate in the first column
            should be evaluated. This can significantly speed up the program.
            If not defined, all covariates are considered as default.
        eval_point: string
            Defining evaluation point for marginal effects. This can be one
            of `"mean"`, `"atmean"`, or `"atmedian"`. The default is `"mean"`.
        window : float
            The share of the standard deviation of `X` to be used for
            evaluation of the marginal effect. The default is `0.1`.
        verbose : bool
            Should the results printed to console? The default is True.

        Returns
        -------
        result : dict
                 Dictionary containing results of marginal effects estimation.
                 Use `result.get("...")` with `"effects"`, `"variances"`,
                 `"std_errors"`, `"t-values"`, `"p-values"`, `"ci-up"` or
                 `"ci-down"` to extract arrays of marginal effects, variances,
                 standard errors, t-values, p-values or upper and lower
                 confidence intervals, respectively. All of these arrays are
                 of shape `(n_samples, nclass)`.

        Notes
        -----
        [`.margins()`](#orf.OrderedForest.margins) evaluates marginal effects
        at the mean, at the median, or the mean marginal effects, depending
        on the `eval_point` argument. For a greater flexibility in the marginal
        effects comptation, the argument `X_eval` controls for which features
        the marginal effects should be evaluated. If not defined, the marginal
        effects of all features are computed which might be computationally
        expensive. Additionally, the evaluation window for the marginal effects
        can be regulated through the `window` argument. Furthermore, the user
        might specify which features/covariates should be explicitly handled
        as categorical via the `X_cat` argument. Moreover, new test data for
        which marginal effects should be evaluated can be supplied via the `X`
        argument as long as it lies within the support of the training `X`
        data. In addition to the estimation of the marginal effects, the
        weight-based inference for the effects is supported as well, this is
        inherited from the `OrderedForest()` class arguments. Note, that the
        inference procedure is a computationally exhausting exercise
        due to the computation of the forest weights. It is advised to increase
        the number of subsampling replications in the supplied `OrderedForest()`
        object as the estimation of the marginal effects is a more demanding
        exercise than a simple Ordered Forest estimation/prediction.

        Examples
        --------
        ```py
        # load packages
        import orf

        # get example data
        features, outcome  = orf.make_ordered_regression(seed=123)

        # estimate Ordered Forest
        oforest = orf.OrderedForest().fit(X=features, y=outcome)

        # estimate default (mean) marginal effects for all features
        marg = oforest.margins()

        # return mean marginal effects as array
        print(marg.get("effects"))

        # estimate mean marginal effects, explicitly defining the second
        # column of the features as categorical
        marg = oforest.margins(X_cat=[1])

        # estimate mean marginal effects for the first and third column of X
        marg = oforest.margins(X_eval=[0,2])

        # estimate marginal effects at the mean and at the median
        marg_atmean = oforest.margins(eval_point="atmean")
        marg_atmedian = oforest.margins(eval_point="atmedian")

        # estimate Ordered Forest using inference
        oforest = orf.OrderedForest(inference=True).fit(X=features, y=outcome)

        # estimate mean marginal effects for the first column of X
        marg = oforest.margins(X_eval=[0])

        # return marginal effects as array
        print(marg.get("effects"))
        # return variances as array
        print(marg.get("variances"))
        # return standard errors as array
        print(marg.get("std_errors"))
        # return t-values as array
        print(marg.get("t-values"))
        # return p-values as array
        print(marg.get("p-values"))
        # return upper confidence intervals as array
        print(marg.get("ci-up"))
        # return lower confidence intervals as array
        print(marg.get("ci-down"))
        ```
        """
        return super().margins(X=X, X_cat=X_cat, X_eval=X_eval,
                               eval_point=eval_point, window=window,
                               verbose=verbose)


    def summary(self, item=None):
        """
        Summary of estimated Ordered Forest object of class `OrderedForest()`.

        Parameters
        ----------
        item : dict or NoneType
               Object that should be summarized: Either prediction or margins
               output or None. If None, then forest parameters will be printed.
               The default is None.

        Returns
        -------
        None. Prints summary to console.

        Notes
        -----
        [`.summary()`](#orf.OrderedForest.summary) provides a short summary of
        the Ordered Forest estimation, including the input information
        regarding the values of hyperparameters as well as the output
        information regarding the prediction accuracy.

        Examples
        --------
        ```py
        # load package
        import orf

        # get example data
        features, outcome  = orf.make_ordered_regression(seed=123)

        # estimate Ordered Forest
        oforest = orf.OrderedForest().fit(X=features, y=outcome)

        # print summary of estimation
        oforest.summary()

        # predict the probabilities with the estimated Ordered Forest
        pred = oforest.predict()

        # print summary of the Ordered Forest predictions
        oforest.summary(pred)

        # estimate marginal effects for first feature
        marg = oforest.margins(X_eval = [0])

        # print summary of the marginal effects
        oforest.summary(marg)
        ```
        """
        return super().summary(item=item)


    def plot(self):
        """
        Plot the probability distributions estimated by the Ordered Forest
        object of class `OrderedForest()`.


        Parameters
        ----------
        None.

        Returns
        -------
        fig : object of type ggplot
            Plot of probability distributions.

        Notes
        -----
        [`.plot()`](#orf.OrderedForest.plot) generates probability
        distributions, i.e. density plots of the estimated ordered
        probabilities by the Ordered Forest for each outcome
        class considered. The plots effectively visualize the estimated
        probability density in contrast to a real observed ordered outcome
        class and as such provide a visual inspection of the overall in-sample
        estimation accuracy. The dashed lines locate the means of the
        respective probability distributions.

        Examples
        --------
        ```py
        # load package
        import orf

        # get example data
        features, outcome  = orf.make_ordered_regression(seed=123)

        # estimate Ordered Forest
        oforest = orf.OrderedForest().fit(X=features, y=outcome)

        # plot the estimated probability distributions
        oforest.plot()
        ```
        """
        return super().plot()


    def performance(self):
        """
        Print the prediction performance of the Ordered Forest object of class
        `OrderedForest()`.

        Parameters
        ----------
        None.

        Returns
        -------
        None. Prints MSE, RPS, Classification accuracy and confusion matrix.

        Notes
        -----
        [.performance()`](#orf.OrderedForest.performance) evaluates the
        probability and class predictions in terms of Ranked Probability
        Score (RPS), Mean Squared Error (MSE) and Classification Accuracy (CA).
        In addition, it prints the confusion matrix.

        Examples
        --------
        ```py
        # load package
        import orf

        # get example data
        features, outcome  = orf.make_ordered_regression(seed=123)

        # estimate Ordered Forest
        oforest = orf.OrderedForest().fit(X=features, y=outcome)

        # print the prediction performance measures
        oforest.performance()
        ```
        """
        return super().performance()
