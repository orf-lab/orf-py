# -*- coding: utf-8 -*-
"""
orf: Ordered Random Forest.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Definition of main user class.

"""

from orf.OrderedForest import OrderedForest

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
