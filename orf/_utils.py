"""
Helper functions for OrderedForest
"""

import numpy as np
import pandas as pd

# Define function to produce data sets of different size
def example_data(n_samples=1000, 
                 y_classes=3,
                 p_cont=1, 
                 p_cat=1, 
                 cat_classes=3,
                 p_binary=1,
                 noise=True, 
                 seed=None):
    """
    Generate example data for Ordered Forest estimation.

    Parameters
    ----------
    n_samples : integer
        The number of observations. The default is 1000.
    y_classes : integer
        The number of classes of the outcome variable. The default is 3.
    p_cont : integer
        The number of continuous covariates drawn from a normal distribution.
        The default is 1.
    p_cat : integer
        The number of categorical covariates drawn from a binomial 
        distribution. The default is 1.
    cat_classes : integer
        The number of classes of the categorical variable(s). The default is 3.
    p_binary : integer
        The number of binary covariates drawn from a binomial distribution.
        The default is 1.
    noise : boolean
        Whether to include a continuous noise variable that does not influence
        the outcome. The default is True.
    seed : integer or NoneType
        Set seed for reproducability. The default is None.

    Returns
    -------
    X : ndarray
        The generated covariates/features.
    y : ndarray
        The generated outcomes.
        
    Notes
    -------
    This functions generates an example dataset of size `n_sample`, consisting
    of an ordered outcome variable with `y_classes` classes and an array of 
    features of different types. The data-generating process (DGP) may include
    continuous (`p_cont`), binary (`p_binary`) and categorical (`p_cat`)
    features. In addition, it is possible to include random noise in the 
    outcome variable by specifying `noise=True`. 
    
    Example
    -------
    ```py
    # load orf package
    import orf

    # generate dataset consisting of 2000 observations and 4 outcome classes
    features, outcome  = orf.example_data(n_samples=2000,
                                          y_classes=4,
                                          seed=123)
    ```
    """

    # Set seed
    seed = np.random.default_rng(seed=seed)
    # Draw covariates
    cont = seed.normal(0, 1, size=(n_samples, p_cont))
    cat = seed.binomial(cat_classes-1, 0.5, size=(n_samples, p_cat))+1
    binary = seed.binomial(1, 0.5, size=(n_samples, p_binary))
    # Combine deterministic covariates
    X_det = np.hstack([cont, cat, binary])
    if noise:
        X = np.hstack([X_det, seed.normal(0, 10, size=(n_samples, 1))])
    else:
        X = X_det
    # Generate continuous outcome with logistic error
    y = np.sum(X_det, axis=1) + seed.logistic(0, 1, n_samples)
    # Thresholds for continuous outcome
    y = pd.qcut(y, y_classes, labels=False)+1
    # Return X and Y
    return X, y

