"""
Helper functions for OrderedForest
"""

import numpy as np
import pandas as pd

# Define function to produce data sets of different size
def example_data(seed=None, 
                 n_samples=1000, 
                 p_cont=1, 
                 p_cat=1, 
                 p_binary=1, 
                 noise=True, 
                 y_classes=3,
                 cat_classes=3):
    """
    Generate example data for OrderedForest.

    Parameters
    ----------
    seed : integer or NoneType
        Set seed for reproducability.
    n_samples : integer
        The number of observations.
    p_cont : integer
        The number of continuous covariates drawn from a normal distribution.
    p_cat : integer
        The number of categorical covariates drawn from a binomial distribution.
    p_binary : integer
        The number of binary covariates drawn from a binomial distribution.
    noise : boolean
        Whether to include a continuous noise variable that does not influence
        the outcome. The default is True.
    y_classes : integer
        The number of classes of the outcome variable. The default is 3.
    cat_classes : integer
        The number of classes of the categorical variable(s). The default is 3.

    Returns
    -------
    X : ndarray
        The generated covariates/features.
    y : ndarray
        The generated outcomes.

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

