# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 08:56:33 2022

@author: FMuny
"""

import numpy as np
import pandas as pd

# Define function to produce data sets of different size
def example_data(seed, n, p_cont, p_cat, p_binary, noise=True, y_cat=3,
                 cat_cat=3):
    """
    Generate example data to test orf

    Parameters
    ----------
    seed : TYPE: integer
        DESCRIPTION: Set seed for reproducability.
    n : TYPE: integer
        DESCRIPTION: The number of observations.
    p_cont : TYPE: integer
        DESCRIPTION: The number of continuous covariates.
    p_cat : TYPE: integer
        DESCRIPTION: The number of categorical covariates.
    p_binary : TYPE: integer
        DESCRIPTION. The number of binary covariates.
    noise : TYPE_ boolean, optional
        DESCRIPTION. Whether to include a continuous noise variable. The
        default is True.
    y_cat : TYPE: integer, optional
        DESCRIPTION. The number of categories of the outcome variable. The
        default is 3.
    cat_cat : TYPE: integer, optional
        DESCRIPTION. The number of categories of the categorical variable. The
        default is 3.

    Returns
    -------
    Example dataset.

    """
    # Set seed
    seed = np.random.default_rng(seed=seed)
    # Draw covariates
    cont = seed.normal(0, 1, size=(n, p_cont))
    cat = seed.binomial(cat_cat-1, 0.5, size=(n, p_cat))+1
    binary = seed.binomial(1, 0.5, size=(n, p_binary))
    # Combine deterministic covariates
    X_det = np.hstack([cont, cat, binary])
    if noise:
        X = np.hstack([X_det, seed.normal(0, 1, size=(n, 1))])
    else:
        X = X_det
    # Generate continuous outcome with logistic error
    Y = np.sum(X_det, axis=1) + seed.logistic(0, 1, n)
    # Thresholds for continuous outcome
    Y = pd.qcut(Y, y_cat, labels=False)+1
    # Return X and Y
    return pd.DataFrame(X), pd.Series(Y)

