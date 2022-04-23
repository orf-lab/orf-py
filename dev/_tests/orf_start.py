# Code to start debugging of orf.py
# After running this code it should be possible to run snippets within the
# OrderedForest class

# import modules
import pandas as pd
import os
import numpy as np
# path = "D:/switchdrive/Projects/ORF_Python/ORFpy"
# path = "/home/okasag/Documents/HSG/ORF/python/ORFpy"
path = '/Users/okasag/Desktop/HSG/orf/python/ORFpy'
os.chdir(path)

# load the ordered forest
from orf.OrderedForest import OrderedForest

# initiate the class with tuning parameters
self = OrderedForest(n_estimators=1000, min_samples_leaf=5, max_features=0.3,
                     replace=False, sample_fraction=0.5, honesty=True,
                     n_jobs=1, pred_method='numpy_loop', inference=False,
                     weight_method='numpy_loop')

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



# Generate data set
X, y = example_data(seed=123, n=1000, p_cont=1, p_cat=1,
                    p_binary=1, noise=True, y_cat=3,
                    cat_cat=3)

class_idx = 1

# save the dataset for comparison reasons as csv
example_df = pd.DataFrame(pd.concat([y, X], axis=1, ignore_index=True)).rename(
    columns={0: 'y', 1: 'x1', 2: 'x2', 3: 'x3', 4: 'x4'})
example_df.to_csv('dev/_R/example_df.csv', index=False)
