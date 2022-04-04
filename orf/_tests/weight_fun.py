"""
orf: Ordered Random Forests.

Test file for weight computation.

"""

# %% import libraries
import time

import numpy as np

from spyder_kernels.utils.iofuncs import load_dictionary
from orf.OrderedRandomForest import OrderedRandomForest
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import shared_memory, Manager  # Requires Python >= 3.8
from multiprocessing import Pool, cpu_count, Lock, shared_memory
from numba import jit

# define functions to parallelize
def _honest_weight_numpy_out(tree, forest_apply, forest_apply_all, n_samples,
                             n_est):
    """Compute the honest weights using numpy."""

    # extract vectors of leaf IDs
    leaf_IDs_honest = forest_apply[:, tree]
    leaf_IDs_all = forest_apply_all[:, tree]
    # Take care of cases where not all train leafs
    # populated by observations from honest sample
    leaf_IDs_honest_u = np.unique(leaf_IDs_honest)
    leaf_IDs_all_u = np.unique(leaf_IDs_all)
    if np.array_equal(leaf_IDs_honest_u, 
                      leaf_IDs_all_u):
        leaf_IDs_honest_ext = leaf_IDs_honest
        leaf_IDs_all_ext = leaf_IDs_all
    else:
        # Find leaf IDs in all that are not in honest
        extra_honest = np.setdiff1d(
            leaf_IDs_all_u, leaf_IDs_honest_u)
        leaf_IDs_honest_ext = np.append(
            leaf_IDs_honest, extra_honest)
        # Find leaf IDs in honest that are not in all
        extra_all = np.setdiff1d(
            leaf_IDs_honest_u, leaf_IDs_all_u)
        leaf_IDs_all_ext = np.append(
            leaf_IDs_all, extra_all)
    # Generate onehot matrices
    onehot_honest = OneHotEncoder(
        sparse=True).fit_transform(
            leaf_IDs_honest_ext.reshape(-1, 1)).T
    onehot_all = OneHotEncoder(
        sparse=True).fit_transform(
            leaf_IDs_all_ext.reshape(-1, 1))
    onehot_all = onehot_all[:n_samples,:]
    # Multiply matrices
    # (n, n_leafs)x(n_leafs, n_est)
    tree_out = onehot_all.dot(onehot_honest).todense()
    # Get leaf sizes
    # leaf size only for honest sample !!!
    leaf_size = tree_out.sum(axis=1)
    # Delete extra observations for unpopulated
    # honest leaves
    if not np.array_equal(
            leaf_IDs_honest_u, leaf_IDs_all_u):
        tree_out = tree_out[:n_samples, :n_est]
    # Compute weights
    tree_out = tree_out/leaf_size

    return tree_out