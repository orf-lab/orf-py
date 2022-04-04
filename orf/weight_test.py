"""
orf: Ordered Random Forests.

Test file for weight computation.

"""

# %% import libraries
import os

path = "/Users/okasag/Desktop/HSG/orf/python/ORFpy"
os.chdir(path)

import time
import orf.weight_fun as weightfuns

import numpy as np

from spyder_kernels.utils.iofuncs import load_dictionary
from orf.OrderedRandomForest import OrderedRandomForest
from sklearn.preprocessing import OneHotEncoder
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import shared_memory, Manager  # Requires Python >= 3.8
from multiprocessing import Pool, cpu_count, Lock, shared_memory
from numba import jit
import multiprocess as mp
from functools import partial
from mpire import WorkerPool

# multiprocessing with shared memory
_lock = Lock()  # initiate lock

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


# using shared memory
def _forest_weights_shared(tree, forest_apply, forest_apply_all, n_samples,
                           n_est, forest_out, lock):
    lock.acquire()
    # perform the parallel task
    forest_out += _honest_weight_numpy_out(tree, forest_apply,
                                               forest_apply_all, n_samples,
                                               n_est)
    lock.release()
    return


# %% load the testing data for weight computation
spydata = '/Users/okasag/Desktop/HSG/orf/python/ORFpy/orf/weight_test_data.spydata'
weight_test_data = load_dictionary(spydata)[0]
# unpack the data to global environment
for key in weight_test_data:
    globals()[key] = weight_test_data[key]


# %% Standard Loop

# generate storage matrix for weights
forest_out_loop = np.zeros((n_samples, n_est))
# Loop over trees
for tree in range(self.n_estimators):
    # get honest tree weights
    forest_out_loop += _honest_weight_numpy_out(
        tree=tree,
        forest_apply=forest_apply,
        forest_apply_all=forest_apply_all,
        n_samples=n_samples,
        n_est=n_est)

# %% standard parallel by joblib
forest_out_joblib = sum(
    Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(
        _honest_weight_numpy_out)(
        tree=tree,
        forest_apply=forest_apply,
        forest_apply_all=forest_apply_all,
        n_samples=n_samples,
        n_est=n_est) for tree in range(self.n_estimators)))

# %% divide and conquer with multiprocessing
# depending on number of cores divide the loops
effective_jobs = self.n_jobs
while (np.mod(
        self.n_estimators, effective_jobs) != 0):
    # decrease number of cores to use
    effective_jobs = effective_jobs - 1
    # break if effective_jobs are equal to 1
    if (effective_jobs == 1):
        break
# use parralel to do effective jobs in chunks
n_chunks = int(self.n_estimators/effective_jobs)
chunk_range = np.arange(self.n_estimators)
start_tree = 0
stop_tree = effective_jobs
# create the shared object of forest weights dim
forest_out_multi = np.zeros((n_samples, n_est))
# serialize chunks in a loop
for tree_chunk in range(n_chunks):
    if __name__ == '__main__':
        # setup the pool for multiprocessing
        pool = mp.Pool(effective_jobs)
        # prepare iterables (need to replicate fixed items)
        args_iter = []
        for tree in chunk_range[start_tree:stop_tree]:
            args_iter.append((tree, forest_apply, forest_apply_all, n_samples, n_est))
        # loop over trees in parallel
        tree_chunk_out = sum(pool.starmap(weightfuns._honest_weight_numpy_out, args_iter))
        pool.close()  # close parallel
        pool.join()  # join parallel
        # adjust start and stop trees
        start_tree += effective_jobs
        stop_tree += effective_jobs
        # sum up all tree weights
        forest_out_multi += tree_chunk_out

# %% divide and conquer with mpire
# depending on number of cores divide the loops
effective_jobs = self.n_jobs
while (np.mod(
        self.n_estimators, effective_jobs) != 0):
    # decrease number of cores to use
    effective_jobs = effective_jobs - 1
    # break if effective_jobs are equal to 1
    if (effective_jobs == 1):
        break
# use parralel to do effective jobs in chunks
n_chunks = int(self.n_estimators/effective_jobs)
chunk_range = np.arange(self.n_estimators)
start_tree = 0
stop_tree = effective_jobs
# create the shared object of forest weights dim
forest_out_mpire = np.zeros((n_samples, n_est))
# serialize chunks in a loop
for tree_chunk in range(n_chunks):
    # define partial function by fixing parameters
    partial_fun = partial(
        weightfuns._honest_weight_numpy_out,
        forest_apply=forest_apply,
        forest_apply_all=forest_apply_all,
        n_samples=n_samples,
        n_est=n_est)
    # set up the worker pool for parallelization
    pool = WorkerPool(n_jobs=effective_jobs)
    # loop over trees in parallel
    tree_chunk_out = sum(pool.map(
        partial_fun, chunk_range[start_tree:stop_tree],
        progress_bar=False,
        concatenate_numpy_output=False))
    # stop and join pool
    pool.stop_and_join()
    # adjust start and stop trees
    start_tree += effective_jobs
    stop_tree += effective_jobs
    # sum up all tree weights
    forest_out_mpire += tree_chunk_out

# %% divide and conquer with joblib
# depending on number of cores divide the loops
effective_jobs = self.n_jobs
while (np.mod(
    self.n_estimators, effective_jobs) != 0):
    # decrease number of cores to use
    effective_jobs = effective_jobs - 1
    # break if effective_jobs are equal to 1
    if (effective_jobs == 1):
        break
# use parralel to do effective jobs in chunks
n_chunks = int(self.n_estimators/effective_jobs)
chunk_range = np.arange(self.n_estimators)
start_tree = 0
stop_tree = effective_jobs
# create the shared object of forest weights dim
forest_out_joblib_2 = np.zeros((n_samples, n_est))
# serialize chunks in a loop
for tree_chunk in range(n_chunks):
    # generate storage matrix for weights
    tree_chunk_out = sum(
        Parallel(n_jobs=effective_jobs,
                 backend='threading')(delayed(
                     self._honest_weight_numpy)(
                         tree=tree,
                         forest_apply=forest_apply,
                         forest_apply_all=forest_apply_all,
                         n_samples=n_samples,
                         n_est=n_est) for tree in chunk_range[start_tree:stop_tree]))
    # adjust start and stop trees
    start_tree += effective_jobs
    stop_tree += effective_jobs
    # sum up all tree weights
    forest_out_joblib_2 += tree_chunk_out

# %% joblib with shared memory
# start paralell for loop
if __name__ == "__main__":
  forest_out_shared = np.zeros([n_samples, n_est])
  Parallel(n_jobs=self.n_jobs, backend="threading")(delayed(_forest_weights_shared)(
      tree=tree, forest_apply=forest_apply, forest_apply_all=forest_apply_all,
      n_samples=n_samples, n_est=n_est, forest_out=forest_out_shared, lock=_lock
      ) for tree in range(self.n_estimators))

# %% compare the outputs of weights
all_weights = [forest_out_mpire, forest_out_multi, forest_out_joblib,
               forest_out_joblib_2, forest_out_shared]
all_methods = ['mpire', 'multi', 'joblib', 'joblib_conquer', 'joblib_shared']
# check equality of results
for idx in range(len(all_methods)):
    if (np.round(np.sum(all_weights[idx] - forest_out_loop), 5) == 0):
        print('Weights from ' + all_methods[idx] + ' are identical.')
    else:
        print('Weights from ' + all_methods[idx] + ' are NOT identical.')
