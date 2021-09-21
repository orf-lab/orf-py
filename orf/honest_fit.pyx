#!python
#cython: language_level=3

import numpy
import cython
cimport numpy

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

# Define function to check equality of values
cdef int equal_to(numpy.int64_t x, long k):
    return x == k

# Define function to compute leaf means
def honest_fit(numpy.ndarray[numpy.int64_t, ndim=2] forest_apply,
               numpy.ndarray[numpy.int32_t, ndim=1] outcome_ind_est,
               int trees,
               long max_id):
    # Define data types
    cdef numpy.ndarray[numpy.float64_t, ndim=1] leaf_means
    cdef long idx
    cdef long n_obs
    cdef long obs
    cdef long outcome_id 
    cdef long n_id 
    # create an empty array to save the leaf means
    leaf_means = numpy.empty(max_id)
    # get the number of observations
    n_obs = forest_apply.shape[0]
    
    # loop over leaf indices
    for idx in range(0, max_id):
        # Reset counters of outcomes and number of obs per leaf
        outcome_id = 0
        n_id = 0
        # Loop over observations
        for obs in range(0, n_obs):
            # Check if leaf index of observation is equal to current idx
            if equal_to(forest_apply[obs, trees], idx):
                # Add outcome to sum of outcomes
                outcome_id += outcome_ind_est[obs]
                # Compute number of observations of the leaf
                n_id += 1
        # If no obs with current leaf idx assign zero to leaf means
        if n_id == 0:
            leaf_means[idx] = 0
        # Otherwise compute leaf mean
        else:
            leaf_means[idx] = outcome_id/n_id
    # Return matrix of leaf means
    return leaf_means
        

                            