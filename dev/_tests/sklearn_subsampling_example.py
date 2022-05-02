# -*- coding: utf-8 -*-
"""
Option for subsampling using sklearn

Here I show how we can implement an sklearn RandomForestRegressor with 
subsampling. The file sklearn_ensemble__forest.py contains the plain code
from https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_forest.py
where I just adjusted the imports at the beginning of the file.

The bootstrapping in the RandomForestRegressor is carried out using the 
function

def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)

    return sample_indices

which is based on the numpy function randint() which draws samples with
replacement from the range between 0 and n_samples. If we replace the function
generating the sample_indices by the numpy function choice()

    sample_indices = random_instance.choice(a=n_samples, size=n_samples_bootstrap, replace=False)

we can sample without replacement from the range 0 to n_samples. This
implements subsampling as showcased in the sample below.

"""
# import modules
import numpy as np

# load forest functions
import sklearn.ensemble as sk

from sklearn.datasets import make_regression
X, y = make_regression(n_features=4, n_informative=2,
                       random_state=0, shuffle=False)

# Get dimensions
n_samples, p = X.shape

"""
I show that subsampling works by 
 1.) fitting a forest with only one tree and bootstrap = False -> should use 
     all observations without sampling
 2.) fitting a forest with only one tree, bootstrap = True and 
     max_samples = n_samples -> should use all observations when sampling with
     replacement
We show that predictions differ between 1.) and 2.) if we use bootstrapping
(with replacement) but the predictions are identical if we use subsampling
without replacement
""" 

# A) BOOTSTRAPPING (original sklearn function)

# Run random forest with only one tree and bootstrap = False
rf1 = sk.RandomForestRegressor(
    n_estimators=1, min_samples_leaf=10, random_state=1, bootstrap=False
)
rf1.fit(X, y)
# Run random forest with only one tree, bootstrap = True and max_samples = n_samples
rf2 = sk.RandomForestRegressor(
    n_estimators=1, min_samples_leaf=10, random_state=1, max_samples=n_samples
)
rf2.fit(X, y)

# get predictions
p1 = rf1.predict(X)
p2 = rf2.predict(X)

# Results differ
print(p1[0:10])
print(p2[0:10])
# compare
np.all(p1 == p2)

# Show indices of X used in bootstrap 
print(sk._forest._generate_sample_indices(random_state=rf2.estimators_[0].random_state,
                                          n_samples=n_samples,
                                          n_samples_bootstrap=n_samples))
# Show unsampled indices
print(sk._forest._generate_unsampled_indices(random_state=rf2.estimators_[0].random_state, 
                                             n_samples=n_samples,
                                             n_samples_bootstrap=n_samples))


# B) SUBSAMPLING WITHOUT REPLACEMENT (adjusted sklearn function)

# Re-define function for generation of sample indices
def _generate_sample_indices(random_state, n_samples, n_samples_bootstrap):
    """
    Private function used to _parallel_build_trees function."""

    random_instance = sk._forest.check_random_state(random_state)
    #sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)
    sample_indices = random_instance.choice(a=n_samples, size=n_samples_bootstrap, 
                                            replace=False, p=None)

    return sample_indices

# monkey patching
sk._forest._generate_sample_indices = _generate_sample_indices

# Run random forest with only one tree and bootstrap = False
rf3 = sk.RandomForestRegressor(
    n_estimators=1, min_samples_leaf=10, random_state=1, bootstrap=False
)
rf3.fit(X, y)
# Run random forest with only one tree, bootstrap = True and max_samples = n_samples
rf4 = sk.RandomForestRegressor(
    n_estimators=1, min_samples_leaf=10, random_state=1, max_samples=n_samples
)
rf4.fit(X, y)

# get predictions
p3 = rf3.predict(X)
p4 = rf4.predict(X)

# Results identical
print(p3[0:10])
print(p4[0:10])
# compare
np.all(p3 == p4)

# Show indices of X used in subsampling
print(sk._forest._generate_sample_indices(random_state=rf4.estimators_[0].random_state,
                                          n_samples=n_samples,
                                          n_samples_bootstrap=n_samples))
# Show unsampled indices: empty!!!
print(sk._forest._generate_unsampled_indices(random_state=rf4.estimators_[0].random_state, 
                                             n_samples=n_samples,
                                             n_samples_bootstrap=n_samples))

# There should also be a way to overwrite the sklearn function directly,
# see https://github.com/scikit-learn/scikit-learn/issues/20177
