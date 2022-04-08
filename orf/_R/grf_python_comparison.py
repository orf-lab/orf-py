"""
Compare Results of the Python Implementation with R Implementation for GRF.
"""

# %% import libraries
import os
path = "/Users/okasag/Desktop/HSG/orf/python/ORFpy"
os.chdir(path)

import pandas as pd
import numpy as np
from econml.grf import CausalForest

# %% Empirical Dataset
# read in empirical test data based on the stevedata package in R
dataset = pd.read_csv('orf/_R/data/empdata_test.csv')
Y = np.array(dataset.y)
T = np.array(dataset.HealthInsurance)
X = np.array(dataset.drop(['HealthInsurance', 'y'], axis=1))

# seed
np.random.seed(123)
# set parameters for causal forest 
causal_forest = CausalForest(n_estimators=2000,
                             criterion="het",
                             min_samples_leaf=5,
                             max_features=0.3,
                             max_samples=0.5,
                             honest=True,
                             inference=True,
                             n_jobs=None)
                      
# fit train data to causal forest model 
causal_forest.fit(X=X, T=T, y=Y)
# estimate the CATEs
cates = causal_forest.predict_and_var(X=X)
# check means of CATEs
np.mean(cates[0])
np.mean(cates[1])

# %% Synthetic Dataset
n = 10000
p_cont = 1
p_cat = 1
p_binary = 1
cat_cat = 3
noise = True
# Set seed
seed = np.random.default_rng(seed=123)
# Draw covariates
cont = seed.normal(0, 1, size=(n, p_cont))
cat = seed.binomial(cat_cat-1, 0.5, size=(n, p_cat))+1
binary = seed.binomial(1, 0.5, size=(n, p_binary))
# Combine deterministic covariates
X_det = np.hstack([cont, cat, binary])
if noise:
    X = np.hstack([X_det, seed.normal(0, 10, size=(n, 1))])
else:
    X = X_det
# Generate continuous outcome with normal error
Y = np.sum(X_det, axis=1) + seed.normal(0, 1, n)
T = X[:, 2]
X = np.delete(X, 2, axis=1)

# seed
np.random.seed(123)
# set parameters for causal forest 
causal_forest = CausalForest(n_estimators=2000,
                             criterion="het",
                             min_samples_leaf=5,
                             max_features=0.3,
                             max_samples=0.5,
                             honest=True,
                             inference=True,
                             n_jobs=None)
                      
# fit train data to causal forest model 
causal_forest.fit(X=X, T=T, y=Y)
# estimate the CATEs
cates = causal_forest.predict_and_var(X=X)
# check means of CATEs
np.mean(cates[0])
np.mean(cates[1])
