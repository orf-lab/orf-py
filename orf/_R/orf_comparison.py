"""
orf: Ordered Random Forests.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Compare Ordered Forest Estimation with R Implementation.
"""

# import modules
import pandas as pd
import os
import numpy as np
# path="D:\switchdrive\Projects\ORF_Python\ORFpy"
path = "/home/okasag/Documents/HSG/ORF/python/ORFpy"
os.chdir(path)

# load the ordered forest
from orf.orf import OrderedForest

# read in example data from the orf package in R
odata = pd.read_csv('orf/R/data/example_df.csv')

# define outcome and features
outcome = odata['y']
features = odata.drop('y', axis=1)

# Ordered Forest estimation

# Set seed
np.random.seed(999)
# initiate the class with tuning parameters
oforest = OrderedForest(n_estimators=1000, min_samples_leaf=5,
                        max_features=0.3, replace=False, sample_fraction=0.5,
                        honesty=True, n_jobs=-1, pred_method='numpy_loop',
                        inference=True)
# fit the model
forest_fit = oforest.fit(X=features, y=outcome)

# check mean predictions
forest_fit.forest['probs']
# check mean variances
forest_fit.forest['variance']
