"""
orf: Ordered Random Forests.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Showcase application of the Ordered Forest estimator.
"""

# import modules
import pandas as pd
import os
import numpy as np
path="D:\switchdrive\Projects\ORF_Python\ORFpy"
# path = "/home/okasag/Documents/HSG/ORF/python/ORFpy"
os.chdir(path)

# load the ordered forest
from orf.OrderedRandomForest import OrderedRandomForest

# read in example data from the orf package in R
odata = pd.read_csv('orf/odata.csv')

# define outcome and features
outcome = odata['Y']
features = odata.drop('Y', axis=1)

# Ordered Forest estimation

# Set seed
np.random.seed(999)
# initiate the class with tuning parameters
oforest = OrderedRandomForest(n_estimators=1000, min_samples_leaf=5, max_features=0.3,
                        replace=False, sample_fraction=0.5, honesty=True,
                        n_jobs=-1, pred_method='numpy_loop',
                        weight_method='numpy_loop', inference=True)
# fit the model
oforest.fit(X=features, y=outcome)
# print summary of estimation
oforest.summary()
# evaluate the prediction performance
oforest.performance()
# plot predictions
oforest.plot()

# predict ordered probabilities
pred = oforest.predict(X=features)
# print summary of predictions
oforest.summary(pred)
# return predictions as array
pred.get("predictions")
# predict ordered classes
oforest.predict(X=features, prob=False).get("predictions")

# evaluate marginal effects
marg = oforest.margin(X=features, eval_point="mean", verbose=True)
# print summary of marginal effects
oforest.summary(marg)
# return marginal effects as array
marg.get("effects")