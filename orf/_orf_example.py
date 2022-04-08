"""
orf: Ordered Random Forests.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Showcase application of the Ordered Forest estimator.
"""

# import modules
import os
import orf
# path="D:\switchdrive\Projects\ORF_Python\ORFpy"
# path = "/home/okasag/Documents/HSG/ORF/python/ORFpy"
path = '/Users/okasag/Desktop/HSG/orf/python/ORFpy'
os.chdir(path)

# get example data
features, outcome  = orf.example_data()

# Ordered Forest estimation

# initiate the class with tuning parameters
oforest = orf.OrderedForest(n_estimators=1000, min_samples_leaf=5,
                            max_features=0.3, replace=False,
                            sample_fraction=0.5, honesty=True,
                            n_jobs=-1, inference=True, random_state=1)
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
marg = oforest.margins(X=features, eval_point="mean", verbose=True)
# print summary of marginal effects
oforest.summary(marg)
# return marginal effects as array
marg.get("effects")
