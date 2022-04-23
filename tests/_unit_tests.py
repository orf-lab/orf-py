"""
orf: Ordered Random Forest.

Python implementation of the Ordered Forest as in Lechner & Okasa (2019).

Overview and Testing of examples from documentation.

"""

"""
Basic example using default settings
"""
# load orf package
import orf

# get example data
features, outcome  = orf.make_ordered_regression(seed=123)

# estimate Ordered Forest with default settings
oforest = orf.OrderedForest()
oforest.fit(X=features, y=outcome)

# show summary of the orf estimation
oforest.summary()

# evaluate the prediction performance
oforest.performance()

# plot the estimated probability distributions
oforest.plot()

# predict ordered probabilities
oforest.predict()

# evaluate marginal effects
oforest.margins()

"""
make_ordered_regression()
"""
# load orf package
import orf

# generate dataset consisting of 2000 observations and 4 outcome classes
features, outcome  = orf.make_ordered_regression(n_samples=2000,
                                                 y_classes=4,
                                                 seed=123)

"""
OrderedForest()
"""
# load orf package
import orf

# initialize Ordered Forest with default parameters
oforest = orf.OrderedForest()

# initialize Ordered Forest with own tuning parameters
oforest = orf.OrderedForest(n_estimators = 2000, min_samples_leaf = 10,
                            max_features = 3)

# initialize Ordered Forest with bootstrapping and without honesty
oforest = orf.OrderedForest(replace = True, honesty = False)

# initialize Ordered Forest with subsampling and with honesty
oforest = orf.OrderedForest(replace = False, honesty = True)

# initialize Ordered Forest with subsampling and with honesty
# with own tuning for subsample fraction and honesty fraction
oforest = orf.OrderedForest(replace = False, sample_fraction = 0.5,
                            honesty = True, honesty_fraction = 0.5)

# initialize Ordered Forest with subsampling, honesty and 
# inference (for inference, subsampling and honesty are required)
oforest = orf.OrderedForest(replace = False, honesty = True,
                            inference = True)

# initialize Ordered Forest with all custom settings
oforest = orf.OrderedForest(n_estimators = 2000, min_samples_leaf = 10,
                            max_features = 3, replace = True,
                            sample_fraction = 1, honesty = False,
                            honesty_fraction = 0, inference = False)

"""
fit()
"""
# load orf package
import orf

# get example data
features, outcome  = orf.make_ordered_regression(seed=123)

# initialize Ordered Forest with default parameters
oforest = orf.OrderedForest()
# estimate Ordered Forest
oforest.fit(X=features, y=outcome)

# initialize Ordered Forest with own tuning parameters
oforest = orf.OrderedForest(n_estimators = 2000, 
                            min_samples_leaf = 10,
                            max_features = 3)
# estimate Ordered Forest
oforest.fit(X=features, y=outcome)

# initialize Ordered Forest with bootstrapping and without honesty
oforest = orf.OrderedForest(replace = True, honesty = False)
# estimate Ordered Forest
oforest.fit(X=features, y=outcome)

# initialize Ordered Forest with subsampling and with honesty
oforest = orf.OrderedForest(replace = False, honesty = True)
# estimate Ordered Forest
oforest.fit(X=features, y=outcome)

# initialize Ordered Forest with subsampling and with honesty
# with own tuning for subsample fraction and honesty fraction
oforest = orf.OrderedForest(replace = False, sample_fraction = 0.5,
                            honesty = True, honesty_fraction = 0.5)
# estimate Ordered Forest
oforest.fit(X=features, y=outcome)

# initialize Ordered Forest with subsampling, honesty and 
# inference (for inference, subsampling and honesty are required)
oforest = orf.OrderedForest(replace = False, honesty = True,
                            inference = True)
# estimate Ordered Forest
oforest.fit(X=features, y=outcome)

# initialize Ordered Forest with all custom settings
oforest = orf.OrderedForest(n_estimators = 2000,
                            min_samples_leaf = 10,
                            max_features = 3, replace = True,
                            sample_fraction = 1, honesty = False,
                            honesty_fraction = 0,
                            inference = False)    
# estimate Ordered Forest
oforest.fit(X=features, y=outcome)

"""
margins()
"""
# load packages
import orf

# get example data
features, outcome  = orf.make_ordered_regression(seed=123)

# estimate Ordered Forest
oforest = orf.OrderedForest().fit(X=features, y=outcome)

# estimate default (mean) marginal effects for all features
marg = oforest.margins()

# return mean marginal effects as array
print(marg.get("effects"))

# estimate mean marginal effects, explicitly defining the second
# column of the features as categorical
marg = oforest.margins(X_cat=[1])

# estimate mean marginal effects for the first and third column of X
marg = oforest.margins(X_eval=[0,2])

# estimate marginal effects at the mean and at the median
marg_atmean = oforest.margins(eval_point="atmean")
marg_atmedian = oforest.margins(eval_point="atmedian")

# estimate Ordered Forest using inference
oforest = orf.OrderedForest(inference=True).fit(X=features, y=outcome)

# estimate mean marginal effects for the first column of X
marg = oforest.margins(X_eval=[0])

# return marginal effects as array
print(marg.get("effects"))
# return variances as array
print(marg.get("variances"))
# return standard errors as array
print(marg.get("std_errors"))
# return t-values as array
print(marg.get("t-values"))
# return p-values as array
print(marg.get("p-values"))
# return upper confidence intervals as array
print(marg.get("ci-up"))
# return lower confidence intervals as array
print(marg.get("ci-down"))

"""
performance()
"""
# load package
import orf

# get example data
features, outcome  = orf.make_ordered_regression(seed=123)

# estimate Ordered Forest
oforest = orf.OrderedForest().fit(X=features, y=outcome)

# print the prediction performance measures
oforest.performance()

"""
plot()
"""
# load package
import orf

# get example data
features, outcome  = orf.make_ordered_regression(seed=123)

# estimate Ordered Forest
oforest = orf.OrderedForest().fit(X=features, y=outcome)

# plot the estimated probability distributions
oforest.plot()

"""
predict()
"""
# load packages
import orf
from sklearn.model_selection import train_test_split

# get example data
features, outcome  = orf.make_ordered_regression(seed=123)

# generate train and test set
X_train, X_test, y_train, y_test = train_test_split(
    features, outcome, test_size=0.2, random_state=123)

# estimate Ordered Forest
oforest = orf.OrderedForest().fit(X=X_train, y=y_train)

# predict the probabilities with the estimated Ordered Forest
pred = oforest.predict(X=X_test)
# return predictions as array
print(pred.get("predictions"))

# predict the classes with estimated Ordered Forest
pred_class = oforest.predict(X=X_test, prob=False)
# return predictions as array
pred_class.get("predictions")

# estimate Ordered Forest using inference
oforest = orf.OrderedForest(inference=True).fit(X=X_train, y=y_train)

# predict the probabilities together with variances
pred_inf = oforest.predict(X=X_test)
# return predictions as array
print(pred_inf.get("predictions"))
# return variances as array
print(pred_inf.get("variances"))

"""
summary()
"""
# load package
import orf

# get example data
features, outcome  = orf.make_ordered_regression(seed=123)

# estimate Ordered Forest
oforest = orf.OrderedForest().fit(X=features, y=outcome)

# print summary of estimation
oforest.summary()

# predict the probabilities with the estimated Ordered Forest
pred = oforest.predict()

# print summary of the Ordered Forest predictions
oforest.summary(pred)

# estimate marginal effects for first feature
marg = oforest.margins(X_eval = [0])

# print summary of the marginal effects
oforest.summary(marg)