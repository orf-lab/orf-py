"""
 Description
 ----------------------------
 A Python implementation of the Ordered Forest estimator as developed in
 Lechner & Okasa (2019). The Ordered Forest flexibly estimates the conditional
 probabilities of models with ordered categorical outcomes (so-called ordered
 choice models). Additionally to common machine learning algorithms the `orf`
 package provides functions for estimating marginal effects as well as
 statistical inference thereof and thus provides similar output as in standard
 econometric models for ordered choice. The core forest algorithm relies on the
 fast forest implementations from the
 [`scikit-learn`](https://scikit-learn.org/stable/){:target="_blank"}
 (Pedregosa et al., 2011) and
 [`EconML`](https://econml.azurewebsites.net/){:target="_blank"}
 (Battocchi et al., 2019) libraries. For the R version of the
 `orf` package (Lechner & Okasa, 2020), see
 [CRAN](https://CRAN.R-project.org/package=orf){:target="_blank"} repository.

 Installation
 ----------------------------
 To install the `orf` package run
 ```
 pip install orf
 ```
 in the terminal. `orf` requires the following dependencies:
     
 * numpy (<1.22.0,>=1.21.0)
 * pandas (>=1.4.2)
 * scipy (<1.8.0,>=1.7.2)
 * scikit-learn (>=1.0.2)
 * econml (>=0.13.0)
 * joblib (>=1.1.0)
 * plotnine (>=0.8.0)
 
 In case of an installation failure due to dependency 
 issues or conflicts with Anaconda distribution,
 consider installing the package in a virtual 
 environment or try `pip install orf --user`.
 
 The implementation relies on Python 3 and is compatible with 
 version 3.8, 3.9 and 3.10.

 Examples
 ----------------------------

 The following examples demonstrate the basic usage of the `orf` package with
 default settings for the Ordered Forest estimator.
 ```
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
 ```

 Authors
 ----------------------------
 Michael Lechner, Fabian Muny & Gabriel Okasa

 References
 ----------------------------
 - Battocchi, K., Dillon, E., Hei, M., Lewis, G., Oka, P., Oprescu, M. &
 Syrgkanis, V. (2019). EconML: A Python Package for ML-Based Heterogeneous
 Treatment Effects Estimation. Version 0.13.0, https://github.com/microsoft/EconML
 - Lechner, M., & Okasa, G. (2019). Random Forest Estimation of the Ordered
 Choice Model. arXiv preprint arXiv:1907.02436. https://arxiv.org/abs/1907.02436
 - Lechner, M., & Okasa, G. (2020). orf: Ordered Random Forests.
 R package version 0.1.3, https://CRAN.R-project.org/package=orf
 - Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12,
 pp. 2825-2830.
"""

from orf.OrderedForest import OrderedForest
from orf._utils import make_ordered_regression
__all__ = ["OrderedForest", "make_ordered_regression"]
__version__ = "0.1.1"
__module__ = 'orf'
__author__ = "Michael Lechner, Fabian Muny & Gabriel Okasa"
__copyright__ = "Copyright (c) 2022, Michael Lechner, Fabian Muny & Gabriel Okasa"
__license__ = "MIT License"
