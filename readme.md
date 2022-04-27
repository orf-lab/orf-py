<h1>orf: ordered random forests 
<a href="https://github.com/orf-lab/orf-py"> <img src='https://raw.githubusercontent.com/fmuny/ORFpy/main/docs/images/orf-logo.png' align="right" height="120" />
</a>
</h1>


Welcome to the repository of the `Python` package `orf` for random forest estimation
of the ordered choice models. For the `R` version of the `orf` package 
[Lechner and Okasa (2020)](https://cran.r-project.org/web/packages/orf/orf.pdf)
please refer to the [CRAN](https://CRAN.R-project.org/package=orf) repository.

## Introduction

The `Python` package `orf` is an implementation of the Ordered Forest estimator
as developed in [Lechner and Okasa (2019)](https://arxiv.org/abs/1907.02436).
The Ordered Forest flexibly estimates the conditional probabilities of models with
ordered categorical outcomes (so-called ordered choice models). Additionally to
common machine learning algorithms the Ordered Forest provides functions for estimating
marginal effects and thus provides similar output as in standard econometric models
for ordered choice. The core Ordered Forest algorithm relies on the fast forest
implementations from the `scikit-learn` ([Pedregosa et al., 2011](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)) and 
`EconML` ([Battocchi et al., 2019](https://econml.azurewebsites.net/))
libraries.

## Installation

In order to install the latest `PyPi` released version run
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
issues, consider installing the package in a virtual 
environment or try `pip install orf --user`.

The implementation relies on Python 3 and is compatible with version 3.8, 3.9 and 3.10.

## Examples

The example below demonstrates the basic functionality of the Ordered Forest.

```python
## Ordered Forest
import orf

# load example data
features, outcome = orf.make_ordered_regression()

# initiate Ordered Forest with custom settings
oforest = orf.OrderedForest(n_estimators=1000, min_samples_leaf=5,
                            max_features=2, replace=False, sample_fraction=0.5,
                            honesty=True, honesty_fraction=0.5, inference=False,
                            n_jobs=-1, random_state=123)

# fit Ordered Forest
oforest.fit(X=features, y=outcome)

# show summary of the Ordered Forest estimation
oforest.summary()

# evaluate the prediction performance
oforest.performance()

# plot the estimated probability distributions
oforest.plot()

# predict ordered probabilities in-sample
oforest.predict(X=None, prob=True)

# evaluate marginal effects for the Ordered Forest
oforest.margins(X=None, X_cat=None, X_eval=None, eval_point='mean', window=0.1)
```

For more detailed examples see the package description.

## References

- Battocchi, K., Dillon, E., Hei, M., Lewis, G., Oka, P., Oprescu, M. & 
  Syrgkanis, V. (2019). EconML: A Python Package for ML-Based Heterogeneous 
 Treatment Effects Estimation. Version 0.13.0, <https://github.com/microsoft/EconML>
- Lechner, M., & Okasa, G. (2019). Random Forest Estimation of the Ordered Choice Model.
  arXiv preprint arXiv:1907.02436. <https://arxiv.org/abs/1907.02436>
- Lechner, M., & Okasa, G. (2020). orf: Ordered Random Forests.
  R package version 0.1.3, <https://CRAN.R-project.org/package=orf>
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.
  <https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html>

The `orf` logo has been created via R-package [hexSticker](https://CRAN.R-project.org/package=hexSticker) using [Tourney](https://fonts.google.com/specimen/Tourney?query=Tyler+Finck&preview.text=orf&preview.text_type=custom) font designed by Tyler Finck, ETC.
