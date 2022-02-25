# ORFpy: Ordered Random Forests in Python

To-do list and comments for the Python implementation of the Ordered Forest estimator.

## To Do:

- implement the plotting function (F)
- implement get_params and set_params function for returning input values (F)
- implement summary function (F)
- check the structure of outputs of functions (predict, margin - tuple or dict, check econML and statsmodels) (F)
- check how to handle categorical variables (condition integer and number of unique values OR optional array indicating categorical variables by the user, change in R package as well) (G)
- implement while loop for marginal effects if window too small to yield different x_up and x_down (change in R package as well) (G)
- decide how to handle parallel computing and which method (for all fit, predict and margin functions) (G,F)
- representative test file to compare with R for big data (G)
- simplify the categorical variable condition in R package to ensure difference of exactly 1 as in Python (G)
- check which functions are internal and which are available to the user (_) (F)
- add optional argument to compute marginal effects only for certain variables (change in R package as well) (G)
- think about classes vs. subclasses to separate code into smaller files (G,F)
- check time comparisons for parallelisation (G)
- code deadline: 25.3.2022
- documentation deadline: 1.4.2022

## Done:

- sorting train and honest indices from 0 to N again, that way we can access the honest and train corresponding outputs with ind_tr and ind_est later on
- make only one honest_variance! By passing in different data and combining outputs afterwards
- simplify the variance function by removing the n_samples argument
- number of trees equal to 1 is not possible (depending on subtrees relevant only for econML honesty)
- normalize by leaf size should be always the honest leaf size! Not combined leaf size! (this is correct now, and has been also)
- now not all the weights sum exactly to 1, as if there are some observations from honest sample repopulating the trained trees that do not fill in all leaves, meaning some of the leaves from the trained trees remain empty, this results in training observations being in the leaf, but there is no observation from the honest sample in that respective leaf and thus it falls out of the weights computation but also from the honest prediction computation. Therefore, it is still ensured that the honest predictions via averaging (numpy loop) are exactly same as the honest predictions based on weights
- for multiprocessing to work the function must be outside of the class, there is a speed up in comparison to joblib
- check if the multiprocessing output probabilities are identical with other methods
- bug fix for numpy based weights computation
- implementation of multiprocessing for weights computation
- set n_jobs for max -1 as default
- improve multiprocessing for weights computation
- checked correctness of multiprocessing output across different weight_methods
- set defaults for pred_method and weightmethod using mpire module for multiprocessing
- implement honesty and inference for the predict function
- implement honesty and inference for the margins function
- check if X_sd for marginal effects is SD from training (honest) sample or from the new evaluation sample (non-existence of SD if evaluation point is just 1 observation)

## Comments:

- 
