# ORFpy: Ordered Random Forests in Python

To-do list and comments for the Python implementation of the Ordered Forest estimator.

## To Do:

- implement honesty and inference for the predict function
- implement honesty and inference for the margins function
- improve multiprocessing for weights computation

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

## Comments:

- 
