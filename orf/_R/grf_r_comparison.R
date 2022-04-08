################################################################################
###             R vs. Python Comparison of the Causal Forest GRF             ###
################################################################################

# load grf
library(grf)
# set the directory to the one of the source file (requires Rstudio)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
path <- getwd()

# ---------------------------------------------------------------------------- #

# Empirical Dataset
# read in empirical test data based on the stevedata package in R
dataset = read.csv(file = paste0(path, '/data/empdata_test.csv'))
Y = dataset$y
W = dataset$HealthInsurance
X = dataset[, -which(colnames(dataset) %in% c('HealthInsurance', 'y'))]

# seed
set.seed(123)
# set parameters for causal forest 
cforest = causal_forest(X=X, Y=Y, W=W,
                        num.trees=2000,
                        min.node.size=5,
                        mtry=0.3,
                        sample.fraction=0.5,
                        honesty=TRUE,
                        num.threads=1)
# estimate the CATEs
cates = predict(cforest, estimate.variance = TRUE)
# check means of CATEs
mean(cates[, 'predictions'])
mean(cates[, 'variance.estimates'])
# get ATE
average_treatment_effect(cforest, target.sample = "all")

# ---------------------------------------------------------------------------- #
# Synthetic Dataset
# set seed for replicability
set.seed(123)
# number of observations
n  <- 10000

# various covariates
X1 <- rnorm(n, 0, 1)    # continuous
X2 <- rbinom(n, 2, 0.5) # categorical
X3 <- rbinom(n, 1, 0.5) # dummy
X4 <- rnorm(n, 0, 10)   # noise

# bind into matrix
X <- as.matrix(cbind(X1, X2, X3, X4))

# deterministic component
deterministic <- X1 + X2 + X3
# generate continuous outcome with logistic error
Y <- deterministic + rnorm(n, 0, 1)
W <- X[, 3]
X <- X[, -3]

# seed
set.seed(123)
# set parameters for causal forest 
cforest = causal_forest(X=X, Y=Y, W=W,
                        num.trees=2000,
                        min.node.size=5,
                        mtry=0.3,
                        sample.fraction=0.5,
                        honesty=TRUE,
                        num.threads=1)
# estimate the ATE
cates = predict(cforest, estimate.variance = TRUE)
# check means of CATEs
mean(cates[, 'predictions'])
mean(cates[, 'variance.estimates'])
# get ATE
average_treatment_effect(cforest, target.sample = "all")
