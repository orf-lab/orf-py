##################################################################################
###          R vs. Python Comparison of the Ordered Forest Estimation          ###
##################################################################################

# set the directory to the one of the source file (requires Rstudio)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# load orf package
library(orf)

# load example data
odata <- read.csv('data/example_df.csv')
# load example data
# data(odata)

# specify response and covariates
Y <- as.numeric(odata[, 1])
X <- as.matrix(odata[, -1])

# estimate Ordered Forest with subsampling and with honesty and with inference
set.seed(999)
# (for inference, subsampling and honesty are required)
orf_fit <- orf(X, Y, num.trees = 1000, min.node.size = 5, mtry = 0.3, replace = FALSE, honesty = TRUE, inference = TRUE)
# check mean of predictions
colMeans(orf_fit$predictions)
# check mean of variances
colMeans(orf_fit$variances)
