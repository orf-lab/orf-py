################################################################################
###         R vs. Python Comparison of orf timing                            ###
################################################################################

# set the directory to the one of the source file (requires Rstudio)
path <- ("D:/switchdrive/Projects/ORF_Python/ORFpy/dev/_R")

# load packages
library(tidyverse)
library(stevedata)
library(orf)

# Synthetic Data
# generate example data using the DGP from orf package data
set.seed(123) # set seed for replicability

# number of observations (at least 10k for reliable comparison)
n  <- 100000

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
Y <- deterministic + rlogis(n, 0, 1)
# thresholds for continuous outcome
cuts <- quantile(Y, c(0, 1/3, 2/3, 1))
# discretize outcome into ordered classes 1, 2, 3
Y <- as.numeric(cut(Y, breaks = cuts, include.lowest = TRUE))

# save data as a dataframe
odata <- as.data.frame(cbind(Y, X))
# save data to disk
write.csv(odata, file = paste0(path, '/data/odata_test.csv'), row.names = F)

# ---------------------------------------------------------------------------- #

# Benchmark settings:
replace_options <- list(FALSE, TRUE)
honesty_options <- list(FALSE, TRUE)
inference_options <- list(FALSE)
n_options <- list(1000,10000, 50000)
ntrees_options <- list(100,1000, 2000)

repetitions <- 1

results <- data.frame(matrix(ncol=6,nrow=0, dimnames=list(NULL, c("n", 
                                                                  "ntrees",
                                                                  "inference",
                                                                  "honesty",
                                                                  "replace",
                                                                  "time"))))
# start benchmarks
for (n in n_options) {
  # specify response and covariates
  Y <- as.numeric(odata[1:n, 1])
  X <- as.matrix(odata[1:n, -1])
  for (ntrees in ntrees_options){
    # loop through different settings and save the results
    for (inference_idx in inference_options) {
      # loop through honesty options
      for (honesty_idx in honesty_options) {
        # check if the setting is admissible
        if (inference_idx == TRUE & honesty_idx == FALSE) {
          next
        }
        # lastly loop through subsampling option
        for (replace_idx in replace_options) {
          # check if the setting is admissible (for comparison with python)
          if (honesty_idx == TRUE & replace_idx == TRUE) {
            next
          }
          # print current iteration
          print(paste('n:', n,
                      'ntrees:', ntrees,
                      'inference:', inference_idx,
                      'honesty:',honesty_idx,
                      'replace:', replace_idx, sep = " "))
          orf_time_sum = 0
          for (r in 1:repetitions){
            # set seed for replicability
            set.seed(123)
            # fit orf (at least 2000 trees for reliable comparison)
            orf_time <- system.time({
              orf(X, Y, num.trees = ntrees, min.node.size = 5, mtry = 0.3,
                           replace = replace_idx,
                           honesty = honesty_idx,
                           inference = inference_idx)})
            print(orf_time)
            orf_time_sum = orf_time_sum + orf_time['elapsed']
          }
          orf_time_mean = orf_time_sum/repetitions
          
          results_it <- data.frame(matrix(c(n, ntrees, inference_idx, honesty_idx, 
                                     replace_idx, orf_time_mean),nrow = 1))
          colnames(results_it) <- c("n", 
                                    "ntrees",
                                    "inference",
                                    "honesty",
                                    "replace",
                                    "time")
          results = rbind(results, results_it)
        }
      }
    }
  }
}

# save the results
write.csv(results,
          file = paste0(path, '/results/R_timing.csv'),
          row.names = FALSE)