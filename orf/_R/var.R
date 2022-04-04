##################################################################################
### R vs. Python Comparison of the Variance Computation for the Ordered Forest ###
##################################################################################

# set the directory to the one of the source file (requires Rstudio)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# prepare comparison data
honest_pred = list()

honest_pred[[1]] = c(0.569929,
                     0.457652,
                     0.200977,
                     0.163428,
                     0.321162,
                     0.187856,
                     0.557985,
                     0.202383,
                     0.430742,
                     0.24974,
                     0.196137
)

honest_pred[[2]] = c(0.894521,
                     0.86706,
                     0.609612,
                     0.515869,
                     0.681767,
                     0.511826,
                     0.915146,
                     0.556205,
                     0.793036,
                     0.682363,
                     0.641027
)

honest_weights=list()
honest_weights[[1]]=matrix(c(
c(0.00346975, 0.0110608, 0.00899232, 0.0158936, 0.0301387, 0, 0, 0.0158936, 0, 0.00390239, 0.016946),
c(0.0068159, 0.00177689, 0.00236292, 0.00220057, 0.0065358, 0, 0.0012479, 0.00220057, 0, 0, 0.00508244),
c(0.00722998, 0.000196078, 0.000196078, 0.000196078, 0.00196392, 0.00652776, 0.000418301, 0.000196078, 0.00165952, 0, 0.00112622),
c(0.00389689, 0.0003125, 0.0003125, 0, 0, 0.00332004, 0.00614804, 0, 0.00585058, 0.00499115, 0),
c(0.00114338, 0.000196078, 0.00314779, 0.00110744, 0.000196078, 0.00476587, 0.000607402, 0.00110744, 0.00578674, 0, 0.000196078),
c(0, 0.0003125, 0.0003125, 0, 0, 0.00982205, 0.00376463, 0, 0.0163425, 0.00327193, 0),
c(0.000873597, 0.0112818, 0.0101538, 0.0103377, 0.010198, 0.000172414, 0.00210625, 0.0103377, 0.000172414, 0.00181341, 0.0100045),
c(0.000942953, 0, 0, 0.000227273, 0.0003125, 0.0102191, 0.0123371, 0.000227273, 0.0120904, 0.000753163, 0),
c(0.00313037, 0, 0.00124086, 0.00032002, 0.0034221, 0.000181818, 0.00669657, 0.00032002, 0, 0.00109054, 0.00166949),
c(0.0176566, 0.0011673, 0.000185185, 0.000844722, 0.00157705, 0, 0.00159, 0.000844722, 0, 0, 0.0112805),
c(0.0184682, 0.000636223, 0.000322497, 0.00184519, 0.000322497, 0.00193601, 0.00856475, 0.00184519, 0.000879279, 0, 0)
), nrow=11, byrow=TRUE)


honest_weights[[2]]=matrix(c(
c(0.00710746, 0.00642752, 0.00605589, 0.0138432, 0.0286242, 0, 0.000479303, 0.0134236, 0, 0.00196262, 0.0173823),
c(0.0077141, 0.00340493, 0.00471814, 0.00320477, 0.0113724, 0, 0.000192308, 0.00301246, 0, 0, 0.0101862),
c(0.0074748, 0.000358879, 0.0000980392, 0.000358879, 0.00468363, 0.00625764, 0, 0.000358879, 0.0019216, 0, 0.00262136),
c(0.00421359, 0.000699271, 0.000429, 0, 0, 0.00248581, 0.00630264, 0, 0.00484261, 0.00352792, 0),
c(0.000340848, 0.00186834, 0.00531596, 0.00318137, 0.000448769, 0.00456733, 0.00143355, 0.00269094, 0.00578633, 0, 0.000221496),
c(0.00028665, 0.00143795, 0.00143795, 0, 0, 0.0104452, 0.00421068, 0, 0.0168282, 0.00567508, 0),
c(0.00112149, 0.0114419, 0.0102851, 0.0119951, 0.00863355, 0, 0.00348319, 0.0109123, 0, 0.0010737, 0.0104381),
c(0.000988275, 0, 0, 0.000263158, 0, 0.00922252, 0.0140849, 0, 0.0127809, 0.000147059, 0),
c(0.00404495, 0, 0.00111091, 0, 0.0049544, 0.000511762, 0.00583617, 0, 0, 0.000839181, 0.00159972),
c(0.0231542, 0.000537781, 0.0002152, 0.00130081, 0.00441213, 0, 0.000608974, 0.00130081, 0, 0, 0.013501),
c(0.016871, 0.00156384, 0.00120933, 0.00342018, 0.00114496, 0.00070298, 0.00541731, 0.00319291, 0.000591337, 0, 0.000650806)
), nrow=11, byrow=TRUE)

Y_ind_honest=list()

Y_ind_honest[[1]] = c(0,
                      0,
                      1,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0,
                      0
)

Y_ind_honest[[2]] = c(1,
                      1,
                      1,
                      0,
                      1,
                      0,
                      0,
                      1,
                      1,
                      1,
                      0
)

# train and honest indices
ind_tr = c(1,2,3,4,5,6)
ind_est = c(7,8,9,10,11)

#' Predict ORF Variance
#'
#' predict variance of ordered random forest predictions based on honest sample
#' splitting as described in Lechner (2018)
#'
#' @param honest_pred list of vectors of honest forest predictions
#' @param honest_weights list of n x n matrices of honest forest weights
#' @param Y_ind_honest list of vectors of 0-1 outcomes for the honest sample
#'
#' @return vector of ORF variances
#'
#' @keywords internal
#'

pred_orf_variance <- function(honest_pred, honest_weights, Y_ind_honest) {
  
  # ----------------------------------------------------------------------------------- #
  
  # get categories
  categories <- seq(1:(length(Y_ind_honest)+1))
  
  # ----------------------------------------------------------------------------------- #
  
  ## single variances computation
  # compute the conditional means (predictions): already have this as forest_pred
  # divide it by N to get the "mean"
  honest_pred_mean <- lapply(honest_pred, function(x) x/length(Y_ind_honest[[1]]))
  
  # calculate standard multiplication of weights and outcomes: honest_weights*y_ind_honest (note with seq_along: as many rows as honest_pred or honest_weights)
  honest_multi <- mapply(function(x,y) lapply(seq_along(x[, 1]), function(i) x[i, ] * y), honest_weights, Y_ind_honest, SIMPLIFY = FALSE)
  
  # subtract the mean from each obs i
  honest_multi_demeaned <- mapply(function(x,y) mapply(function(x,y) x-y, x, y, SIMPLIFY = FALSE), honest_multi, honest_pred_mean, SIMPLIFY = FALSE)
  
  ## now do the single variances for each category m
  # square the demeaned
  honest_multi_demeaned_sq <- lapply(honest_multi_demeaned, function(x) lapply(x, function(x) x^2))
  
  # sum all obs i together
  honest_multi_demeaned_sq_sum <- lapply(honest_multi_demeaned_sq, function(x) lapply(x, function(x) sum(x)))
  
  # multiply by N/N-1 (normalize)
  honest_multi_demeaned_sq_sum_norm <- lapply(honest_multi_demeaned_sq_sum, function(x) lapply(x, function(x) x*(length(honest_pred[[1]])/(length(honest_pred[[1]])-1)) ))
  
  # put it into a shorter named object
  honest_variance <- honest_multi_demeaned_sq_sum_norm
  
  # ----------------------------------------------------------------------------------- #
  
  ##  covariances computation
  # multiply forest_var_multi_demeaned according to formula for covariance (shifted categories needed for computational convenience)
  # honest sample
  honest_multi_demeaned_0_last <- append(honest_multi_demeaned, list(rep(list(rep(0, length(honest_multi_demeaned[[1]][[1]]))), length(honest_multi_demeaned[[1]]))))
  honest_multi_demeaned_0_first <- append(list(rep(list(rep(0, length(honest_multi_demeaned[[1]][[1]]))), length(honest_multi_demeaned[[1]]))), honest_multi_demeaned)
  
  # compute the multiplication of category m with m-1 according to the covariance formula
  honest_multi_demeaned_cov <- mapply(function(x,y) mapply(function(x,y) x*y, x, y, SIMPLIFY = FALSE), honest_multi_demeaned_0_first, honest_multi_demeaned_0_last, SIMPLIFY = FALSE)
  
  # sum all obs i together
  honest_multi_demeaned_cov_sum <- lapply(honest_multi_demeaned_cov, function(x) lapply(x, function(x) sum(x)))
  
  # multiply by N/N-1 (normalize)
  honest_multi_demeaned_cov_sum_norm <- lapply(honest_multi_demeaned_cov_sum, function(x) lapply(x, function(x) x*(length(honest_pred[[1]])/(length(honest_pred[[1]])-1)) ))
  
  # multiply by 2
  honest_multi_demeaned_cov_sum_norm_mult2 <- lapply(honest_multi_demeaned_cov_sum_norm, function(x) lapply(x, function(x) x*2 ))
  
  # put it into a shorter named object
  honest_covariance <- honest_multi_demeaned_cov_sum_norm_mult2
  
  # ----------------------------------------------------------------------------------- #
  
  ## put everything together according to the whole variance formula
  # shift variances accordingly for ease of next computations (covariance already has the desired format)
  # honest sample
  honest_variance_last <- append(honest_variance, list(rep(list(0), length(honest_multi_demeaned[[1]])))) # append zero element list
  honest_variance_first <- append(list(rep(list(0), length(honest_multi_demeaned[[1]]))), honest_variance) # prepend zero element list
  
  # put everything together according to formula: var_last + var_first - cov
  honest_variance_final <- mapply(function(x,y,z) mapply(function(x,y,z) x+y-z, x, y, z, SIMPLIFY = FALSE), honest_variance_last, honest_variance_first, honest_covariance, SIMPLIFY = FALSE)
  
  ## output for final variances
  # coerce to a matrix
  honest_var <- sapply(honest_variance_final, function(x) sapply(x, function(x) as.matrix(x)))
  
  # ----------------------------------------------------------------------------------- #
  
  # save as forest_var
  forest_variance <- honest_var
  
  # add names
  colnames(forest_variance) <- sapply(categories, function(x) paste("Category", x, sep = " "))
  
  # ----------------------------------------------------------------------------------- #
  
  ## return the matrix
  output <- forest_variance
  # output
  return(output)
  
  # ----------------------------------------------------------------------------------- #
  
}

#' Get ORF Variance
#'
#' get variance of ordered random forest predictions based on honest sample
#' splitting as described in Lechner (2018)
#'
#' @param honest_pred list of vectors of honest forest predictions
#' @param honest_weights list of n x n matrices of honest forest weights
#' @param train_pred list of vectors of honest forests predictions from train sample
#' @param train_weights list of vectors of honest forests predictions from train sample
#' @param Y_ind_honest list of vectors of 0-1 outcomes for the honest sample
#'
#' @return vector of ORF variances
#'
#' @keywords internal
#'
get_orf_variance <- function(honest_pred, honest_weights, train_pred, train_weights, Y_ind_honest) {
  
  # ----------------------------------------------------------------------------------- #
  
  # first get honest and train rownames
  rows_honest_data <- as.numeric(c("7", "8", "9", "10", "11"))
  rows_train_data <- as.numeric(c("1", "2", "3", "4", "5", "6"))
  # get also categories
  categories <- seq(1:(length(Y_ind_honest)+1))
  
  # ----------------------------------------------------------------------------------- #
  
  ## single variances computation
  # compute the conditional means (predictions): already have this as forest_pred
  # divide it by N to get the "mean"
  honest_pred_mean <- lapply(honest_pred, function(x) x/length(honest_pred[[1]]))
  train_pred_mean <- lapply(train_pred, function(x) x/length(train_pred[[1]]))
  
  # calculate standard multiplication of weights and outcomes: honest_weights*y_ind_honest
  honest_multi <- mapply(function(x,y) lapply(seq_along(x[, 1]), function(i) x[i, ] * y), honest_weights, Y_ind_honest, SIMPLIFY = FALSE)
  train_multi <- mapply(function(x,y) lapply(seq_along(x[, 1]), function(i) x[i, ] * y), train_weights, Y_ind_honest, SIMPLIFY = FALSE)
  
  # subtract the mean from each obs i
  honest_multi_demeaned <- mapply(function(x,y) mapply(function(x,y) x-y, x, y, SIMPLIFY = FALSE), honest_multi, honest_pred_mean, SIMPLIFY = FALSE)
  train_multi_demeaned <- mapply(function(x,y) mapply(function(x,y) x-y, x, y, SIMPLIFY = FALSE), train_multi, train_pred_mean, SIMPLIFY = FALSE)
  
  ## now do the single variances for each category m
  # square the demeaned
  honest_multi_demeaned_sq <- lapply(honest_multi_demeaned, function(x) lapply(x, function(x) x^2))
  train_multi_demeaned_sq <- lapply(train_multi_demeaned, function(x) lapply(x, function(x) x^2))
  
  # sum all obs i together
  honest_multi_demeaned_sq_sum <- lapply(honest_multi_demeaned_sq, function(x) lapply(x, function(x) sum(x)))
  train_multi_demeaned_sq_sum <- lapply(train_multi_demeaned_sq, function(x) lapply(x, function(x) sum(x)))
  
  # multiply by N/N-1 (normalize)
  honest_multi_demeaned_sq_sum_norm <- lapply(honest_multi_demeaned_sq_sum, function(x) lapply(x, function(x) x*(length(honest_pred[[1]])/(length(honest_pred[[1]])-1)) ))
  train_multi_demeaned_sq_sum_norm <- lapply(train_multi_demeaned_sq_sum, function(x) lapply(x, function(x) x*(length(train_pred[[1]])/(length(train_pred[[1]])-1)) ))
  
  # put it into a shorter named object
  honest_variance <- honest_multi_demeaned_sq_sum_norm
  train_variance <- train_multi_demeaned_sq_sum_norm
  
  # ----------------------------------------------------------------------------------- #
  
  ## covariances computation
  # multiply forest_var_multi_demeaned according to formula for covariance (shifted categories needed for computational convenience)
  # honest sample
  honest_multi_demeaned_0_last <- append(honest_multi_demeaned, list(rep(list(rep(0, length(honest_multi_demeaned[[1]][[1]]))), length(honest_multi_demeaned[[1]]))))
  honest_multi_demeaned_0_first <- append(list(rep(list(rep(0, length(honest_multi_demeaned[[1]][[1]]))), length(honest_multi_demeaned[[1]]))), honest_multi_demeaned)
  # train sample
  train_multi_demeaned_0_last <- append(train_multi_demeaned, list(rep(list(rep(0, length(train_multi_demeaned[[1]][[1]]))), length(train_multi_demeaned[[1]]))))
  train_multi_demeaned_0_first <- append(list(rep(list(rep(0, length(train_multi_demeaned[[1]][[1]]))), length(train_multi_demeaned[[1]]))), train_multi_demeaned)
  
  # compute the multiplication of category m with m-1 according to the covariance formula
  honest_multi_demeaned_cov <- mapply(function(x,y) mapply(function(x,y) x*y, x, y, SIMPLIFY = FALSE), honest_multi_demeaned_0_first, honest_multi_demeaned_0_last, SIMPLIFY = FALSE)
  train_multi_demeaned_cov <- mapply(function(x,y) mapply(function(x,y) x*y, x, y, SIMPLIFY = FALSE), train_multi_demeaned_0_first, train_multi_demeaned_0_last, SIMPLIFY = FALSE)
  
  # sum all obs i together
  honest_multi_demeaned_cov_sum <- lapply(honest_multi_demeaned_cov, function(x) lapply(x, function(x) sum(x)))
  train_multi_demeaned_cov_sum <- lapply(train_multi_demeaned_cov, function(x) lapply(x, function(x) sum(x)))
  
  # multiply by N/N-1 (normalize)
  honest_multi_demeaned_cov_sum_norm <- lapply(honest_multi_demeaned_cov_sum, function(x) lapply(x, function(x) x*(length(honest_pred[[1]])/(length(honest_pred[[1]])-1)) ))
  train_multi_demeaned_cov_sum_norm <- lapply(train_multi_demeaned_cov_sum, function(x) lapply(x, function(x) x*(length(train_pred[[1]])/(length(train_pred[[1]])-1)) ))
  
  # multiply by 2
  honest_multi_demeaned_cov_sum_norm_mult2 <- lapply(honest_multi_demeaned_cov_sum_norm, function(x) lapply(x, function(x) x*2 ))
  train_multi_demeaned_cov_sum_norm_mult2 <- lapply(train_multi_demeaned_cov_sum_norm, function(x) lapply(x, function(x) x*2 ))
  
  # put it into a shorter named object
  honest_covariance <- honest_multi_demeaned_cov_sum_norm_mult2
  train_covariance <- train_multi_demeaned_cov_sum_norm_mult2
  
  # ----------------------------------------------------------------------------------- #
  
  ## put everything together according to the whole variance formula
  # shift variances accordingly for ease of next computations (covariance already has the desired format)
  # honest sample
  honest_variance_last <- append(honest_variance, list(rep(list(0), length(honest_multi_demeaned[[1]]) ))) # append zero element list
  honest_variance_first <- append(list(rep(list(0), length(honest_multi_demeaned[[1]]) )), honest_variance) # prepend zero element list
  # train sample
  train_variance_last <- append(train_variance, list(rep(list(0), length(train_multi_demeaned[[1]]) ))) # append zero element list
  train_variance_first <- append(list(rep(list(0), length(train_multi_demeaned[[1]]) )), train_variance) # prepend zero element list
  
  # put everything together according to formula: var_last + var_first - cov
  honest_variance_final <- mapply(function(x,y,z) mapply(function(x,y,z) x+y-z, x, y, z, SIMPLIFY = FALSE), honest_variance_last, honest_variance_first, honest_covariance, SIMPLIFY = FALSE)
  train_variance_final <- mapply(function(x,y,z) mapply(function(x,y,z) x+y-z, x, y, z, SIMPLIFY = FALSE), train_variance_last, train_variance_first, train_covariance, SIMPLIFY = FALSE)
  
  ## output for final variances
  # coerce to a matrix
  honest_var <- sapply(honest_variance_final, function(x) sapply(x, function(x) as.matrix(x)))
  train_var <- sapply(train_variance_final, function(x) sapply(x, function(x) as.matrix(x)))
  
  ## put it together according to rownames
  rownames(honest_var) <- rows_honest_data # rownames
  rownames(train_var) <- rows_train_data # rownames
  # combine and sort
  forest_variance <- rbind(honest_var, train_var)
  # sort according to rownames
  forest_variance <- forest_variance[order(as.numeric(row.names(forest_variance))), ]
  
  # add names
  colnames(forest_variance) <- sapply(categories, function(x) paste("Category", x, sep = " "))
  
  # ----------------------------------------------------------------------------------- #
  
  ## return the matrix
  output <- forest_variance
  # output
  return(output)
  
  # ----------------------------------------------------------------------------------- #
  
}

# pred the variances (out-of-sample)
orf_var_R <- pred_orf_variance(honest_pred, honest_weights, Y_ind_honest)

# export the variances into a csv file
write.csv(orf_var_R, file = 'data/orf_var_R.csv', row.names = FALSE)

# prepare in-sample inputs
train_pred <- lapply(honest_pred, function(x) x[ind_tr])
honest_pred <- lapply(honest_pred, function(x) x[ind_est])
train_weights <- lapply(honest_weights, function(x) x[ind_tr, ind_est])
honest_weights <- lapply(honest_weights, function(x) x[ind_est, ind_est])
Y_ind_honest <- lapply(Y_ind_honest, function(x) x[ind_est])

# get the variances (in-of-sample)
orf_var_R_in <- get_orf_variance(honest_pred, honest_weights, train_pred, train_weights, Y_ind_honest)

# export the variances into a csv file
write.csv(orf_var_R_in, file = 'data/orf_var_R_in.csv', row.names = FALSE)

