

# JANSEN ESTIMATOR FUNCTION ####################################################

jansen_fun <- function(d, N, params) {
  
  m <- matrix(d, nrow = N)
  k <- length(params)
  Y_A <- m[, 1]
  Y_AB <- m[, -1]
  f0 <- (1 / length(Y_A)) * sum(Y_A)
  VY <- 1 / length(Y_A) * sum((Y_A - f0) ^ 2)
  value <- (1 / (2 * N) * Rfast::colsums((Y_A - Y_AB) ^ 2)) / VY
  
  output <- data.table(value = value, parameters = params)
  
  return(output)
}

################################################################################