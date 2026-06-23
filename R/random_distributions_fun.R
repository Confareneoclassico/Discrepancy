
# RANDOM DISTRIBUTIONS FUNCTION ################################################

# Truncated distributions ------------------------------------------------------

truncated_normal <- function(x, min, max, mean, sd) {
  
  a <- pnorm(min, mean, sd)
  b <- pnorm(max, mean, sd)
  
  uniform <- qunif(x, a, b)
  out <- qnorm(uniform, mean, sd)
  
  return(out)
}

truncated_beta <- function(x, min, max, shape1, shape2) {
  
  a <- pbeta(min, shape1, shape2)
  b <- pbeta(max, shape1, shape2)
  
  uniform <- qunif(x, a, b)
  out <- qbeta(uniform, shape1, shape2)
  
  return(out)
}

truncated_logitnorm <- function(x, min, max, mu, sigma) {
  
  a <- logitnorm::plogitnorm(min, mu = mu, sigma = sigma)
  b <- logitnorm::plogitnorm(max, mu = mu, sigma = sigma)
  
  uniform <- qunif(x, a, b)
  out <- logitnorm::qlogitnorm(uniform, mu = mu, sigma = sigma)
  
  return(out)
}

# Random distributions  --------------------------------------------------------

sample_distributions <- list(
  
  "uniform" = function(x) x,
  "normal" = function(x) truncated_normal(x, 0, 1, 0.5, 0.15),
  "beta" = function(x) truncated_beta(x, 0, 1, 8, 2),
  "beta2" = function(x) truncated_beta(x, 0, 1, 2, 8),
  "beta3" = function(x) truncated_beta(x, 0, 1, 2, 0.8),
  "beta4" = function(x) truncated_beta(x, 0, 1, 0.8, 2),
  "logitnormal" = function(x) truncated_logitnorm(x, 0, 1, 0, 3.16)
  
)

# RANDOM DISTRIBUTIONS FUN #####################################################

random_distributions_fun <- function(X, phi) {
  
  names_ff <- names(sample_distributions)
  
  if(!phi == length(names_ff) + 1) {
    
    out <- sample_distributions[[names_ff[phi]]](X)
    
  } else {
    
    temp <- sample(names_ff, ncol(X), replace = TRUE)
    
    out <- sapply(seq_along(temp), function(x) 
      sample_distributions[[temp[x]]](X[, x]))
    
  }
  return(out)
}

################################################################################
