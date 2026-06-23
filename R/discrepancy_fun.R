
# DISCREPANCY FUNCTIONS ########################################################

# Original discrepancy modified (output is 1 - Discrepancy now -----------------)

s_ersatz <- function(mat) {
  
  N <- nrow(mat)
  
  s <- ceiling(sqrt(N))
  
  # Create the zero matrix
  
  mat_zeroes <- matrix(0, s, s)
  
  # Compute index for x_i
  
  m <- ceiling(mat[, 1] * s)
  
  # Compute index for y
  
  x <- mat[, 2]
  n_norm <- (x-min(x))/(max(x)-min(x)) # Scale y to 0, 1
  n <- ceiling(n_norm * s)
  
  # Turn y==0 to y == 1
  
  n <- ifelse(n == 0, 1, n)
  
  # Merge and identify which cells are occupied by points
  
  ind <- cbind(m, n)
  mat_zeroes[ind] <- 1
  
  # Compute discrepancy
  
  S <- 1 - (sum(mat_zeroes==1) / prod(dim(mat_zeroes)))
  
  return(S)
  
}

# New discrepancy adjusted -----------------------------------------------------

s_ersatz_adj <- function(mat) {
  
  N <- nrow(mat)
  
  s <- ceiling(sqrt(N))
  
  # Compute index for x_i
  
  m <- ceiling(mat[, 1] * s)
  
  # Compute index for y
  
  x <- mat[, 2]
  n_norm <- (rank(x, ties.method = "first")-1) / length(x)
  n <- ceiling(n_norm * s)
  
  # Turn y==0 to y == 1
  
  n <- ifelse(n == 0, 1, n)
  
  # Merge and identify which cells are occupied by points
  
  mat_zeroes <- matrix(0, s, s)
  ind <- cbind(m, n)
  mat_zeroes[ind] <- 1
  mat_zeroes = update_matrix(mat_zeroes)
  S <- 1 - sum(mat_zeroes==1) / prod(dim(mat_zeroes))
  
  return(S)
  
}

# DISCREPANCY WRAPPER WITH THE TWO DISCREPANCIES ###############################

discrepancy_wrapper_fun <- function(mat, Y, params, type = "adjusted") {
  
  if (type == "not.adjusted") {
    
    value <- sapply(1:ncol(mat), function(j) {
      
      design <- cbind(mat[, j], Y)
      value <- s_ersatz(mat = design)
      
    })
    
  } else if (type == "adjusted") {
    
    value <- sapply(1:ncol(mat), function(j) {
      
      design <- cbind(mat[, j], Y)
      value <- s_ersatz_adj(mat = design)
      
    })
    
  } else {
    
    stop("type should be adjusted or not.adjusted")
    
  }
  
  out <- data.table::data.table(value)[, parameters:= params]
  
  return(out)
  
}

################################################################################