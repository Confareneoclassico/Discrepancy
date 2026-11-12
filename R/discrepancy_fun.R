
# DISCREPANCY FUNCTIONS ########################################################

# Corrections in version 2.0.0 (see CHANGELOG.md), so that the results agree
# with the Python implementation in src/discrepancy.py:
# - an input value equal to 0 is assigned to the first row (it was dropped);
# - the column of the rank grid is computed in exact integer arithmetic
#   (floating-point ceiling misplaced ranks on block boundaries, e.g. N = 2500);
# - a constant output puts every point in the first column of the S-ersatz
#   grid (the min-max scaling divided by zero);
# - update_matrix() is defined here, so that this file is self-contained.

# Row (input) index of the s x s grid: ceiling(x s), with x = 0 in row 1
row_index <- function(x, s) pmax(1, ceiling(x * s))

# Moore fill (non-cascading): an empty cell is filled when at least a share
# `threshold` of its in-grid neighbours (8, 5 on an edge, 3 in a corner) is
# occupied in the original matrix M
update_matrix <- function(M, threshold = 0.5) {

  nr <- nrow(M)
  nc <- ncol(M)
  M_new <- M

  for (i in 1:nr) {
    for (j in 1:nc) {

      if (M[i, j] == 0) {

        block <- M[max(1, i - 1):min(nr, i + 1),
                   max(1, j - 1):min(nc, j + 1), drop = FALSE]

        # the cell itself is empty, so sum(block) counts the neighbours
        if (length(block) > 1 && sum(block) / (length(block) - 1) >= threshold) {
          M_new[i, j] <- 1
        }
      }
    }
  }

  return(M_new)

}

# Original discrepancy modified (output is 1 - Discrepancy now -----------------)

s_ersatz <- function(mat) {

  N <- nrow(mat)

  s <- ceiling(sqrt(N))

  # Create the zero matrix

  mat_zeroes <- matrix(0, s, s)

  # Compute index for x_i

  m <- row_index(mat[, 1], s)

  # Compute index for y: min-max scaling to [0, 1], y == 0 in column 1

  x <- mat[, 2]
  range_x <- max(x) - min(x)
  n_norm <- if (range_x > 0) (x - min(x)) / range_x else rep(0, length(x))
  n <- pmax(1, ceiling(n_norm * s))

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

  m <- row_index(mat[, 1], s)

  # Compute index for y: column max(1, ceiling((r - 1) s / N)) of the rank r
  # (ties broken by order of appearance), in exact integer arithmetic

  r <- rank(mat[, 2], ties.method = "first")
  n <- pmax(1, ((r - 1) * s + N - 1) %/% N)

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
