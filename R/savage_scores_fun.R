
# SAVAGE SCORES ################################################################

# The input of rank r (1 = largest value) receives sum_{j=r}^{k} 1/j, and
# tied inputs receive the average of their scores (Iman & Conover 1987), as in
# src/discrepancy.py. Correction in version 2.0.0: the previous version used
# the average ranks of tied values as indices and as reciprocals, which gave
# wrong scores whenever two values were equal.

savage_scores_fun <- function(x) {

  k <- length(x)
  base <- rev(cumsum(1 / (k:1)))          # base[r] = sum_{j=r}^{k} 1/j
  out <- base[rank(-x, ties.method = "first")]
  out <- ave(out, x, FUN = mean)          # average over ties

  return(out)
}

################################################################################
