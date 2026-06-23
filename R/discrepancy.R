### Function

rm(list = ls())
library(sensobol)
library(ggplot2)
library(gridExtra)
library(truncnorm)
library(xtable)
library(qrng)

sobol_mat <- function(matrices = c("A", "B", "AB"), N, params, order = "first", type = "QRN") {
  
  k <- length(params)
  n.matrices <- ifelse(any(stringr::str_detect(matrices, "C")) == FALSE, 2, 3)
  
  # Selection of the sample matrix type
  # -----------------------------------------------------------------
  
  if (type == "QRN") {
    df <- qrng::sobol(n = N, d = k * n.matrices, randomize = "Owen", seed = 123)
    
  } else if (type == "R") {
    df <- replicate(k * n.matrices, stats::runif(N))
    
  } else if (type == "LHS") {
    df <- lhs::randomLHS(N, n.matrices * k)
    
  } else {
    stop ("method should be either QRN, R or LHS")
  }
  
  # Construction of A, B, ... matrices
  # -----------------------------------------------------------------
  
  A <- df[, 1:k, drop = FALSE]
  B <- df[, (k + 1) : (k * 2), drop = FALSE]
  
  if (n.matrices == 3) {
    C <- df[, ((k * 2) + 1):(k * 3), drop = FALSE]
    
  } else {
    C <- NULL
  }
  
  # Construction of Ab, Ba, ... matrices
  # -----------------------------------------------------------------
  
  out <- scrambled_sobol(matrices = matrices,
                         A = A, B = B, C = C,
                         order = order)
  A.mat <- "A" %in% matrices
  B.mat <- "B" %in% matrices
  C.mat <- "C" %in% matrices
  
  # Set NULL to those not used
  # -----------------------------------------------------------------
  
  if (A.mat == FALSE) {
    A <- NULL
  }
  
  if (B.mat == FALSE) {
    B <- NULL
  }
  
  if (C.mat == FALSE) {
    C <- NULL
  }
  
  # Bind and output
  # -----------------------------------------------------------------
  
  final <- rbind(A, B, C, out)
  colnames(final) <- params
  return(final)
  
}

scrambled_sobol <- function(matrices, A, B, C, order) {
  
  first <- 1:ncol(A)
  N <- nrow(A)
  
  # Vectors with the columns
  # -----------------------------------------------------------------
  
  if(order == "first") {
    loop <- first
    
  } else if (order == "second") {
    
    second <- c(first, utils::combn(1:ncol(A), 2, simplify = FALSE))
    loop <- second
    
  } else if (order == "third") {
    
    second <- c(first, utils::combn(1:ncol(A), 2, simplify = FALSE))
    third <- c(second, utils::combn(1:ncol(A), 3, simplify = FALSE))
    loop <- third
    
  } else if (order == "fourth") {
    
    second <- c(first, utils::combn(1:ncol(A), 2, simplify = FALSE))
    third <- c(second, utils::combn(1:ncol(A), 3, simplify = FALSE))
    fourth <- c(third, utils::combn(1:ncol(A), 4, simplify = FALSE))
    loop <- fourth
    
  } else {
    
    stop("order should be either first, second, third or fourth")
  }
  
  # Check which matrices have been selected
  # -----------------------------------------------------------------
  
  AB.mat <- "AB" %in% matrices
  BA.mat <- "BA" %in% matrices
  CB.mat <- "CB" %in% matrices
  
  # Construct Ab, Ba matrices, etc
  # -----------------------------------------------------------------
  
  if (AB.mat == TRUE) {
    X <- rbind(A, B)
    
    for(i in loop) {
      AB <- A
      AB[, i] <- B[, i]
      X <- rbind(X, AB)
    }
    AB <- X[(2 * N + 1):nrow(X), , drop = FALSE]
    
  } else if (AB.mat == FALSE) {
    AB <- NULL
  }
  
  if (BA.mat == TRUE) {
    W <- rbind(A, B)
    
    for (i in loop) {
      BA <- B
      BA[, i] <- A[, i]
      W <- rbind(W, BA)
    }
    BA <- W[(2 * N + 1) : nrow(W), , drop = FALSE]
    
  } else if (BA.mat == FALSE) {
    BA <- NULL
  }
  
  if (CB.mat == TRUE) {
    Z <- rbind(A, B)
    
    for (i in loop) {
      CB <- C
      CB[, i] <- B[, i]
      Z <- rbind(Z, CB)
    }
    CB <- Z[(2 * N + 1) : nrow(Z), , drop = FALSE]
    
  } else if (CB.mat == FALSE) {
    CB <- NULL
  }
  
  # Merge and output
  # -----------------------------------------------------------------
  
  final <- rbind(AB, BA, CB)
  return(final)
  
}

update_matrix <- function(M) {
  
  nr <- nrow(M)
  nc <- ncol(M)
  M_new <- M
  
  for (i in 1:nr) {
    for (j in 1:nc) {
      
      if (M[i, j] == 0) {
        
        # limiti dei vicini
        r_min <- max(1, i - 1)
        r_max <- min(nr, i + 1)
        c_min <- max(1, j - 1)
        c_max <- min(nc, j + 1)
        
        neighbors <- M[r_min:r_max, c_min:c_max]
        
        # rimuove la cella centrale
        neighbors <- neighbors[-which(row(neighbors) == i - r_min + 1 &
                                        col(neighbors) == j - c_min + 1)]
        
        if (mean(neighbors) >= 0.5) {
          M_new[i, j] <- 1
        }
      }
    }
  }
  
  return(M_new)
  
}

jansen_fun <- function(d, N, params) {
  
  m <- matrix(d, nrow = N)
  k <- length(params)
  Y_A <- m[, 1]
  Y_AB <- m[, -1]
  f0 <- (1 / length(Y_A)) * sum(Y_A)
  VY <- 1 / length(Y_A) * sum((Y_A - f0) ^ 2)
  value <- (1 / (2 * N) * Rfast::colsums((Y_A - Y_AB) ^ 2)) / VY
  
  output <- data.frame(parameters = params, value = value)
  
  return(output)

}

savage_scores_fun <- function(x) {
  
  true.ranks <- rank(-x)
  p <- sort(1 / true.ranks)
  mat <- matrix(rep(p, length(p)), nrow = length(p), byrow = TRUE)
  mat[upper.tri(mat)] <- 0
  out <- sort(rowSums(mat), decreasing = TRUE)[true.ranks]
  
  return(out)
  
}

discrepancy_ersatz <- function(mat, Y, params, adj = 0) {
  
  if (adj == 0) {
    
    value <- sapply(1:ncol(mat), function(j) {
      
      design <- cbind(mat[, j], Y)
      value <- s_ersatz(mat = design)
      
    })
    
  } else {
    
    value <- sapply(1:ncol(mat), function(j) {
      
      design <- cbind(mat[, j], Y)
      value <- s_ersatz_adj(mat = design, j)
      
    })
    
  }

  out <- data.frame(params = params, value = value)
  
  return(out)
  
}

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
  
  S <- 1 - sum(mat_zeroes==1) / prod(dim(mat_zeroes))
  
  return(S)
  
}

s_ersatz_adj <- function(mat, j) {
  
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
  
  coords <- which(mat_zeroes == 1, arr.ind = TRUE)
  coords <- coords[!duplicated(rbind(ind, coords))[(nrow(ind)+1):(nrow(ind)+nrow(coords))], ]
  x1 <- ind[,1]  
  y1 <- ind[,2]
  x2 <- coords[,1]  
  y2 <- coords[,2]
  plot(x1, y1, cex.lab = 1.5, cex.axis = 1.5, xlab = paste("x", j), ylab = "Y")#, pch = 16)
  points(x2, y2, col ="red", pch = 4)
  mean_y <- tapply(c(y1, y2), c(x1, x2), mean)
  points(as.numeric(names(mean_y)), mean_y,
         col = "red", pch = 19, cex = 2)
  
  return(S)
  
}

Ti = readRDS("/Users/alessiolachi/Documents/3) UniCamillus/Articoli/8) discrepancy GSA/AE_df.rds")
Ti = as.matrix(Ti)

### Bratley 1988 function

N = 2^9
params = paste("x", 1:8, sep = "")
mat_dis = sobol_mat(N = N, params = params, matrices = "A")
Y_dis = bratley1988_Fun(mat_dis)
sob = Ti[7,]
dis = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 0)
dis_adj = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 1)
tab = round(cbind(sobol = sob, 
                  ersatz = dis[,2], 
                  ersatz_adj = dis_adj[,2]), 3)
tab = rbind(tab, c(NaN, round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,2]))),3),
                        round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,3]))),3)))
tab = rbind(tab, c(NaN, round(apply(abs(tab[1:length(params),2:3] - tab[1:length(params),1]), 2, mean), 3)))
y_true <- ifelse(tab[1:length(params),1] > 0.05, 1, 0)
y_pred <- ifelse(tab[1:length(params),2:3] > 0.05, 1, 0)
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 1 & y_true == 0) / sum(y_true == 0),3),
                        round(sum(y_pred[,2] == 1 & y_true == 0) / sum(y_true == 0),3)))
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 0 & y_true == 1) / sum(y_true == 1),3),
                        round(sum(y_pred[,2] == 0 & y_true == 1) / sum(y_true == 1),3)))
params = c(paste("$x_", 1:8, "$", sep = ""), "$\\rho$", "MAE", "$\\alpha$", "$\\beta$")
tab = data.frame(params, tab)
print(xtable(tab), include.rownames = FALSE, sanitize.text.function = identity, booktabs = TRUE)

# mat = mat_dis; Y = Y_dis; params = params; adj = 1
# j = 1; mat = cbind(mat[, j], Y)

### Bratley 1992 function

N = 2^9
params = paste("x", 1:8, sep = "")
mat_dis = sobol_mat(N = N, params = params, matrices = "A")
Y_dis = bratley1992_Fun(mat_dis)
sob = Ti[1,]
dis = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 0)
dis_adj = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 1)
tab = round(cbind(sobol = sob, 
                  ersatz = dis[,2], 
                  ersatz_adj = dis_adj[,2]), 3)
tab = rbind(tab, c(NaN, round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,2]))),3),
                   round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,3]))),3)))
tab = rbind(tab, c(NaN, round(apply(abs(tab[1:length(params),2:3] - tab[1:length(params),1]), 2, mean), 3)))
y_true <- ifelse(tab[1:length(params),1] > 0.05, 1, 0)
y_pred <- ifelse(tab[1:length(params),2:3] > 0.05, 1, 0)
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 1 & y_true == 0) / sum(y_true == 0),3),
                   round(sum(y_pred[,2] == 1 & y_true == 0) / sum(y_true == 0),3)))
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 0 & y_true == 1) / sum(y_true == 1),3),
                   round(sum(y_pred[,2] == 0 & y_true == 1) / sum(y_true == 1),3)))
params = c(paste("$x_", 1:8, "$", sep = ""), "$\\rho$", "MAE", "$\\alpha$", "$\\beta$")
tab = data.frame(params, tab)
print(xtable(tab), include.rownames = FALSE, sanitize.text.function = identity, booktabs = TRUE)

# mat = mat_dis; Y = Y_dis; params = params; adj = 1
# j = 8; mat = cbind(mat[, j], Y)

### Ishigami function

N = 2^9
params = paste("x", 1:3, sep = "")
mat_sob = sobol_matrices(N = N, params = params, matrices = c("A", "AB"))
mat_dis = sobol_mat(N = N, params = params, matrices = "A")
Y_sob = ishigami_Fun(mat_sob)
Y_dis = ishigami_Fun(mat_dis)
sob = jansen_fun(Y_sob, N, params)
dis = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 0)
dis_adj = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 1)
tab = round(cbind(sobol = sob$value, 
                  ersatz = dis[,2], 
                  ersatz_adj = dis_adj[,2]), 3)
tab = rbind(tab, c(NaN, round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,2]))),3),
                   round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,3]))),3)))
tab = rbind(tab, c(NaN, round(apply(abs(tab[1:length(params),2:3] - tab[1:length(params),1]), 2, mean), 3)))
y_true <- ifelse(tab[1:length(params),1] > 0.05, 1, 0)
y_pred <- ifelse(tab[1:length(params),2:3] > 0.05, 1, 0)
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 1 & y_true == 0) / sum(y_true == 0),3),
                   round(sum(y_pred[,2] == 1 & y_true == 0) / sum(y_true == 0),3)))
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 0 & y_true == 1) / sum(y_true == 1),3),
                   round(sum(y_pred[,2] == 0 & y_true == 1) / sum(y_true == 1),3)))
params = c(paste("$x_", 1:3, "$", sep = ""), "$\\rho$", "MAE", "$\\alpha$", "$\\beta$")
tab = data.frame(params, tab)
print(xtable(tab), include.rownames = FALSE, sanitize.text.function = identity, booktabs = TRUE)

# mat = mat_dis; Y = Y_dis; params = params; adj = 1
# j = 1; mat = cbind(mat[, j], Y)

### Ishigami function 2

N = 2^9
params = paste("x", 1:3, sep = "")
mat_sob = sobol_matrices(N = N, params = params, matrices = c("A", "AB"))
mat_sob2 = apply(mat_sob, 2, function(x) x * (pi + pi) - pi)
mat_dis = sobol_mat(N = N, params = params, matrices = "A")
mat_dis2 = apply(mat_dis, 2, function(x) x * (pi + pi) - pi)
Y_sob = sin(mat_sob2[,1]) + 7*sin(mat_sob2[,2])^2 + 0.1*mat_sob2[,3]^4*sin(mat_sob2[,1])
Y_dis = sin(mat_dis2[,1]) + 7*sin(mat_dis2[,2])^2 + 0.1*mat_dis2[,3]^4*sin(mat_dis2[,1])
sob = jansen_fun(Y_sob, N, params)
dis = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 0)
dis_adj = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 1)
tab = round(cbind(sobol = sob$value, 
                  ersatz = dis[,2], 
                  ersatz_adj = dis_adj[,2]), 3)
tab = rbind(tab, c(NaN, round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,2]))),3),
                   round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,3]))),3)))
tab = rbind(tab, c(NaN, round(apply(abs(tab[1:length(params),2:3] - tab[1:length(params),1]), 2, mean), 3)))
y_true <- ifelse(tab[1:length(params),1] > 0.05, 1, 0)
y_pred <- ifelse(tab[1:length(params),2:3] > 0.05, 1, 0)
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 1 & y_true == 0) / sum(y_true == 0),3),
                   round(sum(y_pred[,2] == 1 & y_true == 0) / sum(y_true == 0),3)))
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 0 & y_true == 1) / sum(y_true == 1),3),
                   round(sum(y_pred[,2] == 0 & y_true == 1) / sum(y_true == 1),3)))
params = c(paste("$x_", 1:3, "$", sep = ""), "$\\rho$", "MAE", "$\\alpha$", "$\\beta$")
tab = data.frame(params, tab)
print(xtable(tab), include.rownames = FALSE, sanitize.text.function = identity, booktabs = TRUE)

# mat = mat_dis; Y = Y_dis; params = params; adj = 1
# j = 1; mat = cbind(mat[, j], Y)

### Oakley function

N = 2^9
params = paste("x", 1:15, sep = "")
mat_sob = sobol_matrices(N = N, params = params, matrices = c("A", "AB"))
mat_dis = sobol_mat(N = N, params = params, matrices = "A")
Y_sob = oakley_Fun(mat_sob)
Y_dis = oakley_Fun(mat_dis)
sob = jansen_fun(Y_sob, N, params)
dis = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 0)
dis_adj = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 1)
tab = round(cbind(sobol = sob$value, 
                  ersatz = dis[,2], 
                  ersatz_adj = dis_adj[,2]), 3)
tab = rbind(tab, c(NaN, round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,2]))),3),
                   round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,3]))),3)))
tab = rbind(tab, c(NaN, round(apply(abs(tab[1:length(params),2:3] - tab[1:length(params),1]), 2, mean), 3)))
y_true <- ifelse(tab[1:length(params),1] > 0.05, 1, 0)
y_pred <- ifelse(tab[1:length(params),2:3] > 0.05, 1, 0)
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 1 & y_true == 0) / sum(y_true == 0),3),
                   round(sum(y_pred[,2] == 1 & y_true == 0) / sum(y_true == 0),3)))
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 0 & y_true == 1) / sum(y_true == 1),3),
                   round(sum(y_pred[,2] == 0 & y_true == 1) / sum(y_true == 1),3)))
params = c(paste("$x_", 1:15, "$", sep = ""), "$\\rho$", "MAE", "$\\alpha$", "$\\beta$")
tab = data.frame(params, tab)
print(xtable(tab), include.rownames = FALSE, sanitize.text.function = identity, booktabs = TRUE)

# mat = mat_dis; Y = Y_dis; params = params; adj = 1
# j = 1; mat = cbind(mat[, j], Y)

### Sobol function

N = 2^9
params = paste("x", 1:8, sep = "")
mat_dis = sobol_mat(N = N, params = params, matrices = "A")
Y_dis = sobol_Fun(mat_dis)
sob = Ti[2,]
dis = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 0)
png("/Users/alessiolachi/Documents/3) UniCamillus/Articoli/8) discrepancy GSA/Ris/scatterplot5.png", width = 1200, height = 600)
par(mfrow = c(2,4), mar=c(4,4,2,1))
dis_adj = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 1)
par(mfrow = c(1,1))
dev.off()
tab = round(cbind(sobol = sob, 
                  ersatz = dis[,2], 
                  ersatz_adj = dis_adj[,2]), 3)
tab = rbind(tab, c(NaN, round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,2]))),3),
                   round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,3]))),3)))
tab = rbind(tab, c(NaN, round(apply(abs(tab[1:length(params),2:3] - tab[1:length(params),1]), 2, mean), 3)))
y_true <- ifelse(tab[1:length(params),1] > 0.05, 1, 0)
y_pred <- ifelse(tab[1:length(params),2:3] > 0.05, 1, 0)
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 1 & y_true == 0) / sum(y_true == 0),3),
                   round(sum(y_pred[,2] == 1 & y_true == 0) / sum(y_true == 0),3)))
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 0 & y_true == 1) / sum(y_true == 1),3),
                   round(sum(y_pred[,2] == 0 & y_true == 1) / sum(y_true == 1),3)))
params = c(paste("$x_", 1:8, "$", sep = ""), "$\\rho$", "MAE", "$\\alpha$", "$\\beta$")
tab = data.frame(params, tab)
print(xtable(tab), include.rownames = FALSE, sanitize.text.function = identity, booktabs = TRUE)

png("/Users/alessiolachi/Documents/3) UniCamillus/Articoli/8) discrepancy GSA/Ris/scatterplot.png", width = 1200, height = 600)
n_cols <- ncol(mat_dis)
par(mfrow = c(2,4), mar=c(4,4,2,1))
for(j in 1:n_cols) {
  x <- mat_dis[, j]
  breaks <- seq(min(x), max(x), by = 0.05)
  means <- sapply(1:(length(breaks)-1), function(i) {
    idx <- which(x >= breaks[i] & x < breaks[i+1])
    if(length(idx) > 0) mean(Y_dis[idx]) else NA
  })
  midpoints <- (breaks[-1] + breaks[-length(breaks)])/2
  plot(x, Y_dis, xlab = paste("x", j), ylab = "Y", cex.lab = 1.5, cex.axis = 1.5, main=var(means)/var(Y_dis))
  points(midpoints, means, col = "red", pch = 19, cex = 2)
}
par(mfrow = c(1,1))
dev.off()

png("/Users/alessiolachi/Documents/3) UniCamillus/Articoli/8) discrepancy GSA/Ris/scatterplot2.png", width = 1200, height = 600)
n_cols <- ncol(mat_dis)
par(mfrow = c(2,4), mar=c(4,4,2,1))
for(j in 1:n_cols) {
  x <- mat_dis[, j]
  Y_dis = (Y_dis-min(Y_dis))/(max(Y_dis)-min(Y_dis))
  plot(x, Y_dis, xlab = paste("x", j), ylab = "Y", cex.lab = 1.5, cex.axis = 1.5)
  breaks <- seq(min(x), max(x), by = 0.05)
  means <- sapply(1:(length(breaks)-1), function(i) {
    idx <- which(x >= breaks[i] & x < breaks[i+1])
    if(length(idx) > 0) mean(Y_dis[idx]) else NA
  })
  midpoints <- (breaks[-1] + breaks[-length(breaks)])/2
  points(midpoints, means, col = "red", pch = 19, cex = 2)
}
par(mfrow = c(1,1))
dev.off()

png("/Users/alessiolachi/Documents/3) UniCamillus/Articoli/8) discrepancy GSA/Ris/scatterplot3.png", width = 1200, height = 600)
n_cols <- ncol(mat_dis)
par(mfrow = c(2,4), mar=c(4,4,2,1))
for(j in 1:n_cols) {
  x <- mat_dis[, j]
  Y_dis = (rank(Y_dis, ties.method = "first")-1) / length(Y_dis)
  plot(x, Y_dis, xlab = paste("x", j), ylab = "Y", cex.lab = 1.5, cex.axis = 1.5)
  breaks <- seq(min(x), max(x), by = 0.05)
  means <- sapply(1:(length(breaks)-1), function(i) {
    idx <- which(x >= breaks[i] & x < breaks[i+1])
    if(length(idx) > 0) mean(Y_dis[idx]) else NA
  })
  midpoints <- (breaks[-1] + breaks[-length(breaks)])/2
  points(midpoints, means, col = "red", pch = 19, cex = 2)
}
par(mfrow = c(1,1))
dev.off()

# mat = mat_dis; Y = Y_dis; params = params; adj = 1
# j = 1; mat = cbind(mat[, j], Y)

### Play function

inverse_cdf_uniform <- function(u, a = 0, b = 1) {
  return(a + (b - a) * u)
}

inverse_cdf_normal <- function(u, mu = 0, sigma = 1) {
  return(qtruncnorm(u, a = 0, b = 1, mean = mu, sd = sigma))
}

model = function(X) {
  if (X[5] == 1) {
    x1 = inverse_cdf_normal(X[1], mu = 1/2, sigma = 1/12)      
    x2 = inverse_cdf_normal(X[2], mu = 1/2, sigma = 1/12)      
    x3 = inverse_cdf_normal(X[3], mu = 1/2, sigma = 1/12)      
  } else {
    x1 = inverse_cdf_uniform(X[1], a = 0, b = 1)  
    x2 = inverse_cdf_uniform(X[2], a = 0, b = 1)   
    x3 = inverse_cdf_uniform(X[3], a = 0, b = 1)   
  }
  if        (X[4] == 0) {
    return((x1 + x2 + x3)^(1/3))
  } else if (X[4] == 1) {
    return((x1 + x2 * x3)^(1/3))  
  } else if (X[4] == 2) {
    return((x1 * x2 + x3)^(1/3))  
  } else if (X[4] == 3) {
    return((x1 * x3 + x2)^(1/3))  
  } else if (X[4] == 4) {
    return((x1 * (x2 + x3))^(1/3))  
  } else if (X[4] == 5) {
    return((x2 * (x1 + x3))^(1/3))  
  } else if (X[4] == 6) {
    return((x3 * (x1 + x2))^(1/3))  
  } else if (X[4] == 7) {
    return((x1 * x2 * x3)^(1/3))  
  }
}

N = 2^9
params = c("\u03B81", "\u03B82", "\u03B83", "\u03BE", "\u03B6")

mat_sob = sobol_mat(N = N, params = params)
mat_sob2 = mat_sob
mat_sob2[,4] = floor(mat_sob2[,4] * 8) 
mat_sob2[,5] = round(mat_sob2[,5], 0)
Y_sob = NULL
for (i in 1:nrow(mat_sob2)) {
  Y_sob = c(Y_sob, model(mat_sob2[i,]))
}

mat_dis = sobol_mat(N = N, params = params, matrices = "A")
mat_dis2 = mat_dis
mat_dis2[,4] = floor(mat_dis2[,4] * 8) 
mat_dis2[,5] = round(mat_dis2[,5], 0)
Y_dis = NULL
for (i in 1:nrow(mat_dis2)) {
  Y_dis = c(Y_dis, model(mat_dis2[i,]))
}

sob = sobol_indices(Y = Y_sob, N = N, params = params)
dis = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 0)
dis_adj = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 1)
tab = round(cbind(sobol = sob$results$original[6:10], 
                  ersatz = dis[,2], 
                  ersatz_adj = dis_adj[,2]), 3)
tab = rbind(tab, c(NaN, round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,2]))),3),
                        round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,3]))),3)))
tab = rbind(tab, c(NaN, round(apply(abs(tab[1:length(params),2:3] - tab[1:length(params),1]), 2, mean), 3)))
y_true <- ifelse(tab[1:length(params),1] > 0.05, 1, 0)
y_pred <- ifelse(tab[1:length(params),2:3] > 0.05, 1, 0)
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 1 & y_true == 0) / sum(y_true == 0),3),
                        round(sum(y_pred[,2] == 1 & y_true == 0) / sum(y_true == 0),3)))
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 0 & y_true == 1) / sum(y_true == 1),3),
                        round(sum(y_pred[,2] == 0 & y_true == 1) / sum(y_true == 1),3)))
params = c("$\\theta_1$", "$\\theta_2$", "$\\theta_3$", "$\\zeta$", "$\\xi$", "$\\rho$", "MAE", "$\\alpha$", "$\\beta$")
tab = data.frame(params, tab)
print(xtable(tab), include.rownames = FALSE, sanitize.text.function = identity, booktabs = TRUE)

# mat = mat_dis; Y = Y_dis; params = params; adj = 1
# j = 1; mat = cbind(mat[, j], Y)

### Hidrology function

N = 50000
params = paste("x", 1:5, sep = "")
mat_dis = sobol_mat(N = N, params = params, matrices = "A")
mat_dis2 = mat_dis
mat_dis2[,1] = qunif(mat_dis[,1], min = 1, max = 500)
mat_dis2[,2] = qunif(mat_dis[,2], min = 0.1, max = 2)
mat_dis2[,3] = qunif(mat_dis[,3], min = 0.1, max = 0.98) 
mat_dis2[,4] = qunif(mat_dis[,4], min = 0, max = 0.1) 
mat_dis2[,5] = qunif(mat_dis[,5], min = 0.1, max = 0.98) 
write.csv(mat_dis2, file="/Users/alessiolachi/Documents/3) UniCamillus/Articoli/8) discrepancy GSA/hydro.csv")
Y_dis = read.csv("/Users/alessiolachi/Documents/3) UniCamillus/Articoli/8) discrepancy GSA/hydro2.csv")[,7]

dis = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 0)
dis_adj = discrepancy_ersatz(mat = mat_dis, Y = Y_dis, params = params, adj = 1)

tab = round(cbind(sobol = c(0.506923, 0.017083, 0.072756, 0.002433, 0.72544),
                  ersatz = dis[,2], 
                  ersatz_adj = dis_adj[,2]), 3)
tab = rbind(tab, c(NaN, round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,2]))),3),
                   round(abs(cor(savage_scores_fun(tab[,1]), savage_scores_fun(tab[,3]))),3)))
tab = rbind(tab, c(NaN, round(apply(abs(tab[1:length(params),2:3] - tab[1:length(params),1]), 2, mean), 3)))
y_true <- ifelse(tab[1:length(params),1] > 0.05, 1, 0)
y_pred <- ifelse(tab[1:length(params),2:3] > 0.05, 1, 0)
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 1 & y_true == 0) / sum(y_true == 0),3),
                   round(sum(y_pred[,2] == 1 & y_true == 0) / sum(y_true == 0),3)))
tab = rbind(tab, c(NaN, round(sum(y_pred[,1] == 0 & y_true == 1) / sum(y_true == 1),3),
                   round(sum(y_pred[,2] == 0 & y_true == 1) / sum(y_true == 1),3)))
params = c(params, "$\\rho$", "MAE", "$\\alpha$", "$\\beta$")
tab = data.frame(params, tab)
print(xtable(tab), include.rownames = FALSE, sanitize.text.function = identity, booktabs = TRUE)

# mat = mat_dis; Y = Y_dis; params = params; adj = 1
# j = 1; mat = cbind(mat[, j], Y)

params = paste("x", 1:5, sep = "")
Ns <- 2^(5:15)
res <- data.frame(N = Ns,
                  rho = NA_real_,
                  mae = NA_real_)
for (i in seq_along(Ns)) {
  N <- Ns[i]
  dis <- discrepancy_ersatz(mat = mat_dis[1:N, ], Y = Y_dis[1:N], params = params, adj = 1)
  rho_val <- abs(cor(
    savage_scores_fun(c(0.506923, 0.017083, 0.072756, 0.002433, 0.72544)),
    savage_scores_fun(dis[,2])
  ))
  mae_val <- mean(abs(c(0.506923, 0.017083, 0.072756, 0.002433, 0.72544) - dis[,2]))
  res$rho[i] <- rho_val
  res$mae[i] <- mae_val
}
plot = ggplot(res, aes(x = N)) +
  geom_line(aes(y = rho, colour = "rho")) +
  geom_point(aes(y = rho, colour = "rho")) +
  geom_line(aes(y = mae, colour = "MAE")) +
  geom_point(aes(y = mae, colour = "MAE")) +
  scale_x_log10(breaks = Ns) +        # scala log per N
  labs(x = "Sample size (N)",
       y = "Value",
       colour = "Metric",
       title = expression("Trend of " * rho * " and MAE vs sample size")) +
  theme_minimal() +
  theme(
    axis.text  = element_text(size = 18),   # numeri sugli assi
    axis.title = element_text(size = 18),   # titoli degli assi
    plot.title = element_text(size = 18, face = "bold") # titolo del grafico
  )
plot
ggsave("/Users/alessiolachi/Documents/3) UniCamillus/Articoli/8) discrepancy GSA/Ris/trend_adj.pdf", plot = plot,
       width = 10, height = 5, dpi = 300)

params = paste("x", 1:5, sep = "")
Ns <- 2^(5:15)
res <- data.frame(N = Ns,
                  rho = NA_real_,
                  mae = NA_real_)
for (i in seq_along(Ns)) {
  N <- Ns[i]
  dis <- discrepancy_ersatz(mat = mat_dis[1:N, ], Y = Y_dis[1:N], params = params, adj = 0)
  rho_val <- abs(cor(
    savage_scores_fun(c(0.506923, 0.017083, 0.072756, 0.002433, 0.72544)),
    savage_scores_fun(dis[,2])
  ))
  mae_val <- mean(abs(c(0.506923, 0.017083, 0.072756, 0.002433, 0.72544) - dis[,2]))
  res$rho[i] <- rho_val
  res$mae[i] <- mae_val
}
plot = ggplot(res, aes(x = N)) +
  geom_line(aes(y = rho, colour = "rho")) +
  geom_point(aes(y = rho, colour = "rho")) +
  geom_line(aes(y = mae, colour = "MAE")) +
  geom_point(aes(y = mae, colour = "MAE")) +
  scale_x_log10(breaks = Ns) +        # scala log per N
  labs(x = "Sample size (N)",
       y = "Value",
       colour = "Metric",
       title = expression("Trend of " * rho * " and MAE vs sample size")) +
  theme_minimal() +
  theme(
    axis.text  = element_text(size = 18),   # numeri sugli assi
    axis.title = element_text(size = 18),   # titoli degli assi
    plot.title = element_text(size = 18, face = "bold") # titolo del grafico
  )
plot
ggsave("/Users/alessiolachi/Documents/3) UniCamillus/Articoli/8) discrepancy GSA/Ris/trend.pdf", plot = plot,
       width = 10, height = 5, dpi = 300)
