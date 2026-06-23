rm(list = ls())

#   PRELIMINARY FUNCTIONS ######################################################

sensobol::load_packages(c("data.table", "tidyverse", "sensobol", "benchmarkme", 
                          "here", "Rfast", "parallel", "scales", "qrng"))

sobol_mat <- function(matrices = c("A", "B", "AB"),
                      N, params, order = "first",
                      type = "QRN") {
  
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

# Create custom theme -----------------------------------------------------------

theme_AP <- function() {
  theme_bw() +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          legend.background = element_rect(fill = "transparent",
                                           color = NA),
          legend.key = element_rect(fill = "transparent",
                                    color = NA), 
          strip.background = element_rect(fill = "white"), 
          legend.text = element_text(size = 7.3), 
          axis.title = element_text(size = 10),
          legend.key.width = unit(0.4, "cm"), 
          legend.key.height = unit(0.4, "cm"), 
          legend.key.spacing.y = unit(0, "lines"),
          legend.box.spacing = unit(0, "pt"),
          legend.title = element_text(size = 7.3), 
          axis.text.x = element_text(size = 7), 
          axis.text.y = element_text(size = 7), 
          axis.title.x = element_text(size = 7.3), 
          axis.title.y = element_text(size = 7.3),
          plot.title = element_text(size = 8),
          strip.text.x = element_text(size = 7.4), 
          strip.text.y = element_text(size = 7.4)) 
}

# Select color palette ----------------------------------------------------------

selected.palette <- "Darjeeling1"

# SOURCE ALL R FUNCTIONS NEEDED FOR THE STUDY ###################################

# Source all .R files in the "functions" folder --------------------------------

r_functions <- list.files(path = here("functions"), pattern = "\\.R$", full.names = TRUE)
lapply(r_functions, source)

# CREATE SAMPLE MATRIX #########################################################

# DEFINE SETTINGS --------------------------------------------------------------

N <- 2^9
params <- c("epsilon", "phi", "k", "tau", "base.sample.size")
mat <- sobol_mat(matrices = "A", N = N, params = params)

# DEFINE DISTRIBUTIONS ---------------------------------------------------------

mat[, "epsilon"] <- floor(qunif(mat[, "epsilon"], 1, 200))
mat[, "phi"] <- floor(mat[, "phi"] * 8) + 1
mat[, "k"] <- floor(qunif(mat[, "k"], 3, 50))
mat[, "tau"] <- floor(mat[, "tau"] * 2) + 1
mat[, "base.sample.size"] <- floor(qunif(mat[, "base.sample.size"], 10, 100))

# RE-ARRANGE COST OF ANALYSIS --------------------------------------------------

cost.jansen <- mat[, "base.sample.size"] * (mat[, "k"] + 1)
cost.saltelli <- mat[, "base.sample.size"] * (mat[, "k"] + 2)
cost.discrepancy <- cost.jansen

final.mat <- cbind(mat, cost.jansen, cost.saltelli, cost.discrepancy)

# RUN SIMULATIONS ##############################################################

y <- mclapply(1:nrow(final.mat), function(i) {
  triggers_fun(tau = final.mat[i, "tau"],
            epsilon = final.mat[i, "epsilon"], 
            base.sample.size = final.mat[i, "base.sample.size"], 
            cost.discrepancy = final.mat[i, "cost.discrepancy"], 
            phi = final.mat[i, "phi"], 
            k = final.mat[i, "k"])}, 
  mc.cores = floor(detectCores() * 0.75))

# ARRANGE OUTPUT ###############################################################

final.dt <- do.call(rbind, y) %>%
  cbind(final.mat, .) %>%
  data.table() %>%
  .[, id:= .I]

colnames(final.dt)[9] = "Adjusted"
colnames(final.dt)[10] = "Non-Adjusted"

colnames(final.dt)[11] = "Adjusted "
colnames(final.dt)[12] = "Non-Adjusted "

colnames(final.dt)[13] = "Adjusted  "
colnames(final.dt)[14] = "Non-Adjusted  "

colnames(final.dt)[15] = "Adjusted   "
colnames(final.dt)[16] = "Non-Adjusted   "

median(final.dt$`Non-Adjusted`)
median(final.dt$`Non-Adjusted `)
median(final.dt$`Non-Adjusted  `)
median(final.dt$`Non-Adjusted   `)

median(final.dt$Adjusted)
median(final.dt$`Adjusted `)
median(final.dt$`Adjusted  `)
median(final.dt$`Adjusted   `)

# PLOT OUTPUT ##################################################################

disc.type <- c("Adjusted", "Non-Adjusted")
boxplots <-  melt(final.dt, measure.vars = disc.type) %>%
  ggplot(., aes(variable, value)) +
  geom_boxplot() +
  labs(y = expression(rho), x = "") +
  scale_y_continuous(breaks = pretty_breaks(n = 3)) +
  ylim(0,1) + 
  theme_bw() +
  theme(
    legend.position = "none",
    axis.title.x = element_text(size = 16),  # xlab
    axis.title.y = element_text(size = 16),  # ylab
    axis.text.x = element_text(size = 16),   # valori asse x
    axis.text.y = element_text(size = 16)    # valori asse y
  )
boxplots

disc.type <- c("Adjusted ", "Non-Adjusted ")
boxplots <-  melt(final.dt, measure.vars = disc.type) %>%
  ggplot(., aes(variable, value)) +
  geom_boxplot() +
  labs(y = "MAE", x = "") +
  scale_y_continuous(breaks = pretty_breaks(n = 3)) +
  ylim(0,1) + 
  theme_bw() +
  theme(
    legend.position = "none",
    axis.title.x = element_text(size = 16),  # xlab
    axis.title.y = element_text(size = 16),  # ylab
    axis.text.x = element_text(size = 16),   # valori asse x
    axis.text.y = element_text(size = 16)    # valori asse y
  )
boxplots

disc.type <- c("Adjusted  ", "Non-Adjusted  ")
boxplots <-  melt(final.dt, measure.vars = disc.type) %>%
  ggplot(., aes(variable, value)) +
  geom_boxplot() +
  labs(y = expression(alpha), x = "") +
  scale_y_continuous(breaks = pretty_breaks(n = 3)) +
  ylim(0,1) + 
  theme_bw() +
  theme(
    legend.position = "none",
    axis.title.x = element_text(size = 16),  # xlab
    axis.title.y = element_text(size = 16),  # ylab
    axis.text.x = element_text(size = 16),   # valori asse x
    axis.text.y = element_text(size = 16)    # valori asse y
  )
boxplots

disc.type <- c("Adjusted   ", "Non-Adjusted   ")
boxplots <-  melt(final.dt, measure.vars = disc.type) %>%
  ggplot(., aes(variable, value)) +
  geom_boxplot() +
  labs(y = expression(beta), x = "") +
  scale_y_continuous(breaks = pretty_breaks(n = 3)) +
  ylim(0,1) + 
  theme_bw() +
  theme(
    legend.position = "none",
    axis.title.x = element_text(size = 16),  # xlab
    axis.title.y = element_text(size = 16),  # ylab
    axis.text.x = element_text(size = 16),   # valori asse x
    axis.text.y = element_text(size = 16)    # valori asse y
  )
boxplots
