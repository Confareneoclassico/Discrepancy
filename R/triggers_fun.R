# DEFINE MODEL #################################################################

triggers_fun <- function(tau, epsilon, base.sample.size, cost.discrepancy, phi, k) {
  
  params <- paste("X", 1:k, sep = "")
  
  # Define vectors to loop through --------------------------------------------
  
  simulations <- c("discrepancy", "jansen")
  disc.type <- c("adjusted", "not.adjusted")
  
  # Select the sampling method -------------------------------------------------
  
  if (tau == 1) {
    
    type <- "R"
    
  } else if (tau == 2) {
    
    type <- "QRN"
  }
  
  # If statements to select matrices and N as a function
  # of the estimator used -----------------------------------------------------
  
  mat <- y <- ind <- disc <- output <- jansen.results <- jansen.results2 <- jansen.results3 <-jansen.results4 <- list()
  
  for (i in simulations) {
    
    if (i == "discrepancy") {
      
      matrices <- "A"
      N <- cost.discrepancy
      
    } else if (i == "jansen") {
      matrices <- c("A", "AB") 
      N <- base.sample.size
      
    }
    
    # Construct the sample matrix, randomly transform it
    # according to phi and run the metafunction -------------------------------
    
    if (i == "discrepancy") {
      
      set.seed(epsilon)
      mat.uniform <- sobol_mat(matrices = matrices, N = N, params = params, 
                                    type = type)
      
      set.seed(epsilon)
      mat[[i]] <- random_distributions_fun(sobol_mat(matrices = matrices, 
                                                          N = N, params = params, 
                                                          type = type), phi = phi)
      
    } else if (i == "jansen") {
      
      set.seed(epsilon)
      mat[[i]] <- random_distributions_fun(sobol_mat(matrices = matrices, 
                                                          N = N, params = params, 
                                                          type = type), phi = phi)
    }
    
    set.seed(epsilon)
    y[[i]] <- sensobol::metafunction(data = mat[[i]], epsilon = epsilon)
    
    # Calculate first (saltelli) and total order (jansen) indices 
    # and discrepancy values --------------------------------------------------
    
    if (i == "jansen") {
      
      ind[[i]] <- jansen_fun(d = y[[i]], N = base.sample.size, params = params)
      
    } 
    
    if (i == "discrepancy") {
      
      discrepancy.val <- lapply(disc.type, function(x) 
        discrepancy_wrapper_fun(mat = mat.uniform, Y = y[[i]], params = params, type = x))
      
    }
  }
  
  # Arrange output -------------------------------------------------------------
  
  names(discrepancy.val) <- disc.type
  all.simulations <- c(ind, discrepancy.val)

  # Savage scores --------------------------------------------------------------
  
  all.simulations.savage <- list()
  all.simulations.savage2 <- list()
  
  for (i in names(all.simulations)) {
    
    all.simulations.savage[[i]] <- savage_scores_fun(all.simulations[[i]][["value"]])
    all.simulations.savage2[[i]] <- all.simulations[[i]][["value"]]
    
  }
  
  # Correlation between indices and discrepancy measures -----------------------
  
  for (i in disc.type) {
    
    jansen.results[[i]] <- cor(all.simulations.savage$jansen, all.simulations.savage[[i]])
    jansen.results2[[i]] <-  mean(abs(all.simulations.savage2[[i]] - all.simulations.savage2$jansen))
    y_pred <- ifelse(all.simulations.savage2[[i]] > 0.05, 1, 0)
    y_true <- ifelse(all.simulations.savage2$jansen > 0.05, 1, 0)
    jansen.results3[[i]] <- sum(y_pred == 1 & y_true == 0) / sum(y_true == 0)
    jansen.results4[[i]] <- sum(y_pred == 0 & y_true == 1) / sum(y_true == 1)
    if (is.nan(jansen.results3[[i]]) == T) {
      jansen.results3[[i]] = 0
    } 
    if (is.nan(jansen.results4[[i]]) == T) {
      jansen.results4[[i]] = 0
    } 
    
  }
  
  # Arrange output -------------------------------------------------------------
  
  output <- unlist(jansen.results)
  output2 <- unlist(jansen.results2)
  output3 <- unlist(jansen.results3)
  output4 <- unlist(jansen.results4)
  
  return(c(output, output2, output3, output4))
  
}

################################################################################