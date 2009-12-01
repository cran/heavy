heavy.lme <-
function(fixed,
    random,
    groups,
    data = sys.frame(sys.parent()),
    family = Student(df = 4),
    subset,
    na.action = na.fail,
    control)
  UseMethod("heavy.lme")

heavy.lme.formula <-
function(fixed,
    random,
	  groups,
	  data = sys.frame(sys.parent()),
	  family = Student(df = 4),
	  subset,
	  na.action = na.fail, 
	  control = list())
{
  Call <- match.call()
  ## checking arguments
  if(!inherits(fixed, "formula") || length(fixed) != 3) {
    stop("\nFixed-effects model must be a formula of the form \"resp ~ pred\"")
  }
  if(missing(random)) {
    Call[["random"]] <- fixed[-2]
    random <- fixed[[3]]
  }
  groups <- asOneSidedFormula(groups)
  ## extract a data frame with enough information to evaluate 
  ## fixed, random and groups
  mfArgs <- list(formula = as.OneFormula(fixed, random, groups),
    data = data, na.action = na.action)
  if (!missing(subset)) {
    mfArgs[["subset"]] <- asOneSidedFormula(subset)[[2]]
  }
  dataMix <- do.call("model.frame", mfArgs)
  origOrder <- row.names(dataMix)	# preserve the original order
  y <- eval(fixed[[2]], dataMix)
  Fitted <- Resid <- data.frame(population = y, row.names = row.names(dataMix))
  ## sort the model.frame by groups and get the matrices and parameters
  ## used in the estimation procedures
  grp <- as.factor(eval(groups[[2]], dataMix))
  ord <- sort.list(grp) # order in which to sort the groups
  ## sorting the model.frame by groups and getting the matrices and parameters
  ## used in the estimation procedures
  grp <- grp[ord] # sorted groups
  glen <- tabulate(grp) # groups lengths
  ugrp <- as.character(unique(grp)) # unique groups
  n <- length(ugrp) # number of groups
  dataMix <- dataMix[ord,]  # sorted model.frame
  y <- y[ord] # sorted response vector
  N <- sum(glen)
  ZXrows <- nrow(dataMix)
  if(length(all.vars(fixed)) > 1) {
    X <- model.matrix(fixed, model.frame(fixed, dataMix))
  } else {
    X <- matrix(1, N, 1)
    dimnames(X) <- list(row.names(dataMix), "(Intercept)")
  }
  Xnames <- dimnames(X)[[2]]
  p <- ncol(X)
  if(length(all.vars(random))) {
    Z <- model.matrix(random, model.frame(random, dataMix))
  } else {
    Z <- X[,1,drop=F]
    dimnames(Z) <- list(dimnames(X)[[1]],"(Intercept)")
    Z[,1] <- 1
  }
  Znames <- dimnames(Z)[[2]];
  q <- ncol(Z)
  ZXcols <- p + q
  qq <- q^2
  ## generating parameters used throughout the calculations
  ZX <- array(c(Z,X), c(ZXrows, ZXcols), list(rep("",N), c(Znames, Xnames)))
  dims <- c(n, q, p, N, ZXrows, ZXcols)
  ## constructing the lmeData list
  lmeData <- list(ZXy = cbind(ZX,y), dims = dims, glen = glen, grp = grp,
                  ugrp = ugrp)
  ## initial estimates
  fit <- lsfit(X, y, intercept = FALSE)[1:2]
  sigma2 <- sum(fit$residuals^2) / N
  theta <- apply(Z, 2, function(x) sum(x^2))
  theta <- 0.375 * sqrt(theta / ZXrows)
  theta <- diag(theta, nrow = length(theta))
  theta <- as.vector(c(theta, sigma2))
  ## extract family info
  if (!inherits(family, "heavy.family"))
    stop("Use only with 'heavy.family' objects")
  if (is.null(family$family))
    stop("'family' not recognized")
  fltype <- family$which
  if ((fltype < 0) || (fltype > 3))
    stop("not valid 'family' object")
  settings <- c(fltype, family$npars, unlist(family$pars))
  ## set control values 
  control <- heavy.control()
  control <- c(unlist(control), 0, 0)
  ## Call fitter
  fit <- .C("heavy_lme_fit",
            ZXy = as.double(lmeData$ZXy),
            dims = as.integer(lmeData$dims),
            glen = as.integer(lmeData$glen),
            settings = as.double(settings),
            coefficients = as.double(fit$coefficients),
            theta = as.double(theta),
            ranef = as.double(rep(0, n * q)),
            OmegaHalf = as.double(rep(0, n * qq)),
            distances = as.double(rep(0, n)),
            weights = as.double(rep(0, n)),
            logLik = as.double(0),
            control = as.double(control))
  ## compute residuals
  res <- .C("heavy_lme_resid",
            ZXy = as.double(lmeData$ZXy),
            dims = as.integer(lmeData$dims),
            glen = as.integer(lmeData$glen),
            coefficients = as.double(fit$coefficients),
            ranef = as.double(fit$ranef),
            conditional = as.double(rep(0, ZXrows)),
            marginal = as.double(rep(0, ZXrows)))
  ## putting back in original order
  res$conditional <- res$conditional[origOrder]
  res$marginal <- res$marginal[origOrder]
  ## creating the output object
  out <- list(lmeData = lmeData,
              call = Call,
              family = family,
              coefficients = fit$coefficients,
              theta = matrix(fit$theta[1:qq], ncol = q),
              sigma2 = fit$theta[qq + 1],
              logLik = fit$logLik,
              numIter = fit$control[3],
              control = fit$control,
              ranef = matrix(fit$ranef, ncol = q, byrow = TRUE),
              weights = fit$weights,
              distances = fit$distances,
              OmegaHalf = matrix(fit$OmegaHalf, ncol = q, byrow = TRUE),
              residuals = list(conditional = res$conditional, marginal = res$marginal))
  names(out$coefficients) <- Xnames
  dimnames(out$theta) <- list(Znames, Znames)
  dimnames(out$ranef) <- list(ugrp, Znames)
  names(out$weights) <- ugrp
  names(out$distances) <- ugrp
  class(out) <- "heavy.lme"
  out
}

print.heavy.lme <-
function(x, digits = 4, ...)
{
  cat("Call:\n")
  cat("  Fixed:", deparse(x$call$fixed), "\n")
  cat(" Random:", deparse(x$call$random), "\n")
  cat(" Groups:", deparse(x$call$groups), "\n")
  cat("   Data:", as.name(x$call$data), "\n")
  print(x$family)
  cat("\nLog-likelihood:", format(x$logLik), "\n")
  cat("\nFixed Effects Estimate(s):\n ")
  print(x$coefficients)
  cat("\nRandom effects scale matrix\n")
  print.symmetric(x$theta)
  cat("\nWithin-Group scale parameter:", format(x$sigma2), "\n")
  cat("\nNumber of Observations:", x$lmeData$dims[4], "\n")
  cat("Number of Groups:", x$lmeData$dims[1], "\n")
}
