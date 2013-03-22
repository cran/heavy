heavyLm <- 
function(formula, data, family = Student(df = 4), subset, na.action, 
  control = heavy.control(), model = TRUE, x = FALSE, y = FALSE, contrasts = NULL)
{
  ret.x <- x
  ret.y <- y
  Call <- match.call()
  mf <- match.call(expand.dots = FALSE)
  mf$family <- mf$control <- mf$model <- mf$x <- mf$y <- mf$contrasts <- NULL
  mf$drop.unused.levels <- TRUE
  mf[[1]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  Terms <- attr(mf, "terms")
  y <- model.response(mf, "numeric")
  x <- model.matrix(Terms, mf, contrasts)
  xnames <- dimnames(x)[[2]]
  dx <- dim(x)
  n <- dx[1]
  p <- dx[2]

  ## initial estimates
  fit <- lsfit(x, y, intercept = FALSE)[1:2]
  res <- fit$residuals
  cf <- fit$coefficients
  sigma2 <- sum(res^2) / n

  ## extract family info
  if (!inherits(family, "heavy.family"))
    stop("Use only with 'heavy.family' objects")
  if (is.null(family$family))
    stop("'family' not recognized")
  kind <- family$which
  if ((kind < 0) || (kind > 4))
    stop("not valid 'family' object")
  settings <- c(kind, family$npars, unlist(family$pars))

  ## set control values
  if (missing(control))
    control <- heavy.control()
  ctrl <- unlist(control)[1:4]
  ctrl <- c(ctrl, 0)

  ## Call fitter
  now <- proc.time()
  fit <- .C("lm_fit",
            y = as.double(y),
            x = as.double(x),
            dims = as.integer(dx),
            settings = as.double(settings),
            coefficients = as.double(cf),
            sigma2 = as.double(sigma2),
            fitted = double(n),
            residuals = as.double(res),
            distances = double(n),
            weights = as.double(rep(1, n)),
            logLik = double(1),
            acov = double(p^2),
            control = as.double(ctrl))
  speed <- proc.time() - now

  ## creating the output object
  out <- list(call = Call,
              dims = dx,
              family = family,
              settings = fit$settings,
              coefficients = fit$coefficients,
              sigma2 = fit$sigma2,
              fitted.values = fit$fitted,
              residuals = fit$residuals,
              logLik = fit$logLik,
              numIter = fit$control[5],
              control = control,
              weights = fit$weights,
              distances = fit$distances,
              acov = matrix(fit$acov, ncol = p),
              speed = speed,
              converged = FALSE)
  if (!control$fix.shape) {
    if ((kind > 1) && (kind < 4)) {
      df <- signif(out$settings[3], 6)
      out$family$call <- call(out$family$family, df = df)
    }
  }
  if (out$numIter < control$maxIter)
    out$converged <- TRUE
  names(out$coefficients) <- xnames
  out$na.action <- attr(mf, "na.action")
  out$contrasts <- attr(x, "contrasts")
  out$xlevels <- .getXlevels(Terms, mf)
  out$terms <- Terms
  if (model)
    out$model <- mf
  if (ret.y)
    out$y <- y
  if (ret.x)
    out$x <- x
  class(out) <- "heavyLm"
  out
}

print.heavyLm <-
function(x, digits = 4, ...)
{
  cat("Call:\n")
  x$call$family <- x$family$call
  dput(x$call, control = NULL)
  if (x$converged)
    cat("Converged in", x$numIter, "iterations\n")
  else
    cat("Maximum number of iterations exceeded\n")
  cat("\nCoefficients:\n ")
  print(format(round(x$coef, digits = digits)), quote = F, ...)
  nobs <- x$dims[1]
  rdf <- nobs - x$dims[2]
  cat("\nDegrees of freedom:", nobs, "total;", rdf, "residual")
  cat("\nScale estimate:", format(x$sigma2), "\n")
  invisible(x)
}

summary.heavyLm <-
function (object, ...)
{
  z <- object
  se <- sqrt(diag(z$acov))
  est <- z$coefficients
  zval <- est / se
  ans <- z[c("call", "terms")]
  ans$dims <- z$dims
  ans$family <- z$family
  ans$logLik <- z$logLik
  ans$sigma2 <- z$sigma2
  ans$residuals <- z$residuals
  ans$coefficients <- cbind(est, se, zval, 2 * pnorm(abs(zval), lower.tail = FALSE))
  dimnames(ans$coefficients) <- list(names(z$coefficients),
        c("Estimate", "Std.Error", "Z value", "p-value"))
  ans$correlation <- z$acov / outer(se, se)
  dimnames(ans$correlation) <- dimnames(ans$coefficients)[c(1,1)]
  class(ans) <- "summary.heavyLm"
  ans
}

print.summary.heavyLm <-
function(x, digits = 4, ...)
{
  cat("Linear model under heavy-tailed distributions\n")
  cat(" Data:", paste(as.name(x$call$data), ";", sep = ""))
  print(x$family)
  resid <- x$residuals
  nobs <- x$dims[1]
  p <- x$dims[2]
  rdf <- nobs - p
  if (rdf > 5) {
    cat("\nResiduals:\n")
		rq <- quantile(resid)
		names(rq) <- c("Min", "1Q", "Median", "3Q", "Max")
		print(rq, digits = digits, ...)
	}
	else if(rdf > 0) {
	 cat("\nResiduals:\n")
	 print(resid, digits = digits, ...)
  }
  cat("\nCoefficients:\n ")
  print(format(round(x$coef, digits = digits)), quote = F, ...)
  cat("\nDegrees of freedom:", nobs, "total;", rdf, "residual")
  cat("\nScale estimate:", format(x$sigma2))
  cat("\nLog-likelihood:", format(x$logLik), "on", p + 1, "degrees of freedom\n")
  invisible(x)
}
