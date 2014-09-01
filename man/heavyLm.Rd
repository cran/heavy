\name{heavyLm}
\alias{heavyLm}

\title{Linear models under heavy-tailed distributions}
\description{
  This function is used to fit linear models considering heavy-tailed errors.
}
\usage{
heavyLm(formula, data, family = Student(df = 4), subset, na.action, 
  control, model = TRUE, x = FALSE, y = FALSE, contrasts = NULL)
}

\arguments{
  \item{formula}{ an object of class \code{"formula"}: a symbolic description of 
  the model to be fitted.}
  \item{data}{ an optional data frame containing the variables in the model. If 
  not found in \code{data}, the variables are taken from \code{environment(formula)}, 
  typically the environment from which \code{heavyLm} is called.}
  \item{family}{ a description of the error distribution to be used in the model. 
  By default the Student-t distribution with 4 degrees of freedom is considered.}
  \item{subset}{ an optional expression indicating the subset of the rows of 
  data that should be used in the fitting process.}
  \item{na.action}{ a function that indicates what should happen when the data contain NAs.}
  \item{control}{ a list of control values for the estimation algorithm to replace 
  the default values returned by the function \code{\link{heavy.control}}.}
  \item{model, x, y}{ logicals. If \code{TRUE} the corresponding components of 
  the fit (the model frame, the model matrix, the response) are returned.}
  \item{contrasts}{ an optional list. See the \code{contrasts.arg} of \code{\link{model.matrix.default}}.}
}

\value{
  An object of class \code{heavyLm} representing the linear model fit. Generic 
  functions \code{print} and \code{summary}, show the results of the fit.
  
  The following components must be included in a legitimate \code{heavyLm} object.
  \item{call}{  a list containing an image of the \code{heavyLm} call that produced the object.}
  \item{family}{  the \code{\link{heavy.family}} object used, with the estimated shape parameters (if requested).}
  \item{coefficients}{  final estimate of the coefficients vector.}
  \item{sigma2}{  final scale estimate of the random error.}
  \item{fitted.values}{  the fitted mean values.}
  \item{residuals}{  the residuals, that is response minus fitted values.}
  \item{logLik}{  the log-likelihood at convergence.}
  \item{numIter}{  the number of iterations used in the iterative algorithm.}
  \item{weights}{  estimated weights corresponding to the assumed heavy-tailed distribution.}
  \item{distances}{  squared of scaled residuals.}
  \item{acov}{  asymptotic covariance matrix of the coefficients estimates.}
}

\references{
  Dempster, A.P., Laird, N.M., and Rubin, D.B. (1980). 
  Iteratively reweighted least squares for linear regression when errors are Normal/Independent distributed. 
  In P.R. Krishnaiah (Ed.), \emph{Multivariate Analysis V}, p. 35-57. North-Holland.
}

\examples{
data(ereturns)
fit <- heavyLm(m.marietta ~ CRSP, data = ereturns, family = Student(df = 5))
summary(fit)
}
\keyword{regression}
\keyword{models}
