\name{l1fit}
\alias{l1fit}

\title{Minimum absolute residual (L1) regression}
\description{
  Performs an L1 regression on a matrix of explanatory variables and a vector of responses.
}
\usage{
l1fit(x, y, intercept = TRUE, tolerance = 1e-7, print.it = TRUE)
}

\arguments{
  \item{x}{ vector or matrix of explanatory variables. Each row corresponds to 
  an observation and each column to a variable. The number of rows of \code{x} 
  should equal the number of data values in \code{y}, and there should be fewer 
  columns than rows. Missing values are not allowed.}
  \item{y}{ numeric vector containing the response. Missing values are not allowed.}
  \item{intercept}{ whether or not an intercept term should be used.}
  \item{tolerance}{ numerical value used to test for singularity in the regression.}
  \item{print.it}{ logical flag: if \code{TRUE}, then warnings about non-unique 
  solutions and rank deficiency are given.}
}

\value{
  list defining the regression (compare with the function \code{\link{lsfit}}). 
  \item{coefficients}{ vector of coefficients. If intercept is TRUE, the initial 
  element is the term for the intercept.}
  \item{sigma2}{ estimation of the scale parameter.}
  \item{minimum}{ the minimum value of objective function, i.e. the minimum sum of 
  absolute values of the residuals.}
  \item{fitted.values}{ the fitted values from the fit.}
  \item{residuals}{ the residuals from the fit.}
  \item{rank}{ the computed rank (number of linearly independent columns in the 
  model matrix).}
  \item{numIter}{ the number of iterations used to compute the estimates.}
  \item{info}{ exit code with the value 1 if a solution has been calculated 
  successfully, 2 if the calculations are terminated prematurely. An exit code 
  of 1 indicates that the solution is unique, while an exit code of 0 indicates 
  that the solution almost certainly is not unique.}
  \item{message}{ vector of one or two character strings stating whether a 
  non-unique solution is possible, or if the x matrix (plus the intercept) 
  was found to be rank deficient. This component is present only if one of 
  these conditions exists.}
}

\details{
  The Barrodale-Roberts algorithm, which is a specialized linear programming 
  algorithm, is used.
  
  The L1 or Least Absolute Deviations technique has a long history (it was 
  proposed prior to least squares) but due to the computational complexity, 
  was used very little until recently. Just as least squares minimizes the 
  sum of squares of the residuals, L1 minimizes the sum of absolute values 
  of the residuals. 

  L1 is the maximum likelihood estimate when the errors are distributed as a 
  double exponential (Laplace), and is the logical objective in some situations. 
  One reason for the increased interest in L1 regression is that it is robust 
  with respect to heavy-tailed distributions. It is, however, susceptible to 
  high leverage points, and has asymptotic efficiency of about 64\% at the 
  Gaussian distribution.
}

\author{Fortran code by Barrodale and Roberts.}

\references{
  Barrodale, I. and Roberts, F.D.K. (1973). 
  An Improved Algorithm for Discrete L1 Linear Approximations. 
  \emph{SIAM Journal of Numerical Analysis} \bold{10}, 839-848.
  
  Barrodale, I. and Roberts, F.D.K. (1974). 
  Solution of an Overdetermined System of Equations in the L1 Norm. 
  \emph{Communications of the ACM} \bold{17}, 319-320.
  
  Bloomfield, P. and Steiger, W.L. (1983). 
  \emph{Least Absolute Deviations: Theory, Applications, and Algorithms.}
  Birkhauser, Boston, Mass.
}

\examples{
l1fit(stack.x, stack.loss)
}
\keyword{regression}
\keyword{models}
