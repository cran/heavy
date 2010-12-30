\name{heavy.control}
\alias{heavy.control}

\title{Set control parameters}
\description{
  Allows users to set parameters for \code{\link{heavyLm}} and \code{\link{heavyLme}}.
}
\usage{
heavy.control(maxIter = 2000, tolerance = 1e-6, ndraws = 6000, 
  algorithm = c("EM", "NEM"), ncycles = 5)
}

\arguments{
  \item{maxIter}{ maximum number of iterations. The default is 2000.}
  \item{tolerance}{ the relative tolerance in the iterative algorithm.}
  \item{ndraws}{ independent draws for Monte Carlo integration. The default is 6000.}
  \item{algorithm}{ character string specifying the algorithm to use. The default 
  algorithm is an EM algorithm. Other possible value is "NEM" for the nested EM algorithm.}
  \item{ncycles}{ number of cycles used in the nested EM algorithm, not used if \code{algorithm = "EM"}. 
  The default is 5.}
}

\value{
  A list of control arguments to be used in a call to \code{\link{heavyLm}} or \code{\link{heavyLme}}.

  A call to \code{heavy.control} can be used directly in the control argument 
  of a call to \code{\link{heavyLm}} or \code{\link{heavyLme}}. 
}

\author{ Felipe Osorio.}

\examples{
ctrl <- heavy.control(maxIter = 100, tol = 1e-04, ndraws = 8000)
data(dental)
heavyLme(distance ~ age * Sex, random = ~ age, groups = ~ Subject, 
  data = dental, control = ctrl)
}
\keyword{models}
