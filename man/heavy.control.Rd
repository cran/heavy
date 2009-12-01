\name{heavy.control}
\alias{heavy.control}

\title{Set control parameters}
\description{
  Allows users to set parameters for \code{heavy.lme}.
}
\usage{
heavy.control(maxIter = 8000, tolerance = 1e-06)
}

\arguments{
  \item{maxIter}{ maximum number of iterations. The default is 8000.}
  \item{tolerance}{ the relative tolerance in the iterative algorithm.}
}

\value{
  A list of control arguments to be used in a call to \code{heavy.lme}.

  A call to \code{heavy.control} can be used directly in the control argument of a call to \code{heavy.lme}. 
}

\author{ Felipe Osorio.}

\examples{
ctrl <- heavy.control(maxIter = 100, tol = 1e-04)
data(dental)
heavy.lme(distance ~ age * Sex, random = ~ age, groups = ~ Subject, 
  data = dental, control = ctrl)
}
\keyword{models}
