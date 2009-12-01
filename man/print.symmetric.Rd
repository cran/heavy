\name{print.symmetric}
\alias{print.symmetric}

\title{Print a symmetric matrix}
\description{
  Print only the lower triangular part of \code{x}.
}

\usage{
print.symmetric(x, digits = 4, \dots)
}
\arguments{
 \item{x}{a symmetric matrix.}
 \item{digits}{minimal number of significant digits.}
 \item{\dots}{further arguments passed to \code{print.default}.}
}

\author{
  The design was inspired by function \code{print.correlation} in v 1.15 of \code{lme} 
  library by Jose Pinheiro and Douglas Bates.
}

\examples{
a <- matrix(c(5,2,2,1), ncol = 2)
print.symmetric(a)
}
\keyword{models}
