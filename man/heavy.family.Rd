\name{heavy.family}
\alias{heavy.family}
\alias{normal}
\alias{Student}
\alias{slash}
\alias{contaminated}

\title{Family objects for heavy-tailed distributions}
\description{
  Provide a convenient way to specify the details of the models used by functions such as \code{heavy.lme}.
}
\usage{
normal()
Student(df = 4)
slash(df = 2)
contaminated(epsilon = 0.05, vif = 0.1)
}

\arguments{
  \item{df}{ degrees of freedom for Student-t and slash distributions.}
  \item{epsilon, vif}{ outliers contamination percentage and variance inflation factor.}
}
\details{
  \code{family.heavy} is a generic function to create info about some heavy-tailed distributions which 
  is passed to the estimation algorithm.
}
\author{ The design was inspired by S functions to handle \code{glm} models.}

\seealso{ \code{\link{family}} }
\examples{
CauchyFmly <- Student(df = 1)
}
\keyword{models}
