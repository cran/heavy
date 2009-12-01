\name{as.OneFormula}
\alias{as.OneFormula}

\title{Combine formulas of a set of objects}
\usage{
as.OneFormula(\dots, omit)
}

\arguments{
 \item{\dots}{objects, or lists of objects, from which a formula can be
   extracted.}
 \item{omit}{an optional character vector with the names of variables to
   be omitted from the returned formula. Defaults to c(".", "pi").}
}
\description{
  The names of all variables used in the formulas extracted from the
  objects defined in \code{\dots} are converted into a single linear
  formula, with the variables names separated by \code{+}. 
}
\value{
  a one-sided linear formula with all variables named in the formulas
  extracted from the objects in \code{\dots}, except the ones listed in
  \code{omit}. 
}
\author{
  This function was extracted from v 1.15 of \code{lme} library by 
  Jose Pinheiro and Douglas Bates.
}

\examples{
as.OneFormula(y ~ x + z, list(~ w, ~ t * sin(2 * pi)))
}
\keyword{models}
