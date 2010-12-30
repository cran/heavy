\name{ereturns}
\docType{data}
\alias{ereturns}

\title{Excess returns for Martin Marietta and American Can companies}
\description{
  Data from the Martin Marietta and American Can companies collected over a period 
  of 5 years on a monthly basis
}
\usage{data(ereturns)}
\format{
  A data frame with 60 observations on the following 4 variables.
  \describe{
    \item{Date}{the month the data was collected}
    \item{am.can}{a numeric vector}
    \item{m.marietta}{excess returns from the Martin Marietta company}
    \item{CRSP}{an index for the excess rate returns for the New York stock exchange}
  }
}
\source{
  Butler, R.J., McDonald, J.B., Nelson, R.D., and White, S.B. (1990).
  Robust and partially adaptive estimation of regression models.
  \emph{Review of Economic Statistics} \bold{72}, 321-327.  
}
\keyword{datasets}