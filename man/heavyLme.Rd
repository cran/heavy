\name{heavyLme}
\alias{heavyLme}
\alias{heavyLme.formula}

\title{Linear mixed-effects models under heavy-tailed distributions}
\description{
  This function fits a linear mixed-effects model under heavy-tailed errors
  using the formulation described in Pinheiro et al. (2001).
}
\usage{
heavyLme(fixed, random, groups, data, family = Student(df = 4), 
  subset, na.action, control)
}

\arguments{
  \item{fixed}{ a two-sided linear formula object describing the fixed-effects part of the model, 
  with the response on the left of a ~ operator and the terms, separated by + operators.}
  \item{random}{  a one-sided formula of the form ~x1+...+xn specifying the model for the random effects.}
  \item{groups}{ a one-sided formula for specifying the grouping variable.}
  \item{data}{ an optional data frame containing the variables named in \code{fixed}, \code{random} and \code{group}. 
  By default the variables are taken from the environment from which heavy.lme is called.}
  \item{family}{ a description of the error distribution to be used in the model. 
  By default the Student-t distribution with 4 degrees of freedom is considered.}
  \item{subset}{ an optional expression indicating the subset of the rows of data that should be used in the fit.}
  \item{na.action}{ a function that indicates what should happen when the data contain NAs.}
  \item{control}{ a list of control values for the estimation algorithm to replace the default 
  values returned by the function \code{\link{heavy.control}}.}
}

\value{
  an object of class \code{heavyLme} representing the linear mixed-effects model fit. Generic function print, 
  show the results of the fit.
}

\author{ The design was inspired by version 1.15 of function \code{\link[nlme]{lme}} by Jose Pinheiro and Douglas Bates.}
\references{
  Pinheiro, J.C., Liu, C. and Wu, Y.N. (2001).
  Efficient algorithms for robust estimation in linear mixed-effects models using the multivariate \emph{t} distribution.
  \emph{Journal of Computational and Graphical Statistics} \bold{10}, 249--276.
}

\examples{
data(dental)
heavyLme(distance ~ age * Sex, random = ~ age, groups = ~ Subject, 
  data = dental, family = Student(df = 4))
}
\keyword{models}
