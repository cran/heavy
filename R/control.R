heavy.control <-
function(maxIter = 2000, tolerance = 1e-6, ndraws = 6000, algorithm = c("EM", "NEM"), 
  ncycles = 5)
{
  algorithm <- match.arg(algorithm)
  choice <- switch(algorithm, "EM" = 0, "NEM" = 1)
  if (!choice)
    ncycles <- 1
  list(maxIter = maxIter, tolerance = tolerance, ndraws = ndraws, 
       algorithm = choice, ncycles = ncycles)
}
