print.symmetric <-
function(x, digits = 4, ...)
{
  ll <- lower.tri(x, diag = TRUE)
  x[ll] <- format(x[ll], ...)
  x[!ll] <- ""
  print(x, ..., quote = F)
  invisible(x)
}
