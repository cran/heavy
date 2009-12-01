as.OneFormula <-
function(..., omit = c(".", "pi"))
{
  func <- function(x) { # should make all.vars generic
    if (is.list(x)) {
      return(unlist(lapply(x, all.vars)))
    } 
    all.vars(x)
  }
  names <- unique(unlist(lapply(list(...), func)))
  names <- names[is.na(match(names, omit))]
  if(length(names)) {
    eval(parse(text = paste("~", paste(names, collapse = "+")))[[1]])
  } else NULL
}

print.symmetric <-
function(x, digits = 4, ...)
{
  ll <- lower.tri(x, diag = TRUE)
  x[ll] <- format(x[ll], ...)
  x[!ll] <- ""
  print(x, ..., quote = F)
  invisible(x)
}
