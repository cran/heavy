#ifndef HEAVY_SPECFUN_H
#define HEAVY_SPECFUN_H

#include "base.h"

/* normal approximation of the incomplete gamma integral */
extern double pgamma_asymp(double, double, double);

/* derivative of the incomplete gamma integral */
extern double pgamma_derivative(double, double, double);

#endif /* HEAVY_SPECFUN_H */
