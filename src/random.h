#ifndef HEAVY_RANDOM_H
#define HEAVY_RANDOM_H

#include "matrix.h"

/* multivariate symmetric random generation (to be called by R) */
extern void rand_sphere(double *, int *);
extern void rand_norm(double *, int *, double *, double *);
extern void rand_cauchy(double *, int *, double *, double *);
extern void rand_student(double *, int *, double *, double *, double *);
extern void rand_slash(double *, int *, double *, double *, double *);
extern void rand_contaminated(double *, int *, double *, double *, double *, double *);

/* spherical random generation */
extern void rand_spherical_norm(double *, int, int);
extern void rand_spherical_cauchy(double *, int, int);
extern void rand_spherical_student(double *, double, int, int);
extern void rand_spherical_slash(double *, double, int, int);
extern void rand_spherical_contaminated(double *, double, double, int, int);

/* uniformly distributed random vectors */
extern void rand_unif_sphere(double *, int, int);

/* right truncated Gamma distribution */
extern double rtgamma_right_standard(double, double);
extern double rtgamma_right(double, double, double);

#endif /* HEAVY_RANDOM_H */

