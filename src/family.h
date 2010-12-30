#ifndef HEAVY_FAMILY_H
#define HEAVY_FAMILY_H

#include "base.h"
#include "random.h"

/* functions for dealing with 'family' objects */
extern FAMILY family_init(double *);
extern void family_free(FAMILY);

/* routines for computation of weights */
double weight_normal();
double weight_cauchy(double, double);
double weight_student(double, double, double);
double weight_slash(double, double, double);
double weight_contaminated(double, double, double, double);
extern double do_weight(FAMILY, double, double);

/*  functions for evaluation of the log-likelihood */
double logLik_normal(DIMS, double *);
double logLik_cauchy(DIMS, double *, double *);
double logLik_student(DIMS, double *, double, double *);
double logLik_slash(DIMS, double *, double, double *);
double logLik_contaminated(DIMS, double *, double, double, double *);
double logLik_kernel(FAMILY, DIMS, double *, double *);

/* scale factor for the Fisher information matrix */
double acov_scale_normal();
double acov_scale_cauchy(double);
double acov_scale_student(double, double);
double acov_scale_slash(double, double, int);
double acov_scale_contaminated(double, double, double, int);
double acov_scale(FAMILY, double, int);

#endif /* HEAVY_FAMILY_H */
