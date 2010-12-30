#ifndef HEAVY_LM_FIT_H
#define HEAVY_LM_FIT_H

#include "base.h"
#include "matrix.h"
#include "family.h"
#include "random.h"

/* structure to hold model results */
typedef struct LM_struct {
    DIMS dd;        /* dimension data info */
    FAMILY family;  /* family data and info */
    double
      *y,           /* responses */
      *x,           /* model matrix */
      *settings,    /* settings */
      *coef,        /* coefficients estimates */
      *scale,       /* scale estimate */
      *fitted,      /* fitted values */
      *residuals,   /* residuals */
      *distances,   /* Mahalanobis distances */
      *weights,     /* weights for heavy tailed distributions */
      *acov,        /* coefficients covariance matrix */
      *control;     /* control settings for estimation algorithm */
    int
      maxIter,      /* maximun number of iterations */
      ndraws;       /* independent draws for Monte Carlo integration */
    double
      tolerance;    /* convergence tolerance */
} LM_struct, *LMStruct;

/* estimation in linear models under heavy tailed distributions */
extern void heavyLm_fit(double *, double *, int *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *);

/* routines for estimation in linear models */
LMStruct heavyLm_init(double *, double *, int *, double *, double *, double *, double *, double *, double *, double *, double *, double *);
void heavyLm_free(LMStruct);
int IRLS(double *y, double *, DIMS, FAMILY, double *, double *, double *, double *, double *, double *, int, double);
void IRLS_increment(double *, double *, DIMS, double *, double *, double *, double *, double *, double *);
void qr_fitted(DIMS, double *, double *, double *, double *, double *);
double heavyLm_logLik(FAMILY, DIMS, double *, double *);
void heavyLm_acov(FAMILY, DIMS, double *, double *, int, double *);

#endif /* HEAVY_LM_FIT_H */
