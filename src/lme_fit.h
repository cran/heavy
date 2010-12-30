#ifndef HEAVY_LME_FIT_H
#define HEAVY_LME_FIT_H

#include "base.h"
#include "family.h"
#include "matrix.h"
#include "random.h"

/* structure to hold estimation results */
typedef struct LME_struct {
    DIMS dd;            /* dimension data info */
    LENGTHS lengths;    /* lenghts and offsets object */
    FAMILY family;      /* family information */
    double
      *ZX,          /* model matrix */
      *y,           /* observed responses */
      *qraux,       /* auxiliar info for pre-decomposition */
      *settings,    /* settings for family object */
      *coef,        /* coefficients estimates */
      *theta,       /* scale parameters estimates */
      *scale,       /* scale estimate */
      *ranef,       /* random effects */
      *distances,   /* mahalanobis distances */
      *weights,     /* weights for heavy tailed distributions */
      *control,     /* control settings for estimation algorithm */
      *Delta,       /* relative precision factor */
      *Root;        /* triangular factors */
    int
      *dims,        /* dimensions passed from R */
      npar,         /* total length of the mixture parameters */
      maxIter,      /* maximun number of iterations */
      ndraws,       /* independent draws for Monte Carlo integration */
      algorithm,    /* logical flag, EM = 0, nested EM = 1 */
      ncycles;      /* number of cycles for the nested EM algorithm */
    double
      tolerance;    /* convergence tolerance */
} LME_struct, *LMEStruct;

/* structure to hold fitted values */
typedef struct FITTED_struct {
    DIMS dd;            /* dimension data info */
    LENGTHS lengths;    /* lenghts and offsets object */
    double
      *ZX,          /* model matrix */
      *coef,        /* coefficient estimates */
      *ranef,       /* random effects */
      *conditional, /* conditional fitted values */
      *marginal;    /* marginal fitted values */
} FITTED_struct, *FITTED;

/* structure to hold the covariance of coefficients */
typedef struct ACOV_struct {
    DIMS dd;            /* dimension data info */
    LENGTHS lengths;    /* lenghts and offsets object */
    FAMILY family;      /* family information */
    double
      *ZX,          /* model matrix */
      *Root,        /* triangular factors */
      *scale,       /* scale estimate */
      *control,     /* control settings */
      *acov;        /* coefficients covariance matrix */
    int
      ndraws;       /* independent draws for Monte Carlo integration */
} ACOV_struct, *ACOV;

/* estimation in lme under heavy tailed distributions (to be called by R) */
extern void heavyLme_fit(double *, double *, double *, int *, int *, int *, double *, double *, double *, double *, double *, double *, double *, double *, double *, double *);
extern void heavyLme_fitted(double *, int *, int *, int *, double *, double *, double *, double *);
extern void heavyLme_acov(double *, int *, int *, int *, double *, double *, double *, double *, double *);

/* initialization and pre-decomposition */
LMEStruct heavyLme_init(double *, double *, double *, int *, int *, int *, double *, double *, double *, double *, double *, double *, double *, double *, double *);
void heavyLme_free(LMEStruct);
void pre_decomp(double *, double *, double *, DIMS, LENGTHS);

/* routines used in the iterative procedure */
int heavyLme_iterate(LMEStruct);
void internal_EMcycle(LMEStruct);
void internal_Estep(LMEStruct);
void outer_Estep(LMEStruct);

/* routines used in (internal) EM cycles */
void append_decomp(double *, double *, int, int, int, double *, int, double *, int);
void random_effects(double *, int, DIMS, double *, double *);
double mahalanobis(double *, int, DIMS, double *, double *);
void working_response(double *, double *, int, DIMS, double *, double *);
void update_coef(LMEStruct);
void update_scale(LMEStruct);
void update_theta(LMEStruct);

/* relative precision factor and evaluation of marginal log-likelihood */
void relative_precision(double *, int, double *, double *);
double heavyLme_logLik(LMEStruct);

/* Fisher information matrix for the coefficients */
ACOV heavyLme_acov_init(double *, int *, int *, int *, double *, double *, double *, double *, double *);
void heavyLme_acov_free(ACOV);
void heavyLme_acov_coef(ACOV);

/* routines for computation of fitted values */
FITTED heavyLme_fitted_init(double *, int *, int *, int *, double *, double *, double *, double *);
void heavyLme_fitted_free(FITTED);
void heavyLme_fitted_values(FITTED);

#endif /* HEAVY_LME_FIT_H */
