#ifndef HEAVY_LME_FIT_H
#define HEAVY_LME_FIT_H

#include "base.h"
#include "family.h"
#include "matrix.h"

/* structure to hold model results */
typedef struct MODEL_struct {
    DIMS dd;              /* dimension data info */
    FamilyStruct family;  /* family information */
    double
      *ZXy,         /* model matrix */
      *settings,    /* settings for family object */
      *coef,        /* coefficient estimates */
      *theta,       /* scale parameters estimates */
      *ranef,       /* random effects */
      *distances,   /* mahalanobis distances */
      *weights,     /* weights for heavy tailed distributions */
      *control,     /* control settings for estimation algorithm */
      *parameters,  /* parameter estimates */
      *newpars,     /* used in the estimation algorithm */
      *Delta,       /* relative precision factor */
      *OmegaHalf,   /* triangular factors */
      *working;     /* working responses */
    int
      npar,         /* total length of the parameter */
      maxiter;      /* maximun number of iterations */
    double tol;     /* convergence tolerance */
} MODEL_struct, *MODEL;

/* structure to hold residuals */
typedef struct RESID_struct {
    DIMS dd;              /* dimension data info */
    double
      *ZXy,         /* model matrix */
      *coef,        /* coefficient estimates */
      *ranef,       /* random effects */
      *conditional, /* conditional residuals */
      *marginal;    /* marginal residuals */
} RESID_struct, *RESID;

/* estimation in lme under heavy tailed distributions (to be called by R) */
extern void heavy_lme_fit(double *, int *, int *, double *, double *, double *, double *, double *, double *, double *, double *, double *);
extern void heavy_lme_resid(double *, int *, int *, double *, double *, double *, double *);

/* model initialization */
MODEL heavy_lme_init(double *, int *, int *, double *, double *, double *, double *, double *, double *, double *, double *);
void heavy_lme_free(MODEL);

/* routines for dealing with dims objects */
DIMS dims(int *, int *);
void free_dims(DIMS);

/* routines for initial manipulation */
void model_matrix(double *, DIMS);
void pre_decomp(double *, DIMS);

/* misc */
void relative_precision_factor(double *, int, double, double *);
double lme_logLik(MODEL);

/* routines used in the iterative procedure */
int lme_iterate(MODEL);
double conv_criterion(double *, double *, int);
void lme_ECMsteps(MODEL);
void expectation_step(MODEL);
void update_coef(MODEL);
void update_theta(MODEL);
void update_sigma2(MODEL);

/* routines used in EM steps */
void random_effects(double *, int, DIMS, double *, double *);
double mahalanobis(double *, int, DIMS, double *, double *);
void working_response(double *, int, DIMS, double *, double *);

/* routines for computation of residuals */
RESID lme_resid_init(double *, int *, int *, double *, double *, double *, double *);
void lme_resid_free(RESID);
void lme_residuals(RESID);

#endif /* HEAVY_LME_FIT_H */
