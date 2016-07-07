#include "lm_fit.h"

/* static functions.. */

/* functions to deal with dims objects */
static DIMS dims(int *);
static void dims_free(DIMS);

/* routines for estimation in linear models */
static LM lm_init(double *, double *, int *, double *, double *, double *, double *, double *, double *, double *, double *, double *);
static void lm_free(LM);
static int IRLS(double *, double *, DIMS, FAMILY, double *, double *, double *, double *, double *, double *, int, double, int);
static void IRLS_increment(double *, double *, DIMS, double *, double *, double *, double *, double *);

/* routines for evaluation of marginal log-likelihood and Fisher information matrix */
static double lm_logLik(FAMILY, DIMS, double *, double *);
static void lm_acov(FAMILY, DIMS, double *, double *, int, double *);

/* ..end declarations */

void
lm_fit(double *y, double *x, int *pdims, double *settings, double *coef,
  double *scale, double *fitted, double *resid, double *distances, double *weights,
  double *logLik, double *acov, double *control)
{ /* fitter for univariate linear regression under heavy-tailed errors */
  LM model;

  model = lm_init(y, x, pdims, settings, coef, scale, fitted, resid, distances,
                  weights, acov, control);
  control[4] = (double) IRLS(model->y, model->x, model->dm, model->family,
                             model->coef, model->scale, model->resid, model->fitted,
                             model->distances, model->weights, model->maxIter,
                             model->tolerance, model->fixShape);
  *logLik = lm_logLik(model->family, model->dm, model->distances, model->scale);
  lm_acov(model->family, model->dm, model->x, model->scale, model->ndraws, model->acov);
  lm_free(model);
}

static DIMS
dims(int *pdims)
{ /* dims object for linear models */
  DIMS ans;

  ans = (DIMS) Calloc(1, DIMS_struct);
  ans->N = (int) pdims[0];
  ans->n = ans->N;
  ans->p = (int) pdims[1];
  return ans;
}

static void
dims_free(DIMS this)
{ /* destructor for a dims object */
  Free(this);
}

static LM
lm_init(double *y, double *x, int *pdims, double *settings, double *coef,
  double *scale, double *fitted, double *resid, double *distances, double *weights,
  double *acov, double *control)
{ /* constructor for a linear model object */
  LM model;

  model = (LM) Calloc(1, LM_struct);
  model->dm = dims(pdims);
  model->settings = settings;
  model->family = family_init(settings);
  model->y = y;
  model->x = x;
  model->coef = coef;
  model->scale = scale;
  model->fitted = fitted;
  model->resid = resid;
  model->distances = distances;
  model->weights = weights;
  model->acov = acov;
  model->control = control;
  model->maxIter = (int) control[0];
  model->tolerance = control[1];
  model->fixShape = (int) control[2];
  model->ndraws = (int) control[3];
  return model;
}

static void
lm_free(LM this)
{ /* destructor for a model object */
  dims_free(this->dm);
  family_free(this->family);
  Free(this);
}

static int
IRLS(double *y, double *x, DIMS dm, FAMILY family, double *coef, double *scale,
  double *resid, double *fitted, double *distances, double *weights, int maxit,
  double tolerance, int fixShape)
{ /* iteratively reweighted LS algorithm */
  int i, iter = 0, rdf = dm->n - dm->p;
  double conv, RSS, tol = R_pow(tolerance, 2./3.), newRSS, *lengths, *working;

  /* initialization */
  lengths = (double *) Calloc(dm->n, double);
  working = (double *) Calloc(dm->n, double);
  for (i = 0; i < dm->n; i++)
    lengths[i] = 1.;
  RSS = norm_sqr(resid, 1, dm->n);

  /* main loop */
  repeat {
    /* E-step */
    for (i = 0; i < dm->n; i++) {
      distances[i] = SQR(resid[i]) / *scale;
      weights[i] = do_weight(family, 1., distances[i]);
    }
    /* M-step */
    IRLS_increment(y, x, dm, coef, resid, fitted, weights, working);
    newRSS = norm_sqr(working + dm->p, 1, rdf);
    *scale = newRSS / dm->n;
    if (!fixShape)
      update_mixture(family, dm, distances, lengths, weights, tol);

    iter++;

    /* eval convergence */
    conv = fabs((newRSS - RSS) / (newRSS + ABSTOL));
    if (conv < tolerance)
      break; /* successful completion */
    if (iter >= maxit)
      break; /* maximum number of iterations exceeded */
    RSS = newRSS;
  }
  Free(lengths); Free(working);
  return iter;
}

static void
IRLS_increment(double *y, double *x, DIMS dm, double *coef, double *resid,
  double *fitted, double *weights, double *working)
{ /* increment for direction search in IRLS */
  int i, j, job = 1, info = 0;
  double stepsize = 1., wts, *incr, *qraux, *z;
  QRStruct qr;

  /* initialization */
  incr  = (double *) Calloc(dm->p, double);
  qraux = (double *) Calloc(dm->p, double);
  z     = (double *) Calloc(dm->n * dm->p, double);

  /* transformed model matrix and working residuals */
  for (i = 0; i < dm->n; i++) {
    wts = sqrt(weights[i]);
    working[i] = wts * resid[i];
    for (j = 0; j < dm->p; j++)
      z[i + j * dm->n] = wts * x[i + j * dm->n];
  }

  /* solve the transformed LS-problem */
  qr = QR_decomp(z, dm->n, dm->n, dm->p, qraux, &info);
  if (info)
    error("DGEQR2 in IRLS_increment gave code %d", info);
  QR_qty(qr, working, dm->n, dm->n, 1);
  Memcpy(incr, working, dm->p);
  backsolve(z, dm->n, dm->p, incr, dm->p, 1, job, &info);
  if (info)
    error("DTRTRS in IRLS_increment gave code %d", info);

  /* update coefficients */
  ax_plus_y(stepsize, incr, 1, coef, 1, dm->p);

  /* compute fitted values */
  for (i = 0; i < dm->n; i++)
    fitted[i] = 0.0;
  triangle_mult_vec(fitted, z, dm->n, dm->p, coef, job);
  QR_qy(qr, fitted, dm->n, dm->n, 1);

  /* fitted values and residuals in original scale */
  for (i = 0; i < dm->n; i++) {
    wts = sqrt(weights[i]);
    fitted[i] /= wts;
    resid[i] = y[i] - fitted[i];
  }

  QR_free(qr); Free(incr); Free(qraux); Free(z);
}

static double
lm_logLik(FAMILY family, DIMS dm, double *distances, double *scale)
{ /* evaluate the log-likelihood function for linear models */
  double krnl, *lengths;
  int i;

  lengths = (double *) Calloc(dm->n, double);
  for (i = 0; i < dm->n; i++)
    lengths[i] = 1.;
  krnl = logLik_kernel(family, dm, lengths, distances);
  Free(lengths);
  return (krnl - .5 * dm->n * log(*scale));
}

static void
lm_acov(FAMILY family, DIMS dm, double *x, double *scale, int ndraws, double *acov)
{ /* evaluate the Fisher information matrix */
  int info = 0, job = 1;
  double factor = 1., *qraux, *R;
  QRStruct qr;

  /* initialization */
  qraux = (double *) Calloc(dm->p, double);
  R     = (double *) Calloc(SQR(dm->p), double);

  /* unscaled Fisher information matrix */
  qr = QR_decomp(x, dm->n, dm->n, dm->p, qraux, &info);
  if (info)
    error("DGEQR2 in lm_acov gave code %d", info);
  QR_store_R(qr, R, dm->p);
  invert_triangular(R, dm->p, dm->p, job, &info);
  if (info)
    error("DTRTRI in lm_acov gave code %d", info);
  outerprod(acov, R, dm->p, dm->p, dm->p, R, dm->p, dm->p, dm->p);

  Free(qraux); QR_free(qr);

  /* scaling */
  factor = *scale / acov_scale(family, 1., ndraws);
  scale_mat(acov, dm->p, acov, dm->p, dm->p, dm->p, factor);
  Free(R);
}
