#include "lm_fit.h"

void
heavyLm_fit(double *y, double *x, int *pdims, double *settings, double *coef,
    double *scale, double *fitted, double *residuals, double *distances,
    double *weights, double *logLik, double *acov, double *control)
{   /* fitter for linear models under heavy-tailed errors */
    LMStruct model;

    model = heavyLm_init(y, x, pdims, settings, coef, scale, fitted, residuals,
                         distances, weights, acov, control);
    control[3] = (double) IRLS(model->y, model->x, model->dd, model->family,
                               model->coef, model->scale, model->fitted,
                               model->residuals, model->distances, model->weights,
                               model->maxIter, model->tolerance);
    *logLik = heavyLm_logLik(model->family, model->dd, model->distances, model->scale);
    heavyLm_acov(model->family, model->dd, model->x, model->scale, model->ndraws,
                 model->acov);
    heavyLm_free(model);
}

LMStruct
heavyLm_init(double *y, double *x, int *pdims, double *settings, double *coef,
    double *scale, double *fitted, double *residuals, double *distances,
    double *weights, double *acov, double *control)
{   /* constructor for a linear model object */
    LMStruct model;

    model = (LMStruct) Calloc(1, LM_struct);
    model->dd = dims(0, pdims);
    model->settings = settings;
    model->family = family_init(settings);
    model->y = y;
    model->x = x;
    model->coef = coef;
    model->scale = scale;
    model->fitted = fitted;
    model->residuals = residuals;
    model->distances = distances;
    model->weights = weights;
    model->acov = acov;
    model->control = control;
    model->maxIter = (int) control[0];
    model->tolerance = control[1];
    model->ndraws = (int) control[2];
    return model;
}

void
heavyLm_free(LMStruct this)
{   /* destructor for a model object */
    dims_free(this->dd);
    family_free(this->family);
    Free(this);
}

int
IRLS(double *y, double *x, DIMS dd, FAMILY family, double *coef, double *scale,
    double *fitted, double *residuals, double *distances, double *weights,
    int maxIter, double tolerance)
{   /* iteratively reweighted LS algorithm */
    int i, iter, rdf = dd->n - dd->p;
    double conv, RSS, newRSS, *incr, *working;

    /* initialization */
    incr    = (double *) Calloc(dd->p, double);
    working = (double *) Calloc(dd->n, double);
    RSS = norm_sqr(residuals, dd->n, 1);

    /* main loop */
    for (iter = 1; iter <= maxIter; iter++) {
        /* E-step */
        for (i = 0; i < dd->n; i++) {
            distances[i] = SQR(residuals[i]) / *scale;
            weights[i] = do_weight(family, 1.0, distances[i]);
        }
        /* M-step */
        IRLS_increment(y, x, dd, fitted, residuals, weights, coef, incr, working);
        newRSS = norm_sqr(working + dd->p, rdf, 1);
        *scale = newRSS / dd->n;

        /* eval convergence */
        conv = fabs((newRSS - RSS) / (newRSS + ETA_CONV));
        if (conv < tolerance) { /* successful completion */
            Free(incr); Free(working);
            return iter;
        }
        RSS = newRSS;
    }
    Free(incr); Free(working);
    return (iter - 1);
}

void
IRLS_increment(double *y, double *x, DIMS dd, double *fitted, double *residuals,
    double *weights, double *coef, double *incr, double *working)
{   /* increment for direction search in IRLS */
    int i, j, info = 0, one = 1;
    char *uplo = "U", *diag = "N", *side = "L", *notrans = "N", *trans = "T";
    double stepsize = 1.0, wts, *z, *qraux, *work;

    /* initialization */
    z     = (double *) Calloc(dd->n * dd->p, double);
    qraux = (double *) Calloc(dd->p, double);
    work  = (double *) Calloc(dd->p, double);

    /* transformed model matrix and working residuals */
    for (i = 0; i < dd->n; i++) {
        wts = sqrt(weights[i]);
        working[i] = wts * residuals[i];
        for (j = 0; j < dd->p; j++)
            z[i + j * dd->n] = wts * x[i + j * dd->n];
    }

    /* solve the transformed LS-problem */
    F77_CALL(dgeqrf)(&(dd->n), &(dd->p), z, &(dd->n), qraux, work, &(dd->p),
                     &info);
    if (info)
        error("DGEQRF in IRLS_increment gave code %d", info);
    F77_CALL(dormqr)(side, trans, &(dd->n), &one, &(dd->p), z, &(dd->n), qraux,
                     working, &(dd->n), work, &(dd->p), &info);
    if (info)
        error("DORMQR in IRLS_increment gave code %d", info);
    Memcpy(incr, working, dd->p);
    F77_CALL(dtrtrs)(uplo, notrans, diag, &(dd->p), &one, z, &(dd->n), incr,
                     &(dd->p), &info);
    if (info)
        error("DTRTRS in IRLS_increment gave code %d", info);
    /* update coefficients */
    F77_CALL(daxpy)(&(dd->p), &stepsize, incr, &one, coef, &one);

    /* fitted values and residuals */
    qr_fitted(dd, z, coef, fitted, qraux, work);
    for (i = 0; i < dd->n; i++) {
        wts = sqrt(weights[i]);
        fitted[i] /= wts;
        residuals[i] = y[i] - fitted[i];
    }
    Free(z); Free(qraux); Free(work);
}

void
qr_fitted(DIMS dd, double *z, double *coef, double *fitted, double *qraux,
    double *work)
{   /* compute the fitted values */
    int i, info = 0, one = 1;
    char *uplo = "U", *diag = "N", *side = "L", *notrans = "N";

    for (i = 0; i < dd->n; i++) 
        fitted[i] = 0.0;
    Memcpy(fitted, coef, dd->p);
    F77_CALL(dtrmv)(uplo, notrans, diag, &(dd->p), z, &(dd->n), fitted, &one);
    F77_CALL(dormqr)(side, notrans, &(dd->n), &one, &(dd->p), z, &(dd->n),
                     qraux, fitted, &(dd->n), work, &(dd->p), &info);
    if (info)
        error("DORMQR in qr_fitted gave code %d", info);
}

double
heavyLm_logLik(FAMILY family, DIMS dd, double *distances, double *scale)
{   /* evaluate the log-likelihood function for linear models */
    double ans, *lengths;
    int i;

    lengths = (double *) Calloc(dd->n, double);
    for (i = 0; i < dd->n; i++)
        lengths[i] = 1.0;
    ans = -0.5 * dd->n * log(*scale);
    ans += logLik_kernel(family, dd, lengths, distances);
    Free(lengths);
    return ans;
}

void
heavyLm_acov(FAMILY family, DIMS dd, double *x, double *scale, int ndraws,
    double *acov)
{   /* evaluate the Fisher information matrix */
    int job = 1;
    double factor, *qraux, *R;
    QRStruct qr;

    /* initialization */
    qraux = (double *) Calloc(dd->p, double);
    R     = (double *) Calloc(dd->p * dd->p, double);

    /* unscaled Fisher information matrix */
    qr = QR_decomp(x, dd->n, dd->n, dd->p, qraux);
    QR_store_R(qr, R, dd->p);
    invert_triangular(job, R, dd->p, dd->p);
    outerprod(R, dd->p, dd->p, dd->p, R, dd->p, dd->p, dd->p, acov);
    Free(qraux); QR_free(qr);

    /* scaling */
    factor = *scale / acov_scale(family, 1.0, ndraws);
    scale_mat(acov, dd->p, acov, dd->p, dd->p, dd->p, factor);
    Free(R);
}
