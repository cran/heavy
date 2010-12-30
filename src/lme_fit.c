#include "lme_fit.h"

void
heavyLme_fit(double *ZX, double *y, double *qraux, int *pdims, int *lengths,
    int *DcLen, double *settings, double *coef, double *theta, double *scale,
    double *ranef, double *Root, double *distances, double *weights,
    double *logLik, double *control)
{   /* fitter for linear mixed-effects models under heavy-tailed errors */
    LMEStruct model;

    model = heavyLme_init(ZX, y, qraux, pdims, lengths, DcLen, settings, coef,
                          theta, scale, ranef, Root, distances, weights, control);
    pre_decomp(model->ZX, model->y, model->qraux, model->dd, model->lengths);
    control[5] = (double) heavyLme_iterate(model);
    *logLik = heavyLme_logLik(model);
    heavyLme_free(model);
}

LMEStruct
heavyLme_init(double *ZX, double *y, double *qraux, int *pdims, int *lengths,
    int *DcLen, double *settings, double *coef, double *theta, double *scale,
    double *ranef, double *Root, double *distances, double *weights,
    double *control)
{   /* constructor for a model object */
    LMEStruct model;
    int qsq;

    model = (LMEStruct) Calloc(1, LME_struct);
    model->ZX = ZX;
    model->y = y;
    model->qraux = qraux;
    model->dims = pdims;
    model->dd = dims(1, pdims);
    model->lengths = setLengths(model->dd, lengths, DcLen);
    model->settings = settings;
    model->family = family_init(settings);
    model->coef = coef;
    model->theta = theta;
    model->scale = scale;
    model->ranef = ranef;
    model->Root = Root;
    model->distances = distances;
    model->weights = weights;
    model->control = control;
    /* some definitions */
    qsq = SQR(model->dd->q);
    model->npar = model->dd->p + qsq + 1;
    model->Delta = (double *) Calloc(qsq, double);
    model->maxIter = (int) control[0];
    model->tolerance = control[1];
    model->ndraws = (int) control[2];
    model->algorithm = (int) control[3];
    model->ncycles = (int) control[4];
    return model;
}

void
heavyLme_free(LMEStruct this)
{   /* destructor for a model object */
    dims_free(this->dd);
    lengths_free(this->lengths);
    family_free(this->family);
    Free(this->Delta);
    Free(this);
}

void
pre_decomp(double *ZX, double *y, double *qraux, DIMS dd, LENGTHS glen)
{   /* return the pre-decomposition for ZXy */
    QRStruct qr;
    int i;

    for (i = 0; i < dd->n; i++) {
        double *mat = ZX + (glen->offsets)[i];
        double *tau = qraux + i * dd->ZXcols;
        qr = QR_decomp(mat, dd->ZXrows, (glen->glen)[i], dd->ZXcols, tau);
        double *rsp = y + (glen->offsets)[i];
        QR_qty(qr, rsp, (glen->glen)[i], (glen->glen)[i], 1);
        QR_free(qr);
    }
}

int
heavyLme_iterate(LMEStruct model)
{   /* */
    int iter, cycle;
    double conv, RSS, newRSS;

    /* initialization */
    RSS = (double) model->dd->N;

    /* main loop */
    for (iter = 1; iter <= model->maxIter; iter++) {
        /* internal EM cycles */
        for (cycle = 1; cycle <= model->ncycles; cycle++) 
            internal_EMcycle(model);
        /* outer E-step */
        outer_Estep(model);
        newRSS  = sum_abs(model->distances, model->dd->n, 1);
        newRSS *= (model->scale)[0];
        
        /* eval convergence */
        conv = fabs((newRSS - RSS) / (newRSS + ETA_CONV));
        if (conv < model->tolerance) /* successful completion */
            return iter;
        RSS = newRSS;
    }
    return (iter - 1);
}

void
internal_EMcycle(LMEStruct model)
{
    internal_Estep(model);
    update_coef(model);
    update_theta(model);
    update_scale(model);
}

void
internal_Estep(LMEStruct model)
{
    DIMS dd = model->dd;
    LENGTHS glen = model->lengths;
    double *Root;
    int i;

    Root = (double *) Calloc(SQR(dd->ZXcols + 1), double);
    relative_precision(model->theta, dd->q, model->scale, model->Delta);
    for (i = 0; i < dd->n; i++) {
        double *R = model->ZX + (glen->offsets)[i];
        double *c = model->y + (glen->offsets)[i];
        append_decomp(R, c, (glen->glen)[i], dd->ZXrows, dd->ZXcols,
                      model->Delta, dd->q, Root, dd->ZXcols + 1);
        double *ranef = model->ranef + i * dd->q;
        random_effects(Root, dd->ZXcols + 1, dd, model->coef, ranef);
        (model->distances)[i] = mahalanobis(Root, dd->ZXcols + 1, dd, model->coef,
                                            model->scale);
        double *Half = model->Root + i * dd->q * dd->q;
        upper_tri(Half, dd->q, Root, dd->ZXcols + 1, dd->q, dd->q);
        zero_mat(Root, dd->ZXcols + 1, dd->ZXcols + 1, dd->ZXcols + 1);
    }
    Free(Root);
}

void
outer_Estep(LMEStruct model)
{
    DIMS dd = model->dd;
    LENGTHS glen = model->lengths;
    int i;

    for (i = 0; i < dd->n; i++)
        (model->weights)[i] = do_weight(model->family, (glen->glen)[i],
                                        (model->distances)[i]);
}

void
update_coef(LMEStruct model)
{   /* compute the fixed effect estimates */
    DIMS dd = model->dd;
    LENGTHS glen = model->lengths;
    double wts, *R, *X, *working;
    int i;

    X = (double *) Calloc(dd->DcRows * dd->p, double);
    working = (double *) Calloc(dd->DcRows, double);
    for (i = 0; i < dd->n; i++) {
        R = (double *) Calloc((glen->DcLen)[i] * dd->ZXcols, double);
        double *ZXy = model->ZX + (glen->offsets)[i];
        double *rsp = model->y + (glen->offsets)[i];
        double *ranef = model->ranef + i * dd->q;
        wts = sqrt((model->weights)[i]);
        upper_tri(R, (glen->DcLen)[i], ZXy, dd->ZXrows, (glen->DcLen)[i], dd->ZXcols);
        working_response(R, rsp, (glen->DcLen)[i], dd, ranef, working + (glen->DcOff)[i]);
        scale_mat(X + (glen->DcOff)[i], dd->DcRows, R + (glen->DcLen)[i] * dd->q,
                  (glen->DcLen)[i], (glen->DcLen)[i], dd->p, wts);
        scale_mat(working + (glen->DcOff)[i], 0, working + (glen->DcOff)[i], 0,
                  (glen->DcLen)[i], 1, wts);
        Free(R);
    }
    lsfit(X, dd->DcRows, dd->DcRows, dd->p, working, dd->DcRows, 1, model->coef);
    Free(X); Free(working);
}

void
update_scale(LMEStruct model)
{   /* update the within-groups scale parameter */
    DIMS dd = model->dd;
    double SSQ;

    SSQ  = dot_product(model->distances, 1, model->weights, 1, dd->n);
    SSQ += dd->n * dd->q;
    (model->scale)[0] *= SSQ;
    (model->scale)[0] /= dd->N + dd->n * dd->q;
}

void
update_theta(LMEStruct model)
{   /* update the scale matrix of random effects */
    DIMS dd = model->dd;
    int i, job = 1, qsq = SQR(dd->q);
    double *accum, *Omega, scale, wts;

    scale = (model->scale)[0];
    accum = (double *) Calloc(qsq, double);
    Omega = (double *) Calloc(qsq, double);
    for (i = 0; i < dd->n; i++) {
        double *Root = model->Root + i * qsq;
        invert_triangular(job, Root, dd->q, dd->q);
        outerprod(Root, dd->q, dd->q, dd->q, Root, dd->q, dd->q, dd->q, Omega);
        scale_mat(Omega, dd->q, Omega, dd->q, dd->q, dd->q, scale);
        wts = (model->weights)[i];
        double *ranef = model->ranef + i * dd->q;
        rank1_update(Omega, dd->q, dd->q, dd->q, wts, ranef, ranef);
        add_mat(accum, dd->q, 1.0, Omega, dd->q, dd->q, dd->q);
        zero_mat(Omega, dd->q, dd->q, dd->q);
    }
    scale_mat(model->theta, dd->q, accum, dd->q, dd->q, dd->q, 1.0 / dd->n);
    Free(accum); Free(Omega);
}

void
relative_precision(double *Psi, int q, double *scale, double *Delta)
{
    int job = 0; /* Delta is lower triangular */

    zero_mat(Delta, q, q, q);
    lower_tri(Delta, q, Psi, q, q, q);
    chol_decomp(job, Delta, q, q);
    invert_triangular(job, Delta, q, q);
    scale_mat(Delta, q, Delta, q, q, q, sqrt(*scale));
}

void
append_decomp(double *R, double *c, int glen, int nrow, int ncol,
    double *Delta, int q, double *Store, int ldStr)
{   /* apply a QR decomposition to the augmented matrix rbind(Rc, Delta),
     * triangular part is returned in Store */
    int arow = glen + q, acol = ncol + 1;
    double *aug, *qraux;
    QRStruct qr;

    aug   = (double *) Calloc(arow * acol, double);
    qraux = (double *) Calloc(acol, double);
    upper_tri(aug, arow, R, nrow, glen, ncol);
    Memcpy(aug + arow * ncol, c, glen);
    lower_tri(aug + glen, arow, Delta, q, q, q);
    qr = QR_decomp(aug, arow, arow, acol, qraux);
    QR_store_R(qr, Store, ldStr);
    QR_free(qr); Free(aug); Free(qraux);
}

void
random_effects(double *Root, int ldRoot, DIMS dd, double *coef, double *ranef)
{   /* compute random effects estimates */
    int job = 1;

    Memcpy(ranef, Root + ldRoot * dd->ZXcols, dd->q);
    gaxpy(ranef, -1.0, Root + ldRoot * dd->q, ldRoot, dd->q, dd->p, coef, 1.0);
    backsolve(job, Root, ldRoot, dd->q, ranef, dd->q, 1);
}

double
mahalanobis(double *Root, int ldRoot, DIMS dd, double *coef, double *scale)
{   /* Mahalanobis distances */
    double ans, *z;

    z = (double *) Calloc(dd->p + 1, double);
    Memcpy(z, Root + ldRoot * dd->ZXcols + dd->q, dd->p + 1);
    gaxpy(z, -1.0, Root + ldRoot * dd->q + dd->q, ldRoot, dd->p, dd->p, coef, 1.0);
    ans = norm_sqr(z, dd->p + 1, 1) / *scale;
    Free(z);
    return ans;
}

void
working_response(double *R, double *c, int DcLen, DIMS dd, double *ranef,
    double *working)
{   /* compute the working response */
    Memcpy(working, c, DcLen);
    gaxpy(working, -1.0, R, DcLen, dd->q, dd->q, ranef, 1.0);
}

double
heavyLme_logLik(LMEStruct model)
{   /* evaluate the log-likelihood function */
    DIMS dd = model->dd;
    int i, qsq = SQR(dd->q);
    double *lengths, accum = 0.0, scale, ans;

    scale = (model->scale)[0];
    relative_precision(model->theta, dd->q, model->scale, model->Delta);
    lengths = (double *) Calloc(dd->n, double);
    for (i = 0; i < dd->n; i++) {
        double *Root = model->Root + i * qsq;
        accum += logAbsDet(Root, dd->q, dd->q);
        lengths[i] = (double) (model->lengths->glen)[i];
    }
    ans  = -.5 * dd->N * log(scale) + dd->n * logAbsDet(model->Delta, dd->q, dd->q);
    ans += accum;
    ans += logLik_kernel(model->family, dd, lengths, model->distances);
    Free(lengths);
    return ans;
}

void
heavyLme_fitted(double *ZX, int *pdims, int *lengths, int *DcLen,
    double *coef, double *ranef, double *conditional, double *marginal)
{
    FITTED ans;

    ans = heavyLme_fitted_init(ZX, pdims, lengths, DcLen, coef, ranef,
                               conditional, marginal);
    heavyLme_fitted_values(ans);
    heavyLme_fitted_free(ans);
}

FITTED
heavyLme_fitted_init(double *ZX, int *pdims, int *lengths, int *DcLen,
    double *coef, double *ranef, double *conditional, double *marginal)
{   /* constructor for a fitted object */
    FITTED ans;

    ans = (FITTED) Calloc(1, FITTED_struct);
    ans->ZX = ZX;
    ans->dd = dims(1, pdims);
    ans->lengths = setLengths(ans->dd, lengths, DcLen);
    ans->coef = coef;
    ans->ranef = ranef;
    ans->conditional = conditional;
    ans->marginal = marginal;
    return(ans);
}

void
heavyLme_fitted_free(FITTED this)
{   /* destructor for a fitted object */
    dims_free(this->dd);
    lengths_free(this->lengths);
    Free(this);
}

void
heavyLme_fitted_values(FITTED object)
{
    DIMS dd = object->dd;
    LENGTHS glen = object->lengths;
    int i;

    /* marginal fitted values */
    gaxpy(object->marginal, 1.0, object->ZX + dd->ZXrows * dd->q, dd->ZXrows,
          dd->ZXrows, dd->p, object->coef, 0.0);
    /* conditional fitted values */
    Memcpy(object->conditional, object->marginal, dd->ZXrows);
    for (i = 0; i < dd->n; i++) {
        double *Z = object->ZX + (glen->offsets)[i];
        double *yFit = object->conditional + (glen->offsets)[i];
        double *ranef = object->ranef + i * dd->q;
        gaxpy(yFit, 1.0, Z, dd->ZXrows, (glen->glen)[i], dd->q, ranef, 1.0);
    }
}

void
heavyLme_acov(double *ZX, int *pdims, int *lengths, int *DcLen, double *settings,
    double *Root, double *scale, double *control, double *acov)
{
    ACOV ans;

    ans = heavyLme_acov_init(ZX, pdims, lengths, DcLen, settings, Root, scale,
                             control, acov);
    heavyLme_acov_coef(ans);
    heavyLme_acov_free(ans);
}

ACOV
heavyLme_acov_init(double *ZX, int *pdims, int *lengths, int *DcLen,
    double *settings, double *Root, double *scale, double *control, double *acov)
{   /* constructor for a covariance object */
    ACOV ans;

    ans = (ACOV) Calloc(1, ACOV_struct);
    ans->ZX = ZX;
    ans->dd = dims(1, pdims);
    ans->lengths = setLengths(ans->dd, lengths, DcLen);
    ans->family = family_init(settings);
    ans->Root = Root;
    ans->scale = scale;
    ans->control = control;
    ans->ndraws = (int) control[2];
    ans->acov = acov;
    return(ans);
}

void
heavyLme_acov_free(ACOV this)
{   /* destructor for a fitted object */
    dims_free(this->dd);
    lengths_free(this->lengths);
    family_free(this->family);
    Free(this);
}

void
heavyLme_acov_coef(ACOV object)
{   /* evaluate the Fisher information matrix */
    DIMS dd = object->dd;
    LENGTHS glen = object->lengths;
    FAMILY family = object->family;
    int i = 0, qsq = SQR(dd->q);
    double factor, *accum, *cross, *prod, *outer, *Z, *R;

    accum = (double *) Calloc(SQR(dd->p), double);
    cross = (double *) Calloc(SQR(dd->p), double);
    prod  = (double *) Calloc(dd->p * dd->q, double);
    outer = (double *) Calloc(SQR(dd->p), double);
    for (i = 0; i < dd->n; i++) {
        R = (double *) Calloc((glen->glen)[i] * dd->ZXcols, double);
        double *ZX = object->ZX + (glen->offsets)[i];
        upper_tri(R, (glen->glen)[i], ZX, dd->ZXrows, (glen->glen)[i], dd->ZXcols);
        crossprod(R + (glen->glen)[i] * dd->q, (glen->glen)[i], (glen->glen)[i], dd->p,
                  R + (glen->glen)[i] * dd->q, (glen->glen)[i], (glen->glen)[i], dd->p,
                  cross);
        Z = (double *) Calloc((glen->glen)[i] * dd->q, double);
        double *Root = object->Root + i * qsq;
        mult_mat(R, (glen->glen)[i], (glen->glen)[i], dd->q, Root, dd->q, dd->q, dd->q, Z);
        crossprod(R + (glen->glen)[i] * dd->q, (glen->glen)[i], dd->q, dd->p,
                  Z, (glen->glen)[i], dd->q, dd->q, prod);
        outerprod(prod, dd->p, dd->p, dd->q, prod, dd->p, dd->p, dd->q, outer);
        add_mat(cross, dd->p, -1.0, outer, dd->p, dd->p, dd->p);
        factor  = acov_scale(family, (glen->glen)[i], object->ndraws);
        factor /= (glen->glen)[i];
        add_mat(accum, dd->p, factor, cross, dd->p, dd->p, dd->p);
        Free(Z); Free(R);
    }
    copy_mat(object->acov, dd->p, accum, dd->p, dd->p, dd->p);
    Free(accum); Free(prod); Free(cross); Free(outer);
}
