#include "lme_fit.h"

MODEL
heavy_lme_init(double *ZXy, int *pdims, int *lengths, double *settings,
    double *coef, double *theta, double *ranef, double *OmegaHalf,
    double *distances, double *weights, double *control)
{   /* constructor for a model object */
    MODEL model;
    int qq;

    model = (MODEL) Calloc(1, MODEL_struct);
    model->ZXy = ZXy;
    model->dd = dims(pdims, lengths);
    model->settings = settings;
    model->family = init_family(settings);
    model->coef = coef;
    model->theta = theta;
    model->ranef = ranef;
    model->OmegaHalf = OmegaHalf;
    model->distances = distances;
    model->weights = weights;
    model->control = control;
    /* some definitions */
    qq = model->dd->q * model->dd->q;
    model->npar = model->dd->p + qq + 1;
    model->parameters = (double *) Calloc(model->npar, double);
    model->newpars    = (double *) Calloc(model->npar, double);
    model->Delta = (double *) Calloc(qq, double);
    model->working = (double *) Calloc(model->dd->ZXrows, double);
    model->maxiter = (int) control[0];
    model->tol = control[1];
    /* initial computations */
    model_matrix(model->ZXy, model->dd);
    Memcpy(model->parameters, coef, model->dd->p);
    Memcpy(model->parameters + model->dd->p, theta, qq + 1);
    return(model);
}

void
heavy_lme_free(MODEL this)
{   /* destructor for a model object */
    free_dims(this->dd);
    free_family(this->family);
    Free(this->parameters);
    Free(this->newpars);
    Free(this->Delta);
    Free(this->working);
    Free(this);
}

DIMS
dims(int *pdims, int *lengths)
{   /* constructor for a dims object */
    DIMS ans;
    int i, accum1 = 0, accum2 = 0;

    ans = (DIMS) Calloc(1, DIMS_struct);
    ans->n = (int) pdims[0];
    ans->q = (int) pdims[1];
    ans->p = (int) pdims[2];
    ans->N = (int) pdims[3];
    ans->ZXrows = (int) pdims[4];
    ans->ZXcols = (int) pdims[5];
    ans->lengths = lengths;
    ans->offsets = (int *) Calloc(ans->n, int);
    ans->ZXlen = (int *) Calloc(ans->n, int);
    ans->ZXoff = (int *) Calloc(ans->n, int);
    (ans->ZXlen)[0] = (ans->lengths)[0] * (ans->ZXcols + 1);
    for (i = 1; i < ans->n; i++) {
        (ans->ZXlen)[i] = (ans->lengths)[i] * (ans->ZXcols + 1);
        accum1 += (ans->lengths)[i-1];
        (ans->offsets)[i] = accum1;
        accum2 += (ans->ZXlen)[i-1];
        (ans->ZXoff)[i] = accum2;
    }
    return ans;
}

void
free_dims(DIMS this)
{   /* destructor for a dims object */
    Free(this->offsets);
    Free(this->ZXlen);
    Free(this->ZXoff);
    Free(this);
}

void
model_matrix(double *ZXy, DIMS dd)
{   /* re-arrange the ZXy matrix */
    int i;
    double *tmp;

    tmp = (double *) Calloc(dd->ZXrows * (dd->ZXcols + 1), double);
    for (i = 0; i < dd->n; i++) {
        copy_mat(tmp + (dd->ZXoff)[i], (dd->lengths)[i], ZXy + (dd->offsets)[i],
                 dd->ZXrows, (dd->lengths)[i], dd->ZXcols + 1);
    }
    Memcpy(ZXy, tmp, dd->ZXrows * (dd->ZXcols + 1));
    Free(tmp);
}

void
pre_decomp(double *ZXy, DIMS dd)
{   /* return the pre-decomposition for ZXy */
    QRStruct qr;
    int i;

    for (i = 0; i < dd->n; i++) {
        double *mat = ZXy + (dd->ZXoff)[i];
        qr = QR_decomp(mat, (dd->lengths)[i], (dd->lengths)[i], dd->ZXcols);
        double *rsp = mat + (dd->lengths)[i] * dd->ZXcols;
        QR_qty(qr, rsp, (dd->lengths)[i], 1);
        QR_free(qr);
        /* FIXME: required for computations on pre-decomposed ZXy */
        zero_lower(mat, (dd->lengths)[i], (dd->lengths)[i], dd->ZXcols);
    }
}

void
relative_precision_factor(double *theta, int q, double sigma2, double *Delta)
{   /* compute the relative precision factor */
    int job = 0; /* Delta is lower triangular */
    double inv = 1.0 / sigma2;

    scale_lower(Delta, q, inv, theta, q, q, q);
    chol_decomp(job, Delta, q, q);
    invert_lower(Delta, q, q);
}

void
heavy_lme_fit(double *ZXy, int *pdims, int *lengths, double *settings,
    double *coef, double *theta, double *ranef, double *OmegaHalf,
    double *distances, double *weights, double *logLik, double *control)
{   /* estimation in the linear mixed model using an ECM algorithm */
    MODEL model;

    model = heavy_lme_init(ZXy, pdims, lengths, settings, coef, theta, ranef,
                           OmegaHalf, distances, weights, control);
    pre_decomp(model->ZXy, model->dd);
    control[2] = (double) lme_iterate(model);
    *logLik = lme_logLik(model);
    heavy_lme_free(model);
}

int
lme_iterate(MODEL model)
{
    int iter, qq = model->dd->q * model->dd->q;
    double crit;

    (model->control)[3] = 0; /* ok */
    for (iter = 1; iter <= model->maxiter; iter++) {
        lme_ECMsteps(model);
        Memcpy(model->newpars, model->coef, model->dd->p);
        Memcpy(model->newpars + model->dd->p, model->theta, qq + 1);
        crit = conv_criterion(model->parameters, model->newpars, model->npar);
        /* cleanup */
        zero_mat(model->Delta, model->dd->q, model->dd->q, model->dd->q);
        zero_mat(model->working, 0, model->dd->ZXrows, 1);
        if (crit < model->tol) return(iter); /* successful completion */
    }
    (model->control)[3] = 1; /* maximum number of iterations exceeded */
    return (iter - 1);
}

double
conv_criterion(double *old, double *newpars, int npar)
{   /* evaluate the convergence criterion, then copy newpars to old */
    int i;
    double conv, ans = -1.0;

    for (i = 0; i < npar; i++) {
        conv = (old[i] - newpars[i]) / ((newpars[i] == 0.0) ? 1.0 : newpars[i]);
        ans = MAX(ans, fabs(conv));
    }
    Memcpy(old, newpars, npar);
    return ans;
}

void
lme_ECMsteps(MODEL model)
{
    expectation_step(model);
    update_coef(model);
    update_theta(model);
    update_sigma2(model);
}

void
expectation_step(MODEL model)
{
    DIMS dd = model->dd;
    int i, nrow, ncol = dd->ZXcols + 1;
    double *G, sigma2;

    sigma2 = (model->theta)[dd->q * dd->q];
    relative_precision_factor(model->theta, dd->q, sigma2, model->Delta);
    G = (double *) Calloc(ncol * ncol, double);
    for (i = 0; i < dd->n; i++) {
        double *Rc = model->ZXy + (dd->ZXoff)[i];
        nrow = (dd->lengths)[i];
        append_decomp(Rc, nrow, nrow, ncol, model->Delta, dd->q, G, ncol);
        double *OmegaHalf = model->OmegaHalf + i * dd->q * dd->q;
        copy_mat(OmegaHalf, dd->q, G, ncol, dd->q, dd->q);
        double *ranef = model->ranef + i * dd->q;
        random_effects(G, ncol, dd, model->coef, ranef);
        (model->distances)[i] = mahalanobis(G + ncol * dd->q + dd->q, ncol, dd, model->coef, model->theta);
        (model->weights)[i] = heavy_weights(model->family, nrow, (model->distances)[i]);
        zero_mat(G, ncol, ncol, ncol);
    }
    Free(G);
}

void
random_effects(double *G, int ldG, DIMS dd, double *coef, double *ranef)
{   /* compute random effects estimates */
    int job = 1;

    Memcpy(ranef, G + ldG * dd->ZXcols, dd->q);
    gaxpy(ranef, -1.0, G + ldG * dd->q, ldG, dd->q, dd->p, coef, 1.0);
    backsolve(job, G, ldG, dd->q, ranef, dd->q, 1);
}

double
mahalanobis(double *G, int ldG, DIMS dd, double *coef, double *theta)
{
    double ans, sigma2, ssq, *t;

    sigma2 = theta[dd->q * dd->q];
    t = (double *) Calloc(dd->p + 1, double);
    Memcpy(t, G + ldG * dd->p, dd->p + 1);
    gaxpy(t, -1.0, G, ldG, dd->p, dd->p, coef, 1.0);
    ssq = norm_sqr(t, dd->p + 1, 1);
    ans = ssq / sigma2;
    Free(t);
    return ans;
}

void
update_coef(MODEL model)
{   /* compute the fixed effect estimate */
    DIMS dd = model->dd;
    int i, inc = 1, nrow;
    double wts, *X;
    QRStruct qr;
    
    X = (double *) Calloc(dd->ZXrows * dd->p, double);
    for (i = 0; i < dd->n; i++) {
        double *Rc = model->ZXy + (dd->ZXoff)[i];
        nrow = MIN((dd->lengths)[i], dd->ZXcols);
        wts  = sqrt((model->weights)[i]);
        scale_mat(X + (dd->offsets)[i], dd->ZXrows, wts, Rc + (dd->lengths)[i] * dd->q, (dd->lengths)[i], nrow, dd->p);
        double *resp = model->working + (dd->offsets)[i];
        double *ranef = model->ranef + i * dd->q;
        working_response(Rc, (dd->lengths)[i], dd, ranef, resp);
        F77_CALL(dscal)(&nrow, &wts, resp, &inc);
    }
    qr = QR_decomp(X, dd->ZXrows, dd->ZXrows, dd->p);
    QR_coef(qr, model->working, dd->ZXrows, 1, model->coef);
    QR_free(qr);
    Free(X);
}

void
working_response(double *Rc, int ldRc, DIMS dd, double *ranef, double *working)
{   /* compute the working response */
    int n;

    n = MIN(ldRc, dd->ZXcols);
    Memcpy(working, Rc + ldRc * dd->ZXcols, n);
    gaxpy(working, -1.0, Rc, ldRc, dd->q, dd->q, ranef, 1.0);
}

void
update_sigma2(MODEL model)
{
    DIMS dd = model->dd;
    double sigma2, ssq;

    sigma2 = (model->theta)[dd->q * dd->q];
    ssq  = dot_product(model->distances, 1, model->weights, 1, dd->n);
    ssq += dd->n * dd->q;
    sigma2 *= ssq;
    sigma2 /= dd->N + dd->n * dd->q;
    (model->theta)[dd->q * dd->q] = sigma2;
}

void
update_theta(MODEL model)
{   /* update the covariance of random effects */
    DIMS dd = model->dd;
    int i, inc = 1, qq = dd->q * dd->q;
    double *accum, *Omega, one = 1.0, sigma2, wts;

    sigma2 = (model->theta)[qq];
    accum = (double *) Calloc(qq, double);
    Omega = (double *) Calloc(qq, double);
    for (i = 0; i < dd->n; i++) {
        double *OmegaHalf = model->OmegaHalf + i * dd->q * dd->q;
        invert_upper(OmegaHalf, dd->q, dd->q);
        outerprod(OmegaHalf, dd->q, dd->q, dd->q, OmegaHalf, dd->q, dd->q, dd->q, Omega);
        scale_mat(Omega, dd->q, sigma2, Omega, dd->q, dd->q, dd->q);
        wts = (model->weights)[i];
        double *ranef = model->ranef + i * dd->q;
        rank1_update(Omega, dd->q, dd->q, dd->q, wts, ranef, ranef);
        F77_CALL(daxpy)(&qq, &one, Omega, &inc, accum, &inc);
        zero_mat(Omega, dd->q, dd->q, dd->q);
    }
    scale_mat(model->theta, dd->q, 1.0 / dd->n, accum, dd->q, dd->q, dd->q);
    Free(accum); Free(Omega);
}

double
lme_logLik(MODEL model)
{   /* evaluate the log-likelihood function */
    DIMS dd = model->dd;
    int i;
    double accum = 0.0, sigma2, ans;

    sigma2 = (model->theta)[dd->q * dd->q];
    relative_precision_factor(model->theta, dd->q, sigma2, model->Delta);
    for (i = 0; i < dd->n; i++) {
        double *OmegaHalf = model->OmegaHalf + i * dd->q * dd->q;
        accum += logAbsDet(OmegaHalf, dd->q, dd->q);
    }
    ans  = -.5 * dd->N * log(sigma2) + dd->n * logAbsDet(model->Delta, dd->q, dd->q);
    ans += accum;
    ans += logLik_kernel(model->family, dd, model->distances);
    return ans;
}

RESID
lme_resid_init(double *ZXy, int *pdims, int *lengths, double *coef, double *ranef,
    double *conditional, double *marginal)
{   /* constructor for a resid object */
    RESID ans;

    ans = (RESID) Calloc(1, RESID_struct);
    ans->ZXy = ZXy;
    ans->dd = dims(pdims, lengths);
    ans->coef = coef;
    ans->ranef = ranef;
    ans->conditional = conditional;
    ans->marginal = marginal;
    /* initial computations */
    model_matrix(ans->ZXy, ans->dd);
    return(ans);
}

void
lme_resid_free(RESID this)
{   /* destructor for a resid object */
    Free(this);
}

void
heavy_lme_resid(double *ZXy, int *pdims, int *lengths, double *coef, double *ranef,
    double *conditional, double *marginal)
{
    RESID resid;

    resid = lme_resid_init(ZXy, pdims, lengths, coef, ranef, conditional, marginal);
    lme_residuals(resid);
    lme_resid_free(resid);
}

void
lme_residuals(RESID resid)
{
    DIMS dd = resid->dd;
    int i, nrow;

    for (i = 0; i < dd->n; i++) {
        nrow = (dd->lengths)[i];
        double *Z = resid->ZXy + (dd->ZXoff)[i];
        double *X = Z + nrow * dd->q;
        double *y = Z + nrow * dd->ZXcols;
        double *r = resid->marginal + (dd->offsets)[i];
        double *e = resid->conditional + (dd->offsets)[i];
        double *ranef = resid->ranef + i * dd->q;
        Memcpy(r, y, nrow);
        gaxpy(r, -1.0, X, nrow, nrow, dd->p, resid->coef, 1.0);
        Memcpy(e, r, nrow);
        gaxpy(e, -1.0, Z, nrow, nrow, dd->q, ranef, 1.0);
    }
}
