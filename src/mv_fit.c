#include "mv_fit.h"

/* declaration of static functions */

/* functions to deal with dims objects */
static DIMS dims(int *);
static void dims_free(DIMS);

/* routines for estimation in linear models */
static MV mv_init(double *, int *, double *, double *, double *, double *, double *, double *, double *);
static void mv_free(MV);
static int mv_iterate(MV);
static void mv_Estep(MV);
static double mahalanobis(double *, int, double *, double *);
static void update_center(MV);
static void update_Scatter(MV);

/* routines for evaluation of marginal log-likelihood and Fisher information matrix */
static void mv_acov(FAMILY, DIMS, double *, int, double *);
static double mv_logLik(FAMILY, DIMS, double *, double *);

/* ..end declarations */

void
mv_fit(double *y, int *pdims, double *settings, double *center, double *Scatter,
    double *distances, double *weights, double *logLik, double *aCov, double *control)
{   /* estimation for multivariate heavy-tailed distributions */
    MV model;

    model = mv_init(y, pdims, settings, center, Scatter, distances, weights, aCov,
                    control);
    control[4] = (double) mv_iterate(model);
    *logLik = mv_logLik(model->family, model->dm, model->distances, model->Scatter);
    mv_acov(model->family, model->dm, model->Scatter, model->ndraws, model->aCov);
    mv_free(model);
}

static DIMS
dims(int *pdims)
{   /* dims object for multivariate models */
    DIMS ans;

    ans = (DIMS) Calloc(1, DIMS_struct);
    ans->N = (int) pdims[0];
    ans->n = ans->N;
    ans->p = (int) pdims[1];
    return ans;
}

static void
dims_free(DIMS this)
{   /* destructor for a dims object */
    Free(this);
}

static MV
mv_init(double *y, int *pdims, double *settings, double *center, double *Scatter,
    double *distances, double *weights, double *aCov, double *control)
{   /* constructor for a multivariate object */
    MV model;

    model = (MV) Calloc(1, MV_struct);
    model->dm = dims(pdims);
    model->settings = settings;
    model->family = family_init(settings);
    model->y = y;
    model->center = center;
    model->Scatter = Scatter;
    model->distances = distances;
    model->weights = weights;
    model->aCov = aCov;
    model->control = control;
    model->maxIter = (int) control[0];
    model->tolerance = control[1];
    model->fixShape = (int) control[2];
    model->ndraws = (int) control[3];
    return model;
}

static void
mv_free(MV this)
{   /* destructor for a model object */
    dims_free(this->dm);
    family_free(this->family);
    Free(this);
}

static int
mv_iterate(MV model)
{
    int i, iter = 0;
    double conv, tol = R_pow(model->tolerance, 2./3.), *lengths, RSS, newRSS;

    /* initialization */
    lengths = (double *) Calloc(model->dm->n, double);
    for (i = 0; i < model->dm->n; i++)
        lengths[i] = (double) model->dm->p;
    RSS = (double) model->dm->n * model->dm->p;

    /* main loop */
    repeat {
        /* E-step */
        mv_Estep(model);
        
        /* CM-steps */
        update_center(model);
        update_Scatter(model);
        if (!(model->fixShape))
            update_mixture(model->family, model->dm, model->distances, lengths,
                           model->weights, tol);
        newRSS = dot_product(model->weights, 1, model->distances, 1, model->dm->n);
        
        iter++;
        
        /* eval convergence */
        conv = fabs((newRSS - RSS) / (newRSS + ABSTOL));
        if (conv < model->tolerance) { /* successful completion */
            Free(lengths);
            return iter;
        }
        if (iter >= model->maxIter)
            break; /* maximum number of iterations exceeded */
        RSS = newRSS;
    }
    Free(lengths);
    return (iter - 1);
}

static void
mv_Estep(MV model)
{
    DIMS dm = model->dm;
    double *Root;
    int i, info = 0, job = 0;

    Root = (double *) Calloc(SQR(dm->p), double);
    copy_mat(Root, dm->p, model->Scatter, dm->p, dm->p, dm->p);
    chol_decomp(Root, dm->p, dm->p, job, &info);
    if (info)
        error("chol_decomp in mv_Estep gave code %d", info);
    for (i = 0; i < dm->n; i++) {
        (model->distances)[i] = mahalanobis(model->y + i * dm->p, dm->p, model->center, Root);
        (model->weights)[i] = do_weight(model->family, (double) dm->p, (model->distances)[i]);
    }
    Free(Root);
}

static double
mahalanobis(double *y, int p, double *center, double *Root)
{   /* Mahalanobis distances */
    double ans, minus = -1., *z;
    int inc = 1, info = 0, job = 0;

    z = (double *) Calloc(p, double);
    Memcpy(z, y, p);
    F77_CALL(daxpy)(&p, &minus, center, &inc, z, &inc);
    backsolve(Root, p, p, z, p, 1, job, &info);
    if (info)
        error("backsolve in mahalanobis gave code %d", info);
    ans = norm_sqr(z, p, 1);
    Free(z);
    return ans;
}

static void
update_center(MV model)
{   /* compute the center estimate */
    DIMS dm = model->dm;
    double factor = 1., wts, *center;
    int i, inc = 1;

    center = (double *) Calloc(dm->p, double);
    for (i = 0; i < dm->n; i++) {
        wts = (model->weights)[i];
        F77_CALL(daxpy)(&(dm->p), &wts, model->y + i * dm->p, &inc, center, &inc);
    }
    factor /= F77_CALL(dasum)(&(dm->n), model->weights, &inc);
    F77_CALL(dscal)(&(dm->p), &factor, center, &inc);
    Memcpy(model->center, center, dm->p);
    Free(center);
}

static void
update_Scatter(MV model)
{   /* update the scatter matrix estimate */
    DIMS dm = model->dm;
    double minus = -1., wts, *Scatter, *z;
    int i, inc = 1;

    Scatter = (double *) Calloc(SQR(dm->p), double);
    z = (double *) Calloc(dm->p, double);
    for (i = 0; i < dm->n; i++) {
        wts = (model->weights)[i];
        Memcpy(z, model->y + i * dm->p, dm->p);
        F77_CALL(daxpy)(&(dm->p), &minus, model->center, &inc, z, &inc);
        rank1_update(Scatter, dm->p, dm->p, dm->p, wts, z, z);
    }
    scale_mat(model->Scatter, dm->p, Scatter, dm->p, dm->p, dm->p, 1. / dm->n);
    Free(z); Free(Scatter);
}

static double
mv_logLik(FAMILY family, DIMS dm, double *distances, double *Scatter)
{   /* evaluate the log-likelihood function for multivariate distributions */
    double ans = 0., *lengths, *Root;
    int i, info = 0, job = 0;

    lengths = (double *) Calloc(dm->n, double);
    for (i = 0; i < dm->n; i++)
        lengths[i] = (double) dm->p;
    Root = (double *) Calloc(SQR(dm->p), double);
    copy_mat(Root, dm->p, Scatter, dm->p, dm->p, dm->p);
    chol_decomp(Root, dm->p, dm->p, job, &info);
    if (info)
        error("chol_decomp in mv_logLik gave code %d", info);
    ans -= dm->n * logAbsDet(Root, dm->p, dm->p);
    ans += logLik_kernel(family, dm, lengths, distances);
    Free(lengths); Free(Root);
    return ans;
}

static void 
mv_acov(FAMILY family, DIMS dm, double *Scatter, int ndraws, double *aCov)
{   /* evaluate the Fisher information matrix */
    double factor, length;

    /* scaling */
    length = (double) dm->p;
    factor = 1. / acov_scale(family, length, ndraws);
    scale_mat(aCov, dm->p, Scatter, dm->p, dm->p, dm->p, factor);
}
