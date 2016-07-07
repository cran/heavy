#include "random.h"

/* static functions.. */
static int ncomp_optimal(double);
static DIMS dims(int *);
static void dims_free(DIMS);
/* ..end declarations */

/* 'dims' functions */

static DIMS
dims(int *pdims)
{   /* dims object */
    DIMS ans;

    ans = (DIMS) Calloc(1, DIMS_struct);
    ans->n = (int) pdims[0];
    ans->p = (int) pdims[1];
    return ans;
}

static void
dims_free(DIMS this)
{   /* destructor for a dims object */
    Free(this);
}

/* random vector generation uniformly located on a spherical surface */

void
rand_sphere(double *y, int *pdims)
{   /* random vector generation on the sphere (to be called by R) */
    DIMS dm;

    dm = dims(pdims);
    GetRNGstate();
    rand_unif_sphere(y, dm->n, dm->p);
    PutRNGstate();
    dims_free(dm);
}

void
rand_unif_sphere(double *y, int n, int p)
{   /* random vector generation uniformly on the sphere */
    int i, j, inc = 1;
    double radial;

    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++)
            y[j] = norm_rand();
        radial = 1. / F77_CALL(dnrm2)(&p, y, &inc);
        F77_CALL(dscal)(&p, &radial, y, &inc);
        y += p;
    }
}

/* multivariate normal random generation */

void
rand_norm(double *y, int *pdims, double *center, double *Scatter)
{   /* multivariate normal random generation */
    DIMS dm;
    char *side = "L", *uplo = "U", *trans = "T", *diag = "N";
    double one = 1.;
    int i, inc = 1, info = 0, job = 1;
    
    dm = dims(pdims);
    GetRNGstate();
    chol_decomp(Scatter, dm->p, dm->p, job, &info);
    if (info)
        error("DPOTRF in cholesky decomposition gave code %d", info);
    rand_spherical_norm(y, dm->n, dm->p);
    F77_CALL(dtrmm)(side, uplo, trans, diag, &(dm->p), &(dm->n), &one, Scatter,
                    &(dm->p), y, &(dm->p));
    for (i = 0; i < dm->n; i++) {
        F77_CALL(daxpy)(&(dm->p), &one, center, &inc, y, &inc);
        y += dm->p;
    }
    PutRNGstate();
    dims_free(dm);
}

void
rand_spherical_norm(double *y, int n, int p)
{   /* independent standard normal variates */
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++)
            y[j] = norm_rand();
        y += p;
    }
}

/* multivariate Cauchy random generation */

void
rand_cauchy(double *y, int *pdims, double *center, double *Scatter)
{   /* multivariate normal random generation */
    DIMS dm;
    char *side = "L", *uplo = "U", *trans = "T", *diag = "N";
    double one = 1.;
    int i, inc = 1, info = 0, job = 1;
    
    dm = dims(pdims);
    GetRNGstate();
    chol_decomp(Scatter, dm->p, dm->p, job, &info);
    if (info)
        error("DPOTRF in cholesky decomposition gave code %d", info);
    rand_spherical_cauchy(y, dm->n, dm->p);
    F77_CALL(dtrmm)(side, uplo, trans, diag, &(dm->p), &(dm->n), &one, Scatter,
                    &(dm->p), y, &(dm->p));
    for (i = 0; i < dm->n; i++) {
        F77_CALL(daxpy)(&(dm->p), &one, center, &inc, y, &inc);
        y += dm->p;
    }
    PutRNGstate();
    dims_free(dm);
}

void
rand_spherical_cauchy(double *y, int n, int p)
{   /* standard Cauchy variates */
    int i, j, inc = 1;
    double tau, radial;

    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++)
            y[j] = norm_rand();
        tau = rgamma(.5, 2.);
        radial = R_pow(tau, -.5);
        F77_CALL(dscal)(&p, &radial, y, &inc);
        y += p;
    }
}

/* multivariate Student-t random generation */

void
rand_student(double *y, int *pdims, double *center, double *Scatter, double *df)
{   /* multivariate Student-t random generation */
    DIMS dm;
    char *side = "L", *uplo = "U", *trans = "T", *diag = "N";
    double one = 1.;
    int i, inc = 1, info = 0, job = 1;
    
    dm = dims(pdims);
    GetRNGstate();
    chol_decomp(Scatter, dm->p, dm->p, job, &info);
    if (info)
        error("DPOTRF in cholesky decomposition gave code %d", info);
    rand_spherical_student(y, *df, dm->n, dm->p);
    F77_CALL(dtrmm)(side, uplo, trans, diag, &(dm->p), &(dm->n), &one, Scatter,
                    &(dm->p), y, &(dm->p));
    for (i = 0; i < dm->n; i++) {
        F77_CALL(daxpy)(&(dm->p), &one, center, &inc, y, &inc);
        y += dm->p;
    }
    PutRNGstate();
    dims_free(dm);
}

void
rand_spherical_student(double *y, double df, int n, int p)
{   /* standard Student-t variates */
    int i, j, inc = 1;
    double tau, radial;

    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++)
            y[j] = norm_rand();
        tau = rgamma(df / 2., 2. / df);
        radial = R_pow(tau, -.5);
        F77_CALL(dscal)(&p, &radial, y, &inc);
        y += p;
    }
}

/* multivariate slash random generation */

void
rand_slash(double *y, int *pdims, double *center, double *Scatter, double *df)
{   /* multivariate slash random generation */
    DIMS dm;
    char *side = "L", *uplo = "U", *trans = "T", *diag = "N";
    double one = 1.;
    int i, inc = 1, info = 0, job = 1;
    
    dm = dims(pdims);
    GetRNGstate();
    chol_decomp(Scatter, dm->p, dm->p, job, &info);
    if (info)
        error("DPOTRF in cholesky decomposition gave code %d", info);
    rand_spherical_slash(y, *df, dm->n, dm->p);
    F77_CALL(dtrmm)(side, uplo, trans, diag, &(dm->p), &(dm->n), &one, Scatter,
                    &(dm->p), y, &(dm->p));
    for (i = 0; i < dm->n; i++) {
        F77_CALL(daxpy)(&(dm->p), &one, center, &inc, y, &inc);
        y += dm->p;
    }
    PutRNGstate();
    dims_free(dm);
}

void
rand_spherical_slash(double *y, double df, int n, int p)
{   /* standard slash variates */
    int i, j, inc = 1;
    double tau, radial;

    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++)
            y[j] = norm_rand();
        tau = rbeta(df, 1.);
        radial = R_pow(tau, -.5);
        F77_CALL(dscal)(&p, &radial, y, &inc);
        y += p;
    }
}

/* multivariate contaminated normal random generation */

void
rand_contaminated(double *y, int *pdims, double *center, double *Scatter,
    double *eps, double *vif)
{   /* multivariate slash random generation */
    DIMS dm;
    char *side = "L", *uplo = "U", *trans = "T", *diag = "N";
    double one = 1.;
    int i, inc = 1, info = 0, job = 1;
    
    dm = dims(pdims);
    GetRNGstate();
    chol_decomp(Scatter, dm->p, dm->p, job, &info);
    if (info)
        error("DPOTRF in cholesky decomposition gave code %d", info);
    rand_spherical_contaminated(y, *eps, *vif, dm->n, dm->p);
    F77_CALL(dtrmm)(side, uplo, trans, diag, &(dm->p), &(dm->n), &one, Scatter,
                    &(dm->p), y, &(dm->p));
    for (i = 0; i < dm->n; i++) {
        F77_CALL(daxpy)(&(dm->p), &one, center, &inc, y, &inc);
        y += dm->p;
    }
    PutRNGstate();
    dims_free(dm);
}

void
rand_spherical_contaminated(double *y, double eps, double vif, int n, int p)
{   /* standard contaminated normal variates */
    int i, j, inc = 1;
    double radial, unif;

    radial = 1. / vif;
    for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++)
            y[j] = norm_rand();
        unif = unif_rand();
        if (unif > 1. - eps)
            F77_CALL(dscal)(&p, &radial, y, &inc);
        y += p;
    }
}

/* random number generation of right truncated Gamma distribution using mixtures. 
 * Original C code from Anne Philippe (1997).
 */

static int
ncomp_optimal(double b)
{   /* optimal number of components for p = 0.95 fixed */
    double ans, q = 1.644853626951;
    ans = 0.25 * R_pow_di(q * sqrt(q * q + 4. * b), 2);
    return ((int) ftrunc(ans));
}

double
rtgamma_right_standard(double a, double b)
{   /* random number generation from the gamma with right truncation point t = 1, i.e. TG^-(a,b,1) */
    int n, i, j;
    double x, u, y, z, yy, zz;
    double *wl, *wlc;

    n = ncomp_optimal(b);
    wl  = (double *) Calloc(n + 2, double);
    wlc = (double *) Calloc(n + 2, double);

    wl[0] = 1.0; wlc[0] = 1.0;
    for (i = 1; i <= n; i++) {
        wl[i]  = wl[i-1] * b / (a + i);
        wlc[i] = wlc[i-1] + wl[i];
    }
    for (i = 0; i <= n; i++)
        wlc[i] = wlc[i] / wlc[n];
    y = 1.0; yy = 1.0;
    for (i = 1; i <= n; i++) {
        yy *= b / i;
        y  += yy;
    }

    repeat {
        u = unif_rand();
        j = 0;
        while (u > wlc[j])
            j += 1;
        x = rbeta(a, (double) j + 1);
        u = unif_rand();
        z = 1.0; zz = 1.0;
        for (i = 1; i <= n; i++) {
            zz *= (1 - x) * b / i;
            z  += zz;
        }
        z = exp(-b * x) * y / z;
        if (u <= z)
            break;
    }
    Free(wl); Free(wlc);
    return x;
}

double
rtgamma_right(double shape, double rate, double truncation)
{   /* random number generation from the gamma with right truncation point, i.e. TG^-(a,b,t) */
    double x;
    x = rtgamma_right_standard(shape, rate * truncation) * truncation;
    return x;
}
