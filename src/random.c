#include "random.h"

/* elliptically contoured random generation */

void
elliptical_rand(double *y, int n, int p, FAMILY family, double *mu,
    double *Sigma, int isRoot)
{
    char *side = "L", *uplo = "U", *trans = "T", *diag = "N";
    double one = 1.0, *Root;
    int i, inc = 1, info = 0;

    if (isRoot)
	Root = Sigma;
    else {
	Root = (double *) Calloc(p * p, double);
	copy_mat(Root, p, Sigma, p, p, p);
	F77_CALL(dpotrf)(uplo, &p, Root, &p, &info);
	if (info)
	    error("DPOTRF in generation of elliptical deviates gave code %d", info);
    }
    spherical_rand(y, n, p, family);
    F77_CALL(dtrmm)(side, uplo, trans, diag, &p, &n, &one, Root, &p, y, &p);
    for (i = 0; i < n; i++) {
	F77_CALL(daxpy)(&p, &one, mu, &inc, y, &inc);
	y += p;
    }
    if (!isRoot)
	Free(Root);
}

void
spherical_rand(double *y, int n, int p, FAMILY family)
{
    double df, epsilon, vif;

    switch (family->fltype) {
        case NORMAL:
            norm_spherical_rand(y, n, p);
            break;
        case CAUCHY:
            cauchy_spherical_rand(y, n, p);
            break;
        case STUDENT:
            df = (family->nu)[0];
            student_spherical_rand(y, df, n, p);
            break;
        case SLASH:
            df = (family->nu)[0];
            slash_spherical_rand(y, df, n, p);
            break;
        case CONTAMINATED:
            epsilon = (family->nu)[0];
            vif = (family->nu)[1];
            contaminated_spherical_rand(y, epsilon, vif, n, p);
            break;
        default:
            norm_spherical_rand(y, n, p);
            break;
    }
}

void
norm_spherical_rand(double *y, int n, int p)
{   /* independent standard normal variates */
    int i, j;

    for (i = 0; i < n; i++) {
	for (j = 0; j < p; j++)
	    y[j] = norm_rand();
	y += p;
    }
}

void
cauchy_spherical_rand(double *y, int n, int p)
{   /* standard Cauchy variates */
    int i, j, one = 1;
    double tau, radial;

    for (i = 0; i < n; i++) {
	for (j = 0; j < p; j++)
	    y[j] = norm_rand();
	tau = rgamma(0.5, 2.0);
	radial = R_pow(tau, -0.5);
	F77_CALL(dscal)(&p, &radial, y, &one);
	y += p;
    }
}

void
student_spherical_rand(double *y, double df, int n, int p)
{   /* standard Student-t variates */
    int i, j, one = 1;
    double tau, radial;

    for (i = 0; i < n; i++) {
	for (j = 0; j < p; j++)
	    y[j] = norm_rand();
	tau = rgamma(df / 2.0, 2.0 / df);
	radial = R_pow(tau, -0.5);
	F77_CALL(dscal)(&p, &radial, y, &one);
	y += p;
    }
}

void
slash_spherical_rand(double *y, double df, int n, int p)
{   /* standard slash variates */
    int i, j, one = 1;
    double tau, radial;

    for (i = 0; i < n; i++) {
	for (j = 0; j < p; j++)
	    y[j] = norm_rand();
	tau = rbeta(df, 1.0);
	radial = R_pow(tau, -0.5);
	F77_CALL(dscal)(&p, &radial, y, &one);
	y += p;
    }
}

void
contaminated_spherical_rand(double *y, double epsilon, double vif, int n, int p)
{   /* standard contaminated normal variates */
    int i, j, one = 1;
    double radial, unif;

    radial = 1.0 / vif;
    for (i = 0; i < n; i++) {
	for (j = 0; j < p; j++)
	    y[j] = norm_rand();
	unif = unif_rand();
	if (unif > 1.0 - epsilon)
	    F77_CALL(dscal)(&p, &radial, y, &one);
	y += p;
    }
}

void
power_spherical_rand(double *y, double shape, int n, int p)
{   /* standard power exponential variates */
    int i, j, one = 1;
    double tau, radial;

    shape *= 2.0;
    for (i = 0; i < n; i++) {
	for (j = 0; j < p; j++)
	    y[j] = norm_rand();
	tau = rgamma(p / shape, 2.0);
	radial = R_pow(tau, 1.0 / shape) / F77_CALL(dnrm2)(&p, y, &one);
	F77_CALL(dscal)(&p, &radial, y, &one);
	y += p;
    }
}

/* uniformly distributed random vectors */

void
unif_sphere_rand(double *y, int n, int p)
{   /* random vector generation uniformly on the sphere */
    int i, j, one = 1;
    double radial;

    for (i = 0; i < n; i++) {
	for (j = 0; j < p; j++)
	    y[j] = norm_rand();
	radial = 1.0 / F77_CALL(dnrm2)(&p, y, &one);
	F77_CALL(dscal)(&p, &radial, y, &one);
        y += p;
    }
}

void
unif_ball_rand(double *y, int n, int p)
{   /* random vector generation uniformly in the ball */
    int i, j, one = 1;
    double radial, unif;

    for (i = 0; i < n; i++) {
	for (j = 0; j < p; j++)
	    y[j] = norm_rand();
	unif = unif_rand();
	radial = R_pow(unif, 1.0 / p) / F77_CALL(dnrm2)(&p, y, &one);
	F77_CALL(dscal)(&p, &radial, y, &one);
        y += p;
    }
}

/* random number generation of right truncated Gamma distribution using
 * mixtures. Original C code by from Anne Philippe (1997).
 */

int
ncomp_optimal(double b)
{
    double ans, q = 1.644853626951;
    ans = 0.25 * R_pow_di(q * sqrt(q * q + 4. * b), 2);
    return ((int) ftrunc(ans));
}

double
tgamma_right(double a, double b)
{
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
tgamma_rand(double shape, double rate, double truncation)
{
    double ans;
    ans = tgamma_right(shape, rate * truncation) * truncation;
    return ans;
}
