#include "specfun.h"

/* declaration of static functions */
static double pg_asymp(double, double);
static double pg_continued_fraction(double, double);
static double pg_series_expansion(double, double);

double
pgamma_asymp(double x, double a, double scale)
{   /* Computes the regularized incomplete gamma function using a normal
     * approximation */
    double z;

    x *= scale;
    z  =  3. * sqrt(a) * (R_pow(x / a, 1. / 3.) + 1. / (9. * a) - 1.);

    return pnorm(z, 0., 1., 1, 0);
}

double
pgamma_derivative(double x, double a, double scale)
{   /* Computes the first derivative of the incomplete gamma function.
     * Algorithm 187: Applied Statistics 31, 1982, pp. 330-335 */
    double ans;

    if (a <= 0. || scale <= 0.)
	return 0.;	/* should not happen */

    x *= scale;
    
    if (a > 600.)
	ans = pg_asymp(x, a);
    else if (x > 1. && x >= a)
	ans = pg_continued_fraction(x, a);
    else
	ans = pg_series_expansion(x, a);
    
    return ans;
}

static double
pg_asymp(double x, double a)
{   /* Computes the first derivative of the regularized incomplete gamma
     * function using a normal approximation */
    double z, z_dot;
    
    z  =  3. * sqrt(a) * (R_pow(x / a, 1. / 3.) + 1. / (9. * a) - 1.);
    z_dot = .5 * (R_pow(x / a, 1. / 3.) + 1. / (3. * a) - 3.) / sqrt(a);
    
    return z_dot * dnorm(z, 0., 1., 0);
}

static double
pg_continued_fraction(double x, double a)
{   /* Computes the first derivative of the regularized incomplete gamma
     * function using a continued fraction */
    double lgam, psi, xlog;
    double i, b, g, p, s0, c1, c2, c3, c4, c5, c6, d1, d2, d3, d4, d5, d6;
    double f, f_dot, s, s_dot;
    const static double eps = 1.e-06, max_it = 200000., scalefactor = 1.e+30;

    /* set constants */
    xlog = log(x);
    lgam = lgammafn(a);
    psi  = digamma(a);

    /* eval 'factor' and its derivative */
    f = exp(a * xlog - lgam - x);
    f_dot = f * (xlog - psi);
    
    /* Use a continued fraction expansion */
    p  = a - 1.;
    b  = x + 1. - p;
    c1 = 1.;
    c2 = x;
    c3 = x + 1.;
    c4 = x * b;
    s0 = c3 / c4;
    d1 = 0.;
    d2 = 0.;
    d3 = 0.;
    d4 = -x;
    i  = 0.;

    while (i < max_it) {
	i++;
	
	p--;
	b += 2;
	g = i * p;
	
	c5 = b * c3 + g * c1;
	c6 = b * c4 + g * c2;
	
	d5 = b * d3 - c3 + g * d1 + i * c1;
	d6 = b * d4 - c4 + g * d2 + i * c2;
	
	if (fabs(c6) > DBL_EPSILON) {
	    s = c5 / c6;
	    
	    if (fabs(s - s0) <= eps * s) {
		s_dot = (d5 - s * d6) / c6;
		return (-f * s_dot - f_dot * s);
	    }
	    
	    s0 = s;
	}
	    
	c1 = c3; c2 = c4; c3 = c5; c4 = c6;
	d1 = d3; d2 = d4; d3 = d5; d4 = d6;
	
	if (fabs(c5) > scalefactor) {
	    /* re-scale terms in continued fraction if terms are large */
	    c1 /= scalefactor; c2 /= scalefactor;
	    c3 /= scalefactor; c4 /= scalefactor;
	    d1 /= scalefactor; d2 /= scalefactor;
	    d3 /= scalefactor; d4 /= scalefactor;
	}
    }
    
    /* must not reach here */
    warning("non-convergence in pg_continued_fraction");
    s_dot = (d5 - s * d6) / c6;
    return (-f * s_dot - f_dot * s);
}

static double
pg_series_expansion(double x, double a)
{   /* Computes the first derivative of the regularized incomplete gamma
     * function using a series expansion */
    double lgamma_plus_1, psi, psi_plus_1, xlog;
    double f, f_dot, term, term_dot, sum, sum_dot, p, rel;
    const static double max_it = 200.;

    /* set constants */
    xlog = log(x);
    lgamma_plus_1 = lgamma1p(a);
    psi  = digamma(a);
    psi_plus_1 = psi + 1.0 / a;

    /* eval 'factor' and its derivative */
    f = exp(a * xlog - lgamma_plus_1 - x);
    f_dot = f * (xlog - psi_plus_1);

    /* Pearson's series expansion */
    term = 1.;
    sum  = 1.;
    term_dot = 0.;
    sum_dot = 0.;
    p = a;
    
    do {
	p++;
	
	rel = term_dot / term;
	term_dot = rel - 1. / p;
	
	term *= x / p;
	sum += term;
	
	term_dot *= term;
	sum_dot += term_dot;
	
	if (p > max_it + a) { /* convergence of the expansion is not achieved */
	    warning("non-convergence in pg_series_expansion");
	    break;
	}

    } while (term > sum * DBL_EPSILON);

    return (f_dot * sum + f * sum_dot);
} 

