#include "family.h"

/* functions for dealing with 'family' objects */

FAMILY
family_init(double *settings)
{   /* constructor for a family object */
    FAMILY ans;

    ans = (FAMILY) Calloc(1, FAMILY_struct);
    ans->fltype = (int) settings[0];
    ans->npars  = (int) settings[1];
    ans->nu = settings + 2;
    return ans;
}

void
family_free(FAMILY this)
{   /* destructor for a family object*/
    Free(this);
}

/* functions for computation of weights */

double
weight_normal()
{   /* normal weight */
    return 1.0;
}

double
weight_cauchy(double length, double distance)
{   /* Cauchy weight */
    double ans;

    ans = (1.0 + length) / (1.0 + distance);
    return ans;
}

double
weight_student(double length, double df, double distance)
{   /* Student-t weight */
    double ans;

    ans = (df + length) / (df + distance);
    return ans;
}

double
weight_slash(double length, double df, double distance)
{   /* slash weight */
    int lower_tail = 1, log_p = 0;
    double shape, f, ans;

    shape  = length / 2.0 + df;
    if (shape > 245.0) return 1.0;
    f = (length + 2.0 * df) / distance;
    ans  = pgamma(1.0, shape + 1.0, 2.0 / distance, lower_tail, log_p);
    ans /= pgamma(1.0, shape, 2.0 / distance, lower_tail, log_p);
    return (f * ans);
}

double
weight_contaminated(double length, double epsilon, double vif, double distance)
{   /* contaminated normal weight */
    double f, ans;

    f = exp(0.5 * (1.0 - vif) * distance);
    ans  = 1.0 - epsilon + epsilon * f * pow(vif, 0.5 * length + 1.0);
    ans /= 1.0 - epsilon + epsilon * f * pow(vif, 0.5 * length);
    return ans;
}

double
do_weight(FAMILY family, double length, double distance)
{   /* weights dispatcher */
    double df, epsilon, vif, ans;

    switch (family->fltype) {
        case NORMAL:
            ans = weight_normal();
            break;
        case CAUCHY:
            ans = weight_cauchy(length, distance);
            break;
        case STUDENT:
            df = (family->nu)[0];
            ans = weight_student(length, df, distance);
            break;
        case SLASH:
            df = (family->nu)[0];
            ans = weight_slash(length, df, distance);
            break;
        case CONTAMINATED:
            epsilon = (family->nu)[0];
            vif = (family->nu)[1];
            ans = weight_contaminated(length, epsilon, vif, distance);
            break;
        default:
            ans = weight_normal();
            break;
    }
    return ans;
}

/*  functions for evaluation of the log-likelihood */

double
logLik_normal(DIMS dd, double *distances)
{   /* gaussian log-likelihood */
    int i;
    double accum = 0.0, ans;

    for (i = 0; i < dd->n; i++)
        accum += *distances++;
    ans = -0.5 * accum - dd->N * M_LN_SQRT_2PI;
    return ans;
}

double
logLik_cauchy(DIMS dd, double *lengths, double *distances)
{   /* Cauchy log-likelihood */
    int i;
    double accum = 0.0, m, ans;

    for (i = 0; i < dd->n; i++) {
        m = *lengths++;
        m += 1.0;
        accum += lgammafn(0.5 * m);
        accum -= 0.5 * m * log1p(*distances++);
    }
    ans  = -0.5 * (dd->N + dd->n) * M_LN_SQRT_PI;
    ans += accum;
    return ans;
}

double
logLik_student(DIMS dd, double *lengths, double df, double *distances)
{   /* Student-t log-likelihood */
    int i;
    double accum = 0.0, m, ans;

    for (i = 0; i < dd->n; i++) {
        m = *lengths++;
        m += df;
        accum += lgammafn(0.5 * m);
        accum -= 0.5 * m * log1p(*distances++ / df);
    }
    ans  = -0.5 * dd->N * (log(df) + 2.0 * M_LN_SQRT_PI) - dd->n * lgammafn(0.5 * df);
    ans += accum;
    return ans;
}

double
logLik_slash(DIMS dd, double *lengths, double df, double *distances)
{   /* Slash log-likelihood */
    int i, lower_tail = 1, log_p = 1;
    double accum = 0.0, shape, scale, m, ans;

    for (i = 0; i < dd->n; i++) {
        m = *lengths++;
        shape = 0.5 * m + df;
        scale = 2.0 / *distances++;
        accum += shape * log(scale) + lgammafn(shape);
        accum += pgamma(1.0, shape, scale, lower_tail, log_p);
    }
    ans = dd->n * log(df) - dd->N * M_LN_SQRT_2PI;
    ans += accum;
    return ans;
}

double
logLik_contaminated(DIMS dd, double *lengths, double epsilon, double vif, double *distances)
{   /* contaminated-normal log-likelihood */
    int i;
    double accum = 0.0, f, u, ans;

    for(i = 0; i < dd->n; i++) {
        u = *distances++;
        u *= -0.5;
        f  = epsilon * pow(vif, 0.5 * *lengths++) * exp(vif * u);
        f += (1.0 - epsilon) * exp(u);
        accum += log(f);
    }
    ans = -0.5 * dd->N * M_LN_SQRT_2PI;
    ans += accum;
    return ans;
}

double
logLik_kernel(FAMILY family, DIMS dd, double *lengths, double *distances)
{   /* logLik dispatcher */
    double df, epsilon, vif, ans;

    switch (family->fltype) {
        case NORMAL:
            ans = logLik_normal(dd, distances);
            break;
        case CAUCHY:
            ans = logLik_cauchy(dd, lengths, distances);
            break;
        case STUDENT:
            df = (family->nu)[0];
            ans = logLik_student(dd, lengths, df, distances);
            break;
        case SLASH:
            df = (family->nu)[0];
            ans = logLik_slash(dd, lengths, df, distances);
            break;
        case CONTAMINATED:
            epsilon = (family->nu)[0];
            vif = (family->nu)[1];
            ans = logLik_contaminated(dd, lengths, epsilon, vif, distances);
            break;
        default:
            ans = logLik_normal(dd, distances);
            break;
    }
    return ans;
}

/* scale factor required for the Fisher information matrix */

double
acov_scale_normal()
{   /* normal scale */
    return 1.0;
}

double
acov_scale_cauchy(double length)
{   /* Cauchy scale */
    double ans;

    ans = (length + 1.0) / (length + 3.0);
    ans *= length;
    return ans;
}

double
acov_scale_student(double length, double df)
{   /* Student-t scale */
    double ans;

    ans = (df + length) / (df + length + 2.0);
    ans *= length;
    return ans;
}

double
acov_scale_slash(double length, double df, int ndraws)
{   /* slash scale */
    int i;
    double accum = 0.0, u, w, *z;

    if (df > 30.0)
        return 1.0;
    z = (double *) Calloc(length, double);
    GetRNGstate();
    for (i = 0; i < ndraws; i++) {
        slash_spherical_rand(z, df, 1, length);
        u = norm_sqr(z, length, 1);
        w = weight_slash(length, df, u);
        accum += SQR(w) * u;
    }
    PutRNGstate();
    Free(z);
    return (accum / ndraws);
}

double
acov_scale_contaminated(double length, double epsilon, double vif, int ndraws)
{   /* contaminated normal scale */
    int i;
    double accum = 0.0, u, w, *z;

    z = (double *) Calloc(length, double);
    GetRNGstate();
    for (i = 0; i < ndraws; i++) {
        contaminated_spherical_rand(z, epsilon, vif, 1, length);
        u = norm_sqr(z, length, 1);
        w = weight_contaminated(length, epsilon, vif, u);
        accum += SQR(w) * u;
    }
    PutRNGstate();
    Free(z);
    return (accum / ndraws);
}

double
acov_scale(FAMILY family, double length, int ndraws)
{   /* scale factor for the Fisher information matrix */
    double df, epsilon, vif, ans;

    switch (family->fltype) {
        case NORMAL:
            ans = acov_scale_normal();
            break;
        case CAUCHY:
            ans = acov_scale_cauchy(length);
            break;
        case STUDENT:
            df = (family->nu)[0];
            ans = acov_scale_student(length, df);
            break;
        case SLASH:
            df = (family->nu)[0];
            ans = acov_scale_slash(length, df, ndraws);
            break;
        case CONTAMINATED:
            epsilon = (family->nu)[0];
            vif = (family->nu)[1];
            ans = acov_scale_contaminated(length, epsilon, vif, ndraws);
            break;
        default:
            ans = acov_scale_normal();
            break;
    }
    return ans;
}
