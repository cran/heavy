#include "family.h"

/* functions for dealing with 'family' objects */

FamilyStruct
init_family(double *setngs)
{   /* constructor for a family object */
    FamilyStruct value;

    value = (FamilyStruct) Calloc(1, family_struct);
    value->fltype = (int) setngs[0];
    value->npars  = (int) setngs[1];
    value->nu = setngs + 2;
    return value;
}

void
free_family(FamilyStruct this)
{   /* destructor for a family object*/
    Free(this);
}

/* functions for computation of weights */

double
weight_normal()
{   /* normal weight */
    return 1.;
}

double
weight_student(int length, double df, double distance)
{   /* Student-t weight */
    double ans;

    ans = (df + (double) length) / (df + distance);
    return ans;
}

double
weight_slash(int length, double df, double distance)
{   /* slash weight */
    int lower_tail = 1, log_p = 0;
    double shape, f, ans;

    shape  = length / 2. + df;
    if (shape > 245.0) return 1.;
    f = (length + 2. * df) / distance;
    ans  = pgamma(1.0, shape + 1., 2. / distance, lower_tail, log_p);
    ans /= pgamma(1.0, shape, 2. / distance, lower_tail, log_p);
    return (f * ans);
}

double
weight_contaminated(int length, double epsilon, double vif, double distance)
{   /* contaminated normal weight */
    double f, ans;

    f = exp(.5 * (1. - vif) * distance);
    ans  = 1. - epsilon + epsilon * f * pow(vif, .5 * length + 1.);
    ans /= 1. - epsilon + epsilon * f * pow(vif, .5 * length);
    return ans;
}

double
heavy_weights(FamilyStruct family, int length, double distance)
{   /* weights dispatcher */
    double df, epsilon, vif, ans;

    switch (family->fltype) {
        case NORMAL:
            ans = weight_normal();
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
    ans = -.5 * accum - dd->N * M_LN_SQRT_2PI;
    return ans;
}

double
logLik_student(DIMS dd, double df, double *distances)
{   /* Student-t log-likelihood */
    int i;
    double accum1 = 0.0, accum2 = 0.0, length, ans;

    for (i = 0; i < dd->n; i++) {
        length = (double) (dd->lengths)[i];
        accum1 += lgammafn(.5 * (length + df));
        accum2 += (length + df) * log1p(*distances++ / df);
    }
    ans  = -.5 * dd->N * (log(df) + 2. * M_LN_SQRT_PI) - dd->n * lgammafn(.5 * df);
    ans += accum1;
    ans -= .5 * accum2;
    return ans;
}

double
logLik_slash(DIMS dd, double df, double *distances)
{   /* Slash log-likelihood */
    int i, lower_tail = 1, log_p = 1;
    double accum = 0.0, shape, scale, length, ans;

    for (i = 0; i < dd->n; i++) {
        length = (double) (dd->lengths)[i];
        shape = .5 * length + df;
        scale = 2. / *distances++;
        accum += shape * log(scale) + lgammafn(shape);
        accum += pgamma(1.0, shape, scale, lower_tail, log_p);
    }
    ans = accum + dd->n * log(df) - dd->N * M_LN_SQRT_2PI;
    return ans;
}

double
logLik_contaminated(DIMS dd, double epsilon, double vif, double *distances)
{   /* contaminated-normal log-likelihood */
    int i;
    double accum = 0.0, f, length, ans;

    for(i = 0; i < dd->n; i++) {
        length = (double) (dd->lengths)[i];
        f  = epsilon * pow(vif, .5 * length) * exp(-.5 * vif * distances[i]);
        f += (1. - epsilon) * exp(-.5 * distances[i]);
        accum += log(f);
    }
    ans = accum - dd->N * M_LN_SQRT_2PI;
    return ans;
}

double
logLik_kernel(FamilyStruct family, DIMS dd, double *distances)
{   /* logLik dispatcher */
    double df, epsilon, vif, ans;

    switch (family->fltype) {
        case NORMAL:
            ans = logLik_normal(dd, distances);
            break;
        case STUDENT:
            df = (family->nu)[0];
            ans = logLik_student(dd, df, distances);
            break;
        case SLASH:
            df = (family->nu)[0];
            ans = logLik_slash(dd, df, distances);
            break;
        case CONTAMINATED:
            epsilon = (family->nu)[0];
            vif = (family->nu)[1];
            ans = logLik_contaminated(dd, epsilon, vif, distances);
            break;
        default:
            ans = logLik_normal(dd, distances);
            break;
    }
    return ans;
}
