#ifndef HEAVY_RANDOM_H
#define HEAVY_RANDOM_H

#include "base.h"
#include "matrix.h"

/* elliptically contoured random generation */
void elliptical_rand(double *, int, int, FAMILY, double *, double *, int);
void spherical_rand(double *, int, int, FAMILY);

/* spherical random generation */
void norm_spherical_rand(double *, int, int);
void cauchy_spherical_rand(double *, int, int);
void student_spherical_rand(double *, double, int, int);
void slash_spherical_rand(double *, double, int, int);
void contaminated_spherical_rand(double *, double, double, int, int);
void power_spherical_rand(double *, double, int, int);

/* uniformly distributed random vectors */
void unif_sphere_rand(double *, int, int);
void unif_ball_rand(double *, int, int);

/* right truncated Gamma random generation */
int ncomp_optimal(double);
double tgamma_right_rand(double, double);
extern double tgamma_rand(double, double, double);

#endif /* HEAVY_RANDOM_H */
