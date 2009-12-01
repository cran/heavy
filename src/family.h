#ifndef HEAVY_FAMILY_H
#define HEAVY_FAMILY_H

#include "base.h"

typedef enum {
    NORMAL,
    STUDENT,
    SLASH,
    CONTAMINATED
} flclass;

/* heavy tailed family structure */
typedef struct family_struct {
    flclass fltype; /* family type */
    int npars;      /* number of parameters in 'family' */
    double *nu;     /* parameter vector */
} family_struct, *FamilyStruct;

/* functions for dealing with 'family' objects */
extern FamilyStruct init_family(double *);
extern void free_family(FamilyStruct);

/* routines for computation of weights */
double weight_normal();
double weight_student(int, double, double);
double weight_slash(int, double, double);
double weight_contaminated(int, double, double, double);
extern double heavy_weights(FamilyStruct, int, double);

/* functions for evaluation of the log-likelihood */
double logLik_normal(DIMS, double *);
double logLik_student(DIMS, double, double *);
double logLik_slash(DIMS, double, double *);
double logLik_contaminated(DIMS, double, double, double *);
extern double logLik_kernel(FamilyStruct, DIMS, double *);

#endif /* HEAVY_FAMILY_H */
