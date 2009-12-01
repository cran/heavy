#ifndef HEAVY_MATRIX_H
#define HEAVY_MATRIX_H

#include "base.h"

/* basic matrix manipulations */
extern double norm_sqr(double *, int, int);
extern double dot_product(double *, int, double *, int, int);
extern void gaxpy(double *, double, double *, int, int, int, double *, double);
extern void zero_mat(double *, int, int, int);
extern void zero_lower(double *, int, int, int);
extern void copy_mat(double *, int, double *, int, int, int);
extern double * scale_mat(double *, int, double, double *, int, int, int);
extern double * scale_lower(double *, int, double, double *, int, int, int);
extern void mult_mat(double *, int, int, int, double *, int, int, int, double *);
extern void crossprod(double *, int, int, int, double *, int, int, int, double *);
extern void outerprod(double *, int, int, int, double *, int, int, int, double *);
extern void rank1_update(double *, int, int, int, double, double *, double *);
extern double logAbsDet(double *, int, int);

/* kronecker products */
extern void kronecker_IA(double *, int, int, double *, int, int, int);
extern void kronecker_AI(double *, int, int, double *, int, int, int);
extern void kronecker_prod(double *, int, double *, int, int, int, double *, int, int, int);

/* routines for matrix decompositions */
extern void chol_decomp(int, double *, int, int);
extern QRStruct QR_decomp(double *, int, int, int);
extern void QR_free(QRStruct);
extern EIGENStruct eigen_decomp(double *, int, int, double *, double *);
extern void eigen_free(EIGENStruct);

/* orthogonal-triangular operations */
extern void QR_qty(QRStruct, double *, int, int);
extern void QR_qy(QRStruct, double *, int, int);
extern void QR_coef(QRStruct, double *, int, int, double *);
extern void QR_resid(QRStruct, double *, int, int, double *);
extern void QR_fitted(QRStruct, double *, int, int, double *);
extern void QR_store_R(QRStruct, double *, int);

/* matrix inversion and solver */
extern void invert_mat(double *, int, int);
extern void invert_lower(double *, int, int);
extern void invert_upper(double *, int, int);
extern void backsolve(int, double *, int, int, double *, int, int);

/* QR decomposition used in heavy_lme */
extern void append_decomp(double *, int, int, int, double *, int, double *, int);

#endif /* HEAVY_MATRIX_H */
