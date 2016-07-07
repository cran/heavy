#ifndef HEAVY_MATRIX_H
#define HEAVY_MATRIX_H

#include "base.h"

/* basic matrix manipulations and BLAS wrappers */
extern double dot_product(double *, int, double *, int, int);
extern double norm_sqr(double *, int, int);
extern double sum_abs(double *, int, int);
extern void ax_plus_y(double, double *, int, double *, int, int);
extern void copy_vec(double *, int, double *, int, int);
extern void scale_vec(double, double *, int, int);
extern void swap_vec(double *, int, double *, int, int);
extern void zero_mat(double *, int, int, int);
extern void copy_mat(double *, int, double *, int, int, int);
extern void add_mat(double *, int, double, double *, int, int, int);
extern void scale_mat(double *, int, double *, int, int, int, double);
extern void GE_axpy(double *, double, double *, int, int, int, double *, double, int);
extern void lower_tri(double *, int, double *, int, int, int);
extern void upper_tri(double *, int, double *, int, int, int);
extern void triangle_mult_vec(double *, double *, int, int, double *, int);
extern void mult_mat(double *, double *, int, int, int, double *, int, int, int);
extern void crossprod(double *, double *, int, int, int, double *, int, int, int);
extern void outerprod(double *, double *, int, int, int, double *, int, int, int);
extern void rank1_update(double *, int, int, int, double, double *, double *);
extern void rank1_symm_update(double *, int, int, double, double *, int);
extern double logAbsDet(double *, int, int);

/* routines for matrix decompositions (wrappers to LAPACK and Linpack) */
extern void chol_decomp(double *, int, int, int, int *);
extern void svd_decomp(double *, int, int, int, double *, int, double *, double *, int, int, int *);
extern QRStruct QR_decomp(double *, int, int, int, double *, int *);
extern void QR_free(QRStruct);
extern LQStruct LQ_decomp(double *, int, int, int, double *, int *);
extern void LQ_free(LQStruct);

/* orthogonal-triangular operations (wrappers to LAPACK) */
extern void QR_qty(QRStruct, double *, int, int, int);
extern void QR_qy(QRStruct, double *, int, int, int);
extern void QR_store_R(QRStruct, double *, int);
extern void LQ_yqt(LQStruct, double *, int, int, int);
extern void LQ_yq(LQStruct, double *, int, int, int);

/* matrix inversion and linear solver */
extern void invert_mat(double *, int, int, int *);
extern void invert_triangular(double *, int, int, int, int *);
extern void backsolve(double *, int, int, double *, int, int, int, int *);

/* linear least-squares fit */
extern void lsfit(double *, int, int, int, double *, int, int, double *, int *);

#endif /* HEAVY_MATRIX_H */
