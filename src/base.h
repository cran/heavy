#ifndef HEAVY_BASE_H
#define HEAVY_BASE_H

#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R_ext/BLAS.h>
#include <R_ext/Linpack.h>
#include <R_ext/Lapack.h>
#include <R_ext/Applic.h>

/* some definitions */
#define NULLP    (void *) 0
#define MAX(a,b) (((a)>(b)) ? (a) : (b))
#define MIN(a,b) (((a)<(b)) ? (a) : (b))

/* QR structure */
typedef struct QR_struct {
    double *mat, *qraux;
    int ldmat, nrow, ncol;
} QR_struct, *QRStruct;

/* eigen structure */
typedef struct eigen_struct {
    double *mat, *values, *vectors, *work1, *work2;
    int ldmat, n, ierr;
} eigen_struct, *EIGENStruct;

/* dims structure */
typedef struct DIMS_struct {
    int
      N,        /* total number of observations */
      ZXrows,   /* number of rows in ZX */
      ZXcols,   /* number of columns in ZX */
      n,        /* number of groups (Subjects) */
      p,        /* number of fixed effects */
      q,        /* number of random effects */
      *lengths, /* groups lengths */
      *offsets, /* groups offsets */
      *ZXlen,   /* lengths into ZX */
      *ZXoff;   /* offsets into ZX */
} DIMS_struct, *DIMS;

#endif /* HEAVY_BASE_H */
