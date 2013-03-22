#ifndef HEAVY_BASE_H
#define HEAVY_BASE_H

#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>
#include <R_ext/Linpack.h>
#include <R_ext/Applic.h>

/* some definitions */
#define NULLP    (void *) 0
#define MAX(a,b) (((a)>(b)) ? (a) : (b))
#define MIN(a,b) (((a)<(b)) ? (a) : (b))
#define SQR(x)   R_pow_di(x, 2)
#define ETA_CONV 1.0e-2
#define repeat for(;;)

/* dims structure */
typedef struct DIMS_struct {
    int
      N,        /* total number of observations */
      ZXrows,   /* number of rows in ZX */
      ZXcols,   /* number of columns in ZX */
      n,        /* number of groups (Subjects) */
      p,        /* number of fixed effects */
      q,        /* number of random effects */
      deg,      /* degree of the B-spline */
      ord,      /* order of penalty */
      DcRows;   /* number of rows into decomposition */
} DIMS_struct, *DIMS;

/* lengths and offsets structure */
typedef struct LENGTHS_struct {
    int
      *glen,    /* groups lengths */
      *offsets, /* groups offsets */
      *ZXlen,   /* lengths into ZX */
      *ZXoff,   /* offsets into ZX */
      *DcLen,   /* lengths into decomposition */
      *DcOff;   /* offsets into decomposition */
} LENGTHS_struct, *LENGTHS;

/* QR structure */
typedef struct QR_struct {
    double *mat, *qraux;
    int ldmat, nrow, ncol;
} QR_struct, *QRStruct;

#endif /* HEAVY_BASE_H */
