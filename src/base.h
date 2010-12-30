#ifndef HEAVY_BASE_H
#define HEAVY_BASE_H

#include <R.h>
#include <Rmath.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

/* some definitions */
#define NULLP    (void *) 0
#define MAX(a,b) (((a)>(b)) ? (a) : (b))
#define MIN(a,b) (((a)<(b)) ? (a) : (b))
#define SQR(x)   R_pow_di(x, 2)
#define ETA_CONV 1.0e-2
#define repeat for(;;)

/* dims functions */
typedef enum {
    LM,
    LME
} mlclass;

/* dims structure */
typedef struct DIMS_struct {
    int
      N,        /* total number of observations */
      ZXrows,   /* number of rows in ZX */
      ZXcols,   /* number of columns in ZX */
      n,        /* number of groups (Subjects) */
      p,        /* number of fixed effects */
      q,        /* number of random effects */
      DcRows;   /* number of rows into decomposition */
} DIMS_struct, *DIMS;

/* routines for dealing with dims objects */
extern DIMS dims(mlclass, int *);
extern DIMS dims_LM(int *);
extern DIMS dims_LME(int *);
extern void dims_free(DIMS);

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

/* routines for dealing with lenghts objects */
extern LENGTHS setLengths(DIMS, int *, int *);
extern void lengths_free(LENGTHS);

/* QR structure */
typedef struct QR_struct {
    double *mat, *qraux;
    int ldmat, nrow, ncol;
} QR_struct, *QRStruct;

/* available families */
typedef enum {
    NORMAL,
    CAUCHY,
    STUDENT,
    SLASH,
    CONTAMINATED
} flclass;

/* heavy tailed family structure */
typedef struct FAMILY_struct {
    flclass fltype; /* family type */
    int npars;      /* number of parameters in 'family' */
    double *nu;     /* parameter vector */
} FAMILY_struct, *FAMILY;

#endif /* HEAVY_BASE_H */
