#include "matrix.h"

/* basic matrix manipulations */

double
dot_product(double *x, int incx, double *y, int incy, int n)
{ /* sum(x * y) */
  double ans;

  ans = F77_CALL(ddot)(&n, x, &incx, y, &incy);
  return ans;
}

double
norm_sqr(double *x, int inc, int n)
{ /* sum(x * x) */
  double ans;

  ans = F77_CALL(dnrm2)(&n, x, &inc);
  return SQR(ans);
}

double
sum_abs(double *x, int inc, int n)
{ /* sum(abs(x)) */
  double ans;

  ans = F77_CALL(dasum)(&n, x, &inc);
  return ans;
}

void
ax_plus_y(double alpha, double *x, int incx, double *y, int incy, int n)
{ /* y <- alpha * x + y */
  F77_CALL(daxpy)(&n, &alpha, x, &incx, y, &incy);
}

void
copy_vec(double *y, int incy, double *x, int incx, int n)
{ /* y <- x */
  F77_CALL(dcopy)(&n, x, &incx, y, &incy);
}

void
scale_vec(double alpha, double *x, int inc, int n)
{ /* x <- alpha * x */
  F77_CALL(dscal)(&n, &alpha, x, &inc);
}

void
swap_vec(double *x, int incx, double *y, int incy, int n)
{ /* x <-> y */
  F77_CALL(dswap)(&n, x, &incx, y, &incy);
}

void
zero_mat(double *y, int ldy, int nrow, int ncol)
{ /* y[,] <- 0 */
  int i, j;

  for (j = 0; j < ncol; j++) {
    for (i = 0; i < nrow; i++)
      y[i] = 0.0;
    y += ldy;
  }
}

void
copy_mat(double *y, int ldy, double *x, int ldx, int nrow, int ncol)
{ /* y <- x[,] */
  int j;

  for (j = 0; j < ncol; j++) {
    Memcpy(y, x, nrow);
    y += ldy; x += ldx;
  }
}

void
add_mat(double *y, int ldy, double alpha, double *x, int ldx, int nrow, int ncol)
{ /* y <- alpha * x + y */
  int j, inc = 1;

  for (j = 0; j < ncol; j++) {
    ax_plus_y(alpha, x, inc, y, inc, nrow);
    y += ldy; x += ldx;
  }
}

void
scale_mat(double *y, int ldy, double *x, int ldx, int nrow, int ncol, double alpha)
{ /* y <- alpha * x[,] */
  int i, j;

  for (j = 0; j < ncol; j++) {
    for (i = 0; i < nrow; i++)
      y[i] = alpha * x[i];
    y += ldy; x += ldx;
  }
}

void
GE_axpy(double *y, double alpha, double *a, int lda, int nrow, int ncol, double *x, double beta, int job)
{ /* y <- alpha * a %*% x    + beta * y (job = 0), or
   * y <- alpha * t(a) %*% x + beta * y (job = 1) */
  char *trans;
  int inc = 1;

  trans = (job) ? "T" : "N";
  F77_CALL(dgemv)(trans, &nrow, &ncol, &alpha, a, &lda, x, &inc, &beta, y, &inc);
}

void
lower_tri(double *y, int ldy, double *x, int ldx, int nrow, int ncol)
{ /* y <- lower.tri(x) */
  int j, cols;

  cols = MIN(nrow, ncol);
  for (j = 0; j < cols; j++)
    Memcpy(y + j * (ldy + 1), x + j * (ldx + 1), nrow - j);
}

void
upper_tri(double *y, int ldy, double *x, int ldx, int nrow, int ncol)
{ /* y <- upper.tri(x) */
  int j, rows;

  for (j = 0; j < ncol; j++) {
    rows = MIN(j + 1, nrow);
    Memcpy(y + j * ldy, x + j * ldx, rows);
  }
}

void
triangle_mult_vec(double *y, double *a, int lda, int n, double *x, int job)
{ /* y <- lower.tri(a) %*% x (job = 0), or
   * y <- upper.tri(a) %*% x (job = 1) */
  char *uplo, *trans = "N", *diag = "N";
  int inc = 1;

  uplo = (job) ? "U" : "L";
  Memcpy(y, x, n);
  F77_CALL(dtrmv)(uplo, trans, diag, &n, a, &lda, y, &inc);
}

void
mult_mat(double *z, double *x, int ldx, int xrows, int xcols, double *y, int ldy, int yrows, int ycols)
{ /* matrix multiplication of two conformable matrices. z <- x %*% y */
  char *transx = "N", *transy = "N";
  double one = 1.0, zero = 0.0, *tmp = NULL;

  /* use tmp so z can be either x or y */
  tmp = (double *) Calloc(xrows * ycols, double);
  F77_CALL(dgemm)(transx, transy, &xrows, &ycols, &xcols, &one, x, &ldx, y,
                  &ldy, &zero, tmp, &xrows);
  Memcpy(z, tmp, xrows * ycols);
  Free(tmp);
}

void
crossprod(double *z, double *x, int ldx, int xrows, int xcols, double *y, int ldy, int yrows, int ycols)
{ /* cross product of two given matrices. z <- t(x) %*% y */
  char *transx = "T", *transy = "N";
  double one = 1.0, zero = 0.0, *tmp = NULL;

  /* use tmp so z can be either x or y */
  tmp = (double *) Calloc(xcols * ycols, double);
  F77_CALL(dgemm)(transx, transy, &xcols, &ycols, &xrows, &one, x, &ldx, y,
                  &ldy, &zero, tmp, &xcols);
  Memcpy(z, tmp, xcols * ycols);
  Free(tmp);
}

void
outerprod(double *z, double *x, int ldx, int xrows, int xcols, double *y, int ldy, int yrows, int ycols)
{ /* outer product of two given matrices. z <- x %*% t(y) */
  char *transx = "N", *transy = "T";
  double one = 1.0, zero = 0.0, *tmp = NULL;

  /* use tmp so z can be either x or y */
  tmp = (double *) Calloc(xrows * yrows, double);
  F77_CALL(dgemm)(transx, transy, &xrows, &yrows, &xcols, &one, x, &ldx, y,
                  &ldy, &zero, tmp, &xrows);
  Memcpy(z, tmp, xrows * yrows);
  Free(tmp);
}

void
rank1_update(double *a, int lda, int nrow, int ncol, double alpha, double *x, double *y)
{ /* rank 1 operation: a <- alpha * x %*% t(y) + a */
  int inc = 1;

  F77_CALL(dger)(&nrow, &ncol, &alpha, x, &inc, y, &inc, a, &lda);
}

void
rank1_symm_update(double *a, int lda, int n, double alpha, double *x, int job)
{ /* a <- alpha * x %*% t(x) + a, only lower part of a is referenced (job = 0), or
   * a <- alpha * x %*% t(x) + a, only upper part of a is referenced (job = 1) */
  char *uplo;
  int inc = 1;

  uplo = (job) ? "U" : "L";
  F77_CALL(dsyr)(uplo, &n, &alpha, x, &inc, a, &lda);
}

double
logAbsDet(double *a, int lda, int n)
{ /* log(abs(det(upper triangle))) */
  int i;
  double accum = 0.0;

  for (i = 0; i < n; i++)
    accum += log(fabs(a[i * (lda + 1)]));
  return accum;
}

/* DEBUG routine */

void
print_mat(char *msg, double *x, int ldx, int nrow, int ncol)
{ /* print matrix and message (used for printf debugging) */
  int i, j;

  Rprintf( "%s\n", msg);
  for (i = 0; i < nrow; i++) {
    for (j = 0; j < ncol; j++)
      Rprintf( " %10.5g", x[i + j * ldx ]);
    Rprintf( "\n" );
  }
  Rprintf( "\n" );
}

/* routines for matrix decompositions */

void
chol_decomp(double *a, int lda, int p, int job, int *info)
{ /* cholesky factorization of a real symmetric positive definite matrix a.
   * the factorization has the form:
   * a <- l %*% t(l), if job = 0, or
   * a <- t(u) %*% u, if job = 1,
   * where u is an upper triangular matrix and l is lower triangular. */
  char *uplo;

  uplo = (job) ? "U" : "L";
  F77_CALL(dpotrf)(uplo, &p, a, &lda, info);
}

void
svd_decomp(double *mat, int ldmat, int nrow, int ncol, double *u, int ldu, double *d, double *v, int ldv, int job, int *info)
{ /* return the SVD decomposition of mat. */
  double *work, *upper;

  work  = (double *) Calloc(nrow, double);
  upper = (double *) Calloc(ncol, double);
  F77_CALL(dsvdc)(mat, &ldmat, &nrow, &ncol, d, upper, u, &ldu, v, &ldv, work, &job, info);
  Free(work); Free(upper);
}

QRStruct
QR_decomp(double *mat, int ldmat, int nrow, int ncol, double *qraux, int *info)
{ /* return the QR decomposition of mat */
  double *work;
  QRStruct value;

  value = (QRStruct) Calloc(1, QR_struct);
  work  = (double *) Calloc(ncol, double);
  value->mat = mat;
  value->ldmat = ldmat;
  value->nrow  = nrow;
  value->ncol  = ncol;
  value->qraux = qraux;
  F77_CALL(dgeqr2)(&nrow, &ncol, mat, &ldmat, qraux, work, info);
  Free(work);
  return value;
}

void
QR_free(QRStruct this)
{ /* destructor for a QR object */
  Free(this);
}

LQStruct
LQ_decomp(double *mat, int ldmat, int nrow, int ncol, double *lqaux, int *info)
{ /* return the LQ decomposition of mat */
  double *work;
  LQStruct value;

  value = (LQStruct) Calloc(1, LQ_struct);
  work  = (double *) Calloc(nrow, double);
  value->mat   = mat;
  value->ldmat = ldmat;
  value->nrow  = nrow;
  value->ncol  = ncol;
  value->lqaux = lqaux;
  F77_CALL(dgelq2)(&nrow, &ncol, mat, &ldmat, lqaux, work, info);
  Free(work);
  return value;
}

void
LQ_free(LQStruct this)
{ /* destructor for a LQ object */
  Free(this);
}

/* orthogonal-triangular operations */

void
QR_qty(QRStruct this, double *ymat, int ldy, int yrow, int ycol)
{ /* ymat <- qr.qty(this, ymat) */
  int info = 0, lwork, nrflc;
  char *side = "L", *trans = "T";
  double *work;

  nrflc = MIN(this->nrow, this->ncol);
  lwork = MAX(ycol, this->ncol);
  work  = (double *) Calloc(lwork, double);
  F77_CALL(dormqr)(side, trans, &yrow, &ycol, &nrflc, this->mat, &(this->ldmat),
                   this->qraux, ymat, &ldy, work, &lwork, &info);
  Free(work);
  if (info)
    error("DORMQR in QR_qty gave code %d", info);
}

void
QR_qy(QRStruct this, double *ymat, int ldy, int yrow, int ycol)
{ /* ymat <- qr.qy(this, ymat) */
  int info = 0, lwork, nrflc;
  char *side = "L", *notrans = "N";
  double *work;

  nrflc = MIN(this->nrow, this->ncol);
  lwork = MAX(ycol, this->ncol);
  work  = (double *) Calloc(lwork, double);
  F77_CALL(dormqr)(side, notrans, &yrow, &ycol, &nrflc, this->mat, &(this->ldmat),
                   this->qraux, ymat, &ldy, work, &lwork, &info);
  Free(work);
  if (info)
    error("DORMQR in QR_qy gave code %d", info);
}

void
QR_store_R(QRStruct this, double *Dest, int ldDest)
{ /* copy the R part into Dest */
  int j, rows;

  for (j = 0; j < this->ncol; j++) {
    rows = MIN(j + 1, this->nrow);
    Memcpy(Dest + j * ldDest, this->mat + j * this->ldmat, rows);
  }
}

void
LQ_yqt(LQStruct this, double *ymat, int ldy, int yrow, int ycol)
{ /* ymat <- lq.yqt(this, ymat) */
  int info = 0, lwork, nrflc;
  char *side = "R", *trans = "T";
  double *work;

  nrflc = MIN(this->nrow, this->ncol);
  lwork = MAX(yrow, this->nrow);
  work  = (double *) Calloc(lwork, double);
  F77_CALL(dormlq)(side, trans, &yrow, &ycol, &nrflc, this->mat, &(this->ldmat),
                   this->lqaux, ymat, &ldy, work, &lwork, &info);
  Free(work);
  if (info)
    error("DORMLQ in LQ_yqt gave code %d", info);
}

void
LQ_yq(LQStruct this, double *ymat, int ldy, int yrow, int ycol)
{ /* ymat <- lq.yqt(this, ymat) */
  int info = 0, lwork, nrflc;
  char *side = "R", *notrans = "N";
  double *work;

  nrflc = MIN(this->nrow, this->ncol);
  lwork = MAX(yrow, this->nrow);
  work  = (double *) Calloc(lwork, double);
  F77_CALL(dormlq)(side, notrans, &yrow, &ycol, &nrflc, this->mat, &(this->ldmat),
                   this->lqaux, ymat, &ldy, work, &lwork, &info);
  Free(work);
  if (info)
    error("DORMLQ in LQ_yqt gave code %d", info);
}

void
LQ_store_L(LQStruct this, double *Dest, int ldDest)
{ /* copy the L part into Dest */
  int j, cols;

  cols = MIN(this->nrow, this->ncol);
  for (j = 0; j < cols; j++)
    Memcpy(Dest + j * (ldDest + 1), this->mat + j * (this->ldmat + 1), this->nrow - j);
}

/* matrix inversion and linear solver */

void
invert_mat(double *a, int lda, int n, int *info)
{ /* performs matrix inversion */
  int j, lwork = 2 * n;
  char *trans = "N";
  double *b, *work;

  b    = (double *) Calloc(n * n, double);
  work = (double *) Calloc(lwork, double);
  for (j = 0; j < n; j++)
    b[j * (n + 1)] = 1.;
  F77_CALL(dgels)(trans, &n, &n, &n, a, &lda, b, &n, work, &lwork, info);
  Memcpy(a, b, n * n);
  Free(b); Free(work);
}

void
invert_triangular(double *a, int lda, int n, int job, int *info)
{ /* computes the inverse of an upper (job = 1) or lower (job = 0) triangular
   * matrix in place */
  char *diag = "N", *uplo;

  uplo = (job) ? "U" : "L";
  F77_CALL(dtrtri)(uplo, diag, &n, a, &lda, info);
}

void
backsolve(double *r, int ldr, int n, double *b, int ldb, int nrhs, int job, int *info)
{ /* backsolve solve triangular systems of the form r %*% x = b, or t(r) %*% x = b,
   * where r is a triangular and b is a matrix containing the right-hand sides to
   * equations. job specifies what kind of system is to be solved: job = 00, solve
   * r %*% x = b, r lower triangular, job = 01, solve r %*% x = b, r upper triangular,
   * job = 10, solve t(r) %*% x = b, r lower triangular, job = 11, solve t(r) %*% x = b,
   * r upper triangular. */
  char *diag = "N", *uplo, *trans;

  trans = ((job) / 10) ? "T" : "N";
  uplo  = ((job) % 10) ? "U" : "L";
  F77_CALL(dtrtrs)(uplo, trans, diag, &n, &nrhs, r, &ldr, b, &ldb, info);
}

/* linear least-squares fit */

void
lsfit(double *x, int ldx, int nrow, int ncol, double *y, int ldy, int nrhs, double *coef, int *info)
{ /* solve (overdeterminated) least squares problems */
  char *notrans = "N";
  int lwork;
  double *work;

  lwork = ncol + MAX(ncol, nrhs);
  work  = (double *) Calloc(lwork, double);
  F77_CALL(dgels)(notrans, &nrow, &ncol, &nrhs, x, &ldx, y, &ldy, work, &lwork, info);
  copy_mat(coef, ncol, y, ldy, ncol, nrhs);
  Free(work);
}
