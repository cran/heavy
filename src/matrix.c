#include "matrix.h"

/* basic matrix manipulations */

double
sum_abs(double *x, int n, int incx)
{   /* sum(abs(x)) */
    double ans;

    ans = F77_CALL(dasum)(&n, x, &incx);
    return ans;
}

double
norm_sqr(double *x, int n, int incx)
{   /* sum(x * x) */
    double ans;

    ans = F77_CALL(dnrm2)(&n, x, &incx);
    return R_pow_di(ans, 2);
}

double
dot_product(double *x, int incx, double *y, int incy, int n)
{   /* sum(x * y) */
    double ans;

    ans = F77_CALL(ddot)(&n, x, &incx, y, &incy);
    return ans;
}

void
scale_vec(double *x, int n, int inc, double alpha)
{   /* x <- alpha * x */

    F77_CALL(dscal)(&n, &alpha, x, &inc);
}

void
zero_mat(double *y, int ldy, int nrow, int ncol)
{   /* y[,] <- 0 */
    int i, j;

    for (j = 0; j < ncol; j++) {
        for (i = 0; i < nrow; i++)
            y[i] = 0.0;
        y += ldy;
    }
}

void
copy_mat(double *y, int ldy, double *x, int ldx, int nrow, int ncol)
{   /* y <- x[,] */
    int j;

    for (j = 0; j < ncol; j++) {
        Memcpy(y, x, nrow);
        y += ldy; x += ldx;
    }
}

void
add_mat(double *y, int ldy, double alpha, double *x, int ldx, int nrow, int ncol)
{   /* y <- y + alpha * x */
    int j, inc = 1;

    for (j = 0; j < ncol; j++) {
        F77_CALL(daxpy)(&nrow, &alpha, x, &inc, y, &inc);
        y += ldy; x += ldx;
    }
}

void
scale_mat(double *y, int ldy, double *x, int ldx, int nrow, int ncol, double alpha)
{   /* y <- alpha * x[,] */
    int i, j;

    for (j = 0; j < ncol; j++) {
        for (i = 0; i < nrow; i++)
            y[i] = alpha * x[i];
        y += ldy; x += ldx;
    }
}

void
gaxpy(double *y, double alpha, double *a, int lda, int nrow, int ncol,
    double *x, double beta, int job)
{   /* y <- alpha * a %*% x + beta * y or y <- alpha * t(a) %*% x + beta * y */
    char *trans;
    int inc = 1;

    trans = (job) ? "T" : "N";
    F77_CALL(dgemv)(trans, &nrow, &ncol, &alpha, a, &lda, x, &inc, &beta, y, &inc);
}

void
lower_tri(double *y, int ldy, double *x, int ldx, int nrow, int ncol)
{   /* y <- lower.tri(x) */
    int j, cols;

    cols = MIN(nrow, ncol);
    for (j = 0; j < cols; j++)
        Memcpy(y + j * (ldy + 1), x + j * (ldx + 1), nrow - j);
}

void
upper_tri(double *y, int ldy, double *x, int ldx, int nrow, int ncol)
{   /* y <- upper.tri(x) */
    int j, rows;

    for (j = 0; j < ncol; j++) {
        rows = MIN(j + 1, nrow);
        Memcpy(y + j * ldy, x + j * ldx, rows);
    }
}

void
upper_mult_vec(double *a, int lda, int nrow, int ncol, double *x, double *y)
{   /* y <- upper(a) %*% x */
    int j, inc = 1, rows;

    for (j = 0; j < ncol; j++) {
        rows = MIN(j + 1, nrow);
        F77_CALL(daxpy)(&rows, x, a, &inc, y, &inc);
        x++; a += lda;
    }
}

void
mult_mat(double *x, int ldx, int xrows, int xcols, double *y, int ldy, int yrows,
    int ycols, double *z)
{   /* matrix multiplication of two conformable matrices. z <- x %*% y */
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
crossprod(double *x, int ldx, int xrows, int xcols, double *y, int ldy, int yrows,
    int ycols, double *z)
{   /* cross product of two given matrices. z <- t(x) %*% y */
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
outerprod(double *x, int ldx, int xrows, int xcols, double *y, int ldy, int yrows,
    int ycols, double *z)
{   /* outer product of two given matrices. z <- x %*% t(y) */
    char *transx = "N", *transy = "T";
    double one = 1.0, zero = 0.0;

    F77_CALL(dgemm)(transx, transy, &xrows, &yrows, &xcols, &one, x, &ldx, y,
                    &ldy, &zero, z, &xrows);
}

void
rank1_update(double *a, int lda, int nrow, int ncol, double alpha, double *x,
    double *y)
{   /* rank 1 operation a <- alpha * x %*% t(y) + a */
    int inc = 1;

    F77_CALL(dger)(&nrow, &ncol, &alpha, x, &inc, y, &inc, a, &lda);
}

double 
logAbsDet(double *a, int lda, int n)
{   /* log(abs(det(upper triangle))) */
    int i;
    double accum = 0.0;

    for (i = 0; i < n; i++) 
        accum += log(fabs(a[i * (lda + 1)]));
    return accum;
}

/* routines for matrix decompositions */

void
chol_decomp(double *a, int lda, int p, int job, int *info)
{   /* cholesky factorization of a real symmetric positive definite matrix a.
     * the factorization has the form: a <- l  %*% t(l), if job = 0, or
     * a <- t(u) %*% u, if job = 1, where u is an upper triangular matrix and
     * l is lower triangular. */
    char *uplo;
    
    uplo = (job) ? "U" : "L";
    F77_CALL(dpotrf)(uplo, &p, a, &lda, info);
}

void
svd_decomp(double *mat, int ldmat, int nrow, int ncol, double *u, int ldu, double *d,
           double *v, int ldv, int job, int *info)
{   /* return the SVD decomposition of mat. */
    double *work, *upper;

    work  = (double *) Calloc(nrow, double);
    upper = (double *) Calloc(ncol, double);    
    F77_CALL(dsvdc)(mat, &ldmat, &nrow, &ncol, d, upper, u, &ldu, v, &ldv,
                    work, &job, info);
    Free(work); Free(upper);
}

QRStruct
QR_decomp(double *mat, int ldmat, int nrow, int ncol, double *qraux)
{   /* return the QR decomposition of mat */
    int *dummy = NULL, job = 0;
    double *work;
    QRStruct value;

    value = (QRStruct) Calloc(1, QR_struct);
    work  = (double *) Calloc(ncol, double);
    value->mat = mat;
    value->ldmat = ldmat;
    value->nrow  = nrow;
    value->ncol  = ncol;
    value->qraux = qraux;
    F77_CALL(dqrdc)(mat, &ldmat, &nrow, &ncol, qraux, dummy, work, &job);
    Free(work);
    return value;
}

void
QR_free(QRStruct this)
{   /* destructor for a QR object */
    Free(this);
}

/* orthogonal-triangular operations */

void
QR_qty(QRStruct this, double *ymat, int yrow, int ycol, double *qty)
{   /* qty <- qr.qty(this, ymat) */
    int j, job = 1000, info = 0;
    double *dummy = NULL;

    for (j = 0; j < ycol; j++) {
        F77_CALL(dqrsl)(this->mat, &(this->ldmat), &(this->nrow), &(this->ncol),
                        this->qraux, ymat + j * yrow, dummy, qty + j * yrow,
                        dummy, dummy, dummy, &job, &info);
    }
}

void
QR_qy(QRStruct this, double *ymat, int yrow, int ycol, double *qy)
{   /* qy <- qr.qy(this, ymat) */
    int j, job = 10000, info = 0;
    double *dummy = NULL;

    for (j = 0; j < ycol; j++) {
        F77_CALL(dqrsl)(this->mat, &(this->ldmat), &(this->nrow), &(this->ncol),
                        this->qraux, ymat + j * yrow, qy + j * yrow, dummy,
                        dummy, dummy, dummy, &job, &info);
    }
}

void
QR_coef(QRStruct this, double *ymat, int yrow, int ycol, double *coef)
{   /* coef <- qr.coef(this, ymat) */
    int j, job = 100, info = 0;
    double *dummy = NULL;

    for (j = 0; j < ycol; j++) {
        F77_CALL(dqrsl)(this->mat, &(this->ldmat), &(this->nrow), &(this->ncol),
                        this->qraux, ymat + j * yrow, dummy, ymat + j * yrow,
                        coef + j * this->ncol, dummy, dummy, &job, &info);
        if (info)
            error("DQRSL in QR_coef gave code %d", info);
    }
}

void
QR_resid(QRStruct this, double *ymat, int yrow, int ycol, double *resid)
{   /* resid <- qr.resid(this, ymat) */
    int j, job = 10, info = 0;
    double *dummy = NULL;
    
    for (j = 0; j < ycol; j++) {
        F77_CALL(dqrsl)(this->mat, &(this->ldmat), &(this->nrow), &(this->ncol),
                        this->qraux, ymat + j * yrow, dummy, ymat + j * yrow,
                        dummy, resid + j * yrow, dummy, &job, &info);
    }
}

void
QR_fitted(QRStruct this, double *ymat, int yrow, int ycol, double *fitted)
{   /* fitted <- qr.fitted(this, ymat) */
    int j, job = 1, info = 0;
    double *dummy = NULL;

    for (j = 0; j < ycol; j++) {
        F77_CALL(dqrsl)(this->mat, &(this->ldmat), &(this->nrow), &(this->ncol),
                        this->qraux, ymat + j * yrow, dummy, ymat + j * yrow,
                        dummy, dummy, fitted + j * yrow, &job, &info);
    }
}

void
QR_store_R(QRStruct this, double *Dest, int ldDest)
{   /* copy the R part into Dest */
    int j, rows;

    for (j = 0; j < this->ncol; j++) {
        rows = MIN(j + 1, this->nrow);
        Memcpy(Dest + j * ldDest, this->mat + j * this->ldmat, rows);
    }
}

/* matrix inversion and solver */

void
invert_mat(double *a, int lda, int n, int *info)
{   /* performs matrix inversion */
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
{   /* computes the inverse of an upper (job = 1) or lower (job = 0) triangular
     * matrix in place */
    char *diag = "N", *uplo;

    uplo = (job) ? "U" : "L";
    F77_CALL(dtrtri)(uplo, diag, &n, a, &lda, info);
}

void
backsolve(double *r, int ldr, int n, double *b, int ldb, int nrhs, int job, int *info)
{   /* backsolve solve triangular systems of the form r %*% x = b, or t(r) %*% x = b,
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
lsfit(double *x, int ldx, int nrow, int ncol, double *y, int ldy, int nrhs,
    double *coef, int *info)
{   /* solve (overdeterminated) least squares problems */
    char *notrans = "N";
    int lwork;
    double *work;

    lwork = ncol + MAX(ncol, nrhs);
    work  = (double *) Calloc(lwork, double);
    F77_CALL(dgels)(notrans, &nrow, &ncol, &nrhs, x, &ldx, y, &ldy, work, &lwork, info);
    copy_mat(coef, ncol, y, ldy, ncol, nrhs);
    Free(work);
}
