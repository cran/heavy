#include "matrix.h"

/* basic matrix manipulations */

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
gaxpy(double *y, double alpha, double *a, int lda, int nrow, int ncol, double *x, double beta)
{   /* y <- alpha * a %*% x + beta * y */
    char *transa = "N";
    int inc = 1;

    F77_CALL(dgemv)(transa, &nrow, &ncol, &alpha, a, &lda, x, &inc, &beta, y, &inc);
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
zero_lower(double *y, int ldy, int nrow, int ncol)
{   /* lower(y) <- 0 */
    int i, j;

    for (j = 0; j < ncol; j++) {
        for (i = j + 1; i < nrow; i++)
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

double *
scale_mat(double *y, int ldy, double alpha, double *x, int ldx, int nrow, int ncol)
{   /* y <- alpha * x[,] */
    int i, j;
    double *ret = y;

    for (j = 0; j < ncol; j++) {
        for (i = 0; i < nrow; i++)
            y[i] = alpha * x[i];
        y += ldy; x += ldx;
    }
    return ret;
}

double *
scale_lower(double *y, int ldy, double alpha, double *x, int ldx, int nrow, int ncol)
{   /* lower(y) <- alpha * x[,] */
    int i, j;
    double *ret = y;

    for (j = 0; j < ncol; j++) {
        for (i = j; i < nrow; i++)
            y[i] = alpha * x[i];
        y += ldy; x += ldx;
    }
    return ret;
}

void
mult_mat(double *x, int ldx, int xrows, int xcols, double *y, int ldy, int yrows, int ycols, double *z)
{   /* matrix multiplication of two conformable matrices. z <- x %*% y */
    char *transx = "N", *transy = "N";
    double one = 1.0, zero = 0.0, *tmp = NULL;

    /* use tmp so z can be either x or y */
    tmp = (double *) Calloc(xrows * ycols, double);
    F77_CALL(dgemm)(transx, transy, &xrows, &ycols, &xcols, &one, x, &ldx, y, &ldy, &zero, tmp, &xrows);
    Memcpy(z, tmp, xrows * ycols);
    Free(tmp);
}

void
crossprod(double *x, int ldx, int xrows, int xcols, double *y, int ldy, int yrows, int ycols, double *z)
{   /* cross product of two given matrices. z <- t(x) %*% y */
    char *transx = "T", *transy = "N";
    double one = 1.0, zero = 0.0;

    F77_CALL(dgemm)(transx, transy, &xcols, &ycols, &xrows, &one, x, &ldx, y, &ldy, &zero, z, &xcols);
}

void
outerprod(double *x, int ldx, int xrows, int xcols, double *y, int ldy, int yrows, int ycols, double *z)
{   /* outer product of two given matrices. z <- x %*% t(y) */
    char *transx = "N", *transy = "T";
    double one = 1.0, zero = 0.0;

    F77_CALL(dgemm)(transx, transy, &xrows, &yrows, &xcols, &one, x, &ldx, y, &ldy, &zero, z, &xrows);
}

void
rank1_update(double *a, int lda, int nrow, int ncol, double alpha, double *x, double *y)
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

/* kronecker product */

void
kronecker_IA(double *z, int ldz, int n, double *a, int lda, int nrow, int ncol)
{   /* y <- kronecker(In, A) */
    int j;

    for (j = 0; j < n; j++) {
        copy_mat(z, ldz, a, lda, nrow, ncol);
        z += ldz * ncol + nrow;
    }
}

void
kronecker_AI(double *z, int ldz, int n, double *a, int lda, int nrow, int ncol)
{   /* y <- kronecker(A, In) */
    int i, j, k;

    for (j = 0; j < ncol; j++) {
        for (i = 0; i < nrow; i++) {
            double alpha = a[i + j * lda];
            for (k = 0; k < n; k++)
                z[k * (ldz + 1)] = alpha;
            z += n;
        }
        z += ldz * (n - 1);
    }
}

void
kronecker_prod(double *z, int ldz, double *x, int ldx, int xrow, int xcol, double *y, int ldy, int yrow, int ycol)
{   /* FIXME: current version is based on kronecker definition */
    int i, j;

    for (j = 0; j < xcol; j++) {
        for (i = 0; i < xrow; i++) {
            double alpha = x[i + j * ldx];
            scale_mat(z, ldz, alpha, y, ldy, yrow, ycol);
            z += yrow;
        }
        z += ldz * (ycol - 1);
    }
}

/* routines for matrix decompositions */

void
chol_decomp(int job, double *a, int lda, int n)
{   /* cholesky factorization of a real symmetric positive definite matrix a. the
     * factorization has the form: a <- l  %*% t(l), if job = 0, or a <- t(u) %*% u,
     * if job = 1, where u is an upper triangular matrix and l is lower triangular. */
    int info = 0;
    char *uplo;

    uplo = (job) ? "U" : "L";
    F77_CALL(dpotf2)(uplo, &n, a, &lda, &info);
    if (info)
        error("DPOTF2 in cholesky decomposition gave code %d", info);
}

QRStruct
QR_decomp(double *mat, int ldmat, int nrow, int ncol)
{   /* return the QR decomposition of mat */
    int job = 0, *pivot = (int *) NULLP;
    double *work = (double *) NULLP;
    QRStruct value;

    value = (QRStruct) Calloc(1, QR_struct);
    value->mat = mat;
    value->ldmat = ldmat;
    value->nrow  = nrow;
    value->ncol  = ncol;
    value->qraux = (double *) Calloc(ncol, double);
    F77_CALL(dqrdc)(mat, &ldmat, &nrow, &ncol, value->qraux, pivot, work, &job);
    return value;
}

void
QR_free(QRStruct this)
{   /* destructor for a QR object */
    Free(this->qraux);
    Free(this);
}

EIGENStruct
eigen_decomp(double *mat, int ldmat, int n, double *values, double *vectors)
{   /* return the eigen decomposition of mat */
    int job = 1;
    EIGENStruct value;

    value = (EIGENStruct) Calloc(1, eigen_struct);
    value->mat = mat;
    value->ldmat = ldmat;
    value->n = n;
    value->ierr = 0;
    value->values = values;
    value->vectors = vectors;
    value->work1 = (double *) Calloc(n, double);
    value->work2 = (double *) Calloc(n, double);
    F77_CALL(rs)(&ldmat, &n, mat, value->values, &job, value->vectors, value->work1, value->work2, &(value->ierr));
    return value;
}

void
eigen_free(EIGENStruct this)
{   /* destructor for an EIGEN object */
    Free(this->work1);
    Free(this->work2);
    Free(this);
}

/* orthogonal-triangular operations */

void
QR_qty(QRStruct this, double *ymat, int yrow, int ycol)
{   /* ymat <- qr.qty(this, ymat) */
    double *qty;

    qty = (double *) Calloc(yrow * ycol, double);
    F77_CALL(dqrqty)(this->mat, &(this->nrow), &(this->ncol), this->qraux, ymat, &ycol, qty);
    copy_mat(ymat, yrow, qty, yrow, yrow, ycol);
    Free(qty);
}

void
QR_qy(QRStruct this, double *ymat, int yrow, int ycol)
{   /* ymat <- qr.qy(this, ymat) */
    double *qy;

    qy = (double *) Calloc(yrow * ycol, double);
    F77_CALL(dqrqy)(this->mat, &(this->nrow), &(this->ncol), this->qraux, ymat, &ycol, qy);
    copy_mat(ymat, yrow, qy, yrow, yrow, ycol);
    Free(qy);
}

void
QR_coef(QRStruct this, double *ymat, int yrow, int ycol, double *coef)
{   /* coef <- qr.coef(this, ymat) */
    int info = 0;

    F77_CALL(dqrcf)(this->mat, &(this->nrow), &(this->ncol), this->qraux, ymat, &ycol, coef, &info);
    if (info)
        error("DQRSL in QR_coef gave code %d", info);
}

void
QR_resid(QRStruct this, double *ymat, int yrow, int ycol, double *resid)
{   /* resid <- qr.resid(this, ymat) */
    F77_CALL(dqrrsd)(this->mat, &(this->nrow), &(this->ncol), this->qraux, ymat, &ycol, resid);
}

void
QR_fitted(QRStruct this, double *ymat, int yrow, int ycol, double *fitted)
{   /* fitted <- qr.fitted(this, ymat) */
    F77_CALL(dqrxb)(this->mat, &(this->nrow), &(this->ncol), this->qraux, ymat, &ycol, fitted);
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
invert_mat(double *a, int lda, int n)
{   /* performs matrix inversion */
    int i, job = 1;
    double *b;
    QRStruct qr;
    
    b = (double *) Calloc(n * n, double);
    for (i = 0; i < n; i++)
        b[i * (n + 1)] = 1.0;
    qr = QR_decomp(a, lda, n, n);
    QR_qty(qr, b, n, n);
    backsolve(job, a, lda, n, b, n, n);
    copy_mat(a, lda, b, n, n, n);
    QR_free(qr); Free(b);
}

void
invert_lower(double *a, int lda, int n)
{   /* invert an lower triangular matrix in place */
    char *diag = "N", *uplo = "L";
    int info = 0;

    F77_CALL(dtrti2)(uplo, diag, &n, a, &lda, &info);
    if (info)
        error("DTRTI2 in computation of matrix inverse gave code %d", info);
}

void
invert_upper(double *a, int lda, int n)
{   /* invert an upper triangular matrix in place */
    char *diag = "N", *uplo = "U";
    int info = 0;

    F77_CALL(dtrti2)(uplo, diag, &n, a, &lda, &info);
    if (info)
        error("DTRTI2 in computation of matrix inverse gave code %d", info);
}

void
backsolve(int job, double *r, int ldr, int n, double *x, int ldx, int nrhs)
{   /* backsolve solve triangular systems of the form r %*% x = b, or t(r) %*% x = b, where r
     * is a triangular and x is a matrix containing the right-hand sides to equations. job
     * specifies what kind of system is to be solved: job = 00, solve r %*% x = b, r lower
     * triangular, job = 01, solve r %*% x = b, r upper triangular, job = 10, solve t(r) %*% x = b,
     * r lower triangular, job = 11, solve t(r) %*% x = b, r upper triangular. */
    char *diag = "N", *uplo, *trans;
    int info = 0;

    trans = ((job) / 10) ? "T" : "N";
    uplo  = ((job) % 10) ? "U" : "L";
    F77_CALL(dtrtrs)(uplo, trans, diag, &n, &nrhs, r, &ldr, x, &ldx, &info);
    if (info)
        error("DTRTRS in backsolve gave code %d", info);
}

/* QR decomposition used in heavy_lme */

void
append_decomp(double *mat, int ldmat, int nrow, int ncol, double *Delta, int q, double *store, int ldstr)
{   /* apply a QR decomposition to the augmented matrix rbind(mat, Delta) */
    int arow = nrow + q;
    double *aug;
    QRStruct qr;

    aug = (double *) Calloc(arow * ncol, double);
    copy_mat(aug, arow, mat, ldmat, nrow, ncol);
    copy_mat(aug + nrow, arow, Delta, q, q, q);
    qr = QR_decomp(aug, arow, arow, ncol);
    QR_store_R(qr, store, ldstr);
    QR_free(qr); Free(aug);
}
