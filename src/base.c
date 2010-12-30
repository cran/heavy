#include "base.h"

DIMS
dims(mlclass action, int *pdims)
{   /* constructor (dispatcher) for a dims object */
    DIMS ans;

    switch (action) {
        case LM:
            ans = dims_LM(pdims);
            break;
        case LME:
            ans = dims_LME(pdims);
            break;
        default:
            ans = dims_LM(pdims);
            break;
    }
    return ans;
}

DIMS
dims_LM(int *pdims)
{   /* dims object for linear models */
    DIMS ans;

    ans = (DIMS) Calloc(1, DIMS_struct);
    ans->N = (int) pdims[0];
    ans->n = ans->N;
    ans->p = (int) pdims[1];
    return ans;
}

DIMS
dims_LME(int *pdims)
{   /* dims object for LME models */
    DIMS ans;

    ans = (DIMS) Calloc(1, DIMS_struct);
    ans->n = (int) pdims[0];
    ans->q = (int) pdims[1];
    ans->p = (int) pdims[2];
    ans->N = (int) pdims[3];
    ans->ZXrows = (int) pdims[4];
    ans->ZXcols = (int) pdims[5];
    ans->DcRows = (int) pdims[6];
    return ans;
}

void
dims_free(DIMS this)
{   /* destructor for a dims object */
    Free(this);
}

LENGTHS
setLengths(DIMS dd, int *glen, int *DcLen)
{   /* constructor for a lenghts object */
    LENGTHS ans;
    int i, accum1 = 0, accum2 = 0, accum3 = 0;

    ans = (LENGTHS) Calloc(1, LENGTHS_struct);
    ans->glen  = glen;
    ans->DcLen = DcLen;
    ans->offsets = (int *) Calloc(dd->n, int);
    ans->ZXlen   = (int *) Calloc(dd->n, int);
    ans->ZXoff   = (int *) Calloc(dd->n, int);
    ans->DcOff   = (int *) Calloc(dd->n, int);
    (ans->ZXlen)[0] = (ans->glen)[0] * (dd->ZXcols + 1);
    for (i = 1; i < dd->n; i++) {
        (ans->ZXlen)[i] = (ans->glen)[i] * (dd->ZXcols + 1);
        accum1 += (ans->glen)[i-1];
        (ans->offsets)[i] = accum1;
        accum2 += (ans->ZXlen)[i-1];
        (ans->ZXoff)[i] = accum2;
        accum3 += (ans->DcLen)[i-1];
        (ans->DcOff)[i] = accum3;
    }
    return ans;
}

void
lengths_free(LENGTHS this)
{   /* destructor for a lenghts object */
    Free(this->offsets);
    Free(this->ZXlen);
    Free(this->ZXoff);
    Free(this->DcOff);
    Free(this);
}
