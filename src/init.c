#include "lme_fit.h"

static const R_CMethodDef CEntries[]  = {
    {"heavy_lme_fit", (DL_FUNC) &heavy_lme_fit, 12},
    {"heavy_lme_resid", (DL_FUNC) &heavy_lme_resid, 7},
    {NULL, NULL, 0}
};

void R_init_heavy(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
