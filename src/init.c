#include <R_ext/Rdynload.h>
#include "lm_fit.h"
#include "l1_fit.h"
#include "lme_fit.h"

static const R_CMethodDef CEntries[]  = {
    {"heavyLm_fit",     (DL_FUNC) &heavyLm_fit, 13},
    {"heavyLme_fit",    (DL_FUNC) &heavyLme_fit, 16},
    {"heavyLme_fitted", (DL_FUNC) &heavyLme_fitted, 8},
    {"heavyLme_acov",   (DL_FUNC) &heavyLme_acov, 9},
    {NULL, NULL, 0}
};

static const R_FortranMethodDef FortEntries[] = {
    {"l1fit",   (DL_FUNC) &F77_NAME(l1fit), 10},
    {NULL, NULL, 0}
};

void R_init_heavy(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, NULL, FortEntries, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
