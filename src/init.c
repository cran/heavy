#include <R_ext/Rdynload.h>
#include "lm_fit.h"
#include "lme_fit.h"
#include "ps_fit.h"
#include "random.h"

static const R_CMethodDef CEntries[]  = {
    {"lm_fit",              (DL_FUNC) &lm_fit,              13},
    {"lme_fit",             (DL_FUNC) &lme_fit,             16},
    {"lme_fitted",          (DL_FUNC) &lme_fitted,          8},
    {"lme_acov",            (DL_FUNC) &lme_acov,            9},
    {"ps_fit",              (DL_FUNC) &ps_fit,              17},
    {"ps_combined",         (DL_FUNC) &ps_combined,         17},
    {"rand_sphere",         (DL_FUNC) &rand_sphere,         2},
    {"rand_norm",           (DL_FUNC) &rand_norm,           4},
    {"rand_cauchy",         (DL_FUNC) &rand_cauchy,         4},
    {"rand_student",        (DL_FUNC) &rand_student,        5},
    {"rand_slash",          (DL_FUNC) &rand_slash,          5},
    {"rand_contaminated",   (DL_FUNC) &rand_contaminated,   6},
    {NULL, NULL, 0}
};

void R_init_heavy(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
