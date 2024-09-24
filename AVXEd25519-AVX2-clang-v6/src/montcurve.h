#ifndef _MONCURVE_H
#define _MONCURVE_H

#include "gfparith.h"
#include "utils.h"
#include <stdio.h>
// projective point [x, y, z]
typedef struct projective_point
{
    __m256i x[NWORDS];
    __m256i y[NWORDS];
    __m256i z[NWORDS];
} ProPoint;

// affine point [x, y]
typedef struct affine_point
{
    __m256i x[NWORDS];
    __m256i y[NWORDS];
} AffPoint;


void mon_ladder_step(ProPoint *p, ProPoint *q, const __m256i *xd);
// void mon_mul_varbase(uint32_t *r, const uint32_t *k, const uint32_t *x);
void mon_mul_varbase(ProPoint *q1, ProPoint *q2, const __m256i *k, const __m256i *x);
#endif