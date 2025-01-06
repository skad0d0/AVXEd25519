/**
 * @file montcurve.h
 * @brief Header file for elliptic curve operations in projective and affine coordinates.
 *
 * This file is directly migrated from the original AVXECC library. see: https://github.com/ulhaocheng/AVXECC
 */
#ifndef _MONCURVE_H
#define _MONCURVE_H

#include "gfparith.h"
#include "utils.h"
#include <stdio.h>

// projective point (X:Y:Z), where x = X/Z, y = Y/Z
typedef struct projective_point
{
    __m256i x[NWORDS];
    __m256i y[NWORDS];
    __m256i z[NWORDS];
} ProPoint;

// affine point (x, y)
typedef struct affine_point
{
    __m256i x[NWORDS];
    __m256i y[NWORDS];
} AffPoint;


void mon_ladder_step(ProPoint *p, ProPoint *q, const __m256i *xd);
void mon_mul_varbase(ProPoint *q1, ProPoint *q2, const __m256i *k, const __m256i *x);
#endif