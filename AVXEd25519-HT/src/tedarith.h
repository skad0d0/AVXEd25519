/**
 ******************************************************************************
 * @file tedarith.h
 * @version 1.0.0
 * @date 2024-10-01
 * @copyright Copyright © 2024 by University of Luxembourg.
 * @author Developed at SnT APSIA by: Bowen Zhang
 *
 * @brief Header file for Twisted Edwards curve arithmetic operations.
 *
 * @details
 * This file contains the definitions and function prototypes for performing
 * arithmetic operations on Twisted Edwards curves using AVX2 instructions.
 ******************************************************************************
 */
#ifndef _TEDARITH_H
#define _TEDARITH_H

#include "montcurve.h"

#define mask4 0x0F
#define mask8 0XFF

// Extended point [x, y, z, e, h], where e*h = t = x*y / z
typedef struct extended_point {
    __m256i x[NWORDS];
    __m256i y[NWORDS];
    __m256i z[NWORDS];
    __m256i e[NWORDS];
    __m256i h[NWORDS];
} ExtPoint;

// Point in duif representation
typedef struct duif_point {
    uint64_t x[4]; // (y+x)/2
    uint64_t y[4]; // (y-x)/2
    uint64_t z[4]; // d*x*y
} DuifPoint;

// point operations
void ted_add(ExtPoint *r, ExtPoint *p, ProPoint *q);
void ted_pro_add(ProPoint *r, ProPoint *p, ProPoint *q);
void ted_dbl(ExtPoint *r, ExtPoint *p);
void ted_Z1_add(ProPoint *r, ProPoint *a, ProPoint *b);

// scalar multiplication
void ted_mul_varbase(ProPoint *r, ProPoint *p, const __m256i *k);
void ted_mul_fixbase(ProPoint *r, const __m256i *k);

// double scalar multiplication
void ted_sep_double_scalar_mul(ProPoint *r, ProPoint *p, const __m256i *s, const __m256i *k);
void ted_jsf_double_scalar_mul(ProPoint *r, ProPoint *p, const __m256i *s, const __m256i *k);
void ted_naf_double_scalar_mul(ProPoint *r, ProPoint *p, const __m256i *s, const __m256i *k);

// table query
void ted_table_query(ProPoint *r, const int pos, __m256i b);
void jsf_query(ProPoint *r, ProPoint *table, const __m256i d);
void table_query_wA(ProPoint *r, ProPoint *table, __m256i b);
void table_query_wB(ProPoint *r, __m256i b);

// utility functions
void ted_pro_to_aff(AffPoint *r, ProPoint *p);
void ted_copy_ext_to_pro(ProPoint *r, ExtPoint *p);
void compute_proT(ProPoint *t, ProPoint *a);
void compute_duifT(ProPoint *table, ProPoint *a);
void compute_table_A(ProPoint *t, ProPoint *p);
void compute_duiftable_A(ProPoint *table, ProPoint *p);
#endif
