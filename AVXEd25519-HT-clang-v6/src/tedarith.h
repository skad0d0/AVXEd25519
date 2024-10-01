#ifndef _TEDARITH_H
#define _TEDARITH_H

#include "montcurve.h"

#define mask4 0x0F
#define mask8 0XFF


// extended point [x, y, z, e, h], where e*h = t = x*y / z
typedef struct extended_point
{
    __m256i x[NWORDS];
    __m256i y[NWORDS];
    __m256i z[NWORDS];
    __m256i e[NWORDS];
    __m256i h[NWORDS];
} ExtPoint;

// Point in duif representation
typedef struct duif_point
{
    uint64_t x[4]; // (y+x)/2
    uint64_t y[4]; // (y-x)/2
    uint64_t z[4]; // d*x*y
} DuifPoint;

void ted_add(ExtPoint *r, ExtPoint *p, ProPoint *q);
void ted_pro_add(ProPoint *r, ProPoint *p, ProPoint*q);
void ted_dbl(ExtPoint *r, ExtPoint *p);
// void ted_table_query(ProPoint *r, const int pos, __m256i b);
// ----- core point multiplication ----- //
void ted_mul_fixbase(ProPoint *r, const __m256i *k);
void ted_mul_varbase(ProPoint *r, ProPoint *p, const __m256i *k);
void ted_sep_double_scalar_mul(ProPoint *r, ProPoint *p, const __m256i *s, const __m256i *k);
void ted_sep_double_scalar_mul_v2(ProPoint *r, ProPoint *p, const __m256i *s, const __m256i *k);

void ted_mul_fixbase_v2(ProPoint *r, const __m256i *k);
void ted_sim_double_scalar_mul(ProPoint *r, ProPoint *p, const __m256i *s, const __m256i *k);
void ted_sim_double_scalar_mul_v2(ProPoint *r, ProPoint *p, const __m256i *s, const __m256i *k);
void ted_naf_double_scalar_mul(ProPoint *r, ProPoint *p, const __m256i *s, const __m256i *k);
void ted_naf_double_scalar_mul_v2(ProPoint *r, ProPoint *p, const __m256i *s, const __m256i *k);
// ------------------------------------ // 
// void ted_conv_scalar_to_nibble(__m256i *e, __m256i *k);
// void conv_coor_to_29(__m256i *r, __m256i *a);
// void ted_ext_to_pro(ProPoint *r, ExtPoint *p);
// void ted_pro_to_ext(ExtPoint *r, ProPoint *p);
void ted_pro_to_aff(AffPoint *r, ProPoint *p);
// void conv_ted_to_mon(ProPoint *r, ProPoint *p);
// void conv_mon_to_ted(ProPoint *r, ProPoint *p);
// void point_recovery(ProPoint *r, AffPoint *h, ProPoint *q1, ProPoint *q2);
void ted_copy_ext_to_pro(ProPoint *r, ExtPoint *p);
void compute_proT(ProPoint *t, ProPoint *a);
void ted_Z1_add(ProPoint *r, ProPoint *a, ProPoint *b);
void compute_duifT(ProPoint *table, ProPoint *a);
void compute_duifT_v2(ProPoint *table, ProPoint *a);

void compute_table_A(ProPoint *t, ProPoint *p);
// void compute_table_A_v2(ProPoint *t, ProPoint *p);
void compute_duiftable_A(ProPoint *table, ProPoint *p);
void ted_table_query_v2(ProPoint *r, const int pos, __m256i b);
void jsf_query_v2(ProPoint *r, ProPoint *table, const __m256i d);
void table_query_wA(ProPoint *r, ProPoint *table, __m256i b);
void table_query_wB(ProPoint *r, __m256i b);

#endif