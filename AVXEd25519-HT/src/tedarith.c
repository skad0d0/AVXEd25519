#include "tedarith.h"
#include "utils.h"
#include "jsf.h"
#include "wnaf.h"
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

// initialize a Extended point to [0, 1, 1, 0 ,1]
static void ted_ext_initialize(ExtPoint *p)
{
    mpi29_ini_to_zero_avx2(p->x);
    mpi29_ini_to_one_avx2(p->y);
    mpi29_ini_to_one_avx2(p->z);
    mpi29_ini_to_zero_avx2(p->e);
    mpi29_ini_to_one_avx2(p->h);
}

// initialize a protective point to [0, 1, 1]
static void ted_pro_initialize(ProPoint *p)
{
  mpi29_ini_to_zero_avx2(p->x);
  mpi29_ini_to_one_avx2(p->y);
  mpi29_ini_to_one_avx2(p->z);
}

// copy the [x, y, z] coor. of extened point to projective point
void ted_copy_ext_to_pro(ProPoint *r, ExtPoint *p)
{
  mpi29_copy_avx2(r->x, p->x);
  mpi29_copy_avx2(r->y, p->y);
  mpi29_copy_avx2(r->z, p->z);
}

// copy the [x, y, z] coor of projective point to extended point
void ted_copy_pro_to_ext(ExtPoint *r, ProPoint *p)
{
  ted_ext_initialize(r);
  mpi29_copy_avx2(r->x, p->x);
  mpi29_copy_avx2(r->y, p->y);
  mpi29_copy_avx2(r->z, p->z);
}

// Point addition R = P + Q, Q is in duif point, R is in projective point
void ted_add(ExtPoint *r, ExtPoint *p, ProPoint *q)
{
    __m256i t[NWORDS];
    mpi29_gfp_mul_avx2(t, p->e, p->h);
    mpi29_gfp_sub_avx2(r->e, p->y, p->x);
    mpi29_gfp_add_avx2(r->h, p->y, p->x);
    mpi29_gfp_mul_avx2(r->x, r->e, q->y);
    mpi29_gfp_mul_avx2(r->y, r->h, q->x);
    mpi29_gfp_sub_avx2(r->e, r->y, r->x);
    mpi29_gfp_add_avx2(r->h, r->y, r->x);
    mpi29_gfp_mul_avx2(r->x, t, q->z);
    mpi29_gfp_sbc_avx2(t, p->z, r->x);
    mpi29_gfp_add_avx2(r->x, p->z, r->x);
    mpi29_gfp_mul_avx2(r->z, t, r->x);
    mpi29_gfp_mul_avx2(r->y, r->x, r->h);
    mpi29_gfp_mul_avx2(r->x, r->e, t);
}

// Poing doubling R = 2 * P
void ted_dbl(ExtPoint *r, ExtPoint *p)
{   
    __m256i t[NWORDS];
    mpi29_gfp_sqr_avx2(r->e, p->x);
    mpi29_gfp_sqr_avx2(r->h, p->y);
    mpi29_gfp_sbc_avx2(t, r->e, r->h);
    mpi29_gfp_add_avx2(r->h, r->e, r->h);
    mpi29_gfp_add_avx2(r->x, p->x, p->y);
    mpi29_gfp_sqr_avx2(r->e, r->x);
    mpi29_gfp_sub_avx2(r->e, r->h, r->e);
    mpi29_gfp_sqr_avx2(r->y, p->z);
    mpi29_gfp_mul29_avx2(r->y, r->y, 2);
    mpi29_gfp_add_avx2(r->y, t, r->y);
    mpi29_gfp_mul_avx2(r->x, r->e, r->y);
    mpi29_gfp_mul_avx2(r->z, r->y, t);
    mpi29_gfp_mul_avx2(r->y, t, r->h);
}

uint32_t d[NWORDS] = {0x135978A3, 0x0F5A6E50, 0x10762ADD, 0x00149A82, 0x1E898007, 0x003CBBBC, 0x19CE331D, 0x1DC56DFF, 0x0052036C};

// r = p + q, all are projective points
void ted_pro_add(ProPoint *r, ProPoint *p, ProPoint *q)
{
  __m256i d_vec[NWORDS];
  __m256i t1[NWORDS], t2[NWORDS], t3[NWORDS], t4[NWORDS], t5[NWORDS], t6[NWORDS], t7[NWORDS];
  int i;
  for (i = 0; i < NWORDS; i++) d_vec[i] = VSET164(d[i]);
  
  mpi29_gfp_mul_avx2(t1, p->z, q->z); // t1 = z1 * z2 (A)
  mpi29_gfp_sqr_avx2(t2, t1);         // t2 = t1^2 (B)
  mpi29_gfp_mul_avx2(t3, p->x, q->x); // t3 = x1 * x2 (C)
  mpi29_gfp_mul_avx2(t4, p->y, q->y); // t4 = y1 * y2 (D)
  mpi29_gfp_mul_avx2(t5, t3, d_vec);
  mpi29_gfp_mul_avx2(t5, t5, t4);     // t5 = d*C*D (E)
  mpi29_gfp_sbc_avx2(t6, t2, t5);     // t6 = B - E (F)
  mpi29_gfp_add_avx2(t2, t2, t5);     // t2 = B + E (G)

  mpi29_gfp_add_avx2(t5, p->x, p->y);  // t5 = x1 + y1
  mpi29_gfp_add_avx2(t7, q->x, q->y);  // t7 = x2 + y2
  mpi29_gfp_mul_avx2(t5, t5, t7);      // t5 = (x1 + y1) * (x2 + y2)
  mpi29_gfp_sbc_avx2(t5, t5, t3);
  mpi29_gfp_sbc_avx2(t5, t5, t4);
  mpi29_gfp_mul_avx2(t5, t5, t6);
  mpi29_gfp_mul_avx2(r->x, t5, t1);    // x3 = A*F*((X1+Y1)*(X2+Y2)-C-D)
  mpi29_gfp_mul_avx2(r->z, t6, t2);    // z3 = F*G
  mpi29_gfp_add_avx2(t5, t4, t3);
  mpi29_gfp_mul_avx2(t5, t5, t2); 
  mpi29_gfp_mul_avx2(r->y, t5, t1);    // y3 = A*G*(D+C)
}

// R = A + B
// Assumptions: Z2=1.
// Cost: 9M + 1S + 1*a + 1*d + 7add.
void ted_Z1_add(ProPoint *r, ProPoint *a, ProPoint *b)
{
  __m256i t1[NWORDS], t2[NWORDS], t3[NWORDS], t4[NWORDS], t5[NWORDS], t6[NWORDS];
  __m256i d_vec[NWORDS];

  int i;
  for (i = 0; i < NWORDS; i++) d_vec[i] = VSET164(d[i]);
  mpi29_gfp_sqr_avx2(t1, a->z);       // t1 = Z1^2 (B)
  mpi29_gfp_mul_avx2(t2, a->x, b->x); // t2 = x1 * x2 (C) -
  mpi29_gfp_mul_avx2(t3, a->y, b->y); // t3 = y1 * y2 (D) -
  mpi29_gfp_mul_avx2(t4, t2, d_vec);
  mpi29_gfp_mul_avx2(t4, t4, t3);     // t4 = d*C*D (E)
  mpi29_gfp_sbc_avx2(t5, t1, t4);     // t5 = B-E (F) -
  mpi29_gfp_add_avx2(t6, t1, t4);     // t6 = B+E (G) -
  mpi29_gfp_add_avx2(t1, a->x, a->y); // t1 = x1 + y1
  mpi29_gfp_add_avx2(t4, b->x, b->y); // t4 = x2 + y2
  mpi29_gfp_mul_avx2(t1, t1, t4);
  mpi29_gfp_sbc_avx2(t1, t1, t2);
  mpi29_gfp_sbc_avx2(t1, t1, t3);
  mpi29_gfp_mul_avx2(t1, t5, t1);
  mpi29_gfp_mul_avx2(r->x, t1, a->z); // X3 = Z1*F*((X1+Y1)*(X2+Y2)-C-D)
  mpi29_gfp_add_avx2(t1, t2, t3);
  mpi29_gfp_mul_avx2(t1, t1, t6);
  mpi29_gfp_mul_avx2(r->y, t1, a->z); // Y3 = Z1*G*(D+C)
  mpi29_gfp_mul_avx2(r->z, t5, t6);   // Z3 = F*G
}

// convert 256-bit scalar to 64-bit 4 * nibble
void ted_conv_scalar_to_nibble(__m256i *e, __m256i *k)
{
  int i;
    const __m256i eight = VSET164(8);
    const __m256i vmask4 = VSET164(0x0F);
    const __m256i vmask8 = VSET164(0xFF);
    __m256i carry = VZERO;

    // convert scalar to nibbles
    for (i = 0; i < 8; i++) {
      e[8*i] = VAND(k[i], vmask4);
      e[8*i+1] = VAND(VSHR(k[i], 4), vmask4);
      e[8*i+2] = VAND(VSHR(k[i], 8), vmask4);
      e[8*i+3] = VAND(VSHR(k[i], 12), vmask4);
      e[8*i+4] = VAND(VSHR(k[i], 16), vmask4);
      e[8*i+5] = VAND(VSHR(k[i], 20), vmask4);
      e[8*i+6] = VAND(VSHR(k[i], 24), vmask4);
      e[8*i+7] = VAND(VSHR(k[i], 28), vmask4);
    }

    // convert unsigned nibbles to signed
    for (i = 0; i < 63; i++) {
      e[i] = VADD(e[i], carry);
      carry = VADD(e[i], eight);
      carry = VSHR(carry, 4);
      e[i] = VSUB(e[i], VSHL(carry, 4));
      e[i] = VAND(e[i], vmask8);
    }
    e[63] = VADD(e[63], carry);
    e[63] = VAND(e[63], vmask8);
}

// convert the coordinate from 4*64 to 9*29
void conv_coor_to_29(__m256i *r, __m256i *a)
{
    const __m256i mask = VSET164(MASK29);

    r[0] = VAND(a[0], mask); 
    r[1] = VAND(VSHR(a[0], 29), mask);
    r[2] = VSHR(a[0], 58);
    r[2] = VOR(r[2], VSHL(a[1], 6));
    r[2] = VAND(r[2], mask);
    r[3] = VAND(VSHR(a[1], 23), mask);
    r[4] = VSHR(a[1], 52);
    r[4] = VOR(r[4], VSHL(a[2], 12));
    r[4] = VAND(r[4], mask);
    r[5] = VAND(VSHR(a[2], 17), mask);
    r[6] = VSHR(a[2], 46);
    r[6] = VOR(r[6], VSHL(a[3], 18));
    r[6] = VAND(r[6], mask);
    r[7] = VAND(VSHR(a[3], 11), mask);
    r[8] = VAND(VSHR(a[3], 40), mask);
}


// (1/2) in 2^255 -19
static const uint64_t one_half[4] = { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF,
                                      0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF };

static const DuifPoint base[32][8];
static const DuifPoint base_v2[32][9];

// table query of fixed point B
void ted_table_query(ProPoint *r, const int pos, __m256i b)
{
    const __m256i babs = VABS8(b); 
    const __m256i one = VSET164(1);
    const __m256i zero = VZERO;

    __m256i xP[4], yP[4], zP[4];

    __m256i t[NWORDS];
    __m256i temp, bsign, bmask;
    uint64_t xcoor, ycoor, zcoor;
    int i, j;
    uint32_t index0, index1, index2, index3;
    index0 = VEXTR32(babs, 0); index1 = VEXTR32(babs, 2);
    index2 = VEXTR32(babs, 4); index3 = VEXTR32(babs, 6);

    for (i = 0; i < 4; i++)
    {
      load_vector(&xP[i], base_v2[pos][index0].x[i], base_v2[pos][index1].x[i], base_v2[pos][index2].x[i], base_v2[pos][index3].x[i]);
      load_vector(&yP[i], base_v2[pos][index0].y[i], base_v2[pos][index1].y[i], base_v2[pos][index2].y[i], base_v2[pos][index3].y[i]);
      load_vector(&zP[i], base_v2[pos][index0].z[i], base_v2[pos][index1].z[i], base_v2[pos][index2].z[i], base_v2[pos][index3].z[i]);
    }

    // if b<0, bsign = 1, bmask is all 1; if b>0, bsign = 0, bmask is all 0.
    bsign = VSHR(b, 7);
    bmask = VSUB(zero, bsign);

    // conditional negation
    for (i = 0; i < 4; i++)
    {
        temp = VAND(VXOR(xP[i], yP[i]), bmask);
        xP[i] = VXOR(xP[i],temp);
        yP[i] = VXOR(yP[i], temp);
    }
    conv_coor_to_29(r->x, xP);
    conv_coor_to_29(r->y, yP);
    conv_coor_to_29(r->z, zP);

    mpi29_copy_avx2(t, r->z);
    __m256i swap[NWORDS];
    for (i = 0; i < NWORDS; i++) swap[i] = zero;
    mpi29_gfp_sub_avx2(t, swap, t);
    mpi29_cswap_avx2(r->z, t, bsign);
}

// joint table query used in JSF-based method
void jsf_query(ProPoint *r, ProPoint *table, const __m256i d)
{
  const __m256i dabs = VABS8(d);
  const __m256i one = VSET164(1);
  const __m256i zero = VZERO;
  uint32_t xcoor[4][NWORDS], ycoor[4][NWORDS], zcoor[4][NWORDS], index[4];
  __m256i xP[NWORDS], yP[NWORDS], zP[NWORDS], t[NWORDS], temp;
  int i, j;
  /* Extract digits from the vector */
  for (i = 0; i < 4; i++) index[i] = ((uint32_t*) &dabs)[i*2];
  /* Start table query*/ 
  for (i = 0; i < 4; i++) {
    for (j = 0; j < NWORDS; j++) {
      xcoor[i][j] = ((uint32_t*) &table[index[i]].x[j])[i*2];
      ycoor[i][j] = ((uint32_t*) &table[index[i]].y[j])[i*2];
      zcoor[i][j] = ((uint32_t*) &table[index[i]].z[j])[i*2];
    }
  }

  for (i = 0; i < NWORDS; i++)
  {
    xP[i] = VSET64(xcoor[3][i], xcoor[2][i], xcoor[1][i], xcoor[0][i]);
    yP[i] = VSET64(ycoor[3][i], ycoor[2][i], ycoor[1][i], ycoor[0][i]);
    zP[i] = VSET64(zcoor[3][i], zcoor[2][i], zcoor[1][i], zcoor[0][i]);
  }

  __m256i dsign, dmask;
  dsign = VSHR(d, 7);
  dmask = VSUB(zero, dsign);

  // conditional negation
  for (i = 0; i < NWORDS; i++)
  {
      temp = VAND(VXOR(xP[i], yP[i]), dmask);
      xP[i] = VXOR(xP[i],temp);
      yP[i] = VXOR(yP[i], temp);
  }

  mpi29_copy_avx2(r->x, xP);
  mpi29_copy_avx2(r->y, yP);
  mpi29_copy_avx2(r->z, zP);

  mpi29_copy_avx2(t, r->z);
  mpi29_gfp_neg_avx2(t);
  mpi29_cswap_avx2(r->z, t, dsign);
}

// table query for variable point used in naf-based method
void table_query_wA(ProPoint *r, ProPoint *table, __m256i b)
{
  const __m256i babs = VABS32(b);
  const __m256i one = VSET164(1);
  const __m256i zero = VZERO;
  __m256i temp;
  uint32_t xcoor[4][NWORDS], ycoor[4][NWORDS], zcoor[4][NWORDS], index[4];
  __m256i xP[NWORDS], yP[NWORDS], zP[NWORDS], t[NWORDS];
  int i, j;

  /* Extract digits from the vector */
  for (i = 0; i < 4; i++) index[i] = (((uint32_t*) &babs)[i*2] + 1) / 2;
  /* Start table query*/ 
  for (i = 0; i < 4; i++) {
    for (j = 0; j < NWORDS; j++) {
      xcoor[i][j] = ((uint32_t*) &table[index[i]].x[j])[i*2];
      ycoor[i][j] = ((uint32_t*) &table[index[i]].y[j])[i*2];
      zcoor[i][j] = ((uint32_t*) &table[index[i]].z[j])[i*2];
    }
  }
  
  for (i = 0; i < NWORDS; i++)
  {
    xP[i] = VSET64(xcoor[3][i], xcoor[2][i], xcoor[1][i], xcoor[0][i]);
    yP[i] = VSET64(ycoor[3][i], ycoor[2][i], ycoor[1][i], ycoor[0][i]);
    zP[i] = VSET64(zcoor[3][i], zcoor[2][i], zcoor[1][i], zcoor[0][i]);
  }
  __m256i bsign, bmask;
  bsign = VSHR(b, 31);
  bmask = VSUB(zero, bsign);

  // conditional negation
  for (i = 0; i < NWORDS; i++)
  {
      temp = VAND(VXOR(xP[i], yP[i]), bmask);
      xP[i] = VXOR(xP[i],temp);
      yP[i] = VXOR(yP[i], temp);
  }

  mpi29_copy_avx2(r->x, xP);
  mpi29_copy_avx2(r->y, yP);
  mpi29_copy_avx2(r->z, zP);

  mpi29_copy_avx2(t, r->z);
  mpi29_gfp_neg_avx2(t);
  mpi29_cswap_avx2(r->z, t, bsign);
}

static const DuifPoint precomp_B[33];

// table query for fixed point B used in naf-based method
void table_query_wB(ProPoint *r, __m256i b)
{
  const __m256i babs = VABS32(b);
  const __m256i one = VSET164(1);
  const __m256i zero = VZERO;

  __m256i xP[4], yP[4], zP[4];  
  __m256i t[NWORDS];
  __m256i temp, bsign, bmask;
  uint64_t xcoor, ycoor, zcoor;
  int i, j;
  uint32_t index[4];
  /* Extract digits from the vector */
  for (i = 0; i < 4; i++) index[i] = (((uint32_t*) &babs)[i*2] + 1) / 2;

  // table query
  for (i = 0; i < 4; i++)
  {
    load_vector(&xP[i], precomp_B[index[0]].x[i], precomp_B[index[1]].x[i], precomp_B[index[2]].x[i], precomp_B[index[3]].x[i]);
    load_vector(&yP[i], precomp_B[index[0]].y[i], precomp_B[index[1]].y[i], precomp_B[index[2]].y[i], precomp_B[index[3]].y[i]);
    load_vector(&zP[i], precomp_B[index[0]].z[i], precomp_B[index[1]].z[i], precomp_B[index[2]].z[i], precomp_B[index[3]].z[i]);
  }
  // if b<0, bsign = 1, bmask is all 1; if b>0, bsign = 0, bmask is all 0.
  bsign = VSHR(b, 31);
  bmask = VSUB(zero, bsign);
  // conditional negation
  for (i = 0; i < 4; i++)
  {
    temp = VAND(VXOR(xP[i], yP[i]), bmask);
    xP[i] = VXOR(xP[i],temp);
    yP[i] = VXOR(yP[i], temp);
  }
  conv_coor_to_29(r->x, xP);
  conv_coor_to_29(r->y, yP);
  conv_coor_to_29(r->z, zP);

  mpi29_copy_avx2(t, r->z);
  __m256i swap[NWORDS];
  for (i = 0; i < NWORDS; i++) swap[i] = zero;
  mpi29_gfp_sub_avx2(t, swap, t);
  mpi29_cswap_avx2(r->z, t, bsign);
}

// R = k*B
void ted_mul_fixbase(ProPoint *r, const __m256i *k)
{
  ExtPoint h;
  __m256i e[64], kp[8];
  const __m256i t0 = VSET164(0xFFFFFFF8U);
  const __m256i t1 = VSET164(0x7FFFFFFFU);
  const __m256i t2 = VSET164(0x40000000U);
  int i;
  // prune k
  for (i = 0; i < 8; i++) kp[i] = k[i];
  kp[0] = VAND(kp[0], t0);
  kp[7] = VAND(kp[7], t1);
  kp[7] = VOR(kp[7], t2);

  ted_conv_scalar_to_nibble(e, kp);
  ted_ext_initialize(&h);        


  // for odd i
  for (i = 1; i < 64; i += 2)
  {
    ted_table_query(r, i >> 1, e[i]); // pos += 1
    ted_add(&h, &h, r); // accumulate
  }

  // res * 16
  ted_dbl(&h, &h);
  ted_dbl(&h, &h);
  ted_dbl(&h, &h);
  ted_dbl(&h, &h);

  // for even i
  for (i = 0; i < 64; i+=2)
  {
    ted_table_query(r, i >> 1, e[i]);
    ted_add(&h, &h, r);
  }

  mpi29_copy_avx2(r->x, h.x);
  mpi29_copy_avx2(r->y, h.y);
  mpi29_copy_avx2(r->z, h.z);
}



void ted_ext_to_pro(ProPoint *r, ExtPoint *p)
{
  mpi29_copy_avx2(r->x, p->x);
  mpi29_copy_avx2(r->y, p->y);
  mpi29_copy_avx2(r->z, p->z);
}

void ted_pro_to_ext(ExtPoint *r, ProPoint *p)
{
  __m256i t1[NWORDS], t2[NWORDS];
  mpi29_copy_avx2(r->x, p->x);
  mpi29_copy_avx2(r->y, p->y);
  mpi29_copy_avx2(r->z, p->z);
  mpi29_copy_avx2(r->e, p->x);
  mpi29_gfp_inv_avx2(t1, p->z); // t1 = 1/z
  mpi29_gfp_mul_avx2(t2, t1, p->y); // t2 = y/z
  mpi29_copy_avx2(r->h, t2);
}

void ted_pro_to_aff(AffPoint *r, ProPoint *p)
{
  __m256i t[NWORDS];
  mpi29_gfp_inv_avx2(t, p->z); // t = 1/z
  mpi29_gfp_mul_avx2(r->x, t, p->x); // x = X/Z
  mpi29_gfp_mul_avx2(r->y, t, p->y); // y = Y/Z
}

// sqrt(-486664) mod p in radix^29
uint32_t sqrt486664[NWORDS] = {0x1FFFFFEC, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 
                               0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x007FFFFF};

// convert both x and y in affine coordinate (Xt, Yt, Zt) -> (Xm, Ym, Zm)
void conv_ted_to_mon(ProPoint *r, ProPoint *p)
{
  __m256i t1[NWORDS];
  __m256i sqrt486664_vec[NWORDS];
  int i;
  for (i = 0; i < NWORDS; i++) sqrt486664_vec[i] = VSET164(sqrt486664[i]); 
  
  mpi29_gfp_add_avx2(t1, p->z, p->y); // t1 = zt + yt
  mpi29_gfp_mul_avx2(r->x, t1, p->x); // xm = (zt+yt)*xt
  mpi29_gfp_mul_avx2(t1, t1, p->z);   // t1 = (zt+yt)*zt
  mpi29_gfp_mul_avx2(r->y, t1, sqrt486664_vec);  // ym = sqrt(-486664) * (zt+yt)*zt
  mpi29_gfp_sbc_avx2(t1, p->z, p->y); // t1 = zt - yt
  mpi29_gfp_mul_avx2(r->z, t1, p->x); // yt = (zt-yt)*xt
}

// convert point on MontCurve to TedCurve (Um, Vm, Zm) -> (Xt, Yt, Zt)
void conv_mon_to_ted(ProPoint *r, ProPoint *p)
{
  __m256i t1[NWORDS], t2[NWORDS], t3[NWORDS];

  __m256i sqrt486664_vec[NWORDS];
  int i;
  for (i = 0; i < NWORDS; i++) sqrt486664_vec[i] = VSET164(sqrt486664[i]);  
  mpi29_gfp_add_avx2(t1, p->x, p->z);       // t1 = U+Z
  mpi29_gfp_mul_avx2(t2, t1, p->z);
  mpi29_gfp_mul_avx2(t2, t2, p->x);
  mpi29_gfp_mul_avx2(r->x, t2, sqrt486664_vec); // Xt = sqrt(-486664) * U*Z * (U+Z)
  mpi29_gfp_mul_avx2(t2, p->y, p->z);       // t2 = V*Z         
  mpi29_gfp_mul_avx2(r->z, t2, t1);         // Zt = V*Z*(U+Z)
  mpi29_gfp_sbc_avx2(t1, p->x, p->z);       // t1 = U-Z
  mpi29_gfp_mul_avx2(r->y, t1, t2);         // Yt = (U-Z)* V*Z
}

// Full projective point recovery on Mon Curve (u, v, z) <- [(xm, ym), (x1, z1), (x2, z2)]
void point_recovery(ProPoint *r, AffPoint *h, ProPoint *q1, ProPoint *q2)
{
  __m256i u[NWORDS], v[NWORDS], z[NWORDS];
  // Full projective point recovery (u, v, z) <- [(xm, ym), (x1, z1), (x2, z2)]
  // u = 2B*ym*z1*z2*x1
  // v = z2 * [ (x1 + xm*z1 + 2A*z1) * (x1*xm + z1) - 2A*(z1^2) ] - (x1-xm*z1)^2 * x2
  // z = 2B*ym*z1*z2*z1
  __m256i t1[NWORDS], t2[NWORDS], t3[NWORDS], t4[NWORDS]; 
  uint32_t CONST2B[NWORDS] = {0x1FF125DD, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 
                              0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x007FFFFF};
  int i;
  // create vector
  __m256i CONST2B_vec[NWORDS];
  for (i = 0; i < NWORDS; i++) CONST2B_vec[i] = VSET164(CONST2B[i]);

  mpi29_gfp_mul_avx2(t1, h->x, q1->z);      // t1 = xm * z1
  mpi29_gfp_add_avx2(t2, q1->x, t1);        // t2 = x1 + t1
  mpi29_gfp_sbc_avx2(t3, q1->x, t1);        // t3 = x1 - t1
  mpi29_gfp_sqr_avx2(t3,t3);                // t3 = t3^2
  mpi29_gfp_mul_avx2(t3, t3, q2->x);        // t3 = t3 * x2
  mpi29_gfp_mul29_avx2(t1, q1->z, CONST2A); // t1 = 2A * z1
  mpi29_gfp_add_avx2(t2, t2, t1);           // t2 = t2 + t1
  mpi29_gfp_mul_avx2(t4, h->x, q1->x);      // t4 = xm * x1
  mpi29_gfp_add_avx2(t4, t4, q1->z);        // t4 = t4 + z1
  mpi29_gfp_mul_avx2(t2, t2, t4);           // t2 = t2 * t4
  mpi29_gfp_mul_avx2(t1, t1, q1->z);        // t1 = t1 * z1
  mpi29_gfp_sbc_avx2(t2, t2, t1);           // t2 = t2 - t1
  mpi29_gfp_mul_avx2(t2, t2, q2->z);        // t2 = t2 * z2
  mpi29_gfp_sbc_avx2(v, t2, t3);            // v = t2 - t3
  mpi29_gfp_mul_avx2(t1, h->y, CONST2B_vec);  // t1 = 2B * ym
  mpi29_gfp_mul_avx2(t1, t1, q1->z);        // t1 = t1 * z1
  mpi29_gfp_mul_avx2(t1, t1, q2->z);        // t1 = t1 * z2
  mpi29_gfp_mul_avx2(u, t1, q1->x);         // u = t1 * x1
  mpi29_gfp_mul_avx2(z, t1, q1->z);         // z = t1 * z1
  mpi29_copy_avx2(r->x, u);
  mpi29_copy_avx2(r->y, v);
  mpi29_copy_avx2(r->z, z);
}

// R = k*P, R, P are points on TedCurve in projective form (x, y, z)
void ted_mul_varbase(ProPoint *r, ProPoint *p, const __m256i *k)
{
  // convert Ted to Mon in affine point
  ProPoint h;
  AffPoint a;

  conv_ted_to_mon(&h, p); // (xm, ym, zm) <- (Xt, Yt, Zt)
  
  __m256i t[NWORDS];
  // reduced to 1 inversion in total
  mpi29_gfp_inv_avx2(t, h.z); // t = 1/zm
  mpi29_gfp_mul_avx2(a.x, t, h.x); // u = xm/zm
  mpi29_gfp_mul_avx2(a.y, t, h.y); // v = ym/zm

  ProPoint q1, q2, q3;
  mon_mul_varbase(&q1, &q2, k, a.x); // (Q1, Q2) <- MonLadder(P)
  // Point recovery on Mon Curve
  point_recovery(&q3, &a, &q1, &q2);
  // Convert Mon -> Ted
  conv_mon_to_ted(r, &q3);
}

/**
 * Performs a double scalar multiplication on twisted Edwards curve points.
 * 
 * This function computes r = s*B + k*(-A), where B is a fixed base point and A is the input point.
 * 
 * @param r Output point resulting from the double scalar multiplication.
 * @param p Input point A.
 * @param s Scalar multiplier for the fixed base point B.
 * @param k Scalar multiplier for the negated input point -A.
 */
void ted_sep_double_scalar_mul(ProPoint *r, ProPoint *p, const __m256i *s, const __m256i *k)
{
  ProPoint p1, p2;
  ProPoint h;
  mpi29_copy_avx2(h.x, p->x);
  mpi29_copy_avx2(h.y, p->y);
  mpi29_copy_avx2(h.z, p->z);
  // p1 = s*B
  ted_mul_fixbase(&p1, s);

  // Convert A to -A, (x, y, z) -> (-x, y, z)
  mpi29_gfp_neg_avx2(h.x);
  // p2 = k*(-A)
  ted_mul_varbase(&p2, &h, k);

  ted_pro_add(r, &p1, &p2);
}


void compute_proT(ProPoint *t, ProPoint *a)
{
  uint32_t xB[NWORDS] = {0x0F25D51A, 0x0AB16B04, 0x0969ECB2, 0x198EC12A, 0x0DC5C692, 0x1118FEEB, 0x0FFB0293, 0x1A79ADCA, 0x00216936};
  uint32_t yB[NWORDS] = {0x06666658, 0x13333333, 0x19999999, 0x0CCCCCCC, 0x06666666, 0x13333333, 0x19999999, 0x0CCCCCCC, 0x00666666};
  uint32_t zB[NWORDS] = {0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};

  // create base point
  ProPoint b;
  int i;
  for (i = 0; i < NWORDS; i++)
  {
    b.x[i] = VSET164(xB[i]);
    b.y[i] = VSET164(yB[i]);
    b.z[i] = VSET164(zB[i]);
  }

  ProPoint neg_a;
  mpi29_copy_avx2(neg_a.x, a->x);
  mpi29_copy_avx2(neg_a.y, a->y);
  mpi29_copy_avx2(neg_a.z, a->z);
  mpi29_gfp_neg_avx2(neg_a.x);

  mpi29_copy_avx2(t[0].x, neg_a.x);
  mpi29_copy_avx2(t[0].y, neg_a.y);
  mpi29_copy_avx2(t[0].z, neg_a.z);

  ted_Z1_add(&t[1], a, &b);

  mpi29_copy_avx2(t[2].x, b.x);
  mpi29_copy_avx2(t[2].y, b.y);
  mpi29_copy_avx2(t[2].z, b.z);

  ted_Z1_add(&t[3], &neg_a, &b);
}

void compute_duifT(ProPoint *table, ProPoint *a)
{
  __m256i t1[NWORDS], t2[NWORDS], t3[NWORDS];
  uint32_t half[NWORDS] = {0x1FFFFFF7, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x003FFFFF};
  __m256i half_vec[NWORDS], d_vec[NWORDS], zero_vec[NWORDS];
  uint32_t zero[NWORDS] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

  int i;
  for (i = 0; i < NWORDS; i++) half_vec[i] = VSET164(half[i]);
  for (i = 0; i < NWORDS; i++) d_vec[i] = VSET164(d[i]);
  for (i = 0; i < NWORDS; i++) zero_vec[i] = VSET164(zero[i]);
  // Table[0]
  mpi29_copy_avx2(table[0].x, half_vec);
  mpi29_copy_avx2(table[0].y, half_vec);
  mpi29_copy_avx2(table[0].z, zero_vec);


  AffPoint h;
  mpi29_gfp_mul_avx2(t1, a[0].z, a[1].z);
  mpi29_gfp_mul_avx2(t1, t1, a[3].z); // t1 = z1*zs*zd
  mpi29_gfp_inv_avx2(t1, t1);         // t1 = 1 / z1*zs*zd

  // convert Table[1] = -A
  mpi29_gfp_mul_avx2(t2, t1, a[1].z);
  mpi29_gfp_mul_avx2(t2, t2, a[3].z); // t2 = 1/z1
  mpi29_gfp_mul_avx2(h.x, t2, a[0].x); // x1 = X1/Z1
  mpi29_gfp_mul_avx2(h.y, t2, a[0].y); // y1 = Y1/Z1

  mpi29_gfp_add_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[1].x, t2, half_vec);
  mpi29_gfp_sbc_avx2(t2, h.y, h.x);
  mpi29_gfp_mul_avx2(table[1].y, t2, half_vec);
  mpi29_gfp_mul_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[1].z, t2, d_vec);

  // convert Table[2] = B+A
  mpi29_gfp_mul_avx2(t2, t1, a[0].z);
  mpi29_gfp_mul_avx2(t2, t2, a[3].z); // t2 = 1/z2
  mpi29_gfp_mul_avx2(h.x, t2, a[1].x); // x2 = X2/Z2
  mpi29_gfp_mul_avx2(h.y, t2, a[1].y); // y2 = Y2/Z2

  mpi29_gfp_add_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[2].x, t2, half_vec);
  mpi29_gfp_sbc_avx2(t2, h.y, h.x);
  mpi29_gfp_mul_avx2(table[2].y, t2, half_vec);
  mpi29_gfp_mul_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[2].z, t2, d_vec);

  // convert Table[4] = B-A
  mpi29_gfp_mul_avx2(t2, t1, a[0].z);
  mpi29_gfp_mul_avx2(t2, t2, a[1].z); // t2 = 1/z3
  mpi29_gfp_mul_avx2(h.x, t2, a[3].x); // x2 = X3/Z3
  mpi29_gfp_mul_avx2(h.y, t2, a[3].y); // y2 = Y3/Z3

  mpi29_gfp_add_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[4].x, t2, half_vec);
  mpi29_gfp_sbc_avx2(t2, h.y, h.x);
  mpi29_gfp_mul_avx2(table[4].y, t2, half_vec);
  mpi29_gfp_mul_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[4].z, t2, d_vec);

  // convert Table[3] = B
  mpi29_gfp_add_avx2(t1, a[2].x, a[2].y);
  mpi29_gfp_mul_avx2(table[3].x, t1, half_vec);
  mpi29_gfp_sbc_avx2(t1, a[2].y, a[2].x);
  mpi29_gfp_mul_avx2(table[3].y, t1, half_vec);
  mpi29_gfp_mul_avx2(t1, a[2].x, a[2].y);
  mpi29_gfp_mul_avx2(table[3].z, t1, d_vec);

}

// compute table for variable point A at run time, w = 5
void compute_table_A(ProPoint *t, ProPoint *p)
{
  ExtPoint h;
  ted_ext_initialize(&h);
  mpi29_copy_avx2(h.x, p->x);
  mpi29_copy_avx2(h.y, p->y);
  mpi29_copy_avx2(h.z, p->z);
  // compute 2P
  ted_dbl(&h, &h);
  ProPoint p_dbl;
  ted_copy_ext_to_pro(&p_dbl, &h);

  // compute table[P, 3P, 5P, ... , 15P] 8 points in total
  // P
  mpi29_copy_avx2(t[0].x, p->x);
  mpi29_copy_avx2(t[0].y, p->y);
  mpi29_copy_avx2(t[0].z, p->z);
  // 3P
  ted_pro_add(&t[1], &t[0], &p_dbl);
  // 5P
  ted_pro_add(&t[2], &t[1], &p_dbl);
  // 7P
  ted_pro_add(&t[3], &t[2], &p_dbl);
  // 9P
  ted_pro_add(&t[4], &t[3], &p_dbl);
  // 11P
  ted_pro_add(&t[5], &t[4], &p_dbl);
  // 13P
  ted_pro_add(&t[6], &t[5], &p_dbl);
  // 15P
  ted_pro_add(&t[7], &t[6], &p_dbl);
}


// compute duif table for variable point P at run time, w = 5
void compute_duiftable_A(ProPoint *table, ProPoint *p)
{
  __m256i t1[NWORDS], t2[NWORDS], t3[NWORDS];
  uint32_t half[NWORDS] = {0x1FFFFFF7, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x003FFFFF};
  __m256i half_vec[NWORDS], d_vec[NWORDS], zero_vec[NWORDS];
  // const __m256i zero = VZERO;
  uint32_t zero[NWORDS] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
  int i;
  for (i = 0; i < NWORDS; i++) half_vec[i] = VSET164(half[i]);
  for (i = 0; i < NWORDS; i++) d_vec[i] = VSET164(d[i]);
  for (i = 0; i < NWORDS; i++) zero_vec[i] = VSET164(zero[i]);
  AffPoint h;
  mpi29_gfp_mul_avx2(t1, p[0].z, p[1].z);

  for (i = 2; i < 8; i++) mpi29_gfp_mul_avx2(t1, t1, p[i].z); // t1 = z0z1z2z3z4z5z6z7

  mpi29_gfp_inv_avx2(t1, t1);

  // Table[0] 0P
  mpi29_copy_avx2(table[0].x, half_vec);
  mpi29_copy_avx2(table[0].y, half_vec);
  mpi29_copy_avx2(table[0].z, zero_vec);

  // Table[1] P
  mpi29_gfp_mul_avx2(t2, t1, p[1].z);
  mpi29_gfp_mul_avx2(t2, t2, p[2].z);
  mpi29_gfp_mul_avx2(t2, t2, p[3].z);
  mpi29_gfp_mul_avx2(t2, t2, p[4].z);
  mpi29_gfp_mul_avx2(t2, t2, p[5].z);
  mpi29_gfp_mul_avx2(t2, t2, p[6].z);
  mpi29_gfp_mul_avx2(t2, t2, p[7].z); // t2 = 1/z0

  mpi29_gfp_mul_avx2(h.x, t2, p[0].x);
  mpi29_gfp_mul_avx2(h.y, t2, p[0].y);

  mpi29_gfp_add_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[1].x, t2, half_vec);
  mpi29_gfp_sbc_avx2(t2, h.y, h.x);
  mpi29_gfp_mul_avx2(table[1].y, t2, half_vec);
  mpi29_gfp_mul_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[1].z, t2, d_vec);

  // Table[2] 3P
  mpi29_gfp_mul_avx2(t2, t1, p[0].z);
  mpi29_gfp_mul_avx2(t2, t2, p[2].z);
  mpi29_gfp_mul_avx2(t2, t2, p[3].z);
  mpi29_gfp_mul_avx2(t2, t2, p[4].z);
  mpi29_gfp_mul_avx2(t2, t2, p[5].z);
  mpi29_gfp_mul_avx2(t2, t2, p[6].z);
  mpi29_gfp_mul_avx2(t2, t2, p[7].z); // t2 = 1/z1

  mpi29_gfp_mul_avx2(h.x, t2, p[1].x);
  mpi29_gfp_mul_avx2(h.y, t2, p[1].y);

  mpi29_gfp_add_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[2].x, t2, half_vec);
  mpi29_gfp_sbc_avx2(t2, h.y, h.x);
  mpi29_gfp_mul_avx2(table[2].y, t2, half_vec);
  mpi29_gfp_mul_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[2].z, t2, d_vec);

  // Table[3] 5P
  mpi29_gfp_mul_avx2(t2, t1, p[0].z);
  mpi29_gfp_mul_avx2(t2, t2, p[1].z);
  mpi29_gfp_mul_avx2(t2, t2, p[3].z);
  mpi29_gfp_mul_avx2(t2, t2, p[4].z);
  mpi29_gfp_mul_avx2(t2, t2, p[5].z);
  mpi29_gfp_mul_avx2(t2, t2, p[6].z);
  mpi29_gfp_mul_avx2(t2, t2, p[7].z); // t2 = 1/z2

  mpi29_gfp_mul_avx2(h.x, t2, p[2].x);
  mpi29_gfp_mul_avx2(h.y, t2, p[2].y);

  mpi29_gfp_add_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[3].x, t2, half_vec);
  mpi29_gfp_sbc_avx2(t2, h.y, h.x);
  mpi29_gfp_mul_avx2(table[3].y, t2, half_vec);
  mpi29_gfp_mul_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[3].z, t2, d_vec);

  // Table[4] 7P
  mpi29_gfp_mul_avx2(t2, t1, p[0].z);
  mpi29_gfp_mul_avx2(t2, t2, p[1].z);
  mpi29_gfp_mul_avx2(t2, t2, p[2].z);
  mpi29_gfp_mul_avx2(t2, t2, p[4].z);
  mpi29_gfp_mul_avx2(t2, t2, p[5].z);
  mpi29_gfp_mul_avx2(t2, t2, p[6].z);
  mpi29_gfp_mul_avx2(t2, t2, p[7].z); // t2 = 1/z3

  mpi29_gfp_mul_avx2(h.x, t2, p[3].x);
  mpi29_gfp_mul_avx2(h.y, t2, p[3].y);

  mpi29_gfp_add_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[4].x, t2, half_vec);
  mpi29_gfp_sbc_avx2(t2, h.y, h.x);
  mpi29_gfp_mul_avx2(table[4].y, t2, half_vec);
  mpi29_gfp_mul_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[4].z, t2, d_vec);

  // Table[5] 9P
  mpi29_gfp_mul_avx2(t2, t1, p[0].z);
  mpi29_gfp_mul_avx2(t2, t2, p[1].z);
  mpi29_gfp_mul_avx2(t2, t2, p[2].z);
  mpi29_gfp_mul_avx2(t2, t2, p[3].z);
  mpi29_gfp_mul_avx2(t2, t2, p[5].z);
  mpi29_gfp_mul_avx2(t2, t2, p[6].z);
  mpi29_gfp_mul_avx2(t2, t2, p[7].z); // t2 = 1/z4

  mpi29_gfp_mul_avx2(h.x, t2, p[4].x);
  mpi29_gfp_mul_avx2(h.y, t2, p[4].y);

  mpi29_gfp_add_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[5].x, t2, half_vec);
  mpi29_gfp_sbc_avx2(t2, h.y, h.x);
  mpi29_gfp_mul_avx2(table[5].y, t2, half_vec);
  mpi29_gfp_mul_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[5].z, t2, d_vec);

  // Table[6] 11P
  mpi29_gfp_mul_avx2(t2, t1, p[0].z);
  mpi29_gfp_mul_avx2(t2, t2, p[1].z);
  mpi29_gfp_mul_avx2(t2, t2, p[2].z);
  mpi29_gfp_mul_avx2(t2, t2, p[3].z);
  mpi29_gfp_mul_avx2(t2, t2, p[4].z);
  mpi29_gfp_mul_avx2(t2, t2, p[6].z);
  mpi29_gfp_mul_avx2(t2, t2, p[7].z); // t2 = 1/z5

  mpi29_gfp_mul_avx2(h.x, t2, p[5].x);
  mpi29_gfp_mul_avx2(h.y, t2, p[5].y);

  mpi29_gfp_add_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[6].x, t2, half_vec);
  mpi29_gfp_sbc_avx2(t2, h.y, h.x);
  mpi29_gfp_mul_avx2(table[6].y, t2, half_vec);
  mpi29_gfp_mul_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[6].z, t2, d_vec);

// Table[7] 13P
  mpi29_gfp_mul_avx2(t2, t1, p[0].z);
  mpi29_gfp_mul_avx2(t2, t2, p[1].z);
  mpi29_gfp_mul_avx2(t2, t2, p[2].z);
  mpi29_gfp_mul_avx2(t2, t2, p[3].z);
  mpi29_gfp_mul_avx2(t2, t2, p[4].z);
  mpi29_gfp_mul_avx2(t2, t2, p[5].z);
  mpi29_gfp_mul_avx2(t2, t2, p[7].z); // t2 = 1/z6

  mpi29_gfp_mul_avx2(h.x, t2, p[6].x);
  mpi29_gfp_mul_avx2(h.y, t2, p[6].y);

  mpi29_gfp_add_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[7].x, t2, half_vec);
  mpi29_gfp_sbc_avx2(t2, h.y, h.x);
  mpi29_gfp_mul_avx2(table[7].y, t2, half_vec);
  mpi29_gfp_mul_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[7].z, t2, d_vec);

// Table[8] 15P
  mpi29_gfp_mul_avx2(t2, t1, p[0].z);
  mpi29_gfp_mul_avx2(t2, t2, p[1].z);
  mpi29_gfp_mul_avx2(t2, t2, p[2].z);
  mpi29_gfp_mul_avx2(t2, t2, p[3].z);
  mpi29_gfp_mul_avx2(t2, t2, p[4].z);
  mpi29_gfp_mul_avx2(t2, t2, p[5].z);
  mpi29_gfp_mul_avx2(t2, t2, p[6].z); // t2 = 1/z7

  mpi29_gfp_mul_avx2(h.x, t2, p[7].x);
  mpi29_gfp_mul_avx2(h.y, t2, p[7].y);

  mpi29_gfp_add_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[8].x, t2, half_vec);
  mpi29_gfp_sbc_avx2(t2, h.y, h.x);
  mpi29_gfp_mul_avx2(table[8].y, t2, half_vec);
  mpi29_gfp_mul_avx2(t2, h.x, h.y);
  mpi29_gfp_mul_avx2(table[8].z, t2, d_vec);
}

/**
 * @brief Double scalar multiplication using JSF and AVX2.
 *
 * Computes `r = s*B + k*p` using AVX2 instructions.
 *
 * @param[out] r  Resulting point.
 * @param[in]  p  Input point.
 * @param[in]  s  First scalar.
 * @param[in]  k  Second scalar.
 */
void ted_jsf_double_scalar_mul(ProPoint *r, ProPoint *p, const __m256i *s, const __m256i *k)
{
  ExtPoint h;
  JSFResult_avx2 j;
  __m256i d;
  uint32_t half[NWORDS] = {0x1FFFFFF7, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x003FFFFF};
  __m256i half_vec[NWORDS], zero_vec[NWORDS];
  uint32_t zero[NWORDS] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};
  const __m256i three = VSET164(3);
  const __m256i t0 = VSET164(0xFFFFFFF8U);
  const __m256i t1 = VSET164(0x7FFFFFFFU);
  const __m256i t2 = VSET164(0x40000000U);
  __m256i VMASK8 = VSET164(mask8);
  __m256i sp[8], kp[8];
  int i;

  for (i = 0; i < 8; i++)
  {
    sp[i] = s[i];
    kp[i] = k[i];
  }
  // prune the scalar
  kp[0] = VAND(kp[0], t0);
  kp[7] = VAND(kp[7], t1);
  kp[7] = VOR(kp[7], t2);

  sp[0] = VAND(sp[0], t0);
  sp[7] = VAND(sp[7], t1);
  sp[7] = VOR(sp[7], t2);
  
  JSF_conv(&j, sp, kp);

  for (i = 0; i < NWORDS; i++) half_vec[i] = VSET164(half[i]);
  for (i = 0; i < NWORDS; i++) zero_vec[i] = VSET164(zero[i]);
  
  // create table
  ProPoint t[4], lut[5];
  compute_proT(t, p);
  compute_duifT(lut, t);

  // table query
  ted_ext_initialize(&h);
  for (i = 255; i >= 0; i--)
  {
    ted_dbl(&h, &h);
    d = VMUL(three, j.k0[i]);
    d = VADD(d, j.k1[i]);
    d = VAND(d, VMASK8);

    if (is_all_zero(d) == 1) continue;
    jsf_query(r, lut, d);
    ted_add(&h, &h, r);
  }
  mpi29_copy_avx2(r->x, h.x);
  mpi29_copy_avx2(r->y, h.y);
  mpi29_copy_avx2(r->z, h.z);
}


/**
 * @brief Double scalar multiplication using NAF and AVX2.
 *
 * Computes `r = s*B + k*p` using AVX2 instructions.
 *
 * @param[out] r  Resulting point.
 * @param[in]  p  Input point.
 * @param[in]  s  First scalar.
 * @param[in]  k  Second scalar.
 */
void ted_naf_double_scalar_mul(ProPoint *r, ProPoint *p, const __m256i *s, const __m256i *k)
{
  ExtPoint h;
  NAFResult_avx2 n;
  uint32_t half[NWORDS] = {0x1FFFFFF7, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x1FFFFFFF, 0x003FFFFF};
  __m256i half_vec[NWORDS], zero_vec[NWORDS];
  uint32_t zero[NWORDS] = {0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0};

  __m256i sp[8], kp[8];
  int max_len, i;
  const __m256i t0 = VSET164(0xFFFFFFF8U);
  const __m256i t1 = VSET164(0x7FFFFFFFU);
  const __m256i t2 = VSET164(0x40000000U);
  for (i = 0; i < 8; i++)
  {
    sp[i] = s[i];
    kp[i] = k[i];
  }
  // prune the scalar
  kp[0] = VAND(kp[0], t0); kp[7] = VAND(kp[7], t1); kp[7] = VOR(kp[7], t2);
  sp[0] = VAND(sp[0], t0); sp[7] = VAND(sp[7], t1); sp[7] = VOR(sp[7], t2);

  for (i = 0; i < NWORDS; i++) half_vec[i] = VSET164(half[i]);
  for (i = 0; i < NWORDS; i++) zero_vec[i] = VSET164(zero[i]);

  NAF_conv(&n, sp, kp);

  // compute table for variable point 
  ProPoint t[8], lut[9];
  ProPoint a;
  mpi29_copy_avx2(a.x, p->x);
  mpi29_copy_avx2(a.y, p->y);
  mpi29_copy_avx2(a.z, p->z);
  // point negation
  mpi29_gfp_neg_avx2(a.x);

  compute_table_A(t, &a);
  compute_duiftable_A(lut, t);

  // table query
  ted_ext_initialize(&h);
  max_len = n.max_length;
  for (i = max_len-1; i >= 0; i--)
  {
    ted_dbl(&h, &h);
    if (is_all_zero(n.k0[i]) != 1)
    {
      table_query_wB(r, n.k0[i]);
      ted_add(&h, &h, r);
    }
    if (is_all_zero(n.k1[i]) != 1)
    {
      table_query_wA(r, lut, n.k1[i]);
      ted_add(&h, &h, r);
    }
  }
  mpi29_copy_avx2(r->x, h.x);
  mpi29_copy_avx2(r->y, h.y);
  mpi29_copy_avx2(r->z, h.z);
}





// ======================== Precomputed table ======================== //
// precomp_table for base point, w=7

static const DuifPoint precomp_B[33] = 
{
    // 0B
    {
    { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
    { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
    { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    // 1B
    {
    { 0x97DE49E37AC61DB9, 0x67C996E37DC6070C, 0x9385A44C321EA161, 0x43E7CE9D19EA5D32 },
    { 0xCE881C82EBA0489F, 0xFE9CCF82E8A05F59, 0xD2E0C21A3447C504, 0x227E97C94C7C0933 },
    { 0x55E48902C3BD5534, 0x136CF411E655624F, 0x2D0DBEE5EEA1ACC6, 0x3788BDB44F8632D4 },
    },
    // 3B
    {
    { 0x5792D85426774B98, 0x012D4218744325C5, 0x608DA8014F80B399, 0x3D0B270DCD407C7A },
    { 0xAB308FF4527E6929, 0x9DE9A9FEF2E0DD3E, 0x4098F98D10A5EB5E, 0x555C8AC3AAADED31 },
    { 0x0A57499F86E86C3B, 0x2C4A11910E1AED31, 0x68B872A2C6796DA6, 0x6D141357895CDA63 },
    },
    // 5B
    {
    { 0x51095E220452DD90, 0x46A82461E3AF7681, 0xEE8DF5862D5FF622, 0x54A2E678A3710375 },
    { 0xBFC8C161D223EB5D, 0xEA800A68A59394DB, 0xF19E788E5C325043, 0x0AA53F39F58DAAF9 },
    { 0x5E5DEDF8C0954139, 0x93870403E85EE8FE, 0x5A0DB3858DDED396, 0x61D55F34B59DDB4D },
    },
    // 7B
    {
    { 0x358D2E684A2751D6, 0xBA381A9D59CEE069, 0x38D92941142A1724, 0x630DF534941E493F },
    { 0xDD37964D551910CF, 0x365010A99DDD11D3, 0x4EF53B27C90C961D, 0x4EB76EAE97298BF0 },
    { 0x78C1B6E400DC59D1, 0xD981AFA3829F524D, 0xA94E20DD2C3BD6F9, 0x3D4FDD8E3507C853 },
    },
    // 9B
    {
    { 0xCD9733C55354318E, 0x3284F37A8DE2362C, 0xE75919E4E3437ADA, 0x5A5CF699C56EBFAC },
    { 0xF9B710BF01CEC032, 0x4C5040DB7A9020CD, 0xCB65E30473AF5822, 0x24E02D28FD6E4E47 },
    { 0x835A745FC822D784, 0x717FC1F4538CE917, 0x557B7E14C9EA678B, 0x79E0B9010D804583 },
    },
    // 11B
    {
    { 0x97DF80424540156F, 0x72ECFF6781181713, 0x889F42388BB81A03, 0x213AD5712A36C7D7 },
    { 0x98AFAD8124C321A4, 0x9F6B59B4BB8441C0, 0xD1D03AAAB546F5CA, 0x0C55ACC014EAE3BF },
    { 0x6C159662FEB044EB, 0x018F5A5099417252, 0x221888CCDA8D4311, 0x5EE32A915A9EFCA4 },
    },
    // 13B
    {
    { 0xDFB8611151003FAD, 0xDFC259CD5ADE6F6D, 0xA9BD07097D83DD03, 0x51A7EBF761A37920 },
    { 0xA837809D993FDFC0, 0x577E75E4CDBBB7B5, 0xCE8959195556ACB4, 0x4133C4168BB01253 },
    { 0xA9B0508CB99751BC, 0x921BF358EFC6EA38, 0xD1779BFC48D3F299, 0x24BDD37ED504BC31 },
    },
    // 15B
    {
    { 0x9267660189E7F550, 0x432461468C4E1236, 0x96DEDEFD60F96A68, 0x30F1148BF896F395 },
    { 0x0205E6C32346677C, 0x69C14DD2154C886B, 0xBA84180403D928C9, 0x61DAE6A10C682F5F },
    { 0x72ECD3B17CDE85A8, 0x759C57A71B9FEF77, 0x01972D3EC9EB2138, 0x288EB09085726C21 },
    },
    // 17B
    {
    { 0xC9633B77CA874EB7, 0x52A31066E06B8227, 0xD54D9B3237C78924, 0x76992C926EDC2AF1 },
    { 0x0409C32422106F3A, 0xC50E780B5AC976DA, 0x1CFD271394CA1692, 0x78D3FF37F1241408 },
    { 0xB638C15C52E4642A, 0x19FE8A3CFF2F9501, 0xB967AC8C41BBC686, 0x23A3625B2ACF7554 },
    },
    // 19B
    {
    { 0xE9BBBD9E36E34D0C, 0xEF7D5913B7C4FB0B, 0xA2B28E7BDA9D0B5A, 0x6E4D28EF1A7F4FDB },
    { 0x9A42A364323A089A, 0x3E9AD76E877E6424, 0x7FC9CD3B03395199, 0x50CB31A4BEDAF36B },
    { 0xFA8878E7BCF8872A, 0xFFEEEED50F32C28A, 0x84E1D38B880A113B, 0x6402281E304111DD },
    },
    // 21B
    {
    { 0xE2124F681651BFDA, 0xD02CD071D30AD655, 0xC454B76BE4B70711, 0x6A99CC528B2834B6 },
    { 0x1DB410E91D1B68B1, 0xDDDA0553F4CDCF19, 0x2ECF2E721041C523, 0x7B8F04C42C6F262F },
    { 0xCD097AE93C228F66, 0x1D6D2EBCC2C4CE65, 0xA3BFA516CFD2CA84, 0x6D2F68EB47FAD308 },
    },
    // 23B
    {
    { 0x08CA89157F0A8738, 0x67904D12BF259AEC, 0x39C3FC148F388F10, 0x62565C4BEC5FC978 },
    { 0x5D72F062AC2939A3, 0x9C972E0CE56DCEBF, 0x14329E0F6D0E55F4, 0x40CDB009AFF7EE22 },
    { 0x0F30340A2F09A5B8, 0xE27AF3279218260B, 0x283744547E0D1F6B, 0x4A8624FEF35697C9 },
    },
    // 25B
    {
    { 0xC73DF94A84A3889C, 0xAEB7F79CA7BAD328, 0x0857BCE212D38456, 0x3595AD03ADDCCC91 },
    { 0x5C24C31E4E6E5434, 0xE41FA26DDC38A568, 0xFF1F71AB061B0B46, 0x3C536BBC8F02FDE0 },
    { 0xAC5FB825A3D05CBB, 0x5300D9AABA0BA46A, 0x55158FD8EAA17AC8, 0x392E3FFE256AAE80 },
    },
    // 27B
    {
    { 0x7221338AE8E7CCD9, 0xB9A96A888151069A, 0x91E88ABDC589084F, 0x3CA66493BE58F9D1 },
    { 0x48C015FB8E684C60, 0x7F20B65276AF31B3, 0xEFAC2EB8A4814CA6, 0x266AA312FC2AFD73 },
    { 0x257B621361562820, 0xDE4D76D6997B392C, 0x9568197885188810, 0x78041ABDB7E64742 },
    },
    // 29B
    {
    { 0x85C433939C3B9F77, 0x5C66647D4AFDE67D, 0x4696EAD1DCD694DB, 0x4377BF4C28D687B5 },
    { 0xE80DCFDDC12C251A, 0x23D5B231E95A3C95, 0x5B18B1CE2429B101, 0x09D4951B34EB6A14 },
    { 0x6549BB8E602BBEE9, 0xBAA0720F281AEE2E, 0x12340780EC017038, 0x5E14B6EFC5157C35 },
    },
    // 31B
    {
    { 0x57568AFCEC8A5380, 0x5497BDFCC647FC89, 0x57FC118BCFA9EB98, 0x7D4CE9C9A4863BDD },
    { 0xFE75A6975D8F9297, 0xDC4A8863A056DC8F, 0xFE38D1BEE850D682, 0x4544963803A3B8BD },
    { 0x47A976921B5ED1F4, 0x3BD46420ABF403CA, 0x52D4B2B19317CE70, 0x1433B16941817BE9 },
    },
    // 33B
    {
    { 0xA73C1B049E71AD89, 0xC1708C0ED935D54B, 0x0660C969E5E3DC1F, 0x5978ED02354ECE9D },
    { 0xBE2AC715E7177AD5, 0x724C365A33A3DE31, 0x0AA50BCF9DDDC4DC, 0x7B437951EB78BB3D },
    { 0xD546895336ACBE35, 0x4788C9818269C295, 0x1FC8EE39E104D811, 0x2B0982FC54D69453 },
    },
    // 35B
    {
    { 0x88064BC6F649575F, 0x6521EAA1A6B6B9F2, 0x41898D916C23DD24, 0x40557629F1AEA696 },
    { 0xB391661473D86061, 0xB84EF4DDED83AE29, 0x657B46D3EB808530, 0x41850D77962BE636 },
    { 0x3DD8FBB9801D6955, 0x059F94C01590B304, 0x3C10EE432907691F, 0x105F4E0E12032A40 },
    },
    // 37B
    {
    { 0xF0A9C3EC124B39D3, 0xACA1DE16FAA37249, 0x0E3FCD40E1B7B1DA, 0x3A85599B0F8560EF },
    { 0x1070722571012F30, 0x581D9D97E5EE5C9C, 0x882EB1CE7CAD068E, 0x34BB262A2833F188 },
    { 0x8F451941D17C0812, 0xB7976D11DEBFE5F8, 0x5B97E8ADD61712B1, 0x6A7CB59FDB83A820 },
    },
    // 39B
    {
    { 0x07D6F90214B34933, 0x9D6ED1023EBEB925, 0x379ECA41462BB078, 0x5EBFF4E295DBA9CF },
    { 0x8BBED7E30B588F5D, 0xC4BB25CE7D2BB23C, 0x5BD45088737673C2, 0x7C7341CFDF42EDF8 },
    { 0x381996FB9BDC42AC, 0x3AE82EA1820D0BC5, 0x1907FBA55072CF11, 0x78793479A8044121 },
    },
    // 41B
    {
    { 0xB34322C1D8C02F9A, 0x7A9AE2E8B06EBE0C, 0xF4C3A75B8F265803, 0x7E069A2E7D6C44EC },
    { 0x91920890386E79A1, 0x1C0664BF73FE708B, 0xD98EEF769AA95B4C, 0x60272B601CDC625C },
    { 0xAC8F8FA5C63C19C5, 0x501B3558B3F05AF0, 0xAE5E20A95A2F9EA2, 0x106BAA3B155763BB },
    },
    // 43B
    {
    { 0x2F47E1B7E39DDBAC, 0x5672A1D29B1E5DCD, 0x54C9A53EC81DE491, 0x15C78F2379E77631 },
    { 0x4EBA7F589ADCFA98, 0xC259BEF8EF464AB6, 0xF4991583AB89C5D4, 0x5C5C56D43C85A670 },
    { 0x5AE0254E6FA8FCA5, 0x159CA957658FEF56, 0x8E8836C59945B36D, 0x424D7599675D0CA9 },
    },
    // 45B
    {
    { 0xD5283E85BAFE3C8F, 0x07F7C925BD3392E9, 0x0EC12A159CB59C98, 0x7CAF70BA987B3A7E },
    { 0x6BBB3E9E31EE7F3F, 0x904E2CA44BC2B720, 0x5B33B430F0A7BE09, 0x28E332F0646B12FE },
    { 0x92A52D8529765EB7, 0xAEA08FB7701A57F3, 0xF3512686E5772518, 0x7668CDFA4EE2A23B },
    },
    // 47B
    {
    { 0x0FFF3090B2D7E1C3, 0x041515445C6A8D88, 0xBB7B313F104C85D5, 0x2F00D9D3A14F21F3 },
    { 0xBF43B0C8290BCE48, 0x2B8E850305964FC2, 0x40515D54424CB88F, 0x7A9079EDA059731C },
    { 0x9EDA85F1E9C9ABC7, 0x4B3DB66EACCF4A52, 0x0D184D326F988F37, 0x7884964E6779E4C3 },
    },
    // 49B
    {
    { 0x42B5EC563A028EDE, 0x81FB52042ADBD50F, 0x1D2573E5E4BA1E75, 0x60B9D2DDB89BD5EF },
    { 0x29EC291F81B248C6, 0x515A027A1FD5B58E, 0x8405A54F3340F2D2, 0x0750AD81E812BDD3 },
    { 0x0BE2B718F87C90C5, 0x2D34B7158D7E2384, 0xFBC98B347A5978BB, 0x2FE2B2B0A5271D33 },
    },
    // 51B
    {
    { 0x244970F33BC84C47, 0x80EACA878E2E6B91, 0xF1D840CD72C91F76, 0x190A63A04EA3328D },
    { 0x89B72B86E236BD69, 0x07E855662A7C6E47, 0xACAA4F819886D6C3, 0x71388E20A622A550 },
    { 0xB0994C13A03328BB, 0x1DD250334513CA1B, 0x6CDB5C760C2E911E, 0x2DF54A039F65C19E },
    },
    // 53B
    {
    { 0xDA386731F9A1E97C, 0x8033DD4782A1F478, 0x9AED28D0D108BDB7, 0x25683C2CA278DE97 },
    { 0x320EDF8489644DF2, 0xD679C598BEB72BCE, 0x55FF4F017B4BD832, 0x1D566AE0A47B0F76 },
    { 0x42C71D9A6198C177, 0x6E4CE0238398B413, 0x9A042D9769CED446, 0x5D7F8658EC81429E },
    },
    // 55B
    {
      { 0xC9132185FA629A79, 0x34724E09930F9141, 0x84F799BC47E993E3, 0x5667CFB995ECCF3F },
    { 0xC3E2E3F59D10202F, 0xC77188F7F6D6AB64, 0x949297245694EAFC, 0x08873F437A66928E },
    { 0x2BE06C4F6B01FAF2, 0x8944431478581006, 0xA98B9384D0171DDB, 0x02E2ABF05CB49D1B },
    },
    // 57B
    {
    { 0x7BBB5DD844E10758, 0xB0FC2DFB7D07EC2E, 0xDB5C9FA731A210FD, 0x144FF78420C30902 },
    { 0x6C7CE7188FE4BF2E, 0xBD1F931808FCFED7, 0x70ADBF5045F692EE, 0x770AA60BC7F4C3AD },
    { 0xE7B0B19B7F6B4D56, 0xCD8B7273C19AE4A7, 0x89BC4BB2BA9D3FF3, 0x757DFB2154AE518C },
    },
    // 59B
    {
    { 0xAEF2A8387C89D466, 0xBE8E8B3D95867AB0, 0xED14AB5B48756A44, 0x096049E76DC00F6C },
    { 0x3ED46F06317AE957, 0x4C7E1ED258073DCD, 0x3EF5B56D06D6B870, 0x46DA5C28DCA81C62 },
    { 0x87E0A3FC9845C0C7, 0x834B4ED0508D7188, 0x6773AAB96D63EBFE, 0x59D543CCE331AE73 },
    },
    // 61B
    {
    { 0xC1A47AC47E0AB64F, 0x36D15D4D8D053693, 0x711316AE43E52D5B, 0x50966860E46AC4D3 },
    { 0xD787FA8F5E842E79, 0x3C7A8D44B3E99F8F, 0x37615FF0A830019E, 0x119E3794F4710D43 },
    { 0x697A6A883F8C63B7, 0x891766F9293F4E94, 0xD38543151E9E99A0, 0x4EDBBBC488C8A671 },
    },
    // 63B
    {
    { 0xD99CA3B4EEB80D5B, 0x715C6F6A0CE7C6D2, 0x0AEFA0B0FE956429, 0x3D71654500BE925F },
    { 0xEEF9A91CBE35E12E, 0x3D4BF16629EA8089, 0x3E3A7A1D5FBCD198, 0x58D6CBD6937156FE },
    { 0xDBF40BF684905CB1, 0x0F428C661F8CED4E, 0xF248E0A792AB0532, 0x0F68FE29D3311641 },
    },
};

// lookup table
// base[i][j] = (j+1) * (256^i)
static const DuifPoint base_v2[32][9] = 
{
  { // base[0][0] - base[0][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x97DE49E37AC61DB9, 0x67C996E37DC6070C, 0x9385A44C321EA161, 0x43E7CE9D19EA5D32 },
      { 0xCE881C82EBA0489F, 0xFE9CCF82E8A05F59, 0xD2E0C21A3447C504, 0x227E97C94C7C0933 },
      { 0x55E48902C3BD5534, 0x136CF411E655624F, 0x2D0DBEE5EEA1ACC6, 0x3788BDB44F8632D4 },
    },
    {
      { 0xC91273FE499E38E2, 0x4FA34ECB3D07FADA, 0x2D534D32F0EB0381, 0x6C86031FD43E9717 },
      { 0x454CD2B0215A6AD4, 0x4795C0862730567B, 0xF04F11B5D8B71BD5, 0x35DACAD334E492AA },
      { 0x21FD5459D2CDBD26, 0x9B60B5EEAECD67BC, 0xA807D042059EB518, 0x780D7AD89F5285B9 },
    },
    {
      { 0x5792D85426774B98, 0x012D4218744325C5, 0x608DA8014F80B399, 0x3D0B270DCD407C7A },
      { 0xAB308FF4527E6929, 0x9DE9A9FEF2E0DD3E, 0x4098F98D10A5EB5E, 0x555C8AC3AAADED31 },
      { 0x0A57499F86E86C3B, 0x2C4A11910E1AED31, 0x68B872A2C6796DA6, 0x6D141357895CDA63 },
    },
    {
      { 0x1439A8DCC77E04C6, 0xB3B2E37A3EFE929C, 0xE51A469EFD854932, 0x7407488190F2C393 },
      { 0xCAFF028502B40C56, 0x993F44B8AB307D54, 0x61F471E683502839, 0x53C99FA63A22D24D },
      { 0x2D09FDF4E23B7F7B, 0x374F1CA2BDAE60B9, 0xAEEDEE7C8815A24A, 0x7FCE865FB1AA9F15 },
    },
    {
      { 0x51095E220452DD90, 0x46A82461E3AF7681, 0xEE8DF5862D5FF622, 0x54A2E678A3710375 },
      { 0xBFC8C161D223EB5D, 0xEA800A68A59394DB, 0xF19E788E5C325043, 0x0AA53F39F58DAAF9 },
      { 0x5E5DEDF8C0954139, 0x93870403E85EE8FE, 0x5A0DB3858DDED396, 0x61D55F34B59DDB4D },
    },
    {
      { 0x1D067775BB8AB88F, 0x4D938AC4806457C4, 0xC032DB346D2CD39B, 0x68F2BDDB51661C5E },
      { 0xA4CC035B3DBEC652, 0xABADF14213E9139C, 0x5D842E739022A9DC, 0x1C5B2620D720BC42 },
      { 0xC2D61933817525AF, 0x5F387001A0D0DD80, 0xA9F25125841DE0A2, 0x485C748D4F86B0F1 },
    },
    {
      { 0x358D2E684A2751D6, 0xBA381A9D59CEE069, 0x38D92941142A1724, 0x630DF534941E493F },
      { 0xDD37964D551910CF, 0x365010A99DDD11D3, 0x4EF53B27C90C961D, 0x4EB76EAE97298BF0 },
      { 0x78C1B6E400DC59D1, 0xD981AFA3829F524D, 0xA94E20DD2C3BD6F9, 0x3D4FDD8E3507C853 },
    },
    {
      { 0x2CDBACB3026E9F3E, 0xB65981BBF1443816, 0xD899CE332F6CE191, 0x448AF3B030DE7297 },
      { 0xF153AEF6F9C91A63, 0xCB1EBB4070DAAC7C, 0x9613A0D6371E11FD, 0x5D481250990700E1 },
      { 0xF3E0FAECE4D1488D, 0x5C51B8BC45E653EB, 0xB1B2090C875B1519, 0x13483E2E1766274A },
    },
  },
  { // base[1][0] - base[1][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x97666E873197CE05, 0x28E85B4B3B44988A, 0xA96FDBB5D431BD2C, 0x76E9BEA4D007779C },
      { 0x76ADB1AA24D528AF, 0x5432E24F85E3411D, 0xC2860FF4ADA168E2, 0x186BB6B781E98ADC },
      { 0xB622220B9083725A, 0x7DA9EB404946BFB4, 0x5A39CF5234A69F93, 0x48634B88974325D8 },
    },
    {
      { 0x8653155041AC63F9, 0x351EA571BD102123, 0xBA3269D31D88F6EE, 0x41DFCDD7AA840377 },
      { 0x3249E213BEDF2FEF, 0x132EA7D68CD6BF51, 0x87006FE4231822C8, 0x12F30E55F6B37F04 },
      { 0x1F89F094662C3302, 0xB7AC39F65A2CBA3F, 0x505B1EF6E609347A, 0x2B36BC31A2C37116 },
    },
    {
      { 0xD082A142E32D17E8, 0x36320895798B33E1, 0x34057120398D772C, 0x0A7DD2F9A3C9D915 },
      { 0x8B1BD24FCE60841A, 0xDE472B6AD44DE228, 0x8E5AF607BFBFE96D, 0x19CBADE52F661AEC },
      { 0x9E6BA30B34C2FBEA, 0x2C9F2F4264E4002B, 0x17E1F95B3DB0898F, 0x0A414E7541FE2936 },
    },
    {
      { 0x90F38597A738F65C, 0x732B6EDCA0523BF1, 0x5FB2AB67670EA7C0, 0x02FE1DE229AEBDBF },
      { 0xFFA1BDC24BEECAE1, 0xB63A271855275AD3, 0xCF062EB09E42F445, 0x17ECE38F2FBAC0B9 },
      { 0x925C59D72957EF65, 0x9A4AB1C6769D9867, 0x19D25E41D4DF40CA, 0x5B9BB3A3AE328F82 },
    },
    {
      { 0xB1A04AE58A1232C8, 0x77890A200B60AA9A, 0x4F1C0A0644885E30, 0x35FAC82B98483E46 },
      { 0x97DD4CFEA068D6E3, 0x59838B37CB7A6813, 0xA1B1F8290AF81DD7, 0x4FDF52B61D8C7CCC },
      { 0x07D3BC78F0A0ADC5, 0x03204FFBDD61D3BF, 0x37A96BDC4D514D28, 0x01290E7B3D31AD2B },
    },
    {
      { 0x588A33903B97AF72, 0x747C4A58CB03CD67, 0x257C112680564125, 0x000BA9ECFBE6B662 },
      { 0x289FF705854EA94A, 0x47CC73AE07EFAD33, 0xEA30C3445FF083E7, 0x1FD0053F389C1676 },
      { 0x1E349196CB1EED9A, 0x8EEF43ED5A4B9C2C, 0x556BE8FCD048F942, 0x095AFF17D02476DB },
    },
    {
      { 0xEF95BE1356B78F49, 0xA5B36991A825C489, 0x46204EE03A8E45E1, 0x37BF49E103CB63DC },
      { 0xB8F87DE24B7E719D, 0xB9DCC135D6F9ADF6, 0xE9023930FF9462B0, 0x7A4DBB7CB7D89037 },
      { 0x8FAD7B02575356F9, 0x6091A8F8DF724E4C, 0xB0D4045AF77FB5B3, 0x47E7608780F010A8 },
    },
    {
      { 0x1EF9694EE2122719, 0x9581073A49EC6F05, 0x3664033F410610A6, 0x609BBC8B37F55C85 },
      { 0x3226AC5324FF0F22, 0x10FE575118D6BBBF, 0x81220E2D443FE869, 0x2480D538C1E288F9 },
      { 0x0458DBAA460D7C78, 0xE707BD3E12314CDA, 0xFBB0587C8F036C9C, 0x20DDC43DB9368909 },
    },
  },
  { // base[2][0] - base[2][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x3F11A62CBE3348D7, 0xB2444E9E8542DA64, 0xED7164861AA57D73, 0x05438F0386354F0E },
      { 0xA0743EA23A21A35F, 0x0EA46D6A0ADA9592, 0xBE1D450C509DB01F, 0x275B946097E6DEFB },
      { 0x1980DACCA5DE44BB, 0x39B5D71D2DEEA130, 0x06B0D6F10CEACF1E, 0x5F7398079342EA32 },
    },
    {
      { 0xA1FD3CA3C20F3A8C, 0x72E37D2CB1CE236B, 0x50832F0EF18295BA, 0x3EA3E35167DC4818 },
      { 0x7AE92AF24F3EEB52, 0x400B08AE30858F56, 0x1E4CCBAEC970C3E5, 0x49C0ABB14BC892E1 },
      { 0x1FED680A47786B70, 0xCE9F3A4D48AA379E, 0x38F63108135DC0AB, 0x0A467AC69A64F640 },
    },
    {
      { 0xF12B97BECD723AAD, 0xAB61A2DDC479A43F, 0xCFE885B6B4B05446, 0x53C7F5D6A75750DC },
      { 0xA352497B3C9A780A, 0x234CC25F7B420554, 0x2E50DE1544B08C2A, 0x5FF97D0F5EAEDDEA },
      { 0x58D5340FC6499CB3, 0x4610CA4E1014864C, 0x9C88A94890CE9E29, 0x20826E817F4E33BD },
    },
    {
      { 0xC090A7036D84B55C, 0x90D45B648672279A, 0xB2926095204F157A, 0x00B2DAD2477E5240 },
      { 0xB9595FAF08922115, 0x50FD0619CC519D5A, 0xCA65B080FD295B33, 0x16431D8057D7A9EA },
      { 0x78C8523A5042353B, 0x8977FCC26697BE60, 0x34AF14832C5515C7, 0x2C8DB3ECDFFF645C },
    },
    {
      { 0x4CDCD9B8CF8C5AA5, 0xF232F2FD50C6320F, 0xB084089B614F82F6, 0x644DA7C338180945 },
      { 0x1897868E405A4DFD, 0x2CBCA8AF55F9F645, 0xB93819E04F780E44, 0x1EF01763E547BDE5 },
      { 0xE91908169D75C96E, 0x70B129DA308B5430, 0x9EBF55F38C85D512, 0x64FAFDDD24B65F5F },
    },
    {
      { 0x8AAEB1460F4E2B97, 0xC526C35662C423A0, 0xC8D1A97B28ABB1F5, 0x0350D3614433A8AD },
      { 0x984A4D08452DE7EA, 0x6E206EB85E3239F5, 0x49614A60983E068E, 0x2B025436E5FD373A },
      { 0xB94468EA3E0BB25B, 0xB92A08A07020C5A8, 0x4F818D300C567B68, 0x104C4F44FF13A163 },
    },
    {
      { 0x8B3A13C5C2F57617, 0x2B10EE03BD6595EF, 0xB205260B30E5FA2D, 0x3985CCA87B82CAE9 },
      { 0x24CBBBFE9D16E636, 0x1942BE1652A7EC49, 0xD13CEC326903F1D0, 0x4201F68E86533F14 },
      { 0xE4A5969AC3A762A9, 0x62F36467CC1237C6, 0x7BE5A37D0B601AE7, 0x2DEBA2A184181EE6 },
    },
    {
      { 0xC2E249908AF3BC95, 0x632644D15EE6EEE4, 0x4E8F1ED456D1EBB1, 0x2DDBED891833FC16 },
      { 0xBFCD68CA94592661, 0x3FB5AA32B19AE0C0, 0xB35C5B3727E0391B, 0x099D3C0039C056C1 },
      { 0x04B0FA33E365315F, 0x027610EB108CA977, 0x0C11B03BCDEAA3B8, 0x3A06E536AC787069 },
    },
  },
  { // base[3][0] - base[3][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    {
      { 0x918D462B823C219E, 0xDBDA93876140A1CE, 0xEDD54CF571EC83CF, 0x1601FA92B61581EC },
      { 0xEFA47703A967E727, 0x61FFFD798376045B, 0x82B885955CAA2CE2, 0x0B0E92FD4B1F51C6 },
      { 0xBC878C3ABDA9D235, 0x983D809867862C3C, 0x98C81EBB92BF7BFC, 0x74CA345EDECB5DD7 },
    },
    {
      { 0x6C6E9EF335548CA4, 0xA428326117E06966, 0x4DA412331A7EF517, 0x149F0E273625171D },
      { 0xDE8F97A37A6D7F5E, 0xBE77808A523FEB7B, 0xE98FFEED2523D9BF, 0x69290CD239C82BC2 },
      { 0x1BB709A5C9288967, 0xB81BBC5AEE50AED0, 0xD822C4D7A30E1888, 0x6DB02E223F819411 },
    },
    {
      { 0x1DF4FF637873F826, 0x43352BCF3AF1A4B1, 0xAAA1778B0F0EF30D, 0x17897F7A662D5EEA },
      { 0x5CB2C02C90623E3B, 0xF3F80806491DC7E6, 0x000092B2817177BB, 0x5253B6E7545759F7 },
      { 0x852291596FE063A0, 0x0868373FA064D203, 0x63678A20BC67FB34, 0x2F303D928C521BC8 },
    },
    {
      { 0x5016218E52CB678A, 0xF1E216A05769F200, 0x692293401707936D, 0x100F9989CF22B834 },
      { 0x2C598EC7B66F8C0C, 0x1AE7D3A7E1B12C51, 0xF0D9FFA7B3730EB7, 0x2833D655B666EAFB },
      { 0x7EA93FB58401CE9F, 0x0C58A4B200BE0003, 0xEA9107581712D254, 0x5CBE5D44312301BA },
    },
    {
      { 0xBC0AE1FDE409BCEA, 0xD330CA106EF09578, 0x7FD4E07C42D47EEA, 0x7B8DA01160F0E129 },
      { 0x18609849F82CACD9, 0x711D50C6F4D4BCBB, 0x1117EA48B90EAF13, 0x119CE9903B37361D },
      { 0x6C3EECC3289D17CA, 0xFAD64DB8FCEA6784, 0x6835E18D8F5141D9, 0x598D0C490CCB8D3B },
    },
    {
      { 0x1328979D4EBAB94E, 0x2DE5F1443403A54F, 0xC276E0E088C07BE2, 0x4D64B0CFFB24D33D },
      { 0xFA8B37A2FDA7C063, 0x4E1B63EF30E3BAE7, 0xF1EA740DC820EC8E, 0x188B3E35C1DEFF10 },
      { 0xF9159C2129258834, 0x28341A1DF74E74C3, 0xFE4EB8C225312864, 0x30921B1A0F845888 },
    },
    {
      { 0xC5B1A4F18D16931C, 0x4EEFDB804DE9FE9A, 0xBFC5F8DC51D035D2, 0x0A915518BC6C8222 },
      { 0x6CCEA0EDC3A744BD, 0x04FF52F8B603EE10, 0x3C9E9633E807CDDE, 0x6375F1184F2F7FA0 },
      { 0x161C17A9B4B0A49C, 0xED7F204D5B96B688, 0x746419C8DB237913, 0x22FF387A82921836 },
    },
    {
      { 0x31792490644A8A3F, 0x02F803E41FB18651, 0xB7DDA2E97AE4EA5C, 0x4B30CFB6DABD1122 },
      { 0x6D243AD34B0605C6, 0x2DB4683B77871790, 0x03FDA8E79E85C7EA, 0x21468B11D071C96A },
      { 0x0427A52200D18475, 0xD4110CE1BB52E556, 0x6F5C6F2321E8DE3E, 0x4EC0AC96B05E9C63 },
    },
  },
  { // base[4][0] - base[4][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },

    {
      { 0x43B2DB4FBDC2E2F4, 0xB7F833C5E8B45D59, 0x1D3873BE0E9987CD, 0x1D2FB6A8D857C73E },
      { 0xB09B43AB5306D626, 0x8BF017B575D5EE2B, 0x3F8C9F96A66707BE, 0x5011A53BC4F66E78 },
      { 0x3B6906DB38BC5929, 0x038E1A7CEA8F68B0, 0x7B15251059F208B8, 0x3E6B411A9E7FF1B3 },
    },
    {
      { 0xD332E6B034567A70, 0xA16C968C1E6BF1E9, 0xABAC9C4E99B012EC, 0x5F78129D95966C7F },
      { 0x05F0D22DEC43FD5B, 0x154235195D201DB7, 0xECC9080974B73000, 0x141C64431DEE04A1 },
      { 0xE8B5D867A5232818, 0xFD24B5A08AE2BBD5, 0x4167D7457A55A0CE, 0x10EE5C5303541409 },
    },
    {
      { 0xCD46807D5F3B98DD, 0xC101B03F314F0C44, 0xD966011BA1F9ECBF, 0x2EC206DFB637B3C5 },
      { 0xAE300223464ECFE4, 0x12A004B76A1551E5, 0x892DA6A6097717CE, 0x05E1E840CA518ED5 },
      { 0xB8371C06984FF0BC, 0xB75816D35CF0B2E3, 0xABDDDD4CBED71055, 0x5D213B119560CB6E },
    },
    {
      { 0x9DFC60B96DA23F5C, 0x2FE7E20FE31416DE, 0xC0567FE03AD50AFF, 0x43B864F41270D4FC },
      { 0xA5A121964538427D, 0xC4C50CF1EFDCF2A2, 0xDF4F8010CE2C722E, 0x0FF8BBE750B6F5E8 },
      { 0x67B0ECCD22DADAF5, 0xC304C2748D9D3C92, 0x7398048C981F1F44, 0x5CF9327EA0A8058F },
    },
    {
      { 0xE8CDA555FF04BDE7, 0xD236FE70EFF00C94, 0x61E4844A165378FF, 0x72E31093961AF8A7 },
      { 0x53D69A0BEDF3F14E, 0xDECA1BB515CE09CE, 0x50748DC749ACBDD4, 0x0B896B9A34444C20 },
      { 0xF395C4FC6718C9E5, 0xA68819AB5092E05D, 0x020CD49E970E7F41, 0x517CC00558CE7139 },
    },
    {
      { 0xA1014FEECD377ED6, 0xDC89675F1A52A4A0, 0x3207B25CC3DEF9BD, 0x20B8D269C2CC655A },
      { 0xB02D1B451F4F7C5C, 0xF1F4E01152A8238A, 0xAA9EA4582F921247, 0x49FA0B66B23B1372 },
      { 0xFD13AC554CE4A646, 0x118037B7D8005C03, 0xFDE948EED6ED29C9, 0x28410A7D2BA5E8D5 },
    },
    {
      { 0xA30D0ADDA9E801EB, 0x590814445E79E4B2, 0x93E2BB3AB6341D2D, 0x1D3BAC5264365A23 },
      { 0x6101348A9F6B7F1C, 0x532D339CA88EBBE2, 0xE5EF1323160A57CA, 0x517CB07637D5D3A5 },
      { 0x2A4088FB49D7283B, 0x8ED710EF8EFEAA53, 0x091246487988AF32, 0x2ECFE8AFC6F3FA4A },
    },
    {
      { 0x9F922695776BA90F, 0x471D481421974B0A, 0x70B25D3B974E0B6A, 0x1DE0C3FD23F5CC6C },
      { 0x818A0469B6B1B936, 0xB51BCD77EBE3DA99, 0xD4F0C7E2E6577125, 0x59979AC8A7C7DF69 },
      { 0xB6A3808AF5436106, 0x4CC55BE5B6236892, 0xEBBC195A9D3300C4, 0x2286C0E74837DD01 },
    },
  },
  { // base[5][0] - base[5][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0xE83A6C4B0E573A16, 0xFC368C7AF70E31F6, 0x4BDEE2ADF3FA7694, 0x665D693CB31D5884 },
      { 0xB73DDB50D3102931, 0x552790EBA09E4741, 0xB7AB68AAF447AE59, 0x56F12EA5D31A2DF0 },
      { 0x4068C812506B8FDD, 0xE292E1057D94457C, 0xD8D1CBA5AF9D320C, 0x7EBFDE77F1003919 },
    },
    {
      { 0xE6BE2EE2F9E1484A, 0x63C0D14D154882D5, 0xC0630E9B210E182C, 0x27CE68CB6E6C6A6B },
      { 0x7D778F35133593F7, 0x433634626AB9CF8B, 0xFB4517DE0D81BB16, 0x6CBAA1AF43DBAD46 },
      { 0x8CC94BEC353D9BB4, 0xE8682C120D68BD31, 0xDD014E56AE0E060B, 0x3E66E8421C3D0183 },
    },
    {
      { 0xCD86420C33B06640, 0x66D7003D0D59954C, 0x5446F6433105ED0C, 0x5AC9E54240C86522 },
      { 0xEE53211636930202, 0xD70A9EA84A41205E, 0xD4E060DA7DB4633B, 0x6145E876B0E867A9 },
      { 0xC9098C4D2F424D4A, 0xEA6C619AB2EC7D66, 0x46292A2DA9FEDDE8, 0x539CC1846D16B1F3 },
    },
    {
      { 0x5CD0872605381220, 0x87D12C336ABE8DEF, 0xFFDCECDAE693ED7B, 0x6B9614A2A49619FE },
      { 0xA161C69421AF6A00, 0xDEA879B0193C6664, 0xDD83D58D3CED01F7, 0x534ACBD75F4619AA },
      { 0xE3BFE3A2EB66985F, 0x726FF469F1DD577D, 0x51164418552EED06, 0x3FCC2A4C602DE540 },
    },
    {
      { 0x69AB0AA907DFB1A8, 0x04022D22E7A6FDD3, 0xF76127DE439FD061, 0x5879329E6B4D8973 },
      { 0x1C24E7444F85F082, 0x4002D68DBDAA5144, 0x1ED1E1CF91FE490E, 0x7B6176238518F982 },
      { 0xC504649C55608639, 0x230BCDB06D93B5E5, 0x5490600F0737D638, 0x578939F8ACB239ED },
    },
    {
      { 0x982445EBAAD385E0, 0x036B5AD278EA2173, 0x7568D34F5E2CB0B1, 0x1C560CCBF6E2FBC2 },
      { 0xA39CFE3E45700EFF, 0x7EA93A48253555CF, 0x20ECC54143B94797, 0x6ECF2B956C2DB4F9 },
      { 0x03335A8BD3A8D894, 0xBA3E83433F4DC2C6, 0x56566008A2A6EF24, 0x516FE6CE5FF4F34E },
    },
    {
      { 0xAB762CDA081DF047, 0x1771DD76692CFCB4, 0x3CBE594A09FAE699, 0x47F4C3BC1266F239 },
      { 0xC6EDE97061868663, 0xD6C7332FD65DA199, 0xC7B592C619154B0F, 0x75948B602A2460E3 },
      { 0x3F6D9A68855D4894, 0xA751E6C11736D607, 0xB3041EFFB2BC7C0A, 0x66181F983FF8050B },
    },
    {
      { 0x94FE01AC06ECA280, 0x76693D5237DDF649, 0x89850AAFE17153FC, 0x20B58A8D5B8350EA },
      { 0x69851DEB0BD94639, 0xE2E9BBDB9CBB9DF5, 0x636373C60F352E5F, 0x46B0DC7BC5955BE2 },
      { 0x2B546BF7F4E09B58, 0xDE83F2E6AC722590, 0x57F317ED0DABF055, 0x0C8D157BA13BF469 },
    },
  },
  { // base[6][0] - base[6][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x4FF315A1A7A30774, 0x6F6981EA531B03EB, 0xF82910875BD06D12, 0x51BF3EDF002A2DC9 },
      { 0x670B7BA5E29E0A0F, 0x95CB92E7103976EF, 0x5C5CE1B7DAD91F73, 0x7F17072285AE6484 },
      { 0x009ABAF6B380DA18, 0x11884A734F85FE88, 0xBA99078AC1F23F91, 0x38D7D34CD888AAF1 },
    },
    {
      { 0x75211E0E239DA86B, 0xA8F43D0F9D9C7788, 0x4DC25FAFD964DF4A, 0x00398FDE3C7C4D0E },
      { 0x32E737CD9CA9DB05, 0x632C1CF557D0A0F3, 0x07A1AFFED4FBACFF, 0x4108A174E158E147 },
      { 0xF218638C247C0C40, 0xDFCB06112F67608C, 0x5B6D7041B5DD0AF1, 0x2626B799A3F0AC04 },
    },
    {
      { 0x97866EFE4C478CB8, 0x35C8B113D85CFA8D, 0x3763DB623BC8BB5F, 0x1C5FCA805447CFD4 },
      { 0x0C7BF667E0BE8FDB, 0xB63AFAD328A01E0A, 0x6DEF3895FBF7066F, 0x4C9FEED553F23D11 },
      { 0x0FE9649E1BF443AE, 0x517B0F2D0C68A316, 0xA8407AC11C92093B, 0x7537DCCF5F86A4B4 },
    },
    {
      { 0x7758915ADB7211E3, 0x49CEB80879437FC7, 0x485495418EE7AEC6, 0x09B7ED4FA162F588 },
      { 0xB52360DDAB042AEC, 0x920B5D9C7C49F84E, 0x6B8E889BC7B8D660, 0x7AFBB48A518C4B75 },
      { 0x7CA66FD8D182DEDF, 0x079B25CECFFC1604, 0x1543EC52E1DDAC45, 0x4110C1A885F46E5D },
    },
    {
      { 0xCEAD3880A1983D36, 0xD831EF4F623ED22F, 0x115DFF295F493D69, 0x49C3E220FEA02136 },
      { 0x257BB31C2F56968A, 0xD0476C40653E2C18, 0x8689D37308108F1E, 0x35038E70BDC03601 },
      { 0xDAE9E1E8C3CBC57C, 0x3915AD1EBF872209, 0x06BDA4245DA3BE50, 0x18B8D935578F6E49 },
    },
    {
      { 0xD306DBEC594523DF, 0x535F8A6B0BB85278, 0xEA50FC49A9EEDEAC, 0x7628A5319A2121F4 },
      { 0xD49798C84BAB2654, 0x7FBDDC26113AF08C, 0x27AAFF1BD243A8A8, 0x110FEA439E7841AD },
      { 0x919110279D0AB197, 0x7DB9F074DD050196, 0x7E706EA620878187, 0x646D52CB7DC92555 },
    },
    {
      { 0x0A7B0EAEE4264BC0, 0x4CA0FCF1F7A0C103, 0xE6FADC479A313BD6, 0x6C641BFD07453CD4 },
      { 0x376547332E52CE5A, 0x542392A5971C5650, 0x98D7E384690F0BE7, 0x73B6EB7E656C257B },
      { 0x867CB442CB7E482C, 0x0EEE5DF9BDAB500D, 0x6E6173BEA49AEB35, 0x0E27B9F96352BF85 },
    },
    {
      { 0xD9B738377E3E1A42, 0xB9EFE4DA61E0E7B0, 0x758EBCE4BC0E63F2, 0x3822CD6DBED7B3AE },
      { 0x873D27DE982FD054, 0x414EA6702A6331D6, 0x7A10E1C197F19C24, 0x7CAD64068DFB2621 },
      { 0x0DC8EDA4C8DA15D0, 0x2B934B11A5816E65, 0xCFEFCF728FC63C6E, 0x6FF0B14246710FE9 },
    },
  },
  { // base[7][0] - base[7][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x943CC296AEBE5904, 0xDC6F6EB8343EF973, 0xEE05FFD590B43C48, 0x15A26021B3BED51A },
      { 0xA72C90A7F0CA4B0D, 0xA4DF3EE386B8E6A7, 0x498067E91DA87916, 0x23C4EA237E48B919 },
      { 0x8D0E43D583A75BC7, 0xFD6368C74CED7A33, 0x1F565DE6A427C833, 0x3062977795DCD272 },
    },
    {
      { 0xB815E2E13E57367F, 0x2263B4CDAA524655, 0xF7DE202B5D249759, 0x786BB9246CDB33B6 },
      { 0x05AEC4DE1DFEC5EF, 0xD835C91BE4F9AA8D, 0x87260B586A98147A, 0x485E4E189667E555 },
      { 0x554574259F615024, 0xCC34CF7A768BC0F0, 0xBCA289F2384742E8, 0x71BAADE9D4BB7A09 },
    },
    {
      { 0x1EE3880C4BF8D652, 0xAEED3EAF60B2DDEC, 0x28472DCE07D08107, 0x53B1BA8B9BE2952B },
      { 0xDAAFD01F15688420, 0x1AB7BAC84F731AB4, 0x4FFCF8FEDF34DC48, 0x46C660E245E0B7C2 },
      { 0x814A0169B75A0CCB, 0x785A273F3BDA3052, 0xE7D431186A1E24AB, 0x78616EC53D68B373 },
    },
    {
      { 0x48EA4B3EDC76BF00, 0x3A1297856BBB40BD, 0xF204C17006C292B2, 0x595C309C0B529E72 },
      { 0xB2B0CA284FB7F607, 0x77173F54A36328C6, 0xCB99E0F9B3F04DAE, 0x1707D631B1CA424A },
      { 0x3CF3FBDF722466B2, 0x35641D33843C4368, 0xFC4FEA6CD0726D97, 0x20BC90AE39AD27A0 },
    },
    {
      { 0x725719DC9435E69A, 0xDBF7BF5B2ACEEB6E, 0x13C58A0FD9E9C70F, 0x18FD42B31120E143 },
      { 0x46384A73EBEE7695, 0x4BFDC561A3E9CE38, 0x709DF019D4836C81, 0x3801A251866CCEBB },
      { 0xD7C13621171B117A, 0xE09014C3CC19A816, 0xCDE0DBF0959C4891, 0x125D918954CA9244 },
    },
    {
      { 0x20FC06157AFC2E2C, 0xB4394261827D33CA, 0x44A2EFCCD1DD0DD6, 0x468E957CFFF5AE8B },
      { 0x58D4768B996F33D8, 0x1E5A4A0C230DA4A4, 0x475EA1A1BB67DE69, 0x47F71F438F0C4004 },
      { 0xD4ED455099310F66, 0x985C1150AC9132BC, 0x20020CBDD3CD60C9, 0x4B566BCB8C298EBB },
    },
    {
      { 0x64ACE362BC43DB4D, 0x4A70CF56AFC87F5D, 0x0B71273151A17A82, 0x4B2769A58C0B0B80 },
      { 0x396FB95796CD8E95, 0xB1A3151B5219122D, 0x9F67503C8B59CB1B, 0x491F077B5C981184 },
      { 0xA43F6CA60C97F34D, 0xB0D716751D488A89, 0xC3BDFB69DCD26F13, 0x3C6D07E30839F9F5 },
    },
    {
      { 0x514FC078B4061D4A, 0xB8FBBF0A8D74F3F3, 0x888078AC2400BCB9, 0x02A552598B59C6EE },
      { 0x2DF8AE947295E335, 0x9623F18C38780D47, 0x120CD7DE036145EE, 0x1692EF7592B58B9D },
      { 0xEFE423468C933E5C, 0x85943C4E3372A6D7, 0x15758E953337760B, 0x09A3085355BED3B0 },
    },
  },
  { // base[8][0] - base[8][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0xE69532F3BBE8FA81, 0xAA44C8C3C7D53078, 0xD8DB9DDE6D5E0372, 0x72A43C65D4BE64FD },
      { 0xA889C763C6FB587F, 0xA9CBED44F2BAFA8D, 0x04903D0EB8BD78DC, 0x10817EDD15906B28 },
      { 0xCB4F720282AE7347, 0x9B5E53B40928D694, 0x1D0D7A8BD53ED20A, 0x456B92ED94F6595D },
    },
    {
      { 0xFF63DE064D82B7B9, 0x29BEA93473FAFFEB, 0xBBD7E3312189577D, 0x67B3AFA9811CCFEC },
      { 0xEE2133D8C1A71222, 0xDB3AA25AB8670DE2, 0x0D783D05FBE8AF6B, 0x6577E7FDB8D01B28 },
      { 0xE1969B1B020A8B8F, 0x6695F788C4CC241D, 0xC3853756E84A2888, 0x05E65DB9515432B0 },
    },
    {
      { 0x0C36AF26287F094B, 0x701CBDC17F744FBF, 0x1DE3FB62A83818D8, 0x333C7EB488479BE1 },
      { 0x8C2F4B17F558D4E4, 0x4373F31AB28A3EE6, 0x58497018DDADB6F9, 0x20127855ACEB5B9F },
      { 0x0AC37D18B1B431E1, 0x03FB46242B9699F9, 0xA7B9E64FBC4F577E, 0x16A171084756A380 },
    },
    {
      { 0x10B8BD8687A9BAC0, 0xC8A73485898F0326, 0x0DDB43D73A95704F, 0x6105F9D3CDA11E37 },
      { 0x4BFA898ACA6FE944, 0x30AACC2E989FA635, 0x75D09F838422A808, 0x73B593045C696991 },
      { 0x409C5D328E2D959A, 0x4338DB76188D8DC0, 0x3DFF8658DE189AD8, 0x7A2E97FD4E0678F0 },
    },
    {
      { 0x301B6FAB90E9A735, 0x58EDC413CCBDD9E8, 0xE9E104E1E43AB57D, 0x0370ADF2A60EE41C },
      { 0xDFA92D0F15E4E455, 0xF52D93041323CEC0, 0x6A88E3876F80AAED, 0x4D711E75CB067AE8 },
      { 0x2DB92EC38C994CA5, 0x991A8E5AE758ED58, 0xBEE20AA4ED5BE502, 0x2C6F6C3093C760FB },
    },
    {
      { 0x16FDADD45B6164D4, 0xA47777C77A962CC6, 0x19C04883F8968AB9, 0x045D34B5A98EADEC },
      { 0xEC0B9BC9F93362AE, 0x6464BB62E622A724, 0xAE71C17C5E1361D4, 0x17F9CEF42A42FB7C },
      { 0xBBF69F7761F7E2BD, 0x82702A8BEA7FA408, 0x751EBD1FF8D338E5, 0x090319DA4A3E7F2A },
    },
    {
      { 0x415E98A3A4890805, 0x6F11BDB6BF37DF03, 0x708F3B0C88F53CE3, 0x03A19DF1E59C9DEF },
      { 0x05CA4C3C48B08021, 0x2773D89E7675FD74, 0xB85F39CACA785260, 0x1AE9854CDA6AC8C2 },
      { 0xFFBCA2602E74CBFA, 0x2BAE9EF2582E28D1, 0xAC19C0FEAD3B423E, 0x16C39F6F3D7B6D4F },
    },
    {
      { 0xD5310170A72EFCB7, 0x5106AC8BA80AF0FA, 0x8C513AE9DD710EB6, 0x42A1B0C500B00129 },
      { 0x8ABD18B2219B99FB, 0x7D5C5BF77A5540EC, 0x5849FF737AD32403, 0x573B9B2A383FD3DB },
      { 0x86F55EFA4BA611D7, 0x55378512CEE72349, 0x8210165C514D5D16, 0x658A219B1683CB06 },
    },
  },
  { // base[9][0] - base[9][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0xCB3E2A748E294E5C, 0x987B134932631AFD, 0x13A3D7FA3C090CB2, 0x4B81C20C757B37AE },
      { 0x66625BE3DB370FBD, 0x220ABF12FA8617BF, 0x9F7836FE389F578E, 0x2C17A233A96D31FB },
      { 0x6318BDE990192672, 0x54082174522445E2, 0xD90F78C5A72D09B2, 0x06150E25E6D146E4 },
    },
    {
      { 0x76E240A434DEB499, 0x86B6C83EDF0E4691, 0xE31DE9096AAE62D5, 0x6D354D98518A6E41 },
      { 0xE926E3E8378F821A, 0xD9134F1F6DC3E02C, 0xE8AD81397DD96947, 0x7E2AC5E8E37B243B },
      { 0x68760A9269CB2315, 0x095DB14561AD1278, 0xD2861D3C8E5E2FD2, 0x420252E5057DD7E1 },
    },
    {
      { 0xB15E4F0D9520B7DF, 0xDAE37B9471A82CC5, 0x021A1FEC1EAEB4B3, 0x5CA93A8B73FC774C },
      { 0x460FA0038553A1EB, 0x665D6865AD932F74, 0xABA58235B347E96F, 0x231CADFEE56ECB19 },
      { 0x08BFED968D2ECD4E, 0x4E3BA2DE68802E15, 0x77EA5F78AA6AB7F5, 0x3B2BCD14F411680B },
    },
    {
      { 0x999E5A89A95A1A79, 0x6C191424C9EF4070, 0xDAA89443BA869AE7, 0x01628A5D9513BBE0 },
      { 0xA2DB473F24E01502, 0x91E6A8D15E54D1BF, 0x9F6B2F88F611260D, 0x61D1C26E4F02DED8 },
      { 0xB425EAED45F8DB19, 0xFDC5E9BF7B5AA5A9, 0x989C8B6BD4D86929, 0x48B04904B0AA402C },
    },
    {
      { 0xBD1C2B0B1B4DA6DD, 0xBAE01653B2AE1AB1, 0x3EE10DFCEA78C010, 0x57B1BEBA48F37021 },
      { 0xDA268B3494ED67D5, 0xED294FA64209ACC7, 0xF4F7B1E5229EAAAC, 0x1A8F092DE2B4C705 },
      { 0xEA5A4DA30D7B3DDF, 0xEB0181BD6455C4B0, 0x38EF70CFFCD34CFD, 0x3F8C168373E7154D },
    },
    {
      { 0x04A2A5B94710BA91, 0x552C747A6A425C6C, 0xE9AC12A6BFA3481E, 0x22566021920E290B },
      { 0xBD3E47325580B476, 0x65AD252A8AF6E2A1, 0x04AA8CE9A3E6876D, 0x33EA56461A1F49D8 },
      { 0x8E3EB5DDA7BD2BB2, 0x459AFF6A48C189F0, 0xA56E50E364B5A342, 0x6AB68E418956B8DE },
    },
    {
      { 0xC0F833AB588DF407, 0x87D7FC118851F9EE, 0xFC59682AB54CA32E, 0x44BD5F1C66463F82 },
      { 0x0BF7A07186469CC1, 0x18FB839F0AD1FD1A, 0xA790F9E583B9B237, 0x3A3636368EC1277F },
      { 0x8624E4C3BF5296D2, 0x2621B4AACDEE0EA1, 0x81161C04FBE675E9, 0x2BBF0A51A5F7425E },
    },
    {
      { 0xCA7F675F5EA6EB8C, 0xFA3527ED03079108, 0x89252CBBE06468FF, 0x7829825C7D80494A },
      { 0xF871345630D39D85, 0xF97D7D081BC8D2FA, 0x60F09F4135B68074, 0x307D3F74B7EBC7A1 },
      { 0x5B1E8E9AA694B763, 0x79E1829F2FD698EC, 0xB385CAC65A5EA176, 0x109CC70650B1A9FE },
    },
  },
  { // base[10][0] - base[10][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x13CC557CDA5BAAF7, 0xAF563909AE46D6B9, 0x69675530B0DBD011, 0x4DDFD94274C7BEA7 },
      { 0x44FA82C51C1599F0, 0x2D715D05D6A4605A, 0x47C9DA81D29ED9B7, 0x6D51F6CECAD11973 },
      { 0x32B3BBF4E3ECB2A7, 0x6595892A3963C01B, 0x3282994CECA83777, 0x6503F0A72F44ABE6 },
    },
    {
      { 0x9205AC66E23BD244, 0xFE9C6D6F3223F80B, 0x8CC9469953E43556, 0x6857BD76C257D040 },
      { 0xA7720965CC06FCC3, 0xD18AEBB79E3763B8, 0x5DD2F6EF492E3BFE, 0x5F85D61C8E989A01 },
      { 0x3727EF008AFB2DE9, 0x94CC131090B084D9, 0x3C0102C085D6EB6C, 0x4C90D18B5D75E803 },
    },
    {
      { 0x6BAD56CD6CF9E0BC, 0xAB350777B058E0CE, 0x9F4D05D612AE076C, 0x7D824EF6503163FA },
      { 0xC4A117BF6FDC387E, 0x9614B5F5A7BB59DE, 0x039C78EA1B6126FB, 0x322C6FA0F139D758 },
      { 0xEE65F1BD1AA22238, 0xBAC43C9987F6DF49, 0x3C3002618962EEC3, 0x7049EE65E14A8732 },
    },
    {
      { 0x35EF775F3042019C, 0x98CCE15B3C07DC2A, 0x4B99BB55DB16834A, 0x7718C064C5B23EC8 },
      { 0x8FF9CD42C2F0382D, 0x9B6852EC59F39C99, 0x21DCF970B8C7A29D, 0x6BE8F5042413D4BE },
      { 0x773D5B73D094582F, 0x5260ACB6C9D445D5, 0x7BDA6F415910B098, 0x5B1F4CCEEECBDE8C },
    },
    {
      { 0x178C246E7125D763, 0xBB4DB92ADD5E57B0, 0xC8659E371E77F498, 0x118FCBCDE37CD9AA },
      { 0xCB5421E09AF70FE2, 0x4BB759AA84726467, 0xDA17B400DAC66998, 0x24774DBC349D0295 },
      { 0xAE18EF25E61579E3, 0x5825D8187F10468F, 0xDBC6B804E0A7DA33, 0x03CDFD4D843C9209 },
    },
    {
      { 0x79E4F6C0516AA119, 0x855045BC3BFB1CA9, 0xEBB6D631E8842A3A, 0x4F7A7D8ACA3831B5 },
      { 0x71C81D28ED1806FA, 0x421CB2119ED4AD58, 0x769E7896859AB240, 0x01C63BFB4240B8CA },
      { 0x42A72F732D8B3DF6, 0xACAC85214B6866E1, 0x39596F9A4C0810CC, 0x2BAF74952505FFAB },
    },
    {
      { 0x2EA35E2285526BF7, 0x61D78913D299DCEC, 0x9C4F1D9315C48361, 0x50050F3F1C17AC0D },
      { 0xEA604048450C17DE, 0x1870B8614CA44EDE, 0x02DD5EABA97B99EF, 0x61EA73889669FE80 },
      { 0x28C6DCB3F57C9D59, 0xB8DE4C4D82B32960, 0x7F15C2ECAB38CBFA, 0x42876529328F271C },
    },
    {
      { 0xCBD61CBB30733475, 0xCD8CDDFF0A9D5A4B, 0xA658BCDA9A7653CF, 0x30A8E04FD098D72B },
      { 0xE1A18D6F229F864E, 0xF4FA822F7FB81DCD, 0x7E6CBD64F6C23D9E, 0x258773610E2C7A63 },
      { 0x1D7AAE06FEF82ECB, 0xEE931770155A773D, 0x88D95DC3890B8B84, 0x0FF7927D40078185 },
    },
  },
  { // base[11][0] - base[11][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x116957FA984BB5C3, 0xC6C85C0361692302, 0xEE50C4B626F2DD72, 0x14002FF3641A060B },
      { 0x1BEB29FD8D5398CB, 0x07CA4A981FEBB20C, 0x56900584FD9D0BD9, 0x2A26A49497E4309F },
      { 0xB577DD4F9A294344, 0x2E0DFFCA12883ED0, 0x7BADDDE6B36CA59B, 0x397239498798B6FD },
    },
    {
      { 0x03F9FB1AE9953B0A, 0x3D5526C32FB2B378, 0x1E42F3CB94682228, 0x4FF73F8007F0321C },
      { 0x934A90464BC0841E, 0x58A8150591A28770, 0x7ECED75301F7EF01, 0x6D4E97461399D1A6 },
      { 0xBB2982ED01EDFBE9, 0x526D79248A1A66DE, 0xBDA56AE6E9254476, 0x407CA028F70202A1 },
    },
    {
      { 0xEBF7C9DD83D7CBA0, 0xAC1F68679EDBB353, 0x6734CC5FB7058F62, 0x63DBFFE92EEA0229 },
      { 0x469AB591E1E99859, 0x790E45CDD8238D83, 0xD9B618B637215C1E, 0x03EBCE3F45F55886 },
      { 0xC3FDFDCE5E046E89, 0xC503359D70F7614D, 0x06AB9215ED8FE0DF, 0x0E1A9051AF5325DB },
    },
    {
      { 0x66D437A010B5E023, 0x0FDD918E895E6C3F, 0xDA4AB54F0BE384C8, 0x5C3A861DB368972A },
      { 0xC06929D35E65D1A5, 0x1F30E1D09C1C10CD, 0xC861DB00CC4171CB, 0x0E1E82BBAE877337 },
      { 0xB49778A04A11728D, 0xE5E0639E15AEFB38, 0x1080A7F3BA267014, 0x0310F163E998243E },
    },
    {
      { 0x5BD70BCB586DF870, 0x2A6FD7DCF0BE70CB, 0x92C91838F4D551DA, 0x6EC72C4E5080174E },
      { 0x57CC3066412CC1BD, 0x48752460E34FCD6E, 0x3293241BB2AC0F18, 0x4003EB04BDE9D2DE },
      { 0x605F8ECA8421549C, 0xD969E1B1AC47971F, 0x054B0A1C5DA8F177, 0x4AC1EBBC1E0E5FC3 },
    },
    {
      { 0xC801A382664E945A, 0x0E8DB3CF7B9662C7, 0x0B7095AFDF2DC393, 0x64AC032741E2AC05 },
      { 0x76775177AED13D67, 0x2CBE1D0A2AB380BA, 0x64D3150933048B3D, 0x52952F9740F6C7B8 },
      { 0x86944A13283373FD, 0xFE61FBC2983E4635, 0x0DA9ED3C0608897E, 0x43CE0B85EC21D9C4 },
    },
    {
      { 0xE6EB66A8606AE82B, 0x4D7BB436DD81AB9D, 0x1E53391FF9E1F7A4, 0x33B4606B98BDC566 },
      { 0x82837672327D37F6, 0x5F71A18F3102F291, 0x1ABCA11228DC7521, 0x76F602F1A564FD80 },
      { 0xCA5B12F2F8AAE0D0, 0x20BDF9D3CCBDBDC8, 0x61165EEE36B59300, 0x68A22F0A6EE6A97A },
    },
    {
      { 0xC498A3D595DF5221, 0xC629D127C903C894, 0xA5A4FCA45F187BD3, 0x4974C8043727EA1E },
      { 0xABA815A59D8A249F, 0xC733FFB5A225DE59, 0x5C5EB4938B31C2ED, 0x498C3798F1C94AE4 },
      { 0xF8864B59BFEFDD97, 0xCFCD49AF090E757C, 0xEF889B621D2DCC1F, 0x3BD971F82E9F4CD7 },
    },
  },
  { // base[12][0] - base[12][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0xEACC31CE096ED852, 0x52E8CF9860124335, 0x68BE1781AC7E7230, 0x03D0CA8A9704AF45 },
      { 0x94B7D4E2CE17626F, 0x5E45B0DFA7C279E5, 0x8E3B836C8BD47C84, 0x31DBCAFE3D6992AE },
      { 0xD41B47811C4F2FE4, 0xC8219D8167C6F21D, 0xD7D0FEAEE2A09321, 0x1F47F41E8197809B },
    },
    {
      { 0x84382646F477E89E, 0xEFE28D4719F01B98, 0x52CEAED2893066F1, 0x116B044CD312C643 },
      { 0x97C58ADC82B8514A, 0x4A792138338422A4, 0xEF0E2D70B0DDFEC2, 0x3ADD1DBCBFD62003 },
      { 0x311CEDE03866E8CB, 0x307F4545B63EC54D, 0x59C423DE75A00930, 0x0482683DC3BBCF2F },
    },
    {
      { 0xFA1916B3247CA053, 0x834A97865E96861C, 0x0B3B4BD6D040FC98, 0x712055675D7B9536 },
      { 0x5A670FEA6EDD48CE, 0xE798ED9F63A646D5, 0x9631E631D6C36628, 0x21F10A1FDE0EEF03 },
      { 0x7C1A3A4E2DD14AD0, 0x6B4A3E2DE51BE92D, 0x33789DD3F3E498B5, 0x2B5ED791C6DA0656 },
    },
    {
      { 0x898869B660CE9DD9, 0x031535DBB111C35C, 0x3E4DC2C8EBD0A7AE, 0x01D518A83F0F2BAA },
      { 0x1B155CF1FA9A99EC, 0x19C2B46AB75C9EA0, 0x4F070A290EAD2AB9, 0x4E925436C1BA098C },
      { 0x7A763B247FEA6706, 0x7022F5782A56460E, 0xC46912C10E849ABE, 0x61D930EE4D75A42C },
    },
    {
      { 0x8CA89EC5B64A89B2, 0xCA7F38930005FA3D, 0x0146886EEAA7CAB3, 0x015A6AF1214A04B2 },
      { 0xF2AD8F0CC45DBCD4, 0xD04F683EE0BD1ACE, 0x58161771301EF519, 0x59302AE7AD93B5E1 },
      { 0x5A50AAE59468C6F9, 0xF56623230C367284, 0x624E7A49B64121C4, 0x13D36404D72E9A08 },
    },
    {
      { 0x66961385621EB4AA, 0x6EA51F2BB5336559, 0xBCFD2C9234EB81B6, 0x110A81B01EC612CC },
      { 0x45D375E68F86D8C4, 0x1BE9EB9D33AD2DF4, 0x79176FD18AFAAC2D, 0x165B38BA7FB050BF },
      { 0xACF766FC9C85F0E8, 0x54A11022394671F8, 0x41448E333D4A787A, 0x3D8EFA5B9C487A1B },
    },
    {
      { 0x2F97110C03FC7AC6, 0x71AAAE4FEA4A04EA, 0xD95554468FDB5318, 0x3434C122E9A9701E },
      { 0x7249797059D95112, 0x3E364F0315AA88B0, 0x0AF5C7F106BFBD87, 0x30FE77932C7E2CC9 },
      { 0xEDD8AEC2950C0C3D, 0x79F25569C36ED66B, 0xA25D714087FB6241, 0x2367A6239ED780E7 },
    },
    {
      { 0x109E3753F8A4C0A0, 0x3E0F3F7C1C95A42A, 0x924461C62B14E75D, 0x0832D57286C662DD },
      { 0xA13292F6CF6272F3, 0x072F6D008B481981, 0x3958D3F965F2E56E, 0x549C3DE68A75AFA0 },
      { 0x0E162292EF9006A2, 0x2E1D96EB5FE533A5, 0x0503F3D8F0C1A018, 0x74D0CC73278E738B },
    },
  },
  { // base[13][0] - base[13][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0xF080A21A6E62E56D, 0xA3F6AECB1E427D99, 0x3800CABB76C35073, 0x52D934BDE933FCF2 },
      { 0xC83159706C8D3C5E, 0x23E4C44E64284B33, 0xCEFAA5332028385C, 0x39B4F3549249D0DF },
      { 0x4EB39FFD89CC3432, 0x1E52FDECA0AEE3DC, 0x7027661DEF939DAF, 0x0A10341EDAA72669 },
    },
    {
      { 0x9A775DB7E0E62D68, 0xB50D8674CB235645, 0xE9D86D24D335EF29, 0x18F41DA0B0E840E0 },
      { 0x5A3C5E8F124EE8C2, 0x31061A802F2C6081, 0xFD816997E65D562E, 0x705B1DF5FA845396 },
      { 0x4BF463894F03159E, 0xA4F247A79499056C, 0xADF670A5B78C341F, 0x6AE78F5B16AA818B },
    },
    {
      { 0x183B5AF1BEFAC629, 0x6B9D5CEEF3CCE61B, 0xDEC18E71A489F710, 0x0D2B7DD5315D0099 },
      { 0x2C3C880832E11EAC, 0xC5CE8436A84A40CE, 0x712017D48962AFD3, 0x334D32B22B8448EA },
      { 0x4A1F35A82E4EE4F6, 0x9812ABDDD3BE1B8D, 0x4C39D72B209A3B28, 0x09E241B3CCE2C52E },
    },
    {
      { 0x626E7DB52EC5E840, 0xEF75E2762B8D2421, 0xEA597441DC72A9B2, 0x285EE43EE472DC13 },
      { 0xA11D2EA32D59F0D3, 0x7E09E0C3E3F89FB0, 0x8CFC1B32765ADCDB, 0x737C0649D31BDB03 },
      { 0xB0369BC1B76FF07F, 0x191A9F0AF808D5EC, 0x32581D6192DB9DCB, 0x4EEAB222392FEAD7 },
    },
    {
      { 0x614BF300045D644D, 0x3EA67508F570E1F0, 0xF9F1C5F0CFF3CBBE, 0x1D1D2287B1D182E6 },
      { 0xC7D23FFC19B10935, 0xDE4FB56238E6BE0A, 0x3738A2A1A4910645, 0x47322C8910CFB997 },
      { 0x03C79798EC1CA30A, 0x1C4E98C1EF4A5288, 0xE8F1B6368BCCB7C0, 0x58C646C9C9D4D43D },
    },
    {
      { 0xAEB34F14D58EE9CC, 0xFE490B2C1A16CF1D, 0x2AC28EFEF9ACB9E6, 0x284D20E192CA857B },
      { 0x793A2E81957FFF03, 0x864F9E24BF926DB3, 0x5E4C69F1DD42CC77, 0x51263E33CD0EA98A },
      { 0xDEE0376E537C92EB, 0xBC9F79FA320D8F99, 0xC17609404EC19F44, 0x42DFF811945089C4 },
    },
    {
      { 0x3440D06E86E28972, 0x27F386E42252FD7F, 0x0FBA4735C7A52920, 0x2BB13BE6F700D1F5 },
      { 0x9B1909B811E56FFC, 0x2A25678568D667AC, 0xCB3A0824E90D0E44, 0x7C05C661FD152253 },
      { 0x0F79C55E11A79826, 0xCD2BBFDE8A02EF04, 0xAF41528A1A731506, 0x6FFA0C393138DBD0 },
    },
    {
      { 0xF2EDA3F409DB4AA0, 0x79AE951DA1930870, 0xD60F93749C3C093B, 0x14EA6DC6505065B4 },
      { 0x9CC704060BC4EDC5, 0xD3B01012F9F3BC7A, 0x7D4C44A6035E81AE, 0x483501EE12D4B35F },
      { 0xECD685579999A9E8, 0x1C334ED2D66984F2, 0x1E2BB2C564447BF8, 0x2559C52882965F7D },
    },
  },
  { // base[14][0] - base[14][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x6B67E8F7AFEEE04E, 0x74159F7EFBABAEE7, 0x92EAB5AE900B1A61, 0x1820E35D827695CD },
      { 0xED3E1592B3B46AC0, 0xCC60E02BA2116509, 0x78D405EAE505670E, 0x54E6E8D6E0445348 },
      { 0x07F9797CECAB70A4, 0x56F3CBBACF9AB597, 0x8D234C5DAFB6012E, 0x0825DEB40A024D3D },
    },
    {
      { 0xD4AECD2FEB3FF8A8, 0xF495F34EA663AB40, 0xDBFC01266F10792B, 0x502795107D8396FA },
      { 0x28F87E98B478F6AA, 0x16408EE6EC379DE1, 0xA26E2E218269796F, 0x6DF4662B849538A4 },
      { 0x640A1D9E9875D833, 0xBAC48AAD5EB29718, 0xB29E1E18C7B6AE18, 0x52B87D8BE13C8B0F },
    },
    {
      { 0x8C9754AA85DC122D, 0xE4737DD447C82868, 0x3CC37516C452649A, 0x120E2FC8EF00C334 },
      { 0x1F7D1B3F965B0AB1, 0x7AFCB7BB0E6B0136, 0xF4638A1532DA92B1, 0x5EE5B2F529818566 },
      { 0x146C0B94A06F3655, 0xC7DF9678116CB99D, 0x8B6BFE6E91AD80E8, 0x0421076EAFE6F872 },
    },
    {
      { 0x81AC61A7027A0867, 0xDB09ADAD13B70342, 0x2ECB3863F5DC8A90, 0x026B2A7990EDC44E },
      { 0x66FF9055C1B17D25, 0xABF08C6A710D1F37, 0xF18BCB0BFE1CF315, 0x06CD29F7DE0BB4FE },
      { 0xAF3EE08B6EDEDAE1, 0x14AA6F5B46D2EE96, 0x8E5B040B999A5149, 0x653D27930C4C8D6B },
    },
    {
      { 0x1261D948D79B951C, 0x49ED413838C0A3F9, 0xEEC242B24344CF79, 0x654B18A111F07719 },
      { 0xFA538C012FD8AFC1, 0x1EFB2F9A35AE0DC7, 0x66FE784280700889, 0x48DA86266EE98C24 },
      { 0xD37413A204527FEB, 0x39C70BBF4E0ABB6C, 0xBB99A45B1E8159F9, 0x27A5E726E735E628 },
    },
    {
      { 0x987130B7624E85AE, 0x722B38C7E576118B, 0xA475A04DF935A7D3, 0x58216772B0ACAF9B },
      { 0x538FE72D711212C2, 0x1375392B497AC54F, 0x690D04EB8E751E7A, 0x39FE6E8A5B8E00F3 },
      { 0x213F383CA24DD617, 0x42AD71B6DE711885, 0x26573B10AFC20D3E, 0x5C4F3A064D4E70EB },
    },
    {
      { 0xE4DEBC7B2B875614, 0xF2AD859913C8CE70, 0x32FE1F55D0CDC8F6, 0x12E212F2EB131B48 },
      { 0xB27E59D71A6E5CE7, 0xCBA80191F1A46856, 0x22D9F83EB1631C0D, 0x30AA29BCA32D33C4 },
      { 0x1F9F035378EBEF37, 0x9F7CBB13C7031184, 0xC60A7B132745363B, 0x329CD0448AA423AC },
    },
    {
      { 0x6EE26DEA0A5DA503, 0x8CD95E1E4C2127C7, 0x24544FEB9B6538B4, 0x47B29907780CDEC8 },
      { 0xF4E90FBA61E97BB0, 0xE0A82A2092E23422, 0x31272E747CDCCF19, 0x48E2F25562E68C36 },
      { 0xEA4368D8E57EF063, 0x279FF3718B1DA8C0, 0x2CD45786FD7949CD, 0x2655E3DEF6198395 },
    },
  },
  { // base[15][0] - base[15][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0xE047BC479FBC693B, 0xFF18539650A026CF, 0xF93BC5FE67B2E64E, 0x7F724C0B2D659010 },
      { 0x11CF4B12044E0517, 0x63A462601D7F239C, 0x0BEDF6953B27D095, 0x31CDC9F8190E42C1 },
      { 0xBDEA8471C888D0D8, 0x9595C86A40483A44, 0x73E9576157397E8C, 0x476FA49E42DB0153 },
    },
    {
      { 0xB3B3E269423B2080, 0x5048201FFBFAFC1A, 0x8E47E7FD65735F6F, 0x4260062A68EFD1B4 },
      { 0x576640AC2CCDAD34, 0xF52BA787F5D6F107, 0x27F20EBA115B3F83, 0x201DC97180CEA7DA },
      { 0x26E117C0C5A32E7C, 0xB8D079AD0A4077FC, 0x57745FD68263EB2B, 0x1AADD8955930BB7A },
    },
    {
      { 0xD180ED63AD46398C, 0xF6C801CED9E75508, 0x3783BE5F9DD71F96, 0x3A8C757C702956C7 },
      { 0xD38F32663A49DDFA, 0xF2DEC26CF651D861, 0x0535E2867D02F3C2, 0x07CDC0990C176189 },
      { 0x52442CE20DBFB619, 0x8796B05E7A1C194C, 0x0C0AD494E4D8E8EC, 0x23E1C38DDD8BAAE2 },
    },
    {
      { 0x7DF32EA864283358, 0x3176625859D14CD8, 0xF29BAA75220D7470, 0x047F5016746A46AF },
      { 0x28A229CBB8F627A4, 0xFC02D8BEE4C62EB7, 0x7BB1608D23E1E335, 0x005C4DC2BB234CEE },
      { 0xC126EEBB346F7568, 0xE432229025B42E91, 0xDA8A67E6AEC4EB32, 0x239C14D3A7BAEA9B },
    },
    {
      { 0x91ECA99D569C815B, 0x32616EE77781AC47, 0x0A92B9C867F097DA, 0x763345A6A27269C8 },
      { 0x41696D3AA33CE20C, 0x731DEBEC5930C6F8, 0x1AAF77925623F585, 0x103C34262419E35A },
      { 0x9DA46790BD3C4106, 0xFBB5055940939F4B, 0xD4B632D3C64776BD, 0x3A08D302A7C5219F },
    },
    {
      { 0x2BCD729E8C58BADA, 0xB43898ACF9C95081, 0xC22AF65D0F779AFA, 0x0F64D43922C61CC7 },
      { 0xA6B2CE995CCEE42D, 0x02266E3AB01D788A, 0xD9A638966E617244, 0x7E09B2BA7DC09A7F },
      { 0x5C73526A00512844, 0xCDC0EB8105E4415A, 0x2BF3E64DF8CABAB0, 0x5D6EC452E3E6B230 },
    },
    {
      { 0x42E14C6A2C9C9823, 0x47BF1ACC2FFB2CF6, 0x0E965115797B371D, 0x30DD0898D2035390 },
      { 0x55C4ABB85B1AEE79, 0x816FF7B67B360FDE, 0x42A981345F5B68C3, 0x124C94FE6643CF3A },
      { 0xD1E850788B4AC80B, 0x811DB5B65D3F5EC4, 0xBDF8AD1F133C1983, 0x6B1018865DEC7673 },
    },
    {
      { 0xB3235AFA3BF142EB, 0xA0747FB3B647B0C9, 0xD3763988D5DACA6E, 0x3F642379B2C67626 },
      { 0xA944C9A1A49A6B18, 0x5CEDFC035289117A, 0x47B6C3C7E1FA0E11, 0x5BB3B51526CECB98 },
      { 0xCDAF479F8ED1175A, 0x09878EBBB600E689, 0x90A647E7D14C4FDC, 0x76D57B919CCDCEEA },
    },
  },
  { // base[16][0] - base[16][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x2C1D825FD656C751, 0x94DBA1F40A45F442, 0x158F2C1D840862ED, 0x15AA24F2C759DDD5 },
      { 0x2F9D3AB1759EDF1A, 0xFBF51C2A475ED05C, 0x8061F298A2BA394C, 0x498274F38B13EAA8 },
      { 0xBC4C0A69356E4E7F, 0x1E0DD59FC5A46E85, 0xED07F0FFFCBCE305, 0x22346F16BE16EB49 },
    },
    {
      { 0x25CD6C637C3183E7, 0x10889A98A1AE8614, 0x6A543362B2BD3B96, 0x2ED3213F319239A9 },
      { 0x28DD9AAF4A0CA34F, 0x19F36E2611EEE3AA, 0xC9D2DB6B223FCCB1, 0x36673E37FDA25EB1 },
      { 0x0D4A63446F561165, 0xDC83377BDDD70FFC, 0x4456C61C46ACAC07, 0x2C794D5FF3CF9654 },
    },
    {
      { 0x25AD325FB88766FB, 0xD8A6729C2316149E, 0x1B21E82B6A859D5C, 0x357C9B920C2DA438 },
      { 0xF48767D5C6F39F34, 0x2A01B7CF9BBF3B52, 0x7824AD85DF00ACC1, 0x2BBB14E253FA0F1B },
      { 0x9910012284E35444, 0xE9701B09A5AAC4B9, 0xC1F11B119E19944F, 0x380F92DD865760C7 },
    },
    {
      { 0x4E8C7B6CBE5F6080, 0x422503733A5FEDF2, 0x907ADA915627306B, 0x79052DE0284AAF28 },
      { 0xE1D4587C7230B66D, 0x7B8033074F12D43E, 0xB0F1830FFA5E52CE, 0x5706495FDEE205F4 },
      { 0x061F84A1CDC02D11, 0x7427459BB12155FE, 0x348A0BF9AE1149A3, 0x474DCE5D8A277876 },
    },
    {
      { 0xC6F74DEAAED8DF77, 0xE4E1D59B85391FDC, 0x225478DF8E346BC8, 0x1B36A20C8E7E9E6F },
      { 0x7DDD6A47FDAB904D, 0xF740C8B5EDFC8687, 0x6A4098A931AAA1DF, 0x510882759F99BDEC },
      { 0xCF1E0BA1F95E460A, 0x976D137E5AC2B61D, 0x665C17873453FDCB, 0x20B3D2735E2C9922 },
    },
    {
      { 0x615F1332FC6747F7, 0x74B3FF8A74406B16, 0x7897373F179B2777, 0x1A5999B865BF697B },
      { 0xB21DCE943B7B1380, 0x2E8ECEA0073B3475, 0x0DA5A18190FE0342, 0x3C9C5DBF112A9235 },
      { 0xE6E2C8F74340EB66, 0xE701084E76C2D3A9, 0x76BA42E0AC404441, 0x08BB7E3716FF32F2 },
    },
    {
      { 0xEDC87144A4BB875C, 0xCC7DE615567A2051, 0x109AA7FF6F6BC3CD, 0x0FB51F2A7934835B },
      { 0x5A57B6682DCE30C4, 0x16EFE4FA5952C240, 0x1EA7D28175F4A6E2, 0x447E1D2633BEAF9A },
      { 0x305260CCE9839A75, 0xA06042DB188B2E6B, 0xF1199F11FBACC14A, 0x2797D6808B5C8068 },
    },
    {
      { 0x4B166C8EDB9DDB1C, 0xF302BBD57E094E04, 0xB7B0CD9CF9DB0B44, 0x1A28CCAF94A27740 },
      { 0x225F5920CA57272A, 0x2FAA0E288C2BF7B6, 0xD30F35969B46824C, 0x222A42524B977BD5 },
      { 0x48A97E684FF53EBE, 0x2540B64A5849AE7B, 0x92C74D5523942E20, 0x085C4E53021449DB },
    },
  },
  { // base[17][0] - base[17][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0xBA9CA0DF2D22F837, 0x683E5776B6CE2FB2, 0x88BBB5CE397FA8DB, 0x0BE968ECF786A6D4 },
      { 0x1EACA3A4CB8C144E, 0x8975FC6292299F93, 0x01315FE58A61F78A, 0x105C3C6ABBDBA8C7 },
      { 0x93F9578C039F9F35, 0x7E9FF28CEBA90834, 0x9171DB961E530011, 0x3910A7B1E632E353 },
    },
    {
      { 0x8ECEDBDCFA1D94DB, 0x6B02C12527A8C7BA, 0x7960395E9897CEE2, 0x4F925642AD0AA2D8 },
      { 0xDA71BFA02983D340, 0xD5D38A6B9799B3CA, 0xEB7DE853B9BB084C, 0x6FEFA462C0B8E5E4 },
      { 0x126B0419474A82D5, 0x23A460E8860A1077, 0x63FFF22E037D92D1, 0x005D39CF1571CAF3 },
    },
    {
      { 0xD722137AF5445D93, 0x1B033CECC24B9DFD, 0x2E4F81861334A728, 0x3914BEF3EA8C6913 },
      { 0xAC974C6F2E43C86B, 0x72DFDBE9A2E1516F, 0x88AD1DB07CDA4C91, 0x01941D1F33D6BC79 },
      { 0x24120EE3DF065C93, 0x1978CDA6C5B19840, 0xE9EFE48681144984, 0x42F094B423138CA2 },
    },
    {
      { 0xD6DFDDE4121622A8, 0x5E640676681840EC, 0x421AB3537AE46FC9, 0x3C6792E9C12C6726 },
      { 0x5D41775996CE24AD, 0x6777E47E7895DCBE, 0xD816D5D749DAE8F0, 0x1CE0064E09B4C6CD },
      { 0x8AD735C718A44EB4, 0xD5428E55CE15F843, 0x64D3AD4BF8277D02, 0x0035A903B59FFC19 },
    },
    {
      { 0xFAE5BF0B5CE7040D, 0x9A03F8A620BD5E14, 0x6A59B5E715FA53D5, 0x7EF174AB0D4FBAE7 },
      { 0x14F067F0CECABC0E, 0xDB40EF8C4B318871, 0x2BEF9CE9B828B59C, 0x26ABF1A21DE3B091 },
      { 0xEF386A7A5B52AF5C, 0xA400A93FAEC2EDCC, 0xEDE4E22069F74D40, 0x759548578D3014F6 },
    },
    {
      { 0xBBF5F9922DD96C05, 0xEC180DA397DC83CD, 0xE323F37926773999, 0x232C096413B61084 },
      { 0xB491FA7E4D730F42, 0xAB9A940EF01FAFE8, 0x53B25721F376E896, 0x6FEC7A74E8969F25 },
      { 0x26A1DF5915083163, 0xB832FDBA9C18EE0B, 0x0C06A53DEF14B46B, 0x42D996158E58B3C8 },
    },
    {
      { 0x7BFE52163D6AC0C1, 0x990A14372199F9E6, 0xDB614E869A05CBCE, 0x58BB8D242B3983F0 },
      { 0xE4602F666926D475, 0xD0E78D5602EFF7C1, 0xEDDF77F93EFCE6B0, 0x5DAAAB51BDA38F4C },
      { 0x1958629270A6EA41, 0xF6D9A8AA0D15D25B, 0xD1E8B0241415AD79, 0x27E03CE93D399B75 },
    },
    {
      { 0xEE1A45A20643627D, 0x899BE5E4E64A7328, 0x32117BA6B21F1E5C, 0x5208B8615D71E684 },
      { 0xA8E49C5844DF97B6, 0x924BDEB2816FF4D3, 0x7FFFE04E3C407229, 0x4922B3E7657CC749 },
      { 0x9FFCD5C3056239DA, 0xF8488EF70089F21A, 0xA573A83075E36257, 0x1FC3094B36438006 },
    },
  },
  { // base[18][0] - base[18][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0xCE0C7E7D1B024680, 0x148ACED9B9C4CEEE, 0x6E4F9A85CFC96855, 0x537ABF7743C50CEA },
      { 0xAACD0664BC1506EF, 0x2A8EE6D97538C1C2, 0xBFB1432D98F791C6, 0x282553BB3CB9B09E },
      { 0x8655966AAB43F7CF, 0xA8C068B1123D78BD, 0xC2E0AD1A27AD1233, 0x6020CA1ECEDD1834 },
    },
    {
      { 0x2590BBA1D1365565, 0xA3D35A1232455BE7, 0xE58EA7BD01FDE4F1, 0x496C98A14C00680C },
      { 0xE1E0775D21F5E64B, 0x46BA4E4E13754E57, 0xECFD4AF70E3BE663, 0x0A1050ECBB421A07 },
      { 0x80633BCCE99BAC9E, 0xAF1E28A0591D523D, 0x220C142A71AFF9CA, 0x4DA7C918A1ACD009 },
    },
    {
      { 0x99E79818524C334F, 0x128FB9E910AFA42C, 0x55C1552028EF7A7B, 0x6FF8C8EAB7CD11FB },
      { 0x1F2E084EC48A849F, 0x1CE77D4896F4B4B5, 0x1075721FCBAF9810, 0x51CDAB953F8996D7 },
      { 0xC0CF6A19D616C834, 0x9441D5BCAFE4C291, 0xF7A2B9402AC9F59E, 0x010629353AC79B65 },
    },
    {
      { 0x7498F7ACF821663B, 0x962C4E4EC70925DB, 0x56E470C55763ACCB, 0x62967F052B016286 },
      { 0x3BCC1A7C4F6C6DDE, 0x6479557CEE3E5236, 0xD4A9266E51F0D83A, 0x015566230A989C3B },
      { 0xC3507BD0323C3BE6, 0xDDE2321387303E4F, 0xD58BF512F8FD88E4, 0x667DBEBD9825C3BD },
    },
    {
      { 0xF1434CE14BC4F789, 0x15B766B8EFAB8C86, 0xE1A1E42BF664B868, 0x2D8EA65E21A69D62 },
      { 0x395A1EB65C4DBAFF, 0xAA634A6CCE356E40, 0x5C61D51B9F71A64F, 0x0A5A31159C83A9B2 },
      { 0xDB7D930AE6054F93, 0x9D278715DC46E672, 0x8980A4C599B4D382, 0x17CC7B892C2C96E8 },
    },
    {
      { 0x1709572227AA5377, 0x7E7F1F7854E5EBEF, 0x675FC486BAC1AEF0, 0x4EC03174F3B0A2AA },
      { 0x064A53A65A87CF2B, 0xAD8FFA54C7470990, 0xCD156610C11807B3, 0x1D357124EC03557C },
      { 0x32BD6D42D4C83E2D, 0x8D07545AC8DC87B1, 0xC6870EFDEF9A5A74, 0x14C5C67457792FF9 },
    },
    {
      { 0x41BD39750510B2EF, 0x1FD583DA05E7BCFB, 0x290B1B63BB9C5738, 0x35D3138C01D3EBEE },
      { 0x15493CA9F7FB8659, 0x25C4E4953C8AB83B, 0xCA0C22BD1853E7B5, 0x1A5C542026AE7242 },
      { 0xE137765AC1B49991, 0xEAD409EFB1DAFF7E, 0x5149D54D525912B9, 0x78EB15EEA32F0E35 },
    },
    {
      { 0xE696DAED58FBAF71, 0xEBBFCAE78B5832FA, 0x8A2B8FF51FA4F842, 0x4E199B109315959E },
      { 0x3299E61469BC6FC0, 0x7B6DA1BC8507D25A, 0x71B22FFCFB80ED2D, 0x3A6AF98BF98B95D2 },
      { 0xD437F2AA33ECE537, 0x9CC5BE3A9594C61B, 0x6D36844971D6311D, 0x6575E622A3F4ECC6 },
    },
  },
  { // base[19][0] - base[19][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x05A046CF39AA5B08, 0xC0359929ADD42DB7, 0x6DF31D01A52C5103, 0x0B9DECEEE4D0EF96 },
      { 0x0978038D93B680DB, 0xF3DC5D62C3624638, 0x2984094DB8EB7DD4, 0x6EC47DFCAD1EDBC9 },
      { 0x95A8078F7F2C3966, 0x2C6B2C176A1C8C60, 0xF37693C764B39D70, 0x4370E689D8CF518C },
    },
    {
      { 0xA395D7B14F2D81A0, 0x9DD505C813C68223, 0x863C2FA34B21DF93, 0x7F9D350D46C1BD89 },
      { 0x206856A8B78B3788, 0x08C719498FD5B55F, 0x1FF1AF0A50268447, 0x5840301A9370B133 },
      { 0x7BF3221CAE9EBFFC, 0xCAD46AAAE480F6FB, 0x3466BC182C96319C, 0x58687EF69728983F },
    },
    {
      { 0xCE5A4B8F345C23A8, 0xD04AB914B3325DE7, 0xAE46F393397D2095, 0x230A8421A8E2C4EC },
      { 0x702CA68D790919D0, 0x8DEDF3C7786626CE, 0x34B28C3FC7A4CD3B, 0x45490A101604CC34 },
      { 0x5E480CE0575CD017, 0x2AE388868B01A657, 0x0736FA80B2CC9976, 0x1DE506944AE52EFF },
    },
    {
      { 0xCE34475B4F6600D6, 0x785E41D6D32244B7, 0xE516CAAFAFBD4FF1, 0x67545A01C6F94120 },
      { 0x207818DE1E2EB152, 0x8CFE459F67F83D30, 0xCC0C1ED10987DAA2, 0x2B18EF6ED74789E6 },
      { 0x9576A30578E56901, 0xA3182982D2467741, 0xC890BBA2A4F88D2F, 0x126704982A165231 },
    },
    {
      { 0x1FE7D0AAFEF985B9, 0x697B8B471B1B9752, 0x5970326F32497C22, 0x6A4C94539927A140 },
      { 0x0FF4487AFE836083, 0x5AE2341AAEC40879, 0xC13C047F3746579F, 0x20EA71E145036BA5 },
      { 0x79371953B1F70D17, 0xD748F25BE92FFEF5, 0x5E1DE99DE8BFA6B4, 0x248DB36F606E7FB5 },
    },
    {
      { 0xBAF82547686D3247, 0x76911657B3F11425, 0xC11A51BC8FBDBDD2, 0x667B5C585B80C5B3 },
      { 0x4C7AD89EE3F5194A, 0xF1EAFC663F0B6DCC, 0x56055FA965FC6CA3, 0x44799C68642F7256 },
      { 0x61C1D410CC8D39D5, 0xD593DE00EF99063D, 0xE09E998DC23BB831, 0x6986A5417583C54C },
    },
    {
      { 0x36B4B9A2B64D5FCF, 0x12BFD97E24805440, 0x95D67A096467DC28, 0x06D9F3F0065FDEAD },
      { 0x00261B1870FCA409, 0xBF16BC134655A9AD, 0xE3A41191E6427FC5, 0x72F53A9F880BB85C },
      { 0x9EB37E1F7104B1A8, 0x40EB163FB0DAE5B5, 0x87DF022109A21D8D, 0x4152760C90F0D0ED },
    },
    {
      { 0x7AE430B178E7BCA6, 0x08C6430C93772BF9, 0x8B9092428E031ABC, 0x5B6895AEF6033FE7 },
      { 0x2E7312CD1D925C51, 0xDC2BBD6622D7D05C, 0xE665F37445D0381B, 0x1E8A1E28893C04DF },
      { 0x093693C8BC8AA2A2, 0x6AF247AE7E3C1D05, 0x9B5EDB746F8BCDD6, 0x577A8BC42DD4142C },
    },
  },
  { // base[20][0] - base[20][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x4B775FFD982D979F, 0xE9FC9C56C44ACB5C, 0xF87A96E3A36AEE92, 0x6BCB41485D9D004A },
      { 0x231BCBA7462C576E, 0x5CF7917DD5F820D2, 0x70C2ECAB74C038C5, 0x178DBC7D58A1D453 },
      { 0xFB8D5C2185107077, 0xF9C9B2C692787623, 0xE7BA84D437717768, 0x7EE21F1AEE1551F0 },
    },
    {
      { 0x2D3C152E139F4B8C, 0x1ABB634CAF277ECA, 0x87976C028F91BE9F, 0x0227DC0EC16A854C },
      { 0xC2CB3332C43EECD8, 0xE487CD98A5D829AA, 0xE37046FC77903CD8, 0x7F7B900B3AC66097 },
      { 0xE0EF8C62D483F1E3, 0xABD99B8EE72631AC, 0x6538229A5900DDA4, 0x7FBCC11FCE186E97 },
    },
    {
      { 0x354E0FF8347AC3DD, 0x8413C4A70028646F, 0x1E5FCCAABEF6ADF3, 0x3254D8218E036B78 },
      { 0x419A691CD1DA89F4, 0xE09B386A5C8FD46C, 0x095AA09B7AC85E99, 0x052701B9EBC26CDA },
      { 0x9759EB50ADBE9483, 0xD85A7B506A9D411A, 0x38AB6721C4D22EA3, 0x438D3E85670C1A36 },
    },
    {
      { 0xE6061AA91070A20F, 0x06B2CA8384D8A8A0, 0xCD7AB10D904EAF9B, 0x7E34DE7BB0BBAAE9 },
      { 0x698396D56443DCFC, 0x80931482DFD2B177, 0x67AA18016077BB45, 0x561DE638A3753F4E },
      { 0x83F86BF5827414A6, 0x886D8C1297A879BE, 0xF4A8D4D18B8BCC6B, 0x77AD4D399156528E },
    },
    {
      { 0x7394EA75D1ECA25F, 0xC6CF04A0403C57CF, 0x2292AB3D23C34E01, 0x0155CB4077469D92 },
      { 0xC5D0800617A0E359, 0xE24FBCE0867F7DCD, 0x277D23B81E628E4F, 0x64A710D170A3D7E5 },
      { 0xF7D24542EEF286CD, 0x90CD112707DCD124, 0x7D048F8EEC8F7B6C, 0x35AEBB65F5235D9A },
    },
    {
      { 0xF07CA08B8F3C1291, 0x78F3573A01B49B69, 0x20459F51687E63A3, 0x0B7DC34E01EE989F },
      { 0x442BAAB676066CCA, 0xB2396E37AE680EDD, 0x5780B48A47A15A3B, 0x057199FB4293B9AA },
      { 0x94470CCB99DB04B1, 0x127E395A6C55F099, 0xA408FBF684C8E81F, 0x1FC0F1C5C7B8683A },
    },
    {
      { 0x056DBF9AAF8BE412, 0xBA5C91E1EBA14CD2, 0xEABE1F45E5FC757B, 0x0569F169A66F6E1E },
      { 0xBFC887E63F6CD7FF, 0x2A2E5C509232C3A5, 0xD41CBF6925862382, 0x282887E0827A84C9 },
      { 0xB78607E299B71245, 0x3A2F6F0CE198E7EC, 0x796B7E8004F77F0E, 0x493E0AC5F87D0F5F },
    },
    {
      { 0x6F5147E25728DCBA, 0x0ECCB9E9BA26FF4B, 0xB1203405C39C2454, 0x276C123CE8B3EFCA },
      { 0xFB0CBE21174C3CD1, 0xD2256EEA29651B23, 0x4DA09FE0A5A75665, 0x1AA77C3E83F7A7B4 },
      { 0x7F71DA913062ECB1, 0x281A977E75A0D85C, 0xC404561854FB329E, 0x5816C969029C91B6 },
    },
  },
  { // base[21][0] - base[21][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x96DE37DB727078B2, 0x0270DF94D25EB549, 0x2F0CB36A3C3D7B74, 0x476E2FAF5A136830 },
      { 0xBC09E0D15E521415, 0x76B17848D0C31EEC, 0x5763DE5C61347D43, 0x4872E9DBB78E5726 },
      { 0xAA29DFEB29ED472A, 0xF4EE0F761254FB20, 0x5FC3931D81ABC511, 0x62DA36289B0E5D39 },
    },
    {
      { 0x674EA6EEC53FF1F2, 0x5589B22B3B310718, 0xA5ACA7BDD9874CAC, 0x2E0E0577990914EF },
      { 0x54A0155F98A7BFC7, 0x712BF8EE47467A28, 0x8EDDEAA591D45F42, 0x50BBDFD1B6E5B89D },
      { 0x9B840DDE7D3CEDBE, 0xB024408F612FACD9, 0x843D3B32CE419243, 0x65730C9C3EC55ADD },
    },
    {
      { 0x3088BF224C2DFDB8, 0xFE70231538CB189B, 0x41D61A246A12C825, 0x7AB42D5F2DD21EB2 },
      { 0x46EDFB5529A25197, 0x3EC4755A5A0DA03C, 0xAF5875CBA50986B0, 0x0D006C8D8BDF9F01 },
      { 0x374B0499F5B0F959, 0xAA1E87D464FFA4A9, 0x6FB93A883D7B32B4, 0x09AA94DB11D87355 },
    },
    {
      { 0xFAE38B5E711741FF, 0x5A15F58CF404C2E0, 0xF64ED31B8A12A557, 0x2CB975028AC85309 },
      { 0x8C786DEBD6E8EA8C, 0xCBCFBC4467E08F88, 0xC39970F8388A3ACD, 0x3CDADC0D32E51D00 },
      { 0x07EA56106E47BBFF, 0x4D4D694A5626A7D4, 0xE00D96B2599B021A, 0x67BF4E4AC82F9DED },
    },
    {
      { 0xB8E4221E9AA94CFF, 0x45E69D8E6DF5F56B, 0x404924CF78D24A33, 0x0CA1776250A256E4 },
      { 0xB133A5DE2BC09817, 0x6C29079CC4D6EE07, 0x4614CCD729FDECE3, 0x18CC9D6C9731C726 },
      { 0x3ED6298CD711A4C9, 0x160D9EC886751F49, 0xAA9E724A129E0891, 0x15053298A77CE53A },
    },
    {
      { 0xE79B0D669E0E3C9D, 0x97CF5E562D1ADE1D, 0xB0743074D466D355, 0x02AEE1CDB6F50D09 },
      { 0x96DBC9BFFBFC93E1, 0x6DBA0F830BE8531A, 0xACC179D108AAD7BB, 0x267B710C323E16F6 },
      { 0xD88C913E6146ADDB, 0x83F1275E3BA6FFD5, 0x541E3C6772519644, 0x090D183B8855125B },
    },
    {
      { 0xEB2CB89F63BA41DB, 0xC45FF03BDC15CB57, 0x944F1411884BDE69, 0x693DDCA536769D4D },
      { 0xF26DAEAF4F81A542, 0xF0A9FE04981A5E16, 0xA302A348CAA8E9D8, 0x599FE3B63D207296 },
      { 0xAB1ECC954CADA417, 0x9A02E83E371C1C00, 0xA4281AEF17B26C72, 0x35C4834D9053D4FB },
    },
    {
      { 0xA0417D465AE3EDB2, 0x0343437C639A60AA, 0x94F3646CFB73D2BF, 0x4239E98453B1CDE7 },
      { 0x40955020B13810FD, 0xCCAD44FD7C922DA7, 0xFFD6E26728397782, 0x51DE1081D539F5B9 },
      { 0x65773C9301AC4EF9, 0x95A5A109236E6249, 0x0150F7BA7300D4A7, 0x4817B9DFEF021A0D },
    },
  },
  { // base[22][0] - base[22][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x515A6D705AA88E4D, 0x3D64301495FFFF83, 0xCC0F9BAEFAA8211A, 0x1FB5EB92ED275096 },
      { 0x758C5CD5BFABA2E3, 0x811D45772BC3E348, 0xDB93896D16FBD7D4, 0x1B2CBE92F52E009E },
      { 0xB9A6C6BD88302C56, 0xECA02BCF37E3482F, 0x32337C7CC9014996, 0x3DBF660CED306B68 },
    },
    {
      { 0x36D72528D3BE7D44, 0xC1131B2A73D1C328, 0x04DDFFE6C796C16D, 0x41DF6E330DFAE55D },
      { 0x3C611B9E34AE347D, 0x6E92973303214837, 0x4A8EA222257095E9, 0x611AD6BB00BA1CAB },
      { 0x312C658683C4BAF1, 0x2494A12A48C4F94C, 0xD0655A11F171B772, 0x473E715866F83350 },
    },
    {
      { 0xFF537F6FECA5B873, 0xF8986028E0FE5D16, 0xA4416A3F3F97D5C4, 0x70A92B09C576775A },
      { 0xE24A321D624642C8, 0x7E9B0EFA1E309CD6, 0x04ED8BEE9D74A6A4, 0x7337052EC7DA33A5 },
      { 0x955DFB272438657D, 0xE6B2DE785522C5B5, 0x4D5F275D3AF44C2E, 0x7F85E4086A8A6F72 },
    },
    {
      { 0x41D64ED6B9B909D0, 0xCFFB7C5D177B974C, 0x988F176EA1F634AB, 0x0E9D483EEF62D5BA },
      { 0xDC8035D2137A09AE, 0x46B39B4F2BF0181A, 0xE5E46FECA7A31E14, 0x468FC6DE7C776DFA },
      { 0x5D0B49989F6840EE, 0x94994FD6C28D9A40, 0x8094009E018190E5, 0x00008DA2518DFEF1 },
    },
    {
      { 0x0B2B0FB4B50553AE, 0xE0DFB92E2C295EB5, 0x08D46EBFCD3CB356, 0x31ECC45169428813 },
      { 0x1FEFD0361FE33606, 0xAEA071C726EB06E9, 0xBD71C59C134726B8, 0x1D6246C8B741ABF0 },
      { 0x800903A9D7DE9197, 0xF495E75C7EEC7B41, 0x7C0B34D9C27395C8, 0x19FD6A9591B45033 },
    },
    {
      { 0xC6966468621167F4, 0x8395A7BD82D09D65, 0x51FF5B73767B52B7, 0x1E61AAE65C8538F1 },
      { 0xAA0324E362F20F0B, 0x057C3218199FBB9A, 0xD9567E697982F3A3, 0x0B607A14D12B6E53 },
      { 0x74DB4A21C81F488F, 0x5C524A65BD2B1BE7, 0x643E68D25D5D4922, 0x718F57A135D73AB4 },
    },
    {
      { 0x23ECBADCD1B806F4, 0xB94062FDF17C02A9, 0xA9B2C79399722EF0, 0x218F963FB32FC05A },
      { 0x59F482086D337F46, 0xC2EEA5A9360B72D3, 0xDE1ECBB08F7CDFC1, 0x6ACCB2458F548CDA },
      { 0x6B0131A242C7BD83, 0x8A559A97D0F528A5, 0x4480220D104854EB, 0x7D8238AFC8929D93 },
    },
    {
      { 0xD9BB614062735D63, 0x4B8769EEB68ECD85, 0xD84D4AAC2285FCA2, 0x2468567D2BE6F111 },
      { 0x41F6DE94567B5718, 0x431ABE45BEAE3D5A, 0xE02023B4DBF59622, 0x6CD9BDFAE17B2C1F },
      { 0xDB0793723ED5F32F, 0x78E8D0CBB1179D1B, 0xA104673F74CB01CA, 0x4B11A0C899B69DED },
    },
  },
  { // base[23][0] - base[23][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x6EA4CE6B0FF9C320, 0x94E6CDE1831B12D0, 0xA8F16C011EEB9EE1, 0x2512B83D101DC918 },
      { 0xDCF24CEF7B133FFB, 0xBBB9653DBA160421, 0x11D00A9FF4D27958, 0x166FEFF66AE82803 },
      { 0x955BB34529FB76B5, 0x9821212C0EE8B850, 0x20000A261D7100B0, 0x2B90C4B6924724FE },
    },
    {
      { 0x142EA848D0E86D27, 0xA5D537D3DAFF1F04, 0xB1F28BBE70C9C9D9, 0x01E49AD7E258187E },
      { 0x05B72A8BFE8C0DD7, 0xC811314F95DCB1DA, 0x2A84DE7499032312, 0x2BC76EBA7B1E09ED },
      { 0x4CB93B6324958615, 0x23E661626FF102FE, 0x6E694DC26EB11D1E, 0x5F6155AC814463D1 },
    },
    {
      { 0xD3909D04D71968DC, 0x8795C3EFA07AE16A, 0x05D75263740F5594, 0x470DFB36356DD62F },
      { 0xD0D0693DF26C3DD3, 0xD4C5A6F5B09C8D76, 0x4CD06EE839E5CDC1, 0x56EAE12D1007E567 },
      { 0xF155EAF4BC96443F, 0x0D01000C65C936AE, 0xDFDD34E6DD572F8F, 0x3982A459AD7447AF },
    },
    {
      { 0xC02D84A5D0EB719A, 0xDF9F78BB849A9F8C, 0xA11F836583113815, 0x2C2D113BEC3C22EE },
      { 0x621AA8D1E5D45C77, 0x32D1378ED908AF8B, 0x3B07A7A955C61C28, 0x1821A21DA08EDC65 },
      { 0xD0C52FC119EA44B1, 0xB34C625AF63C12BF, 0xD3C737D29B9F20FF, 0x3B2B13C4A877CC0F },
    },
    {
      { 0x70B839D1F54367C5, 0x1D467DDB838AAFEE, 0x2429F3FE18C1C547, 0x545DFA425B09FB0B },
      { 0x1C61E7ACEA8FE460, 0xCDF6E97E82835B79, 0x135F884FD5AB8747, 0x1FA0B05460DC2353 },
      { 0x793097AE3789B63E, 0xD7F56883FB6E88DF, 0xA93F4D6909EF3799, 0x0F3CE59AC0C47BAE },
    },
    {
      { 0xBBF4A9EC7AF040B7, 0xC252862214CEEF6C, 0x6E361686432292F2, 0x63C55A969CE8F97A },
      { 0x009A1B61F77BF1EF, 0x4145B53FFF4F087C, 0xBFFC8472DE7CEF7E, 0x72EBCA8D9D1D9C18 },
      { 0xB3535269C92968A3, 0xF2EEF0DE438D6403, 0x5C1635A05360E4B7, 0x4B6C3D208D10910A },
    },
    {
      { 0x7DD26AF16AA702B8, 0xF10FD7EB975ECCFD, 0x24BD6139B774BBC7, 0x4FCC85ABBD2D36EF },
      { 0xD9DEBF2D21033101, 0xC3CDF1E6862D1260, 0x2BE02ED8EB7CCA5B, 0x547C3E40B2F9C653 },
      { 0xD19A27568DF47BEB, 0x3E8F2875D6753CC7, 0x3BE32B4F2906F029, 0x22C417F0A9A6B69F },
    },
    {
      { 0x6C564C94CA1E37F2, 0xDAFCF8B0D1C1C951, 0x134CED89DF644D79, 0x3EE7C21E7202F83A },
      { 0x33349A2EBABCC1EB, 0xB15B76888BD508D3, 0xBEEE8C2BCC2F0947, 0x3447F2DC7B137B6E },
      { 0xB6486B2425239960, 0xEA90A1FEE52B194C, 0xD9DF1461C8AEE370, 0x339CB43F39938C8D },
    },
  },
  { // base[24][0] - base[24][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x5336E64EE4060D60, 0x4BD02E7A0D9C521B, 0xD3F5F9DF4AEDEBE3, 0x3ED05C7B46BF3ED5 },
      { 0x77BC100A1C2B3AD3, 0xD1324F98557ED4F4, 0xA668F5A82E6FD465, 0x2308AD5D0EA6E059 },
      { 0xEA078CA9E1DAED3B, 0x0ED637B99088CF4D, 0x81E63010FF592CB0, 0x2D2FC43F41B3A5A5 },
    },
    {
      { 0x4F4B1469D05321D3, 0x5AE1E58073619032, 0x4DA98144BE16F619, 0x61F1BD716AE8E386 },
      { 0x47B180E7B8509E7F, 0x67E75C0A9A86E862, 0x7B814BEA525E523F, 0x5B34DB2B72268A1A },
      { 0x1C3F1F8376D37090, 0xB3980EA8CCD09D60, 0x5EAD6C7C1B131C08, 0x7510F366A7EAF4DF },
    },
    {
      { 0xF7A09489334CD968, 0xB8E98423B8468980, 0x192A196808C1585E, 0x629B8D83800F459B },
      { 0x78E30B8518237326, 0xAC38951500691A92, 0xB4EDDE9E46415BAA, 0x6C35FCF8D0CAFFAB },
      { 0xD36D8446AF7C3C7C, 0x293C786E30849BF2, 0xD601A4E930D0B75C, 0x4757D81BC87290BC },
    },
    {
      { 0x28A0402F07BAD705, 0xF6017DF193316618, 0x9675EF8F75491CB6, 0x625719A262A1ADD9 },
      { 0xCB02AAE09BA4020E, 0x10CD20F34105D508, 0x8E40FB9C39A43686, 0x584D6633AD016330 },
      { 0x4E7944DCDDD2A1F7, 0x79BB074EAD64B8A1, 0x8EC172E327C9B055, 0x316A910DBFCA33C7 },
    },
    {
      { 0xBAC2EA131D7BBD1E, 0xEFD73D88FF748A26, 0xD28338402CFB8C9E, 0x0A794D29C1C9101B },
      { 0x292614CE0C6849AD, 0xE435DAB645060D06, 0x51BA82976DA54318, 0x6E077EF25E3AA2B1 },
      { 0x6FB8BF6E12D96BF1, 0x10FCB86DCCDA9820, 0x6D491A5BE1F6A631, 0x6F391B2E3DF7049F },
    },
    {
      { 0x3EC99C831784599F, 0xADCB2CF2EFCF995F, 0xD67F9ED68FCF5EFE, 0x385902AAE5B9A4DB },
      { 0xABADFE03A2B890B6, 0x9BBCB3AE834A6CAD, 0x4D051BDDFA0C8F19, 0x7BF8882623DA755E },
      { 0xDF2889E2AA889626, 0xB344211D4D440FE6, 0x22333BC2AF281DA3, 0x071A1CC7A5032025 },
    },
    {
      { 0x0C4985849F258C94, 0xBEF1F08739F9FB20, 0x7A190BED399CAEB7, 0x37C56F6B651BCE1F },
      { 0xDB3E916C9F675EF4, 0x84D9F42093C11783, 0x3A1FD30FD82DB6C6, 0x2F2A029B451B11B9 },
      { 0xF1A0091EFEDBD94D, 0x243DCBF0D10D5948, 0xFCCB3E817EF34A4F, 0x3C06F3976469EF4B },
    },
    {
      { 0x338FF579807A13B9, 0x47B97595154620D5, 0x14D0BFEBCB9B9949, 0x0EF7E356995AC3D3 },
      { 0x057142A2844D73DE, 0x1C46EF678E3FA683, 0x1C560AA8852408DC, 0x075945FB38C94672 },
      { 0xD7ADDF0D77A8CACA, 0x0A46093BC8BD8AF6, 0x94C8FBFDBD72ED17, 0x633E900DFC6E9433 },
    },
  },
  { // base[25][0] - base[25][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0xDE0F7A5EAB3D73CB, 0x1FB126596B224C5E, 0xF2083269160FA764, 0x577CE2D2DD1C2000 },
      { 0x4AFF48CD3A77A7CD, 0x9D413DF67B518451, 0xCB2700E984D23D80, 0x78E21E27ADD1E3CB },
      { 0x5B7EB6FB7D4F3A5D, 0x78C13C5E7257933D, 0xC12AD9E878F7CC87, 0x6D3AC651C862F949 },
    },
    {
      { 0x467048C58EB0EE4A, 0xC6F69B234D409833, 0xEA735414D7F45569, 0x0539C013FB1CEA1F },
      { 0xD15B93886CA31241, 0x1D5463696ABEA801, 0x71EA005FD05A43E5, 0x56DD712259F5B976 },
      { 0xCC07A517ABFFF0E6, 0x80338686F0C1CC21, 0x082E1FA524FD8AFE, 0x134C6531A893534E },
    },
    {
      { 0x171EB817AF1EEC87, 0x4F1F848C726929C3, 0x2F3B9F7B0126D4B5, 0x1E002586257D1999 },
      { 0xF3B298C419585D3C, 0x1C0C18FBC92E7FC5, 0x84540DC8D0148FE6, 0x0FDA1EE624E57583 },
      { 0xCD54A356037A5C0C, 0x8E514252D4036279, 0x9F69932FE366A3C3, 0x75A1FE80E68FE90B },
    },
    {
      { 0x5AE3A12C1F3B0770, 0xBAEE295CF7055CC8, 0x5F8A13E10395C91F, 0x79A10596B7F86CF8 },
      { 0xE3D3AEA5A34BE2A2, 0x0AFEFC246F87FFDF, 0x14345CF5D5233C2D, 0x2D346B882DA97B8A },
      { 0x57967B65CF428F03, 0xC7AC9C89E3111C62, 0x6D455C4B4CFDF9B9, 0x1EDAB197F51A5E4F },
    },
    {
      { 0x9724C858C14C12E1, 0x76F575C39F4D44C8, 0x77781E9CA638257C, 0x6C8CBF524AEF9587 },
      { 0xFA377715FBAEECEC, 0x868BD8FB1CB3ACD2, 0x8DF96898A4CF3939, 0x02190D6FA4EBAF89 },
      { 0x8270B00CF272AD57, 0x73BDA1BD3F17C974, 0x63E716E0B78ACD52, 0x22F57EE0FA6B8660 },
    },
    {
      { 0xDB07231267E658ED, 0xACEDE1495EAE01CA, 0x18D04E8EEE0240E4, 0x5FB9E7752EAB6CA0 },
      { 0xB4C200C2C022EB8C, 0x26117D5167978328, 0x4A0D1B32B59116E3, 0x6D2F75E401B16D6F },
      { 0xDBD3DFE8852746E3, 0x5F2B803F2264D99C, 0xB060903F8AABD77D, 0x1302C44893310C6D },
    },
    {
      { 0xA640C71E633B72A1, 0x2F21164981E76656, 0x7603E6655A094F84, 0x06F6FD08592221DC },
      { 0x2CFB825341B07F82, 0xE1EC9FEF3B30F37A, 0xC18D953989439AA8, 0x2A5686172730AEAB },
      { 0xF71DB3EADC15A915, 0x9B78B1A34FD2E0F5, 0x52DA69793760CFE9, 0x3176595D53BD4A04 },
    },
    {
      { 0xC903941B57DB143A, 0x2FE6AF42BCF08252, 0x2D5680D6E31850A5, 0x30C89EA83AB31FCC },
      { 0xF2F6BCA9308A9595, 0xA4B11ABE876EEBE8, 0x3A4164685CB5A638, 0x572CFC8CD4B36C5F },
      { 0x06E3169B0D1918ED, 0x7D23AC194A100138, 0x016C00A89FCACA67, 0x1EEDE15098E02EAE },
    },
  },
  { // base[26][0] - base[26][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0xF9D52BD113CB5D8A, 0x441D5D5BCD83ED10, 0xF2A5F10C18D01C8E, 0x2F73FD9C6C1902FC },
      { 0xCD6E07FCE72F629C, 0x81CE1535C6178986, 0x814003E3F87C4A8A, 0x7C4B418A560259B5 },
      { 0xA9C6FEE5A0A23547, 0x52D67ED4A1A49BFC, 0xA357C846931E463C, 0x30E8319E4DE50684 },
    },
    {
      { 0x56D1945E7C7E39E6, 0xF74234AED3781BFE, 0x31BFDA6D9C615484, 0x6D91D616FC033DEE },
      { 0xB1BA249AFFED92B3, 0xE2DEB5C4BC05B45D, 0xB78D99402A9F7601, 0x374B2FEC23D76BFA },
      { 0xCD695CA9F7402934, 0xF4478CD57D6F36C6, 0x07388B820A874167, 0x7CDCDDDCEECAEF6E },
    },
    {
      { 0x68CCBED7474FB9BA, 0x5019517C67DD840B, 0x66B65D0936A22F85, 0x0DD408A305665C1A },
      { 0xF5D9AAA035189361, 0x6931C1D4346461C9, 0xB6063214F2DCBD41, 0x2832F8AC64FE90A3 },
      { 0xB840B4FD86214CAA, 0x70A300566BB767B3, 0x17555CC5387322DD, 0x1CC0F9CF2C527D79 },
    },
    {
      { 0x6422EFD2B6F337EF, 0x70A952801620241D, 0xF4E970B1E3DA7B19, 0x187A22976E5E0DB2 },
      { 0x8C7DC53AAC918540, 0x8E8B47B4B0737A2E, 0x9D42D4A28A549E5A, 0x5C6E041B82D6687E },
      { 0x42B693C162BACBA0, 0xFD09A2B4FCCE5F66, 0xC4227E39E0752738, 0x3196CD0D2C9F9234 },
    },
    {
      { 0x5F84FE88F686424A, 0xB1F838C086CFB49D, 0x10C84616ABE7C3BC, 0x5D2D3EF9457B25D1 },
      { 0xFB5DB58ADC03E5D3, 0x8C11E3EFDE2A786B, 0xDD8ECB81B714B385, 0x05927A4423F6A52B },
      { 0x6E6D6A5F288DF55A, 0xD229C03AF6936679, 0xF0CE7FCF802FCD32, 0x5A7E7BA23AA40FB1 },
    },
    {
      { 0x52DD8ED5BC67D54C, 0x2E76D1338C85B979, 0x4984E48885493047, 0x008CD18217D9BA58 },
      { 0xE0CBF0263C4BB3E5, 0x5C38A6E59C6CA33E, 0xAAEF444141FCAFD4, 0x1E9DEE0B26FD31FB },
      { 0xB3D16C4E74610BB5, 0x334ED2FB344AE860, 0xFAB2CC72D9415158, 0x6B604478F6F10539 },
    },
    {
      { 0xAC0DAFD61279C781, 0x5485F4FF5D71865E, 0xCD10B4814567C978, 0x01C5BF5241AC81C7 },
      { 0x19B69E8888543702, 0x6BF9C41905BAD97D, 0xFC8A99BB128394C4, 0x44B3A635CC8845C3 },
      { 0xCFA77C10CC98B7FC, 0x17A4E9417553C6A7, 0x84B8D2D5AD7798BA, 0x372F18812CB4F5B2 },
    },
    {
      { 0x99827D8731833111, 0xFD9A8344C3D65D1F, 0x5E8C923BC60830D1, 0x182C56A1E8C1C310 },
      { 0xD8B6317AC3F2C9F4, 0xA4CCF6EF652E9F38, 0x5A48E0F00A661F36, 0x447A88A3C4D46DD4 },
      { 0x191E07FEF2BB31E8, 0x02E1EF9C51175308, 0xDEE3C55ED64CA7CD, 0x132A4FD277F1EE4C },
    },
  },
  { // base[27][0] - base[27][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0xEDA342A4D79FB337, 0xEBBFE78278A50752, 0x9EF91FFBD25D0623, 0x1D086FF099671E42 },
      { 0xBA0EAD230F35FCEB, 0x9182D9FE3BBBD2C0, 0xEA2ABA51323A69EC, 0x0C9370EE3200F07F },
      { 0xF03FA745750BE750, 0x97EA8AA31D0FE0FE, 0x8BA9917E98F96078, 0x0FD0E80EC30F2E8A },
    },
    {
      { 0x9C6E560068EFCA4C, 0x173895EEE88406F4, 0xBF89F49F7EEAF131, 0x79FE768C774D00F2 },
      { 0x66402ACA3EACCC19, 0x0F232B6D1BF8AA90, 0xCCFB7BBA2702C990, 0x3B9AB1DE353AE799 },
      { 0x8358F4843189CE50, 0x5249ED33E2D01F66, 0x46BBE76456B1C499, 0x4FA135B80DC327A2 },
    },
    {
      { 0x78BF1AE448D092A9, 0x5BB5C0A9ABAF4E3B, 0x7D41A03786CDB91F, 0x05BB5D8D9FD3F21C },
      { 0xF7E4932620C88DF7, 0xF8D1DBDC0BD11612, 0x2C3AED35F9878A23, 0x670D7A938E98D848 },
      { 0x045C60FCBFC949C4, 0xDF33B8E5EA2255B7, 0x9172B231CCDDC00B, 0x7DB6EB0F5BB954AA },
    },
    {
      { 0x2B9855FCD580E95A, 0x8B7DBB6E200A1D8C, 0x43365F32D065D940, 0x69FD4DB2CDFFB57F },
      { 0xDBD6E0F428799EC9, 0xBCCC7D27B0466AE7, 0xD6CB16DEC6FE2DED, 0x381F4DE7578E97A7 },
      { 0xB60A6474CA442A21, 0xC21D2EB332D76A72, 0x0C0DDB9F5E6B2D78, 0x5CC6C9F2E2630FA8 },
    },
    {
      { 0xE1C3BE306973F1F9, 0x1D9A5550184145D8, 0x941F1373B9CF789C, 0x34CE4E48016182BB },
      { 0x8E25E8B399F12470, 0x5ECF09438ADF852F, 0xEA1FC678508581BB, 0x69D84DAEEF8C8D89 },
      { 0xF9835391ACA378E6, 0xC90B8C5AE672ECBE, 0x9466E923C0DA74BA, 0x28E5798637E6EC83 },
    },
    {
      { 0xB9BD7CCD0C562A5A, 0xC819BC6E628E5987, 0x15C4DE19A6708663, 0x495714E0C4FC74CD },
      { 0xD305D3A13B3A7005, 0x318742B850BD3DF9, 0x1BAC2B1EE7999266, 0x2A82551491C1FED5 },
      { 0x54CF60658F8680DE, 0x06E8F7E61D1A7BD7, 0x2AE53A90E84E2711, 0x6FE8A7F4AC75D2F5 },
    },
    {
      { 0x9E217F2F5FC9E5C7, 0x5F6FD4289B6A2B2F, 0x707842CF44211074, 0x3EEB9FCB0392E894 },
      { 0xDAEE16EF9422D596, 0x034A48D8853FF4C9, 0xA6D579EB200171A3, 0x049FF9372C323A68 },
      { 0xD886927F3402CC0B, 0xBAB983396DD791F2, 0x09B3929D5A2BD614, 0x57ACDE5E435A3852 },
    },
    {
      { 0x3820EAB05B48E177, 0xC2900D9FD6EBF38F, 0x8B6170B18899AAC2, 0x5552AF1E80841458 },
      { 0xC1C23EA14C8B89A6, 0xD68DC88FAB3E81EB, 0x3F3BA46CDF3BD568, 0x6A2C5A171728D7A5 },
      { 0x76A8C9730603A21E, 0x2162A716BA210E88, 0x1A95A6417EDAE432, 0x49F48025453B4332 },
    },
  },
  { // base[28][0] - base[28][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0xDD9700648C9DC3B6, 0xF671D448706E2835, 0x7679DBE01B6F324F, 0x6FA302044C6F4F0D },
      { 0xB9CEC422C197E764, 0xFD1C6B64D735FC31, 0x195E06E55BA7FF7B, 0x79C9BF440A5E722F },
      { 0x5C81B88B14BDFA3D, 0x54E89D916A78341A, 0xF0CB8AABA34B5EE3, 0x567C527448EAF41A },
    },
    {
      { 0x965AA43F0BE835D1, 0x12691C0E1CA80CB5, 0x6BB2CE40C2CBC518, 0x3D37BF9448EB527B },
      { 0xB6C9FEC3838887AA, 0x6EA604E9BE1C5AA4, 0xBE58B5266139B543, 0x5024DEB72C129504 },
      { 0xBE84FEC6B54D779B, 0x7877305F2D9EDC85, 0xA610DA9628CF5FEA, 0x7008D56FE2A2CA0E },
    },
    {
      { 0x31EF68640165FC48, 0xFDE84C6506FFB555, 0x3126857EDCDB76CC, 0x34E70C5BBC9A058F },
      { 0xAFB3C936E7CAFC1E, 0xBE3F42B0B8944838, 0x6B50F3F9CCC7BD2D, 0x37E2E60D85B17CF0 },
      { 0x68F7AA94594C3CDC, 0x6E8D571E6A3F4849, 0x893F02210C4F91A9, 0x4AACB59D72B880F8 },
    },
    {
      { 0x84FF988B3F289265, 0x85F20AC5ECE3A2EF, 0x1495BE913F7AAB72, 0x1D527120D7DB689C },
      { 0xA3139CE91FC8BCD1, 0x7FC18918CBEB6EE7, 0x0983EF5AA9F90A45, 0x06911BB43DAFA6ED },
      { 0x96609C5F951982F1, 0xA42C1FC7D1749361, 0x041D58D12AA4E975, 0x597E55372343D1B6 },
    },
    {
      { 0x1903D23993C3E666, 0x0BF18C847909F1FC, 0xEAD9766BFB06CB27, 0x7A37B19B613005F4 },
      { 0x3DE2B746E2BECD71, 0x9F05E976CEF85EF9, 0x55600A6F1177F251, 0x6313F4E77F5EB52E },
      { 0x9FA579A2D5B64B8E, 0xF14475B94CA1B98F, 0x99ACB54501A20C36, 0x3DA48B803F6B3149 },
    },
    {
      { 0xAA1A0D946EA9D165, 0x550BC82DEFA17E1F, 0x87FAC96CA6E97C7A, 0x4E81B107F04669BE },
      { 0x16C2FDAE55C25832, 0x24BC086944F9DE0A, 0xA3B56E223D8AE706, 0x0915D1BB7C227EBD },
      { 0xE1011966D15A72AA, 0xCF687EA108AE8C3F, 0x1755DA5F3EEA3CEC, 0x016385FA95B47626 },
    },
    {
      { 0xD672995FA2C6B967, 0xADF3B4703E5B9E5A, 0xAB67BECA7745DEF3, 0x75834BF1FF5A1D01 },
      { 0x5143F625AE8597D4, 0x20AE2BC803A44165, 0x7022530F60E840AE, 0x5319A785204F7AF0 },
      { 0xDB6478256FB151E0, 0x9F78007783B6D22E, 0x4E4E5CAC24F86954, 0x0E61BFA1A20D97D7 },
    },
    {
      { 0x6BB32B78E4E75753, 0x0E2D8AFC0C72B2B5, 0x137394194226119A, 0x5D1A37BB978CB41C },
      { 0x2847AB2D2E63991E, 0xE830E26072835491, 0x7D8C55EDAE22D60C, 0x763404E081C018A5 },
      { 0xE96AA889716D3564, 0x74DE8198D8F428F6, 0x4B03A36EC7633931, 0x02C88DCFB77BE2E8 },
    },
  },
  { // base[29][0] - base[29][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0xA9A4D679A897756E, 0x9060A0E98E60E5A4, 0x920C0603D4CD3446, 0x6AAF7CE8E325968B },
      { 0x6099CCC1FAEF8754, 0xE079BAC7A8962656, 0x1678898505D9CC70, 0x759E767CD5138631 },
      { 0x1B53B85D1DB9DE84, 0x3125778451D7DF86, 0xAB9BFFCC5A04A379, 0x33AFA6F099C0BA4E },
    },
    {
      { 0x5097FB6C9DED5985, 0x8392EC07CEB296FF, 0x80CE27F9CD5F4A43, 0x707A285C41669E21 },
      { 0x07162901B58BC17E, 0x3240B640B656C1DA, 0x686E5EECB4B2039F, 0x09ECCEFB80B26290 },
      { 0x00A5AF6190F2E065, 0x27E5B4E4EB8CDFD1, 0x272F8E0C3A8011D0, 0x0E036F4F2AF6D640 },
    },
    {
      { 0xFFEA95A07FB6B4D5, 0x9A29858C6E2024DD, 0xAF252E17D1A6CC4B, 0x3C04B7C73E995D16 },
      { 0x4C87BD6B519F6271, 0xB3047C9C5F177047, 0xCE50A1E2B194228A, 0x2679C50FF616DB06 },
      { 0x5055553286FD2E6A, 0xFCE24F15245AA3C6, 0x2784E63EB801B92D, 0x5B9E569D13048D5F },
    },
    {
      { 0xF8DF547DC4EEDDCD, 0x9DE5965E30D75765, 0x47AC53DD8FCDC6CE, 0x50AA3F6D28895343 },
      { 0x594A31A6C164FABE, 0x0FE5FEF09249A29B, 0x4F4E26D9A0C66DAD, 0x002079ECA2A20CFE },
      { 0x6F7EF49CFEACC360, 0xFA139644A8851C06, 0x5B95D203DD988CDC, 0x71AA8519A512A6FA },
    },
    {
      { 0x4DDD2C22B92A3D9B, 0xF982E37D71620470, 0x30747D34E39A78C6, 0x5CD495D7D53EBB3D },
      { 0xB283EB76DAB4E792, 0x8BC214D806529770, 0xF53E004875B5EB2E, 0x5F753163ED7BC7A8 },
      { 0xCE926389F34993A7, 0x2FB1C2BBB46DE9BA, 0xB8292AB075C559CD, 0x3421B50332E4E266 },
    },
    {
      { 0x8F2B698BF410083E, 0x62933422420574B2, 0x60F050E31907FE3D, 0x29B9B34E48B08A39 },
      { 0x5E011AF410179F8A, 0x63AE0071327CBAD8, 0xC8D274EAD1C6120B, 0x4BDB73FB455BC4FC },
      { 0x2E940A55CD072922, 0x4847904264E559FE, 0xD7E57AC42D968F65, 0x4E5A5AD33C7C3E88 },
    },
    {
      { 0xB5BA5531515003EA, 0xF988F0587838E3D8, 0x2B83F21C0005F111, 0x56E07E96C177B756 },
      { 0xDB3260359CA57E36, 0x06446F124C6D2FD8, 0x27C6818B25E56C1A, 0x1985E53C6F3A1A51 },
      { 0x4C177FC2088CBA27, 0xFCB4AF4B1583A392, 0xE2C560A7DFE4A9FD, 0x1E18DF0D9B4F8E7A },
    },
    {
      { 0x60B45E49FCE5A139, 0x575C388FE3E76DCC, 0xBF8729551A5646BD, 0x20E76084BF3EAADD },
      { 0xD87A4326844A4577, 0x03EE0CF748DD0E37, 0xBCBAE6D7535650AC, 0x1985B089A1316A5D },
      { 0xFBCB0CEBD136C045, 0xDD8FEA4F0ECF0AB6, 0xB9EBE1B66DD0EF93, 0x135A266C8F943BBE },
    },
  },
  { // base[30][0] - base[30][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x70DBF949B13981B8, 0x25A93CFFF5E54516, 0xED7E3BC55FEA098A, 0x7EF5880A4E393087 },
      { 0x28F82423C79C3A31, 0xD92EDE7A4E5F659E, 0x4D5589226CCF902A, 0x56384F360E0852EB },
      { 0x65B157B543B3773D, 0xB365F6022AA9E687, 0x2C40009C0785F25A, 0x0473474FFB167175 },
    },
    {
      { 0x979684EA855C7973, 0xD65C90C6E2AC91EF, 0xA5479A1339BB365C, 0x66589DEB9C7B8CFA },
      { 0x1A56A80525E0984D, 0x469C6DA49E85EA4E, 0xD12E1ECC280544DF, 0x578F9FC3F75D1D84 },
      { 0x7BC2463AF28ADB25, 0x52CA80DD6DA5481C, 0x6106989F9FBA8DA8, 0x0CD0F1A9E0571774 },
    },
    {
      { 0x5A10B966EACB5ED5, 0xC9F022A1CC777E20, 0x4FD8A9A3DA2084DA, 0x79B5E9CC8133571A },
      { 0xBE8E3AB05D7D02D8, 0xD9F0D0506372AF30, 0x71A94B8C606B3239, 0x60AA3588E1061A43 },
      { 0xC2A996A8499A59DA, 0xA37E88A5B040B2B9, 0x662FAF98212E41BA, 0x20914AD15C3FD5AE },
    },
    {
      { 0x9732A930F149F563, 0xC22D49019099D66D, 0x2304BAE5BC804CB5, 0x03B05DC68CAD6EC0 },
      { 0x8CE4CDC47ABF6B6B, 0xA9C9E59336FC6412, 0x2E771909D9856939, 0x4A70A9F5DA96971A },
      { 0x209F0D0BE6F340C5, 0xAB8AB6D4F6B4D042, 0x165F9347A3656658, 0x359A5F4DE19D62F9 },
    },
    {
      { 0x08FE34B2B2B8F960, 0xE364F422A98739BD, 0xF19D73D16A7F281A, 0x40DCE3DB1736E985 },
      { 0x79EF97B21D3C6059, 0xA61F4B8F7917013E, 0xF63E8E2F24E0DAD1, 0x100960C784916E96 },
      { 0xC405AAF2AD644E8B, 0x8A41920FA2D053B1, 0x1E9B77EFE173B60F, 0x4457ADBC2725D6F4 },
    },
    {
      { 0xF1398A6944E6161C, 0x25F25E88D1438BC6, 0x0C6A946B7D19B267, 0x7211E0EAD7ECC137 },
      { 0x141A4CEE440F9290, 0x4E8292ED3BC991DB, 0xC4BD6EFDB39A20FA, 0x595BCEB88B1D0B46 },
      { 0xE642FC6CF6FE59B5, 0x915E61479BA372FC, 0x724EF19C7CF2E9E6, 0x24052F7DE09F16E6 },
    },
    {
      { 0x5B30A67221671106, 0xB70CCEE626029C94, 0x331FDA526E0E5F01, 0x52598EA3B48E4703 },
      { 0x85A8F38580B1102F, 0x835A82E7C58ED7E2, 0x9635D830F7AD55E6, 0x63D513B0065BDF98 },
      { 0x152A0F76E00AFC58, 0x08D27F3F3E349FBE, 0x7857B309A7513C6B, 0x6A2DAC2E8A6ED04A },
    },
    {
      { 0xB102726871D990E7, 0x1DD531BD147F8F4A, 0x858667FEADCCDECF, 0x66916E1F32646838 },
      { 0x33DF93AF506A1CFE, 0x56F3471A044DF75F, 0x214489A66A3CF397, 0x47B17CE1995D2A2A },
      { 0xFE5A32C4EB1DAF93, 0xAE57351FABE5E7B0, 0x7F5D61694A9D7D02, 0x4E07D00D1B1B8A1B },
    },
  },
  { // base[31][0] - base[31][7]
    {
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0xFFFFFFFFFFFFFFF7, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x3FFFFFFFFFFFFFFF },
      { 0x0000000000000000, 0x0000000000000000, 0x0000000000000000, 0x0000000000000000 },
    },
    
    {
      { 0x348415874649B528, 0x7CE4D01AE0ED62DB, 0x37DB9F2A626FDB1A, 0x2002A0CD8E95E0A0 },
      { 0x6963025B114A1EF6, 0x5E465F672267D9D0, 0xAE92A7F9CBC0433C, 0x47D1B0A79D8E535F },
      { 0x5001FF5EDCDF4178, 0x1044E0D79D225648, 0x7C24CFC88CAA7D47, 0x0FDD10C577A055A1 },
    },
    {
      { 0xA79F2B821F3D80CA, 0xD40E9F77046D57BF, 0xE41CE355CCEE6F78, 0x3629AE89FFBBB0EA },
      { 0xD5AA4A247D647A9F, 0xC0FB744D3DD31BA0, 0x3A7EB63EB615AF00, 0x1C971D6554643721 },
      { 0x265E9A749F451ACE, 0x9703C0A22C43F40B, 0x8C98CE3B794D5855, 0x52F0BFF26A85609D },
    },
    {
      { 0xC8AFBFFABB7890CA, 0x61A5191397E6C3F1, 0xE65D17EF268DF293, 0x75DD4147C4B4C4CD },
      { 0x85144DEB8F027B3B, 0x90470E296B2107CA, 0xA8C36C581A348FD5, 0x12ABA8A2154FD9A8 },
      { 0xF168DE33487F1C77, 0x265AA50C504CBD6A, 0x4B8EB48A57C2306A, 0x6ACEA827BFB5BDF2 },
    },
    {
      { 0xCE2448F3FB693375, 0x83A250CD8183BC0D, 0x441C478EB030F11D, 0x491F53519AA5EA87 },
      { 0x53B9C1BC59F5AA61, 0x0EB4E9B352AA9E3E, 0x851367B17C94005D, 0x40D5896AC03F190B },
      { 0x08C68C4820F196CB, 0xDCF6F1E16C18AC24, 0x0F55A138EC1922EC, 0x251CB0F1648C50AA },
    },
    {
      { 0x0193EB2279919F8F, 0xA4CD13071A7E780B, 0xC1DAD38B796D5CBC, 0x34567756CDEA088F },
      { 0x38EE1DF07C735DD0, 0x6B677C1A3F7FF185, 0xD4C9212FF09D23B5, 0x166B5E71FD8EDBB1 },
      { 0x1C5A648779EBE108, 0x984737125BD68206, 0x9C306CF8DBF39F11, 0x2CABB06ADA847ACB },
    },
    {
      { 0x441565F5FE8113C8, 0x44D79982E208ABB0, 0xB2FA4971BE9A39FA, 0x165962EFAA28AD15 },
      { 0xB094DFF0825531C2, 0x47CB00045253FE65, 0x9FC5E044BEC84A2C, 0x784FD21F6E5948D4 },
      { 0xF5852EC631FE9565, 0x6915E0B31734A77F, 0x9391F9B77C65D81D, 0x387814F67864098F },
    },
    {
      { 0x953557D52F085853, 0xBC7851B877820D54, 0xBB9F7DBBD51D6B0F, 0x627652D153A5ECF0 },
      { 0xA30983D997769F10, 0x57021799D22AC0F3, 0x64A224E98CAF81B3, 0x45BEAEC53618A42C },
      { 0x92EA24193DCAEA98, 0xB869C180519A078E, 0x6F0E298E3070E295, 0x53911228963EF4F2 },
    },
    {
      { 0x5FBDDDC52154BAFE, 0x462E1CBBCB56D1AC, 0x713FE3B7E6F6D524, 0x0CB9AFEBFB5E1053 },
      { 0x0D5E4957A4E29A17, 0xFFF76C08D9737D68, 0xF7D14646FE642714, 0x08DAEF8C522662A1 },
      { 0x71D5C86821642133, 0xF5C24707BF8CAA3F, 0x9281D0E832D24BDC, 0x07F7C888C8EFC4AF },
    },
  },
};





// lookup table
// base[i][j] = (j+1) * (256^i)
static const DuifPoint base[32][8] = 
{
  { // base[0][0] - base[0][7]
    {
      { 0x97DE49E37AC61DB9, 0x67C996E37DC6070C, 0x9385A44C321EA161, 0x43E7CE9D19EA5D32 },
      { 0xCE881C82EBA0489F, 0xFE9CCF82E8A05F59, 0xD2E0C21A3447C504, 0x227E97C94C7C0933 },
      { 0x55E48902C3BD5534, 0x136CF411E655624F, 0x2D0DBEE5EEA1ACC6, 0x3788BDB44F8632D4 },
    },
    {
      { 0xC91273FE499E38E2, 0x4FA34ECB3D07FADA, 0x2D534D32F0EB0381, 0x6C86031FD43E9717 },
      { 0x454CD2B0215A6AD4, 0x4795C0862730567B, 0xF04F11B5D8B71BD5, 0x35DACAD334E492AA },
      { 0x21FD5459D2CDBD26, 0x9B60B5EEAECD67BC, 0xA807D042059EB518, 0x780D7AD89F5285B9 },
    },
    {
      { 0x5792D85426774B98, 0x012D4218744325C5, 0x608DA8014F80B399, 0x3D0B270DCD407C7A },
      { 0xAB308FF4527E6929, 0x9DE9A9FEF2E0DD3E, 0x4098F98D10A5EB5E, 0x555C8AC3AAADED31 },
      { 0x0A57499F86E86C3B, 0x2C4A11910E1AED31, 0x68B872A2C6796DA6, 0x6D141357895CDA63 },
    },
    {
      { 0x1439A8DCC77E04C6, 0xB3B2E37A3EFE929C, 0xE51A469EFD854932, 0x7407488190F2C393 },
      { 0xCAFF028502B40C56, 0x993F44B8AB307D54, 0x61F471E683502839, 0x53C99FA63A22D24D },
      { 0x2D09FDF4E23B7F7B, 0x374F1CA2BDAE60B9, 0xAEEDEE7C8815A24A, 0x7FCE865FB1AA9F15 },
    },
    {
      { 0x51095E220452DD90, 0x46A82461E3AF7681, 0xEE8DF5862D5FF622, 0x54A2E678A3710375 },
      { 0xBFC8C161D223EB5D, 0xEA800A68A59394DB, 0xF19E788E5C325043, 0x0AA53F39F58DAAF9 },
      { 0x5E5DEDF8C0954139, 0x93870403E85EE8FE, 0x5A0DB3858DDED396, 0x61D55F34B59DDB4D },
    },
    {
      { 0x1D067775BB8AB88F, 0x4D938AC4806457C4, 0xC032DB346D2CD39B, 0x68F2BDDB51661C5E },
      { 0xA4CC035B3DBEC652, 0xABADF14213E9139C, 0x5D842E739022A9DC, 0x1C5B2620D720BC42 },
      { 0xC2D61933817525AF, 0x5F387001A0D0DD80, 0xA9F25125841DE0A2, 0x485C748D4F86B0F1 },
    },
    {
      { 0x358D2E684A2751D6, 0xBA381A9D59CEE069, 0x38D92941142A1724, 0x630DF534941E493F },
      { 0xDD37964D551910CF, 0x365010A99DDD11D3, 0x4EF53B27C90C961D, 0x4EB76EAE97298BF0 },
      { 0x78C1B6E400DC59D1, 0xD981AFA3829F524D, 0xA94E20DD2C3BD6F9, 0x3D4FDD8E3507C853 },
    },
    {
      { 0x2CDBACB3026E9F3E, 0xB65981BBF1443816, 0xD899CE332F6CE191, 0x448AF3B030DE7297 },
      { 0xF153AEF6F9C91A63, 0xCB1EBB4070DAAC7C, 0x9613A0D6371E11FD, 0x5D481250990700E1 },
      { 0xF3E0FAECE4D1488D, 0x5C51B8BC45E653EB, 0xB1B2090C875B1519, 0x13483E2E1766274A },
    },
  },
  { // base[1][0] - base[1][7]
    {
      { 0x97666E873197CE05, 0x28E85B4B3B44988A, 0xA96FDBB5D431BD2C, 0x76E9BEA4D007779C },
      { 0x76ADB1AA24D528AF, 0x5432E24F85E3411D, 0xC2860FF4ADA168E2, 0x186BB6B781E98ADC },
      { 0xB622220B9083725A, 0x7DA9EB404946BFB4, 0x5A39CF5234A69F93, 0x48634B88974325D8 },
    },
    {
      { 0x8653155041AC63F9, 0x351EA571BD102123, 0xBA3269D31D88F6EE, 0x41DFCDD7AA840377 },
      { 0x3249E213BEDF2FEF, 0x132EA7D68CD6BF51, 0x87006FE4231822C8, 0x12F30E55F6B37F04 },
      { 0x1F89F094662C3302, 0xB7AC39F65A2CBA3F, 0x505B1EF6E609347A, 0x2B36BC31A2C37116 },
    },
    {
      { 0xD082A142E32D17E8, 0x36320895798B33E1, 0x34057120398D772C, 0x0A7DD2F9A3C9D915 },
      { 0x8B1BD24FCE60841A, 0xDE472B6AD44DE228, 0x8E5AF607BFBFE96D, 0x19CBADE52F661AEC },
      { 0x9E6BA30B34C2FBEA, 0x2C9F2F4264E4002B, 0x17E1F95B3DB0898F, 0x0A414E7541FE2936 },
    },
    {
      { 0x90F38597A738F65C, 0x732B6EDCA0523BF1, 0x5FB2AB67670EA7C0, 0x02FE1DE229AEBDBF },
      { 0xFFA1BDC24BEECAE1, 0xB63A271855275AD3, 0xCF062EB09E42F445, 0x17ECE38F2FBAC0B9 },
      { 0x925C59D72957EF65, 0x9A4AB1C6769D9867, 0x19D25E41D4DF40CA, 0x5B9BB3A3AE328F82 },
    },
    {
      { 0xB1A04AE58A1232C8, 0x77890A200B60AA9A, 0x4F1C0A0644885E30, 0x35FAC82B98483E46 },
      { 0x97DD4CFEA068D6E3, 0x59838B37CB7A6813, 0xA1B1F8290AF81DD7, 0x4FDF52B61D8C7CCC },
      { 0x07D3BC78F0A0ADC5, 0x03204FFBDD61D3BF, 0x37A96BDC4D514D28, 0x01290E7B3D31AD2B },
    },
    {
      { 0x588A33903B97AF72, 0x747C4A58CB03CD67, 0x257C112680564125, 0x000BA9ECFBE6B662 },
      { 0x289FF705854EA94A, 0x47CC73AE07EFAD33, 0xEA30C3445FF083E7, 0x1FD0053F389C1676 },
      { 0x1E349196CB1EED9A, 0x8EEF43ED5A4B9C2C, 0x556BE8FCD048F942, 0x095AFF17D02476DB },
    },
    {
      { 0xEF95BE1356B78F49, 0xA5B36991A825C489, 0x46204EE03A8E45E1, 0x37BF49E103CB63DC },
      { 0xB8F87DE24B7E719D, 0xB9DCC135D6F9ADF6, 0xE9023930FF9462B0, 0x7A4DBB7CB7D89037 },
      { 0x8FAD7B02575356F9, 0x6091A8F8DF724E4C, 0xB0D4045AF77FB5B3, 0x47E7608780F010A8 },
    },
    {
      { 0x1EF9694EE2122719, 0x9581073A49EC6F05, 0x3664033F410610A6, 0x609BBC8B37F55C85 },
      { 0x3226AC5324FF0F22, 0x10FE575118D6BBBF, 0x81220E2D443FE869, 0x2480D538C1E288F9 },
      { 0x0458DBAA460D7C78, 0xE707BD3E12314CDA, 0xFBB0587C8F036C9C, 0x20DDC43DB9368909 },
    },
  },
  { // base[2][0] - base[2][7]
    {
      { 0x3F11A62CBE3348D7, 0xB2444E9E8542DA64, 0xED7164861AA57D73, 0x05438F0386354F0E },
      { 0xA0743EA23A21A35F, 0x0EA46D6A0ADA9592, 0xBE1D450C509DB01F, 0x275B946097E6DEFB },
      { 0x1980DACCA5DE44BB, 0x39B5D71D2DEEA130, 0x06B0D6F10CEACF1E, 0x5F7398079342EA32 },
    },
    {
      { 0xA1FD3CA3C20F3A8C, 0x72E37D2CB1CE236B, 0x50832F0EF18295BA, 0x3EA3E35167DC4818 },
      { 0x7AE92AF24F3EEB52, 0x400B08AE30858F56, 0x1E4CCBAEC970C3E5, 0x49C0ABB14BC892E1 },
      { 0x1FED680A47786B70, 0xCE9F3A4D48AA379E, 0x38F63108135DC0AB, 0x0A467AC69A64F640 },
    },
    {
      { 0xF12B97BECD723AAD, 0xAB61A2DDC479A43F, 0xCFE885B6B4B05446, 0x53C7F5D6A75750DC },
      { 0xA352497B3C9A780A, 0x234CC25F7B420554, 0x2E50DE1544B08C2A, 0x5FF97D0F5EAEDDEA },
      { 0x58D5340FC6499CB3, 0x4610CA4E1014864C, 0x9C88A94890CE9E29, 0x20826E817F4E33BD },
    },
    {
      { 0xC090A7036D84B55C, 0x90D45B648672279A, 0xB2926095204F157A, 0x00B2DAD2477E5240 },
      { 0xB9595FAF08922115, 0x50FD0619CC519D5A, 0xCA65B080FD295B33, 0x16431D8057D7A9EA },
      { 0x78C8523A5042353B, 0x8977FCC26697BE60, 0x34AF14832C5515C7, 0x2C8DB3ECDFFF645C },
    },
    {
      { 0x4CDCD9B8CF8C5AA5, 0xF232F2FD50C6320F, 0xB084089B614F82F6, 0x644DA7C338180945 },
      { 0x1897868E405A4DFD, 0x2CBCA8AF55F9F645, 0xB93819E04F780E44, 0x1EF01763E547BDE5 },
      { 0xE91908169D75C96E, 0x70B129DA308B5430, 0x9EBF55F38C85D512, 0x64FAFDDD24B65F5F },
    },
    {
      { 0x8AAEB1460F4E2B97, 0xC526C35662C423A0, 0xC8D1A97B28ABB1F5, 0x0350D3614433A8AD },
      { 0x984A4D08452DE7EA, 0x6E206EB85E3239F5, 0x49614A60983E068E, 0x2B025436E5FD373A },
      { 0xB94468EA3E0BB25B, 0xB92A08A07020C5A8, 0x4F818D300C567B68, 0x104C4F44FF13A163 },
    },
    {
      { 0x8B3A13C5C2F57617, 0x2B10EE03BD6595EF, 0xB205260B30E5FA2D, 0x3985CCA87B82CAE9 },
      { 0x24CBBBFE9D16E636, 0x1942BE1652A7EC49, 0xD13CEC326903F1D0, 0x4201F68E86533F14 },
      { 0xE4A5969AC3A762A9, 0x62F36467CC1237C6, 0x7BE5A37D0B601AE7, 0x2DEBA2A184181EE6 },
    },
    {
      { 0xC2E249908AF3BC95, 0x632644D15EE6EEE4, 0x4E8F1ED456D1EBB1, 0x2DDBED891833FC16 },
      { 0xBFCD68CA94592661, 0x3FB5AA32B19AE0C0, 0xB35C5B3727E0391B, 0x099D3C0039C056C1 },
      { 0x04B0FA33E365315F, 0x027610EB108CA977, 0x0C11B03BCDEAA3B8, 0x3A06E536AC787069 },
    },
  },
  { // base[3][0] - base[3][7]
    {
      { 0x918D462B823C219E, 0xDBDA93876140A1CE, 0xEDD54CF571EC83CF, 0x1601FA92B61581EC },
      { 0xEFA47703A967E727, 0x61FFFD798376045B, 0x82B885955CAA2CE2, 0x0B0E92FD4B1F51C6 },
      { 0xBC878C3ABDA9D235, 0x983D809867862C3C, 0x98C81EBB92BF7BFC, 0x74CA345EDECB5DD7 },
    },
    {
      { 0x6C6E9EF335548CA4, 0xA428326117E06966, 0x4DA412331A7EF517, 0x149F0E273625171D },
      { 0xDE8F97A37A6D7F5E, 0xBE77808A523FEB7B, 0xE98FFEED2523D9BF, 0x69290CD239C82BC2 },
      { 0x1BB709A5C9288967, 0xB81BBC5AEE50AED0, 0xD822C4D7A30E1888, 0x6DB02E223F819411 },
    },
    {
      { 0x1DF4FF637873F826, 0x43352BCF3AF1A4B1, 0xAAA1778B0F0EF30D, 0x17897F7A662D5EEA },
      { 0x5CB2C02C90623E3B, 0xF3F80806491DC7E6, 0x000092B2817177BB, 0x5253B6E7545759F7 },
      { 0x852291596FE063A0, 0x0868373FA064D203, 0x63678A20BC67FB34, 0x2F303D928C521BC8 },
    },
    {
      { 0x5016218E52CB678A, 0xF1E216A05769F200, 0x692293401707936D, 0x100F9989CF22B834 },
      { 0x2C598EC7B66F8C0C, 0x1AE7D3A7E1B12C51, 0xF0D9FFA7B3730EB7, 0x2833D655B666EAFB },
      { 0x7EA93FB58401CE9F, 0x0C58A4B200BE0003, 0xEA9107581712D254, 0x5CBE5D44312301BA },
    },
    {
      { 0xBC0AE1FDE409BCEA, 0xD330CA106EF09578, 0x7FD4E07C42D47EEA, 0x7B8DA01160F0E129 },
      { 0x18609849F82CACD9, 0x711D50C6F4D4BCBB, 0x1117EA48B90EAF13, 0x119CE9903B37361D },
      { 0x6C3EECC3289D17CA, 0xFAD64DB8FCEA6784, 0x6835E18D8F5141D9, 0x598D0C490CCB8D3B },
    },
    {
      { 0x1328979D4EBAB94E, 0x2DE5F1443403A54F, 0xC276E0E088C07BE2, 0x4D64B0CFFB24D33D },
      { 0xFA8B37A2FDA7C063, 0x4E1B63EF30E3BAE7, 0xF1EA740DC820EC8E, 0x188B3E35C1DEFF10 },
      { 0xF9159C2129258834, 0x28341A1DF74E74C3, 0xFE4EB8C225312864, 0x30921B1A0F845888 },
    },
    {
      { 0xC5B1A4F18D16931C, 0x4EEFDB804DE9FE9A, 0xBFC5F8DC51D035D2, 0x0A915518BC6C8222 },
      { 0x6CCEA0EDC3A744BD, 0x04FF52F8B603EE10, 0x3C9E9633E807CDDE, 0x6375F1184F2F7FA0 },
      { 0x161C17A9B4B0A49C, 0xED7F204D5B96B688, 0x746419C8DB237913, 0x22FF387A82921836 },
    },
    {
      { 0x31792490644A8A3F, 0x02F803E41FB18651, 0xB7DDA2E97AE4EA5C, 0x4B30CFB6DABD1122 },
      { 0x6D243AD34B0605C6, 0x2DB4683B77871790, 0x03FDA8E79E85C7EA, 0x21468B11D071C96A },
      { 0x0427A52200D18475, 0xD4110CE1BB52E556, 0x6F5C6F2321E8DE3E, 0x4EC0AC96B05E9C63 },
    },
  },
  { // base[4][0] - base[4][7]
    {
      { 0x43B2DB4FBDC2E2F4, 0xB7F833C5E8B45D59, 0x1D3873BE0E9987CD, 0x1D2FB6A8D857C73E },
      { 0xB09B43AB5306D626, 0x8BF017B575D5EE2B, 0x3F8C9F96A66707BE, 0x5011A53BC4F66E78 },
      { 0x3B6906DB38BC5929, 0x038E1A7CEA8F68B0, 0x7B15251059F208B8, 0x3E6B411A9E7FF1B3 },
    },
    {
      { 0xD332E6B034567A70, 0xA16C968C1E6BF1E9, 0xABAC9C4E99B012EC, 0x5F78129D95966C7F },
      { 0x05F0D22DEC43FD5B, 0x154235195D201DB7, 0xECC9080974B73000, 0x141C64431DEE04A1 },
      { 0xE8B5D867A5232818, 0xFD24B5A08AE2BBD5, 0x4167D7457A55A0CE, 0x10EE5C5303541409 },
    },
    {
      { 0xCD46807D5F3B98DD, 0xC101B03F314F0C44, 0xD966011BA1F9ECBF, 0x2EC206DFB637B3C5 },
      { 0xAE300223464ECFE4, 0x12A004B76A1551E5, 0x892DA6A6097717CE, 0x05E1E840CA518ED5 },
      { 0xB8371C06984FF0BC, 0xB75816D35CF0B2E3, 0xABDDDD4CBED71055, 0x5D213B119560CB6E },
    },
    {
      { 0x9DFC60B96DA23F5C, 0x2FE7E20FE31416DE, 0xC0567FE03AD50AFF, 0x43B864F41270D4FC },
      { 0xA5A121964538427D, 0xC4C50CF1EFDCF2A2, 0xDF4F8010CE2C722E, 0x0FF8BBE750B6F5E8 },
      { 0x67B0ECCD22DADAF5, 0xC304C2748D9D3C92, 0x7398048C981F1F44, 0x5CF9327EA0A8058F },
    },
    {
      { 0xE8CDA555FF04BDE7, 0xD236FE70EFF00C94, 0x61E4844A165378FF, 0x72E31093961AF8A7 },
      { 0x53D69A0BEDF3F14E, 0xDECA1BB515CE09CE, 0x50748DC749ACBDD4, 0x0B896B9A34444C20 },
      { 0xF395C4FC6718C9E5, 0xA68819AB5092E05D, 0x020CD49E970E7F41, 0x517CC00558CE7139 },
    },
    {
      { 0xA1014FEECD377ED6, 0xDC89675F1A52A4A0, 0x3207B25CC3DEF9BD, 0x20B8D269C2CC655A },
      { 0xB02D1B451F4F7C5C, 0xF1F4E01152A8238A, 0xAA9EA4582F921247, 0x49FA0B66B23B1372 },
      { 0xFD13AC554CE4A646, 0x118037B7D8005C03, 0xFDE948EED6ED29C9, 0x28410A7D2BA5E8D5 },
    },
    {
      { 0xA30D0ADDA9E801EB, 0x590814445E79E4B2, 0x93E2BB3AB6341D2D, 0x1D3BAC5264365A23 },
      { 0x6101348A9F6B7F1C, 0x532D339CA88EBBE2, 0xE5EF1323160A57CA, 0x517CB07637D5D3A5 },
      { 0x2A4088FB49D7283B, 0x8ED710EF8EFEAA53, 0x091246487988AF32, 0x2ECFE8AFC6F3FA4A },
    },
    {
      { 0x9F922695776BA90F, 0x471D481421974B0A, 0x70B25D3B974E0B6A, 0x1DE0C3FD23F5CC6C },
      { 0x818A0469B6B1B936, 0xB51BCD77EBE3DA99, 0xD4F0C7E2E6577125, 0x59979AC8A7C7DF69 },
      { 0xB6A3808AF5436106, 0x4CC55BE5B6236892, 0xEBBC195A9D3300C4, 0x2286C0E74837DD01 },
    },
  },
  { // base[5][0] - base[5][7]
    {
      { 0xE83A6C4B0E573A16, 0xFC368C7AF70E31F6, 0x4BDEE2ADF3FA7694, 0x665D693CB31D5884 },
      { 0xB73DDB50D3102931, 0x552790EBA09E4741, 0xB7AB68AAF447AE59, 0x56F12EA5D31A2DF0 },
      { 0x4068C812506B8FDD, 0xE292E1057D94457C, 0xD8D1CBA5AF9D320C, 0x7EBFDE77F1003919 },
    },
    {
      { 0xE6BE2EE2F9E1484A, 0x63C0D14D154882D5, 0xC0630E9B210E182C, 0x27CE68CB6E6C6A6B },
      { 0x7D778F35133593F7, 0x433634626AB9CF8B, 0xFB4517DE0D81BB16, 0x6CBAA1AF43DBAD46 },
      { 0x8CC94BEC353D9BB4, 0xE8682C120D68BD31, 0xDD014E56AE0E060B, 0x3E66E8421C3D0183 },
    },
    {
      { 0xCD86420C33B06640, 0x66D7003D0D59954C, 0x5446F6433105ED0C, 0x5AC9E54240C86522 },
      { 0xEE53211636930202, 0xD70A9EA84A41205E, 0xD4E060DA7DB4633B, 0x6145E876B0E867A9 },
      { 0xC9098C4D2F424D4A, 0xEA6C619AB2EC7D66, 0x46292A2DA9FEDDE8, 0x539CC1846D16B1F3 },
    },
    {
      { 0x5CD0872605381220, 0x87D12C336ABE8DEF, 0xFFDCECDAE693ED7B, 0x6B9614A2A49619FE },
      { 0xA161C69421AF6A00, 0xDEA879B0193C6664, 0xDD83D58D3CED01F7, 0x534ACBD75F4619AA },
      { 0xE3BFE3A2EB66985F, 0x726FF469F1DD577D, 0x51164418552EED06, 0x3FCC2A4C602DE540 },
    },
    {
      { 0x69AB0AA907DFB1A8, 0x04022D22E7A6FDD3, 0xF76127DE439FD061, 0x5879329E6B4D8973 },
      { 0x1C24E7444F85F082, 0x4002D68DBDAA5144, 0x1ED1E1CF91FE490E, 0x7B6176238518F982 },
      { 0xC504649C55608639, 0x230BCDB06D93B5E5, 0x5490600F0737D638, 0x578939F8ACB239ED },
    },
    {
      { 0x982445EBAAD385E0, 0x036B5AD278EA2173, 0x7568D34F5E2CB0B1, 0x1C560CCBF6E2FBC2 },
      { 0xA39CFE3E45700EFF, 0x7EA93A48253555CF, 0x20ECC54143B94797, 0x6ECF2B956C2DB4F9 },
      { 0x03335A8BD3A8D894, 0xBA3E83433F4DC2C6, 0x56566008A2A6EF24, 0x516FE6CE5FF4F34E },
    },
    {
      { 0xAB762CDA081DF047, 0x1771DD76692CFCB4, 0x3CBE594A09FAE699, 0x47F4C3BC1266F239 },
      { 0xC6EDE97061868663, 0xD6C7332FD65DA199, 0xC7B592C619154B0F, 0x75948B602A2460E3 },
      { 0x3F6D9A68855D4894, 0xA751E6C11736D607, 0xB3041EFFB2BC7C0A, 0x66181F983FF8050B },
    },
    {
      { 0x94FE01AC06ECA280, 0x76693D5237DDF649, 0x89850AAFE17153FC, 0x20B58A8D5B8350EA },
      { 0x69851DEB0BD94639, 0xE2E9BBDB9CBB9DF5, 0x636373C60F352E5F, 0x46B0DC7BC5955BE2 },
      { 0x2B546BF7F4E09B58, 0xDE83F2E6AC722590, 0x57F317ED0DABF055, 0x0C8D157BA13BF469 },
    },
  },
  { // base[6][0] - base[6][7]
    {
      { 0x4FF315A1A7A30774, 0x6F6981EA531B03EB, 0xF82910875BD06D12, 0x51BF3EDF002A2DC9 },
      { 0x670B7BA5E29E0A0F, 0x95CB92E7103976EF, 0x5C5CE1B7DAD91F73, 0x7F17072285AE6484 },
      { 0x009ABAF6B380DA18, 0x11884A734F85FE88, 0xBA99078AC1F23F91, 0x38D7D34CD888AAF1 },
    },
    {
      { 0x75211E0E239DA86B, 0xA8F43D0F9D9C7788, 0x4DC25FAFD964DF4A, 0x00398FDE3C7C4D0E },
      { 0x32E737CD9CA9DB05, 0x632C1CF557D0A0F3, 0x07A1AFFED4FBACFF, 0x4108A174E158E147 },
      { 0xF218638C247C0C40, 0xDFCB06112F67608C, 0x5B6D7041B5DD0AF1, 0x2626B799A3F0AC04 },
    },
    {
      { 0x97866EFE4C478CB8, 0x35C8B113D85CFA8D, 0x3763DB623BC8BB5F, 0x1C5FCA805447CFD4 },
      { 0x0C7BF667E0BE8FDB, 0xB63AFAD328A01E0A, 0x6DEF3895FBF7066F, 0x4C9FEED553F23D11 },
      { 0x0FE9649E1BF443AE, 0x517B0F2D0C68A316, 0xA8407AC11C92093B, 0x7537DCCF5F86A4B4 },
    },
    {
      { 0x7758915ADB7211E3, 0x49CEB80879437FC7, 0x485495418EE7AEC6, 0x09B7ED4FA162F588 },
      { 0xB52360DDAB042AEC, 0x920B5D9C7C49F84E, 0x6B8E889BC7B8D660, 0x7AFBB48A518C4B75 },
      { 0x7CA66FD8D182DEDF, 0x079B25CECFFC1604, 0x1543EC52E1DDAC45, 0x4110C1A885F46E5D },
    },
    {
      { 0xCEAD3880A1983D36, 0xD831EF4F623ED22F, 0x115DFF295F493D69, 0x49C3E220FEA02136 },
      { 0x257BB31C2F56968A, 0xD0476C40653E2C18, 0x8689D37308108F1E, 0x35038E70BDC03601 },
      { 0xDAE9E1E8C3CBC57C, 0x3915AD1EBF872209, 0x06BDA4245DA3BE50, 0x18B8D935578F6E49 },
    },
    {
      { 0xD306DBEC594523DF, 0x535F8A6B0BB85278, 0xEA50FC49A9EEDEAC, 0x7628A5319A2121F4 },
      { 0xD49798C84BAB2654, 0x7FBDDC26113AF08C, 0x27AAFF1BD243A8A8, 0x110FEA439E7841AD },
      { 0x919110279D0AB197, 0x7DB9F074DD050196, 0x7E706EA620878187, 0x646D52CB7DC92555 },
    },
    {
      { 0x0A7B0EAEE4264BC0, 0x4CA0FCF1F7A0C103, 0xE6FADC479A313BD6, 0x6C641BFD07453CD4 },
      { 0x376547332E52CE5A, 0x542392A5971C5650, 0x98D7E384690F0BE7, 0x73B6EB7E656C257B },
      { 0x867CB442CB7E482C, 0x0EEE5DF9BDAB500D, 0x6E6173BEA49AEB35, 0x0E27B9F96352BF85 },
    },
    {
      { 0xD9B738377E3E1A42, 0xB9EFE4DA61E0E7B0, 0x758EBCE4BC0E63F2, 0x3822CD6DBED7B3AE },
      { 0x873D27DE982FD054, 0x414EA6702A6331D6, 0x7A10E1C197F19C24, 0x7CAD64068DFB2621 },
      { 0x0DC8EDA4C8DA15D0, 0x2B934B11A5816E65, 0xCFEFCF728FC63C6E, 0x6FF0B14246710FE9 },
    },
  },
  { // base[7][0] - base[7][7]
    {
      { 0x943CC296AEBE5904, 0xDC6F6EB8343EF973, 0xEE05FFD590B43C48, 0x15A26021B3BED51A },
      { 0xA72C90A7F0CA4B0D, 0xA4DF3EE386B8E6A7, 0x498067E91DA87916, 0x23C4EA237E48B919 },
      { 0x8D0E43D583A75BC7, 0xFD6368C74CED7A33, 0x1F565DE6A427C833, 0x3062977795DCD272 },
    },
    {
      { 0xB815E2E13E57367F, 0x2263B4CDAA524655, 0xF7DE202B5D249759, 0x786BB9246CDB33B6 },
      { 0x05AEC4DE1DFEC5EF, 0xD835C91BE4F9AA8D, 0x87260B586A98147A, 0x485E4E189667E555 },
      { 0x554574259F615024, 0xCC34CF7A768BC0F0, 0xBCA289F2384742E8, 0x71BAADE9D4BB7A09 },
    },
    {
      { 0x1EE3880C4BF8D652, 0xAEED3EAF60B2DDEC, 0x28472DCE07D08107, 0x53B1BA8B9BE2952B },
      { 0xDAAFD01F15688420, 0x1AB7BAC84F731AB4, 0x4FFCF8FEDF34DC48, 0x46C660E245E0B7C2 },
      { 0x814A0169B75A0CCB, 0x785A273F3BDA3052, 0xE7D431186A1E24AB, 0x78616EC53D68B373 },
    },
    {
      { 0x48EA4B3EDC76BF00, 0x3A1297856BBB40BD, 0xF204C17006C292B2, 0x595C309C0B529E72 },
      { 0xB2B0CA284FB7F607, 0x77173F54A36328C6, 0xCB99E0F9B3F04DAE, 0x1707D631B1CA424A },
      { 0x3CF3FBDF722466B2, 0x35641D33843C4368, 0xFC4FEA6CD0726D97, 0x20BC90AE39AD27A0 },
    },
    {
      { 0x725719DC9435E69A, 0xDBF7BF5B2ACEEB6E, 0x13C58A0FD9E9C70F, 0x18FD42B31120E143 },
      { 0x46384A73EBEE7695, 0x4BFDC561A3E9CE38, 0x709DF019D4836C81, 0x3801A251866CCEBB },
      { 0xD7C13621171B117A, 0xE09014C3CC19A816, 0xCDE0DBF0959C4891, 0x125D918954CA9244 },
    },
    {
      { 0x20FC06157AFC2E2C, 0xB4394261827D33CA, 0x44A2EFCCD1DD0DD6, 0x468E957CFFF5AE8B },
      { 0x58D4768B996F33D8, 0x1E5A4A0C230DA4A4, 0x475EA1A1BB67DE69, 0x47F71F438F0C4004 },
      { 0xD4ED455099310F66, 0x985C1150AC9132BC, 0x20020CBDD3CD60C9, 0x4B566BCB8C298EBB },
    },
    {
      { 0x64ACE362BC43DB4D, 0x4A70CF56AFC87F5D, 0x0B71273151A17A82, 0x4B2769A58C0B0B80 },
      { 0x396FB95796CD8E95, 0xB1A3151B5219122D, 0x9F67503C8B59CB1B, 0x491F077B5C981184 },
      { 0xA43F6CA60C97F34D, 0xB0D716751D488A89, 0xC3BDFB69DCD26F13, 0x3C6D07E30839F9F5 },
    },
    {
      { 0x514FC078B4061D4A, 0xB8FBBF0A8D74F3F3, 0x888078AC2400BCB9, 0x02A552598B59C6EE },
      { 0x2DF8AE947295E335, 0x9623F18C38780D47, 0x120CD7DE036145EE, 0x1692EF7592B58B9D },
      { 0xEFE423468C933E5C, 0x85943C4E3372A6D7, 0x15758E953337760B, 0x09A3085355BED3B0 },
    },
  },
  { // base[8][0] - base[8][7]
    {
      { 0xE69532F3BBE8FA81, 0xAA44C8C3C7D53078, 0xD8DB9DDE6D5E0372, 0x72A43C65D4BE64FD },
      { 0xA889C763C6FB587F, 0xA9CBED44F2BAFA8D, 0x04903D0EB8BD78DC, 0x10817EDD15906B28 },
      { 0xCB4F720282AE7347, 0x9B5E53B40928D694, 0x1D0D7A8BD53ED20A, 0x456B92ED94F6595D },
    },
    {
      { 0xFF63DE064D82B7B9, 0x29BEA93473FAFFEB, 0xBBD7E3312189577D, 0x67B3AFA9811CCFEC },
      { 0xEE2133D8C1A71222, 0xDB3AA25AB8670DE2, 0x0D783D05FBE8AF6B, 0x6577E7FDB8D01B28 },
      { 0xE1969B1B020A8B8F, 0x6695F788C4CC241D, 0xC3853756E84A2888, 0x05E65DB9515432B0 },
    },
    {
      { 0x0C36AF26287F094B, 0x701CBDC17F744FBF, 0x1DE3FB62A83818D8, 0x333C7EB488479BE1 },
      { 0x8C2F4B17F558D4E4, 0x4373F31AB28A3EE6, 0x58497018DDADB6F9, 0x20127855ACEB5B9F },
      { 0x0AC37D18B1B431E1, 0x03FB46242B9699F9, 0xA7B9E64FBC4F577E, 0x16A171084756A380 },
    },
    {
      { 0x10B8BD8687A9BAC0, 0xC8A73485898F0326, 0x0DDB43D73A95704F, 0x6105F9D3CDA11E37 },
      { 0x4BFA898ACA6FE944, 0x30AACC2E989FA635, 0x75D09F838422A808, 0x73B593045C696991 },
      { 0x409C5D328E2D959A, 0x4338DB76188D8DC0, 0x3DFF8658DE189AD8, 0x7A2E97FD4E0678F0 },
    },
    {
      { 0x301B6FAB90E9A735, 0x58EDC413CCBDD9E8, 0xE9E104E1E43AB57D, 0x0370ADF2A60EE41C },
      { 0xDFA92D0F15E4E455, 0xF52D93041323CEC0, 0x6A88E3876F80AAED, 0x4D711E75CB067AE8 },
      { 0x2DB92EC38C994CA5, 0x991A8E5AE758ED58, 0xBEE20AA4ED5BE502, 0x2C6F6C3093C760FB },
    },
    {
      { 0x16FDADD45B6164D4, 0xA47777C77A962CC6, 0x19C04883F8968AB9, 0x045D34B5A98EADEC },
      { 0xEC0B9BC9F93362AE, 0x6464BB62E622A724, 0xAE71C17C5E1361D4, 0x17F9CEF42A42FB7C },
      { 0xBBF69F7761F7E2BD, 0x82702A8BEA7FA408, 0x751EBD1FF8D338E5, 0x090319DA4A3E7F2A },
    },
    {
      { 0x415E98A3A4890805, 0x6F11BDB6BF37DF03, 0x708F3B0C88F53CE3, 0x03A19DF1E59C9DEF },
      { 0x05CA4C3C48B08021, 0x2773D89E7675FD74, 0xB85F39CACA785260, 0x1AE9854CDA6AC8C2 },
      { 0xFFBCA2602E74CBFA, 0x2BAE9EF2582E28D1, 0xAC19C0FEAD3B423E, 0x16C39F6F3D7B6D4F },
    },
    {
      { 0xD5310170A72EFCB7, 0x5106AC8BA80AF0FA, 0x8C513AE9DD710EB6, 0x42A1B0C500B00129 },
      { 0x8ABD18B2219B99FB, 0x7D5C5BF77A5540EC, 0x5849FF737AD32403, 0x573B9B2A383FD3DB },
      { 0x86F55EFA4BA611D7, 0x55378512CEE72349, 0x8210165C514D5D16, 0x658A219B1683CB06 },
    },
  },
  { // base[9][0] - base[9][7]
    {
      { 0xCB3E2A748E294E5C, 0x987B134932631AFD, 0x13A3D7FA3C090CB2, 0x4B81C20C757B37AE },
      { 0x66625BE3DB370FBD, 0x220ABF12FA8617BF, 0x9F7836FE389F578E, 0x2C17A233A96D31FB },
      { 0x6318BDE990192672, 0x54082174522445E2, 0xD90F78C5A72D09B2, 0x06150E25E6D146E4 },
    },
    {
      { 0x76E240A434DEB499, 0x86B6C83EDF0E4691, 0xE31DE9096AAE62D5, 0x6D354D98518A6E41 },
      { 0xE926E3E8378F821A, 0xD9134F1F6DC3E02C, 0xE8AD81397DD96947, 0x7E2AC5E8E37B243B },
      { 0x68760A9269CB2315, 0x095DB14561AD1278, 0xD2861D3C8E5E2FD2, 0x420252E5057DD7E1 },
    },
    {
      { 0xB15E4F0D9520B7DF, 0xDAE37B9471A82CC5, 0x021A1FEC1EAEB4B3, 0x5CA93A8B73FC774C },
      { 0x460FA0038553A1EB, 0x665D6865AD932F74, 0xABA58235B347E96F, 0x231CADFEE56ECB19 },
      { 0x08BFED968D2ECD4E, 0x4E3BA2DE68802E15, 0x77EA5F78AA6AB7F5, 0x3B2BCD14F411680B },
    },
    {
      { 0x999E5A89A95A1A79, 0x6C191424C9EF4070, 0xDAA89443BA869AE7, 0x01628A5D9513BBE0 },
      { 0xA2DB473F24E01502, 0x91E6A8D15E54D1BF, 0x9F6B2F88F611260D, 0x61D1C26E4F02DED8 },
      { 0xB425EAED45F8DB19, 0xFDC5E9BF7B5AA5A9, 0x989C8B6BD4D86929, 0x48B04904B0AA402C },
    },
    {
      { 0xBD1C2B0B1B4DA6DD, 0xBAE01653B2AE1AB1, 0x3EE10DFCEA78C010, 0x57B1BEBA48F37021 },
      { 0xDA268B3494ED67D5, 0xED294FA64209ACC7, 0xF4F7B1E5229EAAAC, 0x1A8F092DE2B4C705 },
      { 0xEA5A4DA30D7B3DDF, 0xEB0181BD6455C4B0, 0x38EF70CFFCD34CFD, 0x3F8C168373E7154D },
    },
    {
      { 0x04A2A5B94710BA91, 0x552C747A6A425C6C, 0xE9AC12A6BFA3481E, 0x22566021920E290B },
      { 0xBD3E47325580B476, 0x65AD252A8AF6E2A1, 0x04AA8CE9A3E6876D, 0x33EA56461A1F49D8 },
      { 0x8E3EB5DDA7BD2BB2, 0x459AFF6A48C189F0, 0xA56E50E364B5A342, 0x6AB68E418956B8DE },
    },
    {
      { 0xC0F833AB588DF407, 0x87D7FC118851F9EE, 0xFC59682AB54CA32E, 0x44BD5F1C66463F82 },
      { 0x0BF7A07186469CC1, 0x18FB839F0AD1FD1A, 0xA790F9E583B9B237, 0x3A3636368EC1277F },
      { 0x8624E4C3BF5296D2, 0x2621B4AACDEE0EA1, 0x81161C04FBE675E9, 0x2BBF0A51A5F7425E },
    },
    {
      { 0xCA7F675F5EA6EB8C, 0xFA3527ED03079108, 0x89252CBBE06468FF, 0x7829825C7D80494A },
      { 0xF871345630D39D85, 0xF97D7D081BC8D2FA, 0x60F09F4135B68074, 0x307D3F74B7EBC7A1 },
      { 0x5B1E8E9AA694B763, 0x79E1829F2FD698EC, 0xB385CAC65A5EA176, 0x109CC70650B1A9FE },
    },
  },
  { // base[10][0] - base[10][7]
    {
      { 0x13CC557CDA5BAAF7, 0xAF563909AE46D6B9, 0x69675530B0DBD011, 0x4DDFD94274C7BEA7 },
      { 0x44FA82C51C1599F0, 0x2D715D05D6A4605A, 0x47C9DA81D29ED9B7, 0x6D51F6CECAD11973 },
      { 0x32B3BBF4E3ECB2A7, 0x6595892A3963C01B, 0x3282994CECA83777, 0x6503F0A72F44ABE6 },
    },
    {
      { 0x9205AC66E23BD244, 0xFE9C6D6F3223F80B, 0x8CC9469953E43556, 0x6857BD76C257D040 },
      { 0xA7720965CC06FCC3, 0xD18AEBB79E3763B8, 0x5DD2F6EF492E3BFE, 0x5F85D61C8E989A01 },
      { 0x3727EF008AFB2DE9, 0x94CC131090B084D9, 0x3C0102C085D6EB6C, 0x4C90D18B5D75E803 },
    },
    {
      { 0x6BAD56CD6CF9E0BC, 0xAB350777B058E0CE, 0x9F4D05D612AE076C, 0x7D824EF6503163FA },
      { 0xC4A117BF6FDC387E, 0x9614B5F5A7BB59DE, 0x039C78EA1B6126FB, 0x322C6FA0F139D758 },
      { 0xEE65F1BD1AA22238, 0xBAC43C9987F6DF49, 0x3C3002618962EEC3, 0x7049EE65E14A8732 },
    },
    {
      { 0x35EF775F3042019C, 0x98CCE15B3C07DC2A, 0x4B99BB55DB16834A, 0x7718C064C5B23EC8 },
      { 0x8FF9CD42C2F0382D, 0x9B6852EC59F39C99, 0x21DCF970B8C7A29D, 0x6BE8F5042413D4BE },
      { 0x773D5B73D094582F, 0x5260ACB6C9D445D5, 0x7BDA6F415910B098, 0x5B1F4CCEEECBDE8C },
    },
    {
      { 0x178C246E7125D763, 0xBB4DB92ADD5E57B0, 0xC8659E371E77F498, 0x118FCBCDE37CD9AA },
      { 0xCB5421E09AF70FE2, 0x4BB759AA84726467, 0xDA17B400DAC66998, 0x24774DBC349D0295 },
      { 0xAE18EF25E61579E3, 0x5825D8187F10468F, 0xDBC6B804E0A7DA33, 0x03CDFD4D843C9209 },
    },
    {
      { 0x79E4F6C0516AA119, 0x855045BC3BFB1CA9, 0xEBB6D631E8842A3A, 0x4F7A7D8ACA3831B5 },
      { 0x71C81D28ED1806FA, 0x421CB2119ED4AD58, 0x769E7896859AB240, 0x01C63BFB4240B8CA },
      { 0x42A72F732D8B3DF6, 0xACAC85214B6866E1, 0x39596F9A4C0810CC, 0x2BAF74952505FFAB },
    },
    {
      { 0x2EA35E2285526BF7, 0x61D78913D299DCEC, 0x9C4F1D9315C48361, 0x50050F3F1C17AC0D },
      { 0xEA604048450C17DE, 0x1870B8614CA44EDE, 0x02DD5EABA97B99EF, 0x61EA73889669FE80 },
      { 0x28C6DCB3F57C9D59, 0xB8DE4C4D82B32960, 0x7F15C2ECAB38CBFA, 0x42876529328F271C },
    },
    {
      { 0xCBD61CBB30733475, 0xCD8CDDFF0A9D5A4B, 0xA658BCDA9A7653CF, 0x30A8E04FD098D72B },
      { 0xE1A18D6F229F864E, 0xF4FA822F7FB81DCD, 0x7E6CBD64F6C23D9E, 0x258773610E2C7A63 },
      { 0x1D7AAE06FEF82ECB, 0xEE931770155A773D, 0x88D95DC3890B8B84, 0x0FF7927D40078185 },
    },
  },
  { // base[11][0] - base[11][7]
    {
      { 0x116957FA984BB5C3, 0xC6C85C0361692302, 0xEE50C4B626F2DD72, 0x14002FF3641A060B },
      { 0x1BEB29FD8D5398CB, 0x07CA4A981FEBB20C, 0x56900584FD9D0BD9, 0x2A26A49497E4309F },
      { 0xB577DD4F9A294344, 0x2E0DFFCA12883ED0, 0x7BADDDE6B36CA59B, 0x397239498798B6FD },
    },
    {
      { 0x03F9FB1AE9953B0A, 0x3D5526C32FB2B378, 0x1E42F3CB94682228, 0x4FF73F8007F0321C },
      { 0x934A90464BC0841E, 0x58A8150591A28770, 0x7ECED75301F7EF01, 0x6D4E97461399D1A6 },
      { 0xBB2982ED01EDFBE9, 0x526D79248A1A66DE, 0xBDA56AE6E9254476, 0x407CA028F70202A1 },
    },
    {
      { 0xEBF7C9DD83D7CBA0, 0xAC1F68679EDBB353, 0x6734CC5FB7058F62, 0x63DBFFE92EEA0229 },
      { 0x469AB591E1E99859, 0x790E45CDD8238D83, 0xD9B618B637215C1E, 0x03EBCE3F45F55886 },
      { 0xC3FDFDCE5E046E89, 0xC503359D70F7614D, 0x06AB9215ED8FE0DF, 0x0E1A9051AF5325DB },
    },
    {
      { 0x66D437A010B5E023, 0x0FDD918E895E6C3F, 0xDA4AB54F0BE384C8, 0x5C3A861DB368972A },
      { 0xC06929D35E65D1A5, 0x1F30E1D09C1C10CD, 0xC861DB00CC4171CB, 0x0E1E82BBAE877337 },
      { 0xB49778A04A11728D, 0xE5E0639E15AEFB38, 0x1080A7F3BA267014, 0x0310F163E998243E },
    },
    {
      { 0x5BD70BCB586DF870, 0x2A6FD7DCF0BE70CB, 0x92C91838F4D551DA, 0x6EC72C4E5080174E },
      { 0x57CC3066412CC1BD, 0x48752460E34FCD6E, 0x3293241BB2AC0F18, 0x4003EB04BDE9D2DE },
      { 0x605F8ECA8421549C, 0xD969E1B1AC47971F, 0x054B0A1C5DA8F177, 0x4AC1EBBC1E0E5FC3 },
    },
    {
      { 0xC801A382664E945A, 0x0E8DB3CF7B9662C7, 0x0B7095AFDF2DC393, 0x64AC032741E2AC05 },
      { 0x76775177AED13D67, 0x2CBE1D0A2AB380BA, 0x64D3150933048B3D, 0x52952F9740F6C7B8 },
      { 0x86944A13283373FD, 0xFE61FBC2983E4635, 0x0DA9ED3C0608897E, 0x43CE0B85EC21D9C4 },
    },
    {
      { 0xE6EB66A8606AE82B, 0x4D7BB436DD81AB9D, 0x1E53391FF9E1F7A4, 0x33B4606B98BDC566 },
      { 0x82837672327D37F6, 0x5F71A18F3102F291, 0x1ABCA11228DC7521, 0x76F602F1A564FD80 },
      { 0xCA5B12F2F8AAE0D0, 0x20BDF9D3CCBDBDC8, 0x61165EEE36B59300, 0x68A22F0A6EE6A97A },
    },
    {
      { 0xC498A3D595DF5221, 0xC629D127C903C894, 0xA5A4FCA45F187BD3, 0x4974C8043727EA1E },
      { 0xABA815A59D8A249F, 0xC733FFB5A225DE59, 0x5C5EB4938B31C2ED, 0x498C3798F1C94AE4 },
      { 0xF8864B59BFEFDD97, 0xCFCD49AF090E757C, 0xEF889B621D2DCC1F, 0x3BD971F82E9F4CD7 },
    },
  },
  { // base[12][0] - base[12][7]
    {
      { 0xEACC31CE096ED852, 0x52E8CF9860124335, 0x68BE1781AC7E7230, 0x03D0CA8A9704AF45 },
      { 0x94B7D4E2CE17626F, 0x5E45B0DFA7C279E5, 0x8E3B836C8BD47C84, 0x31DBCAFE3D6992AE },
      { 0xD41B47811C4F2FE4, 0xC8219D8167C6F21D, 0xD7D0FEAEE2A09321, 0x1F47F41E8197809B },
    },
    {
      { 0x84382646F477E89E, 0xEFE28D4719F01B98, 0x52CEAED2893066F1, 0x116B044CD312C643 },
      { 0x97C58ADC82B8514A, 0x4A792138338422A4, 0xEF0E2D70B0DDFEC2, 0x3ADD1DBCBFD62003 },
      { 0x311CEDE03866E8CB, 0x307F4545B63EC54D, 0x59C423DE75A00930, 0x0482683DC3BBCF2F },
    },
    {
      { 0xFA1916B3247CA053, 0x834A97865E96861C, 0x0B3B4BD6D040FC98, 0x712055675D7B9536 },
      { 0x5A670FEA6EDD48CE, 0xE798ED9F63A646D5, 0x9631E631D6C36628, 0x21F10A1FDE0EEF03 },
      { 0x7C1A3A4E2DD14AD0, 0x6B4A3E2DE51BE92D, 0x33789DD3F3E498B5, 0x2B5ED791C6DA0656 },
    },
    {
      { 0x898869B660CE9DD9, 0x031535DBB111C35C, 0x3E4DC2C8EBD0A7AE, 0x01D518A83F0F2BAA },
      { 0x1B155CF1FA9A99EC, 0x19C2B46AB75C9EA0, 0x4F070A290EAD2AB9, 0x4E925436C1BA098C },
      { 0x7A763B247FEA6706, 0x7022F5782A56460E, 0xC46912C10E849ABE, 0x61D930EE4D75A42C },
    },
    {
      { 0x8CA89EC5B64A89B2, 0xCA7F38930005FA3D, 0x0146886EEAA7CAB3, 0x015A6AF1214A04B2 },
      { 0xF2AD8F0CC45DBCD4, 0xD04F683EE0BD1ACE, 0x58161771301EF519, 0x59302AE7AD93B5E1 },
      { 0x5A50AAE59468C6F9, 0xF56623230C367284, 0x624E7A49B64121C4, 0x13D36404D72E9A08 },
    },
    {
      { 0x66961385621EB4AA, 0x6EA51F2BB5336559, 0xBCFD2C9234EB81B6, 0x110A81B01EC612CC },
      { 0x45D375E68F86D8C4, 0x1BE9EB9D33AD2DF4, 0x79176FD18AFAAC2D, 0x165B38BA7FB050BF },
      { 0xACF766FC9C85F0E8, 0x54A11022394671F8, 0x41448E333D4A787A, 0x3D8EFA5B9C487A1B },
    },
    {
      { 0x2F97110C03FC7AC6, 0x71AAAE4FEA4A04EA, 0xD95554468FDB5318, 0x3434C122E9A9701E },
      { 0x7249797059D95112, 0x3E364F0315AA88B0, 0x0AF5C7F106BFBD87, 0x30FE77932C7E2CC9 },
      { 0xEDD8AEC2950C0C3D, 0x79F25569C36ED66B, 0xA25D714087FB6241, 0x2367A6239ED780E7 },
    },
    {
      { 0x109E3753F8A4C0A0, 0x3E0F3F7C1C95A42A, 0x924461C62B14E75D, 0x0832D57286C662DD },
      { 0xA13292F6CF6272F3, 0x072F6D008B481981, 0x3958D3F965F2E56E, 0x549C3DE68A75AFA0 },
      { 0x0E162292EF9006A2, 0x2E1D96EB5FE533A5, 0x0503F3D8F0C1A018, 0x74D0CC73278E738B },
    },
  },
  { // base[13][0] - base[13][7]
    {
      { 0xF080A21A6E62E56D, 0xA3F6AECB1E427D99, 0x3800CABB76C35073, 0x52D934BDE933FCF2 },
      { 0xC83159706C8D3C5E, 0x23E4C44E64284B33, 0xCEFAA5332028385C, 0x39B4F3549249D0DF },
      { 0x4EB39FFD89CC3432, 0x1E52FDECA0AEE3DC, 0x7027661DEF939DAF, 0x0A10341EDAA72669 },
    },
    {
      { 0x9A775DB7E0E62D68, 0xB50D8674CB235645, 0xE9D86D24D335EF29, 0x18F41DA0B0E840E0 },
      { 0x5A3C5E8F124EE8C2, 0x31061A802F2C6081, 0xFD816997E65D562E, 0x705B1DF5FA845396 },
      { 0x4BF463894F03159E, 0xA4F247A79499056C, 0xADF670A5B78C341F, 0x6AE78F5B16AA818B },
    },
    {
      { 0x183B5AF1BEFAC629, 0x6B9D5CEEF3CCE61B, 0xDEC18E71A489F710, 0x0D2B7DD5315D0099 },
      { 0x2C3C880832E11EAC, 0xC5CE8436A84A40CE, 0x712017D48962AFD3, 0x334D32B22B8448EA },
      { 0x4A1F35A82E4EE4F6, 0x9812ABDDD3BE1B8D, 0x4C39D72B209A3B28, 0x09E241B3CCE2C52E },
    },
    {
      { 0x626E7DB52EC5E840, 0xEF75E2762B8D2421, 0xEA597441DC72A9B2, 0x285EE43EE472DC13 },
      { 0xA11D2EA32D59F0D3, 0x7E09E0C3E3F89FB0, 0x8CFC1B32765ADCDB, 0x737C0649D31BDB03 },
      { 0xB0369BC1B76FF07F, 0x191A9F0AF808D5EC, 0x32581D6192DB9DCB, 0x4EEAB222392FEAD7 },
    },
    {
      { 0x614BF300045D644D, 0x3EA67508F570E1F0, 0xF9F1C5F0CFF3CBBE, 0x1D1D2287B1D182E6 },
      { 0xC7D23FFC19B10935, 0xDE4FB56238E6BE0A, 0x3738A2A1A4910645, 0x47322C8910CFB997 },
      { 0x03C79798EC1CA30A, 0x1C4E98C1EF4A5288, 0xE8F1B6368BCCB7C0, 0x58C646C9C9D4D43D },
    },
    {
      { 0xAEB34F14D58EE9CC, 0xFE490B2C1A16CF1D, 0x2AC28EFEF9ACB9E6, 0x284D20E192CA857B },
      { 0x793A2E81957FFF03, 0x864F9E24BF926DB3, 0x5E4C69F1DD42CC77, 0x51263E33CD0EA98A },
      { 0xDEE0376E537C92EB, 0xBC9F79FA320D8F99, 0xC17609404EC19F44, 0x42DFF811945089C4 },
    },
    {
      { 0x3440D06E86E28972, 0x27F386E42252FD7F, 0x0FBA4735C7A52920, 0x2BB13BE6F700D1F5 },
      { 0x9B1909B811E56FFC, 0x2A25678568D667AC, 0xCB3A0824E90D0E44, 0x7C05C661FD152253 },
      { 0x0F79C55E11A79826, 0xCD2BBFDE8A02EF04, 0xAF41528A1A731506, 0x6FFA0C393138DBD0 },
    },
    {
      { 0xF2EDA3F409DB4AA0, 0x79AE951DA1930870, 0xD60F93749C3C093B, 0x14EA6DC6505065B4 },
      { 0x9CC704060BC4EDC5, 0xD3B01012F9F3BC7A, 0x7D4C44A6035E81AE, 0x483501EE12D4B35F },
      { 0xECD685579999A9E8, 0x1C334ED2D66984F2, 0x1E2BB2C564447BF8, 0x2559C52882965F7D },
    },
  },
  { // base[14][0] - base[14][7]
    {
      { 0x6B67E8F7AFEEE04E, 0x74159F7EFBABAEE7, 0x92EAB5AE900B1A61, 0x1820E35D827695CD },
      { 0xED3E1592B3B46AC0, 0xCC60E02BA2116509, 0x78D405EAE505670E, 0x54E6E8D6E0445348 },
      { 0x07F9797CECAB70A4, 0x56F3CBBACF9AB597, 0x8D234C5DAFB6012E, 0x0825DEB40A024D3D },
    },
    {
      { 0xD4AECD2FEB3FF8A8, 0xF495F34EA663AB40, 0xDBFC01266F10792B, 0x502795107D8396FA },
      { 0x28F87E98B478F6AA, 0x16408EE6EC379DE1, 0xA26E2E218269796F, 0x6DF4662B849538A4 },
      { 0x640A1D9E9875D833, 0xBAC48AAD5EB29718, 0xB29E1E18C7B6AE18, 0x52B87D8BE13C8B0F },
    },
    {
      { 0x8C9754AA85DC122D, 0xE4737DD447C82868, 0x3CC37516C452649A, 0x120E2FC8EF00C334 },
      { 0x1F7D1B3F965B0AB1, 0x7AFCB7BB0E6B0136, 0xF4638A1532DA92B1, 0x5EE5B2F529818566 },
      { 0x146C0B94A06F3655, 0xC7DF9678116CB99D, 0x8B6BFE6E91AD80E8, 0x0421076EAFE6F872 },
    },
    {
      { 0x81AC61A7027A0867, 0xDB09ADAD13B70342, 0x2ECB3863F5DC8A90, 0x026B2A7990EDC44E },
      { 0x66FF9055C1B17D25, 0xABF08C6A710D1F37, 0xF18BCB0BFE1CF315, 0x06CD29F7DE0BB4FE },
      { 0xAF3EE08B6EDEDAE1, 0x14AA6F5B46D2EE96, 0x8E5B040B999A5149, 0x653D27930C4C8D6B },
    },
    {
      { 0x1261D948D79B951C, 0x49ED413838C0A3F9, 0xEEC242B24344CF79, 0x654B18A111F07719 },
      { 0xFA538C012FD8AFC1, 0x1EFB2F9A35AE0DC7, 0x66FE784280700889, 0x48DA86266EE98C24 },
      { 0xD37413A204527FEB, 0x39C70BBF4E0ABB6C, 0xBB99A45B1E8159F9, 0x27A5E726E735E628 },
    },
    {
      { 0x987130B7624E85AE, 0x722B38C7E576118B, 0xA475A04DF935A7D3, 0x58216772B0ACAF9B },
      { 0x538FE72D711212C2, 0x1375392B497AC54F, 0x690D04EB8E751E7A, 0x39FE6E8A5B8E00F3 },
      { 0x213F383CA24DD617, 0x42AD71B6DE711885, 0x26573B10AFC20D3E, 0x5C4F3A064D4E70EB },
    },
    {
      { 0xE4DEBC7B2B875614, 0xF2AD859913C8CE70, 0x32FE1F55D0CDC8F6, 0x12E212F2EB131B48 },
      { 0xB27E59D71A6E5CE7, 0xCBA80191F1A46856, 0x22D9F83EB1631C0D, 0x30AA29BCA32D33C4 },
      { 0x1F9F035378EBEF37, 0x9F7CBB13C7031184, 0xC60A7B132745363B, 0x329CD0448AA423AC },
    },
    {
      { 0x6EE26DEA0A5DA503, 0x8CD95E1E4C2127C7, 0x24544FEB9B6538B4, 0x47B29907780CDEC8 },
      { 0xF4E90FBA61E97BB0, 0xE0A82A2092E23422, 0x31272E747CDCCF19, 0x48E2F25562E68C36 },
      { 0xEA4368D8E57EF063, 0x279FF3718B1DA8C0, 0x2CD45786FD7949CD, 0x2655E3DEF6198395 },
    },
  },
  { // base[15][0] - base[15][7]
    {
      { 0xE047BC479FBC693B, 0xFF18539650A026CF, 0xF93BC5FE67B2E64E, 0x7F724C0B2D659010 },
      { 0x11CF4B12044E0517, 0x63A462601D7F239C, 0x0BEDF6953B27D095, 0x31CDC9F8190E42C1 },
      { 0xBDEA8471C888D0D8, 0x9595C86A40483A44, 0x73E9576157397E8C, 0x476FA49E42DB0153 },
    },
    {
      { 0xB3B3E269423B2080, 0x5048201FFBFAFC1A, 0x8E47E7FD65735F6F, 0x4260062A68EFD1B4 },
      { 0x576640AC2CCDAD34, 0xF52BA787F5D6F107, 0x27F20EBA115B3F83, 0x201DC97180CEA7DA },
      { 0x26E117C0C5A32E7C, 0xB8D079AD0A4077FC, 0x57745FD68263EB2B, 0x1AADD8955930BB7A },
    },
    {
      { 0xD180ED63AD46398C, 0xF6C801CED9E75508, 0x3783BE5F9DD71F96, 0x3A8C757C702956C7 },
      { 0xD38F32663A49DDFA, 0xF2DEC26CF651D861, 0x0535E2867D02F3C2, 0x07CDC0990C176189 },
      { 0x52442CE20DBFB619, 0x8796B05E7A1C194C, 0x0C0AD494E4D8E8EC, 0x23E1C38DDD8BAAE2 },
    },
    {
      { 0x7DF32EA864283358, 0x3176625859D14CD8, 0xF29BAA75220D7470, 0x047F5016746A46AF },
      { 0x28A229CBB8F627A4, 0xFC02D8BEE4C62EB7, 0x7BB1608D23E1E335, 0x005C4DC2BB234CEE },
      { 0xC126EEBB346F7568, 0xE432229025B42E91, 0xDA8A67E6AEC4EB32, 0x239C14D3A7BAEA9B },
    },
    {
      { 0x91ECA99D569C815B, 0x32616EE77781AC47, 0x0A92B9C867F097DA, 0x763345A6A27269C8 },
      { 0x41696D3AA33CE20C, 0x731DEBEC5930C6F8, 0x1AAF77925623F585, 0x103C34262419E35A },
      { 0x9DA46790BD3C4106, 0xFBB5055940939F4B, 0xD4B632D3C64776BD, 0x3A08D302A7C5219F },
    },
    {
      { 0x2BCD729E8C58BADA, 0xB43898ACF9C95081, 0xC22AF65D0F779AFA, 0x0F64D43922C61CC7 },
      { 0xA6B2CE995CCEE42D, 0x02266E3AB01D788A, 0xD9A638966E617244, 0x7E09B2BA7DC09A7F },
      { 0x5C73526A00512844, 0xCDC0EB8105E4415A, 0x2BF3E64DF8CABAB0, 0x5D6EC452E3E6B230 },
    },
    {
      { 0x42E14C6A2C9C9823, 0x47BF1ACC2FFB2CF6, 0x0E965115797B371D, 0x30DD0898D2035390 },
      { 0x55C4ABB85B1AEE79, 0x816FF7B67B360FDE, 0x42A981345F5B68C3, 0x124C94FE6643CF3A },
      { 0xD1E850788B4AC80B, 0x811DB5B65D3F5EC4, 0xBDF8AD1F133C1983, 0x6B1018865DEC7673 },
    },
    {
      { 0xB3235AFA3BF142EB, 0xA0747FB3B647B0C9, 0xD3763988D5DACA6E, 0x3F642379B2C67626 },
      { 0xA944C9A1A49A6B18, 0x5CEDFC035289117A, 0x47B6C3C7E1FA0E11, 0x5BB3B51526CECB98 },
      { 0xCDAF479F8ED1175A, 0x09878EBBB600E689, 0x90A647E7D14C4FDC, 0x76D57B919CCDCEEA },
    },
  },
  { // base[16][0] - base[16][7]
    {
      { 0x2C1D825FD656C751, 0x94DBA1F40A45F442, 0x158F2C1D840862ED, 0x15AA24F2C759DDD5 },
      { 0x2F9D3AB1759EDF1A, 0xFBF51C2A475ED05C, 0x8061F298A2BA394C, 0x498274F38B13EAA8 },
      { 0xBC4C0A69356E4E7F, 0x1E0DD59FC5A46E85, 0xED07F0FFFCBCE305, 0x22346F16BE16EB49 },
    },
    {
      { 0x25CD6C637C3183E7, 0x10889A98A1AE8614, 0x6A543362B2BD3B96, 0x2ED3213F319239A9 },
      { 0x28DD9AAF4A0CA34F, 0x19F36E2611EEE3AA, 0xC9D2DB6B223FCCB1, 0x36673E37FDA25EB1 },
      { 0x0D4A63446F561165, 0xDC83377BDDD70FFC, 0x4456C61C46ACAC07, 0x2C794D5FF3CF9654 },
    },
    {
      { 0x25AD325FB88766FB, 0xD8A6729C2316149E, 0x1B21E82B6A859D5C, 0x357C9B920C2DA438 },
      { 0xF48767D5C6F39F34, 0x2A01B7CF9BBF3B52, 0x7824AD85DF00ACC1, 0x2BBB14E253FA0F1B },
      { 0x9910012284E35444, 0xE9701B09A5AAC4B9, 0xC1F11B119E19944F, 0x380F92DD865760C7 },
    },
    {
      { 0x4E8C7B6CBE5F6080, 0x422503733A5FEDF2, 0x907ADA915627306B, 0x79052DE0284AAF28 },
      { 0xE1D4587C7230B66D, 0x7B8033074F12D43E, 0xB0F1830FFA5E52CE, 0x5706495FDEE205F4 },
      { 0x061F84A1CDC02D11, 0x7427459BB12155FE, 0x348A0BF9AE1149A3, 0x474DCE5D8A277876 },
    },
    {
      { 0xC6F74DEAAED8DF77, 0xE4E1D59B85391FDC, 0x225478DF8E346BC8, 0x1B36A20C8E7E9E6F },
      { 0x7DDD6A47FDAB904D, 0xF740C8B5EDFC8687, 0x6A4098A931AAA1DF, 0x510882759F99BDEC },
      { 0xCF1E0BA1F95E460A, 0x976D137E5AC2B61D, 0x665C17873453FDCB, 0x20B3D2735E2C9922 },
    },
    {
      { 0x615F1332FC6747F7, 0x74B3FF8A74406B16, 0x7897373F179B2777, 0x1A5999B865BF697B },
      { 0xB21DCE943B7B1380, 0x2E8ECEA0073B3475, 0x0DA5A18190FE0342, 0x3C9C5DBF112A9235 },
      { 0xE6E2C8F74340EB66, 0xE701084E76C2D3A9, 0x76BA42E0AC404441, 0x08BB7E3716FF32F2 },
    },
    {
      { 0xEDC87144A4BB875C, 0xCC7DE615567A2051, 0x109AA7FF6F6BC3CD, 0x0FB51F2A7934835B },
      { 0x5A57B6682DCE30C4, 0x16EFE4FA5952C240, 0x1EA7D28175F4A6E2, 0x447E1D2633BEAF9A },
      { 0x305260CCE9839A75, 0xA06042DB188B2E6B, 0xF1199F11FBACC14A, 0x2797D6808B5C8068 },
    },
    {
      { 0x4B166C8EDB9DDB1C, 0xF302BBD57E094E04, 0xB7B0CD9CF9DB0B44, 0x1A28CCAF94A27740 },
      { 0x225F5920CA57272A, 0x2FAA0E288C2BF7B6, 0xD30F35969B46824C, 0x222A42524B977BD5 },
      { 0x48A97E684FF53EBE, 0x2540B64A5849AE7B, 0x92C74D5523942E20, 0x085C4E53021449DB },
    },
  },
  { // base[17][0] - base[17][7]
    {
      { 0xBA9CA0DF2D22F837, 0x683E5776B6CE2FB2, 0x88BBB5CE397FA8DB, 0x0BE968ECF786A6D4 },
      { 0x1EACA3A4CB8C144E, 0x8975FC6292299F93, 0x01315FE58A61F78A, 0x105C3C6ABBDBA8C7 },
      { 0x93F9578C039F9F35, 0x7E9FF28CEBA90834, 0x9171DB961E530011, 0x3910A7B1E632E353 },
    },
    {
      { 0x8ECEDBDCFA1D94DB, 0x6B02C12527A8C7BA, 0x7960395E9897CEE2, 0x4F925642AD0AA2D8 },
      { 0xDA71BFA02983D340, 0xD5D38A6B9799B3CA, 0xEB7DE853B9BB084C, 0x6FEFA462C0B8E5E4 },
      { 0x126B0419474A82D5, 0x23A460E8860A1077, 0x63FFF22E037D92D1, 0x005D39CF1571CAF3 },
    },
    {
      { 0xD722137AF5445D93, 0x1B033CECC24B9DFD, 0x2E4F81861334A728, 0x3914BEF3EA8C6913 },
      { 0xAC974C6F2E43C86B, 0x72DFDBE9A2E1516F, 0x88AD1DB07CDA4C91, 0x01941D1F33D6BC79 },
      { 0x24120EE3DF065C93, 0x1978CDA6C5B19840, 0xE9EFE48681144984, 0x42F094B423138CA2 },
    },
    {
      { 0xD6DFDDE4121622A8, 0x5E640676681840EC, 0x421AB3537AE46FC9, 0x3C6792E9C12C6726 },
      { 0x5D41775996CE24AD, 0x6777E47E7895DCBE, 0xD816D5D749DAE8F0, 0x1CE0064E09B4C6CD },
      { 0x8AD735C718A44EB4, 0xD5428E55CE15F843, 0x64D3AD4BF8277D02, 0x0035A903B59FFC19 },
    },
    {
      { 0xFAE5BF0B5CE7040D, 0x9A03F8A620BD5E14, 0x6A59B5E715FA53D5, 0x7EF174AB0D4FBAE7 },
      { 0x14F067F0CECABC0E, 0xDB40EF8C4B318871, 0x2BEF9CE9B828B59C, 0x26ABF1A21DE3B091 },
      { 0xEF386A7A5B52AF5C, 0xA400A93FAEC2EDCC, 0xEDE4E22069F74D40, 0x759548578D3014F6 },
    },
    {
      { 0xBBF5F9922DD96C05, 0xEC180DA397DC83CD, 0xE323F37926773999, 0x232C096413B61084 },
      { 0xB491FA7E4D730F42, 0xAB9A940EF01FAFE8, 0x53B25721F376E896, 0x6FEC7A74E8969F25 },
      { 0x26A1DF5915083163, 0xB832FDBA9C18EE0B, 0x0C06A53DEF14B46B, 0x42D996158E58B3C8 },
    },
    {
      { 0x7BFE52163D6AC0C1, 0x990A14372199F9E6, 0xDB614E869A05CBCE, 0x58BB8D242B3983F0 },
      { 0xE4602F666926D475, 0xD0E78D5602EFF7C1, 0xEDDF77F93EFCE6B0, 0x5DAAAB51BDA38F4C },
      { 0x1958629270A6EA41, 0xF6D9A8AA0D15D25B, 0xD1E8B0241415AD79, 0x27E03CE93D399B75 },
    },
    {
      { 0xEE1A45A20643627D, 0x899BE5E4E64A7328, 0x32117BA6B21F1E5C, 0x5208B8615D71E684 },
      { 0xA8E49C5844DF97B6, 0x924BDEB2816FF4D3, 0x7FFFE04E3C407229, 0x4922B3E7657CC749 },
      { 0x9FFCD5C3056239DA, 0xF8488EF70089F21A, 0xA573A83075E36257, 0x1FC3094B36438006 },
    },
  },
  { // base[18][0] - base[18][7]
    {
      { 0xCE0C7E7D1B024680, 0x148ACED9B9C4CEEE, 0x6E4F9A85CFC96855, 0x537ABF7743C50CEA },
      { 0xAACD0664BC1506EF, 0x2A8EE6D97538C1C2, 0xBFB1432D98F791C6, 0x282553BB3CB9B09E },
      { 0x8655966AAB43F7CF, 0xA8C068B1123D78BD, 0xC2E0AD1A27AD1233, 0x6020CA1ECEDD1834 },
    },
    {
      { 0x2590BBA1D1365565, 0xA3D35A1232455BE7, 0xE58EA7BD01FDE4F1, 0x496C98A14C00680C },
      { 0xE1E0775D21F5E64B, 0x46BA4E4E13754E57, 0xECFD4AF70E3BE663, 0x0A1050ECBB421A07 },
      { 0x80633BCCE99BAC9E, 0xAF1E28A0591D523D, 0x220C142A71AFF9CA, 0x4DA7C918A1ACD009 },
    },
    {
      { 0x99E79818524C334F, 0x128FB9E910AFA42C, 0x55C1552028EF7A7B, 0x6FF8C8EAB7CD11FB },
      { 0x1F2E084EC48A849F, 0x1CE77D4896F4B4B5, 0x1075721FCBAF9810, 0x51CDAB953F8996D7 },
      { 0xC0CF6A19D616C834, 0x9441D5BCAFE4C291, 0xF7A2B9402AC9F59E, 0x010629353AC79B65 },
    },
    {
      { 0x7498F7ACF821663B, 0x962C4E4EC70925DB, 0x56E470C55763ACCB, 0x62967F052B016286 },
      { 0x3BCC1A7C4F6C6DDE, 0x6479557CEE3E5236, 0xD4A9266E51F0D83A, 0x015566230A989C3B },
      { 0xC3507BD0323C3BE6, 0xDDE2321387303E4F, 0xD58BF512F8FD88E4, 0x667DBEBD9825C3BD },
    },
    {
      { 0xF1434CE14BC4F789, 0x15B766B8EFAB8C86, 0xE1A1E42BF664B868, 0x2D8EA65E21A69D62 },
      { 0x395A1EB65C4DBAFF, 0xAA634A6CCE356E40, 0x5C61D51B9F71A64F, 0x0A5A31159C83A9B2 },
      { 0xDB7D930AE6054F93, 0x9D278715DC46E672, 0x8980A4C599B4D382, 0x17CC7B892C2C96E8 },
    },
    {
      { 0x1709572227AA5377, 0x7E7F1F7854E5EBEF, 0x675FC486BAC1AEF0, 0x4EC03174F3B0A2AA },
      { 0x064A53A65A87CF2B, 0xAD8FFA54C7470990, 0xCD156610C11807B3, 0x1D357124EC03557C },
      { 0x32BD6D42D4C83E2D, 0x8D07545AC8DC87B1, 0xC6870EFDEF9A5A74, 0x14C5C67457792FF9 },
    },
    {
      { 0x41BD39750510B2EF, 0x1FD583DA05E7BCFB, 0x290B1B63BB9C5738, 0x35D3138C01D3EBEE },
      { 0x15493CA9F7FB8659, 0x25C4E4953C8AB83B, 0xCA0C22BD1853E7B5, 0x1A5C542026AE7242 },
      { 0xE137765AC1B49991, 0xEAD409EFB1DAFF7E, 0x5149D54D525912B9, 0x78EB15EEA32F0E35 },
    },
    {
      { 0xE696DAED58FBAF71, 0xEBBFCAE78B5832FA, 0x8A2B8FF51FA4F842, 0x4E199B109315959E },
      { 0x3299E61469BC6FC0, 0x7B6DA1BC8507D25A, 0x71B22FFCFB80ED2D, 0x3A6AF98BF98B95D2 },
      { 0xD437F2AA33ECE537, 0x9CC5BE3A9594C61B, 0x6D36844971D6311D, 0x6575E622A3F4ECC6 },
    },
  },
  { // base[19][0] - base[19][7]
    {
      { 0x05A046CF39AA5B08, 0xC0359929ADD42DB7, 0x6DF31D01A52C5103, 0x0B9DECEEE4D0EF96 },
      { 0x0978038D93B680DB, 0xF3DC5D62C3624638, 0x2984094DB8EB7DD4, 0x6EC47DFCAD1EDBC9 },
      { 0x95A8078F7F2C3966, 0x2C6B2C176A1C8C60, 0xF37693C764B39D70, 0x4370E689D8CF518C },
    },
    {
      { 0xA395D7B14F2D81A0, 0x9DD505C813C68223, 0x863C2FA34B21DF93, 0x7F9D350D46C1BD89 },
      { 0x206856A8B78B3788, 0x08C719498FD5B55F, 0x1FF1AF0A50268447, 0x5840301A9370B133 },
      { 0x7BF3221CAE9EBFFC, 0xCAD46AAAE480F6FB, 0x3466BC182C96319C, 0x58687EF69728983F },
    },
    {
      { 0xCE5A4B8F345C23A8, 0xD04AB914B3325DE7, 0xAE46F393397D2095, 0x230A8421A8E2C4EC },
      { 0x702CA68D790919D0, 0x8DEDF3C7786626CE, 0x34B28C3FC7A4CD3B, 0x45490A101604CC34 },
      { 0x5E480CE0575CD017, 0x2AE388868B01A657, 0x0736FA80B2CC9976, 0x1DE506944AE52EFF },
    },
    {
      { 0xCE34475B4F6600D6, 0x785E41D6D32244B7, 0xE516CAAFAFBD4FF1, 0x67545A01C6F94120 },
      { 0x207818DE1E2EB152, 0x8CFE459F67F83D30, 0xCC0C1ED10987DAA2, 0x2B18EF6ED74789E6 },
      { 0x9576A30578E56901, 0xA3182982D2467741, 0xC890BBA2A4F88D2F, 0x126704982A165231 },
    },
    {
      { 0x1FE7D0AAFEF985B9, 0x697B8B471B1B9752, 0x5970326F32497C22, 0x6A4C94539927A140 },
      { 0x0FF4487AFE836083, 0x5AE2341AAEC40879, 0xC13C047F3746579F, 0x20EA71E145036BA5 },
      { 0x79371953B1F70D17, 0xD748F25BE92FFEF5, 0x5E1DE99DE8BFA6B4, 0x248DB36F606E7FB5 },
    },
    {
      { 0xBAF82547686D3247, 0x76911657B3F11425, 0xC11A51BC8FBDBDD2, 0x667B5C585B80C5B3 },
      { 0x4C7AD89EE3F5194A, 0xF1EAFC663F0B6DCC, 0x56055FA965FC6CA3, 0x44799C68642F7256 },
      { 0x61C1D410CC8D39D5, 0xD593DE00EF99063D, 0xE09E998DC23BB831, 0x6986A5417583C54C },
    },
    {
      { 0x36B4B9A2B64D5FCF, 0x12BFD97E24805440, 0x95D67A096467DC28, 0x06D9F3F0065FDEAD },
      { 0x00261B1870FCA409, 0xBF16BC134655A9AD, 0xE3A41191E6427FC5, 0x72F53A9F880BB85C },
      { 0x9EB37E1F7104B1A8, 0x40EB163FB0DAE5B5, 0x87DF022109A21D8D, 0x4152760C90F0D0ED },
    },
    {
      { 0x7AE430B178E7BCA6, 0x08C6430C93772BF9, 0x8B9092428E031ABC, 0x5B6895AEF6033FE7 },
      { 0x2E7312CD1D925C51, 0xDC2BBD6622D7D05C, 0xE665F37445D0381B, 0x1E8A1E28893C04DF },
      { 0x093693C8BC8AA2A2, 0x6AF247AE7E3C1D05, 0x9B5EDB746F8BCDD6, 0x577A8BC42DD4142C },
    },
  },
  { // base[20][0] - base[20][7]
    {
      { 0x4B775FFD982D979F, 0xE9FC9C56C44ACB5C, 0xF87A96E3A36AEE92, 0x6BCB41485D9D004A },
      { 0x231BCBA7462C576E, 0x5CF7917DD5F820D2, 0x70C2ECAB74C038C5, 0x178DBC7D58A1D453 },
      { 0xFB8D5C2185107077, 0xF9C9B2C692787623, 0xE7BA84D437717768, 0x7EE21F1AEE1551F0 },
    },
    {
      { 0x2D3C152E139F4B8C, 0x1ABB634CAF277ECA, 0x87976C028F91BE9F, 0x0227DC0EC16A854C },
      { 0xC2CB3332C43EECD8, 0xE487CD98A5D829AA, 0xE37046FC77903CD8, 0x7F7B900B3AC66097 },
      { 0xE0EF8C62D483F1E3, 0xABD99B8EE72631AC, 0x6538229A5900DDA4, 0x7FBCC11FCE186E97 },
    },
    {
      { 0x354E0FF8347AC3DD, 0x8413C4A70028646F, 0x1E5FCCAABEF6ADF3, 0x3254D8218E036B78 },
      { 0x419A691CD1DA89F4, 0xE09B386A5C8FD46C, 0x095AA09B7AC85E99, 0x052701B9EBC26CDA },
      { 0x9759EB50ADBE9483, 0xD85A7B506A9D411A, 0x38AB6721C4D22EA3, 0x438D3E85670C1A36 },
    },
    {
      { 0xE6061AA91070A20F, 0x06B2CA8384D8A8A0, 0xCD7AB10D904EAF9B, 0x7E34DE7BB0BBAAE9 },
      { 0x698396D56443DCFC, 0x80931482DFD2B177, 0x67AA18016077BB45, 0x561DE638A3753F4E },
      { 0x83F86BF5827414A6, 0x886D8C1297A879BE, 0xF4A8D4D18B8BCC6B, 0x77AD4D399156528E },
    },
    {
      { 0x7394EA75D1ECA25F, 0xC6CF04A0403C57CF, 0x2292AB3D23C34E01, 0x0155CB4077469D92 },
      { 0xC5D0800617A0E359, 0xE24FBCE0867F7DCD, 0x277D23B81E628E4F, 0x64A710D170A3D7E5 },
      { 0xF7D24542EEF286CD, 0x90CD112707DCD124, 0x7D048F8EEC8F7B6C, 0x35AEBB65F5235D9A },
    },
    {
      { 0xF07CA08B8F3C1291, 0x78F3573A01B49B69, 0x20459F51687E63A3, 0x0B7DC34E01EE989F },
      { 0x442BAAB676066CCA, 0xB2396E37AE680EDD, 0x5780B48A47A15A3B, 0x057199FB4293B9AA },
      { 0x94470CCB99DB04B1, 0x127E395A6C55F099, 0xA408FBF684C8E81F, 0x1FC0F1C5C7B8683A },
    },
    {
      { 0x056DBF9AAF8BE412, 0xBA5C91E1EBA14CD2, 0xEABE1F45E5FC757B, 0x0569F169A66F6E1E },
      { 0xBFC887E63F6CD7FF, 0x2A2E5C509232C3A5, 0xD41CBF6925862382, 0x282887E0827A84C9 },
      { 0xB78607E299B71245, 0x3A2F6F0CE198E7EC, 0x796B7E8004F77F0E, 0x493E0AC5F87D0F5F },
    },
    {
      { 0x6F5147E25728DCBA, 0x0ECCB9E9BA26FF4B, 0xB1203405C39C2454, 0x276C123CE8B3EFCA },
      { 0xFB0CBE21174C3CD1, 0xD2256EEA29651B23, 0x4DA09FE0A5A75665, 0x1AA77C3E83F7A7B4 },
      { 0x7F71DA913062ECB1, 0x281A977E75A0D85C, 0xC404561854FB329E, 0x5816C969029C91B6 },
    },
  },
  { // base[21][0] - base[21][7]
    {
      { 0x96DE37DB727078B2, 0x0270DF94D25EB549, 0x2F0CB36A3C3D7B74, 0x476E2FAF5A136830 },
      { 0xBC09E0D15E521415, 0x76B17848D0C31EEC, 0x5763DE5C61347D43, 0x4872E9DBB78E5726 },
      { 0xAA29DFEB29ED472A, 0xF4EE0F761254FB20, 0x5FC3931D81ABC511, 0x62DA36289B0E5D39 },
    },
    {
      { 0x674EA6EEC53FF1F2, 0x5589B22B3B310718, 0xA5ACA7BDD9874CAC, 0x2E0E0577990914EF },
      { 0x54A0155F98A7BFC7, 0x712BF8EE47467A28, 0x8EDDEAA591D45F42, 0x50BBDFD1B6E5B89D },
      { 0x9B840DDE7D3CEDBE, 0xB024408F612FACD9, 0x843D3B32CE419243, 0x65730C9C3EC55ADD },
    },
    {
      { 0x3088BF224C2DFDB8, 0xFE70231538CB189B, 0x41D61A246A12C825, 0x7AB42D5F2DD21EB2 },
      { 0x46EDFB5529A25197, 0x3EC4755A5A0DA03C, 0xAF5875CBA50986B0, 0x0D006C8D8BDF9F01 },
      { 0x374B0499F5B0F959, 0xAA1E87D464FFA4A9, 0x6FB93A883D7B32B4, 0x09AA94DB11D87355 },
    },
    {
      { 0xFAE38B5E711741FF, 0x5A15F58CF404C2E0, 0xF64ED31B8A12A557, 0x2CB975028AC85309 },
      { 0x8C786DEBD6E8EA8C, 0xCBCFBC4467E08F88, 0xC39970F8388A3ACD, 0x3CDADC0D32E51D00 },
      { 0x07EA56106E47BBFF, 0x4D4D694A5626A7D4, 0xE00D96B2599B021A, 0x67BF4E4AC82F9DED },
    },
    {
      { 0xB8E4221E9AA94CFF, 0x45E69D8E6DF5F56B, 0x404924CF78D24A33, 0x0CA1776250A256E4 },
      { 0xB133A5DE2BC09817, 0x6C29079CC4D6EE07, 0x4614CCD729FDECE3, 0x18CC9D6C9731C726 },
      { 0x3ED6298CD711A4C9, 0x160D9EC886751F49, 0xAA9E724A129E0891, 0x15053298A77CE53A },
    },
    {
      { 0xE79B0D669E0E3C9D, 0x97CF5E562D1ADE1D, 0xB0743074D466D355, 0x02AEE1CDB6F50D09 },
      { 0x96DBC9BFFBFC93E1, 0x6DBA0F830BE8531A, 0xACC179D108AAD7BB, 0x267B710C323E16F6 },
      { 0xD88C913E6146ADDB, 0x83F1275E3BA6FFD5, 0x541E3C6772519644, 0x090D183B8855125B },
    },
    {
      { 0xEB2CB89F63BA41DB, 0xC45FF03BDC15CB57, 0x944F1411884BDE69, 0x693DDCA536769D4D },
      { 0xF26DAEAF4F81A542, 0xF0A9FE04981A5E16, 0xA302A348CAA8E9D8, 0x599FE3B63D207296 },
      { 0xAB1ECC954CADA417, 0x9A02E83E371C1C00, 0xA4281AEF17B26C72, 0x35C4834D9053D4FB },
    },
    {
      { 0xA0417D465AE3EDB2, 0x0343437C639A60AA, 0x94F3646CFB73D2BF, 0x4239E98453B1CDE7 },
      { 0x40955020B13810FD, 0xCCAD44FD7C922DA7, 0xFFD6E26728397782, 0x51DE1081D539F5B9 },
      { 0x65773C9301AC4EF9, 0x95A5A109236E6249, 0x0150F7BA7300D4A7, 0x4817B9DFEF021A0D },
    },
  },
  { // base[22][0] - base[22][7]
    {
      { 0x515A6D705AA88E4D, 0x3D64301495FFFF83, 0xCC0F9BAEFAA8211A, 0x1FB5EB92ED275096 },
      { 0x758C5CD5BFABA2E3, 0x811D45772BC3E348, 0xDB93896D16FBD7D4, 0x1B2CBE92F52E009E },
      { 0xB9A6C6BD88302C56, 0xECA02BCF37E3482F, 0x32337C7CC9014996, 0x3DBF660CED306B68 },
    },
    {
      { 0x36D72528D3BE7D44, 0xC1131B2A73D1C328, 0x04DDFFE6C796C16D, 0x41DF6E330DFAE55D },
      { 0x3C611B9E34AE347D, 0x6E92973303214837, 0x4A8EA222257095E9, 0x611AD6BB00BA1CAB },
      { 0x312C658683C4BAF1, 0x2494A12A48C4F94C, 0xD0655A11F171B772, 0x473E715866F83350 },
    },
    {
      { 0xFF537F6FECA5B873, 0xF8986028E0FE5D16, 0xA4416A3F3F97D5C4, 0x70A92B09C576775A },
      { 0xE24A321D624642C8, 0x7E9B0EFA1E309CD6, 0x04ED8BEE9D74A6A4, 0x7337052EC7DA33A5 },
      { 0x955DFB272438657D, 0xE6B2DE785522C5B5, 0x4D5F275D3AF44C2E, 0x7F85E4086A8A6F72 },
    },
    {
      { 0x41D64ED6B9B909D0, 0xCFFB7C5D177B974C, 0x988F176EA1F634AB, 0x0E9D483EEF62D5BA },
      { 0xDC8035D2137A09AE, 0x46B39B4F2BF0181A, 0xE5E46FECA7A31E14, 0x468FC6DE7C776DFA },
      { 0x5D0B49989F6840EE, 0x94994FD6C28D9A40, 0x8094009E018190E5, 0x00008DA2518DFEF1 },
    },
    {
      { 0x0B2B0FB4B50553AE, 0xE0DFB92E2C295EB5, 0x08D46EBFCD3CB356, 0x31ECC45169428813 },
      { 0x1FEFD0361FE33606, 0xAEA071C726EB06E9, 0xBD71C59C134726B8, 0x1D6246C8B741ABF0 },
      { 0x800903A9D7DE9197, 0xF495E75C7EEC7B41, 0x7C0B34D9C27395C8, 0x19FD6A9591B45033 },
    },
    {
      { 0xC6966468621167F4, 0x8395A7BD82D09D65, 0x51FF5B73767B52B7, 0x1E61AAE65C8538F1 },
      { 0xAA0324E362F20F0B, 0x057C3218199FBB9A, 0xD9567E697982F3A3, 0x0B607A14D12B6E53 },
      { 0x74DB4A21C81F488F, 0x5C524A65BD2B1BE7, 0x643E68D25D5D4922, 0x718F57A135D73AB4 },
    },
    {
      { 0x23ECBADCD1B806F4, 0xB94062FDF17C02A9, 0xA9B2C79399722EF0, 0x218F963FB32FC05A },
      { 0x59F482086D337F46, 0xC2EEA5A9360B72D3, 0xDE1ECBB08F7CDFC1, 0x6ACCB2458F548CDA },
      { 0x6B0131A242C7BD83, 0x8A559A97D0F528A5, 0x4480220D104854EB, 0x7D8238AFC8929D93 },
    },
    {
      { 0xD9BB614062735D63, 0x4B8769EEB68ECD85, 0xD84D4AAC2285FCA2, 0x2468567D2BE6F111 },
      { 0x41F6DE94567B5718, 0x431ABE45BEAE3D5A, 0xE02023B4DBF59622, 0x6CD9BDFAE17B2C1F },
      { 0xDB0793723ED5F32F, 0x78E8D0CBB1179D1B, 0xA104673F74CB01CA, 0x4B11A0C899B69DED },
    },
  },
  { // base[23][0] - base[23][7]
    {
      { 0x6EA4CE6B0FF9C320, 0x94E6CDE1831B12D0, 0xA8F16C011EEB9EE1, 0x2512B83D101DC918 },
      { 0xDCF24CEF7B133FFB, 0xBBB9653DBA160421, 0x11D00A9FF4D27958, 0x166FEFF66AE82803 },
      { 0x955BB34529FB76B5, 0x9821212C0EE8B850, 0x20000A261D7100B0, 0x2B90C4B6924724FE },
    },
    {
      { 0x142EA848D0E86D27, 0xA5D537D3DAFF1F04, 0xB1F28BBE70C9C9D9, 0x01E49AD7E258187E },
      { 0x05B72A8BFE8C0DD7, 0xC811314F95DCB1DA, 0x2A84DE7499032312, 0x2BC76EBA7B1E09ED },
      { 0x4CB93B6324958615, 0x23E661626FF102FE, 0x6E694DC26EB11D1E, 0x5F6155AC814463D1 },
    },
    {
      { 0xD3909D04D71968DC, 0x8795C3EFA07AE16A, 0x05D75263740F5594, 0x470DFB36356DD62F },
      { 0xD0D0693DF26C3DD3, 0xD4C5A6F5B09C8D76, 0x4CD06EE839E5CDC1, 0x56EAE12D1007E567 },
      { 0xF155EAF4BC96443F, 0x0D01000C65C936AE, 0xDFDD34E6DD572F8F, 0x3982A459AD7447AF },
    },
    {
      { 0xC02D84A5D0EB719A, 0xDF9F78BB849A9F8C, 0xA11F836583113815, 0x2C2D113BEC3C22EE },
      { 0x621AA8D1E5D45C77, 0x32D1378ED908AF8B, 0x3B07A7A955C61C28, 0x1821A21DA08EDC65 },
      { 0xD0C52FC119EA44B1, 0xB34C625AF63C12BF, 0xD3C737D29B9F20FF, 0x3B2B13C4A877CC0F },
    },
    {
      { 0x70B839D1F54367C5, 0x1D467DDB838AAFEE, 0x2429F3FE18C1C547, 0x545DFA425B09FB0B },
      { 0x1C61E7ACEA8FE460, 0xCDF6E97E82835B79, 0x135F884FD5AB8747, 0x1FA0B05460DC2353 },
      { 0x793097AE3789B63E, 0xD7F56883FB6E88DF, 0xA93F4D6909EF3799, 0x0F3CE59AC0C47BAE },
    },
    {
      { 0xBBF4A9EC7AF040B7, 0xC252862214CEEF6C, 0x6E361686432292F2, 0x63C55A969CE8F97A },
      { 0x009A1B61F77BF1EF, 0x4145B53FFF4F087C, 0xBFFC8472DE7CEF7E, 0x72EBCA8D9D1D9C18 },
      { 0xB3535269C92968A3, 0xF2EEF0DE438D6403, 0x5C1635A05360E4B7, 0x4B6C3D208D10910A },
    },
    {
      { 0x7DD26AF16AA702B8, 0xF10FD7EB975ECCFD, 0x24BD6139B774BBC7, 0x4FCC85ABBD2D36EF },
      { 0xD9DEBF2D21033101, 0xC3CDF1E6862D1260, 0x2BE02ED8EB7CCA5B, 0x547C3E40B2F9C653 },
      { 0xD19A27568DF47BEB, 0x3E8F2875D6753CC7, 0x3BE32B4F2906F029, 0x22C417F0A9A6B69F },
    },
    {
      { 0x6C564C94CA1E37F2, 0xDAFCF8B0D1C1C951, 0x134CED89DF644D79, 0x3EE7C21E7202F83A },
      { 0x33349A2EBABCC1EB, 0xB15B76888BD508D3, 0xBEEE8C2BCC2F0947, 0x3447F2DC7B137B6E },
      { 0xB6486B2425239960, 0xEA90A1FEE52B194C, 0xD9DF1461C8AEE370, 0x339CB43F39938C8D },
    },
  },
  { // base[24][0] - base[24][7]
    {
      { 0x5336E64EE4060D60, 0x4BD02E7A0D9C521B, 0xD3F5F9DF4AEDEBE3, 0x3ED05C7B46BF3ED5 },
      { 0x77BC100A1C2B3AD3, 0xD1324F98557ED4F4, 0xA668F5A82E6FD465, 0x2308AD5D0EA6E059 },
      { 0xEA078CA9E1DAED3B, 0x0ED637B99088CF4D, 0x81E63010FF592CB0, 0x2D2FC43F41B3A5A5 },
    },
    {
      { 0x4F4B1469D05321D3, 0x5AE1E58073619032, 0x4DA98144BE16F619, 0x61F1BD716AE8E386 },
      { 0x47B180E7B8509E7F, 0x67E75C0A9A86E862, 0x7B814BEA525E523F, 0x5B34DB2B72268A1A },
      { 0x1C3F1F8376D37090, 0xB3980EA8CCD09D60, 0x5EAD6C7C1B131C08, 0x7510F366A7EAF4DF },
    },
    {
      { 0xF7A09489334CD968, 0xB8E98423B8468980, 0x192A196808C1585E, 0x629B8D83800F459B },
      { 0x78E30B8518237326, 0xAC38951500691A92, 0xB4EDDE9E46415BAA, 0x6C35FCF8D0CAFFAB },
      { 0xD36D8446AF7C3C7C, 0x293C786E30849BF2, 0xD601A4E930D0B75C, 0x4757D81BC87290BC },
    },
    {
      { 0x28A0402F07BAD705, 0xF6017DF193316618, 0x9675EF8F75491CB6, 0x625719A262A1ADD9 },
      { 0xCB02AAE09BA4020E, 0x10CD20F34105D508, 0x8E40FB9C39A43686, 0x584D6633AD016330 },
      { 0x4E7944DCDDD2A1F7, 0x79BB074EAD64B8A1, 0x8EC172E327C9B055, 0x316A910DBFCA33C7 },
    },
    {
      { 0xBAC2EA131D7BBD1E, 0xEFD73D88FF748A26, 0xD28338402CFB8C9E, 0x0A794D29C1C9101B },
      { 0x292614CE0C6849AD, 0xE435DAB645060D06, 0x51BA82976DA54318, 0x6E077EF25E3AA2B1 },
      { 0x6FB8BF6E12D96BF1, 0x10FCB86DCCDA9820, 0x6D491A5BE1F6A631, 0x6F391B2E3DF7049F },
    },
    {
      { 0x3EC99C831784599F, 0xADCB2CF2EFCF995F, 0xD67F9ED68FCF5EFE, 0x385902AAE5B9A4DB },
      { 0xABADFE03A2B890B6, 0x9BBCB3AE834A6CAD, 0x4D051BDDFA0C8F19, 0x7BF8882623DA755E },
      { 0xDF2889E2AA889626, 0xB344211D4D440FE6, 0x22333BC2AF281DA3, 0x071A1CC7A5032025 },
    },
    {
      { 0x0C4985849F258C94, 0xBEF1F08739F9FB20, 0x7A190BED399CAEB7, 0x37C56F6B651BCE1F },
      { 0xDB3E916C9F675EF4, 0x84D9F42093C11783, 0x3A1FD30FD82DB6C6, 0x2F2A029B451B11B9 },
      { 0xF1A0091EFEDBD94D, 0x243DCBF0D10D5948, 0xFCCB3E817EF34A4F, 0x3C06F3976469EF4B },
    },
    {
      { 0x338FF579807A13B9, 0x47B97595154620D5, 0x14D0BFEBCB9B9949, 0x0EF7E356995AC3D3 },
      { 0x057142A2844D73DE, 0x1C46EF678E3FA683, 0x1C560AA8852408DC, 0x075945FB38C94672 },
      { 0xD7ADDF0D77A8CACA, 0x0A46093BC8BD8AF6, 0x94C8FBFDBD72ED17, 0x633E900DFC6E9433 },
    },
  },
  { // base[25][0] - base[25][7]
    {
      { 0xDE0F7A5EAB3D73CB, 0x1FB126596B224C5E, 0xF2083269160FA764, 0x577CE2D2DD1C2000 },
      { 0x4AFF48CD3A77A7CD, 0x9D413DF67B518451, 0xCB2700E984D23D80, 0x78E21E27ADD1E3CB },
      { 0x5B7EB6FB7D4F3A5D, 0x78C13C5E7257933D, 0xC12AD9E878F7CC87, 0x6D3AC651C862F949 },
    },
    {
      { 0x467048C58EB0EE4A, 0xC6F69B234D409833, 0xEA735414D7F45569, 0x0539C013FB1CEA1F },
      { 0xD15B93886CA31241, 0x1D5463696ABEA801, 0x71EA005FD05A43E5, 0x56DD712259F5B976 },
      { 0xCC07A517ABFFF0E6, 0x80338686F0C1CC21, 0x082E1FA524FD8AFE, 0x134C6531A893534E },
    },
    {
      { 0x171EB817AF1EEC87, 0x4F1F848C726929C3, 0x2F3B9F7B0126D4B5, 0x1E002586257D1999 },
      { 0xF3B298C419585D3C, 0x1C0C18FBC92E7FC5, 0x84540DC8D0148FE6, 0x0FDA1EE624E57583 },
      { 0xCD54A356037A5C0C, 0x8E514252D4036279, 0x9F69932FE366A3C3, 0x75A1FE80E68FE90B },
    },
    {
      { 0x5AE3A12C1F3B0770, 0xBAEE295CF7055CC8, 0x5F8A13E10395C91F, 0x79A10596B7F86CF8 },
      { 0xE3D3AEA5A34BE2A2, 0x0AFEFC246F87FFDF, 0x14345CF5D5233C2D, 0x2D346B882DA97B8A },
      { 0x57967B65CF428F03, 0xC7AC9C89E3111C62, 0x6D455C4B4CFDF9B9, 0x1EDAB197F51A5E4F },
    },
    {
      { 0x9724C858C14C12E1, 0x76F575C39F4D44C8, 0x77781E9CA638257C, 0x6C8CBF524AEF9587 },
      { 0xFA377715FBAEECEC, 0x868BD8FB1CB3ACD2, 0x8DF96898A4CF3939, 0x02190D6FA4EBAF89 },
      { 0x8270B00CF272AD57, 0x73BDA1BD3F17C974, 0x63E716E0B78ACD52, 0x22F57EE0FA6B8660 },
    },
    {
      { 0xDB07231267E658ED, 0xACEDE1495EAE01CA, 0x18D04E8EEE0240E4, 0x5FB9E7752EAB6CA0 },
      { 0xB4C200C2C022EB8C, 0x26117D5167978328, 0x4A0D1B32B59116E3, 0x6D2F75E401B16D6F },
      { 0xDBD3DFE8852746E3, 0x5F2B803F2264D99C, 0xB060903F8AABD77D, 0x1302C44893310C6D },
    },
    {
      { 0xA640C71E633B72A1, 0x2F21164981E76656, 0x7603E6655A094F84, 0x06F6FD08592221DC },
      { 0x2CFB825341B07F82, 0xE1EC9FEF3B30F37A, 0xC18D953989439AA8, 0x2A5686172730AEAB },
      { 0xF71DB3EADC15A915, 0x9B78B1A34FD2E0F5, 0x52DA69793760CFE9, 0x3176595D53BD4A04 },
    },
    {
      { 0xC903941B57DB143A, 0x2FE6AF42BCF08252, 0x2D5680D6E31850A5, 0x30C89EA83AB31FCC },
      { 0xF2F6BCA9308A9595, 0xA4B11ABE876EEBE8, 0x3A4164685CB5A638, 0x572CFC8CD4B36C5F },
      { 0x06E3169B0D1918ED, 0x7D23AC194A100138, 0x016C00A89FCACA67, 0x1EEDE15098E02EAE },
    },
  },
  { // base[26][0] - base[26][7]
    {
      { 0xF9D52BD113CB5D8A, 0x441D5D5BCD83ED10, 0xF2A5F10C18D01C8E, 0x2F73FD9C6C1902FC },
      { 0xCD6E07FCE72F629C, 0x81CE1535C6178986, 0x814003E3F87C4A8A, 0x7C4B418A560259B5 },
      { 0xA9C6FEE5A0A23547, 0x52D67ED4A1A49BFC, 0xA357C846931E463C, 0x30E8319E4DE50684 },
    },
    {
      { 0x56D1945E7C7E39E6, 0xF74234AED3781BFE, 0x31BFDA6D9C615484, 0x6D91D616FC033DEE },
      { 0xB1BA249AFFED92B3, 0xE2DEB5C4BC05B45D, 0xB78D99402A9F7601, 0x374B2FEC23D76BFA },
      { 0xCD695CA9F7402934, 0xF4478CD57D6F36C6, 0x07388B820A874167, 0x7CDCDDDCEECAEF6E },
    },
    {
      { 0x68CCBED7474FB9BA, 0x5019517C67DD840B, 0x66B65D0936A22F85, 0x0DD408A305665C1A },
      { 0xF5D9AAA035189361, 0x6931C1D4346461C9, 0xB6063214F2DCBD41, 0x2832F8AC64FE90A3 },
      { 0xB840B4FD86214CAA, 0x70A300566BB767B3, 0x17555CC5387322DD, 0x1CC0F9CF2C527D79 },
    },
    {
      { 0x6422EFD2B6F337EF, 0x70A952801620241D, 0xF4E970B1E3DA7B19, 0x187A22976E5E0DB2 },
      { 0x8C7DC53AAC918540, 0x8E8B47B4B0737A2E, 0x9D42D4A28A549E5A, 0x5C6E041B82D6687E },
      { 0x42B693C162BACBA0, 0xFD09A2B4FCCE5F66, 0xC4227E39E0752738, 0x3196CD0D2C9F9234 },
    },
    {
      { 0x5F84FE88F686424A, 0xB1F838C086CFB49D, 0x10C84616ABE7C3BC, 0x5D2D3EF9457B25D1 },
      { 0xFB5DB58ADC03E5D3, 0x8C11E3EFDE2A786B, 0xDD8ECB81B714B385, 0x05927A4423F6A52B },
      { 0x6E6D6A5F288DF55A, 0xD229C03AF6936679, 0xF0CE7FCF802FCD32, 0x5A7E7BA23AA40FB1 },
    },
    {
      { 0x52DD8ED5BC67D54C, 0x2E76D1338C85B979, 0x4984E48885493047, 0x008CD18217D9BA58 },
      { 0xE0CBF0263C4BB3E5, 0x5C38A6E59C6CA33E, 0xAAEF444141FCAFD4, 0x1E9DEE0B26FD31FB },
      { 0xB3D16C4E74610BB5, 0x334ED2FB344AE860, 0xFAB2CC72D9415158, 0x6B604478F6F10539 },
    },
    {
      { 0xAC0DAFD61279C781, 0x5485F4FF5D71865E, 0xCD10B4814567C978, 0x01C5BF5241AC81C7 },
      { 0x19B69E8888543702, 0x6BF9C41905BAD97D, 0xFC8A99BB128394C4, 0x44B3A635CC8845C3 },
      { 0xCFA77C10CC98B7FC, 0x17A4E9417553C6A7, 0x84B8D2D5AD7798BA, 0x372F18812CB4F5B2 },
    },
    {
      { 0x99827D8731833111, 0xFD9A8344C3D65D1F, 0x5E8C923BC60830D1, 0x182C56A1E8C1C310 },
      { 0xD8B6317AC3F2C9F4, 0xA4CCF6EF652E9F38, 0x5A48E0F00A661F36, 0x447A88A3C4D46DD4 },
      { 0x191E07FEF2BB31E8, 0x02E1EF9C51175308, 0xDEE3C55ED64CA7CD, 0x132A4FD277F1EE4C },
    },
  },
  { // base[27][0] - base[27][7]
    {
      { 0xEDA342A4D79FB337, 0xEBBFE78278A50752, 0x9EF91FFBD25D0623, 0x1D086FF099671E42 },
      { 0xBA0EAD230F35FCEB, 0x9182D9FE3BBBD2C0, 0xEA2ABA51323A69EC, 0x0C9370EE3200F07F },
      { 0xF03FA745750BE750, 0x97EA8AA31D0FE0FE, 0x8BA9917E98F96078, 0x0FD0E80EC30F2E8A },
    },
    {
      { 0x9C6E560068EFCA4C, 0x173895EEE88406F4, 0xBF89F49F7EEAF131, 0x79FE768C774D00F2 },
      { 0x66402ACA3EACCC19, 0x0F232B6D1BF8AA90, 0xCCFB7BBA2702C990, 0x3B9AB1DE353AE799 },
      { 0x8358F4843189CE50, 0x5249ED33E2D01F66, 0x46BBE76456B1C499, 0x4FA135B80DC327A2 },
    },
    {
      { 0x78BF1AE448D092A9, 0x5BB5C0A9ABAF4E3B, 0x7D41A03786CDB91F, 0x05BB5D8D9FD3F21C },
      { 0xF7E4932620C88DF7, 0xF8D1DBDC0BD11612, 0x2C3AED35F9878A23, 0x670D7A938E98D848 },
      { 0x045C60FCBFC949C4, 0xDF33B8E5EA2255B7, 0x9172B231CCDDC00B, 0x7DB6EB0F5BB954AA },
    },
    {
      { 0x2B9855FCD580E95A, 0x8B7DBB6E200A1D8C, 0x43365F32D065D940, 0x69FD4DB2CDFFB57F },
      { 0xDBD6E0F428799EC9, 0xBCCC7D27B0466AE7, 0xD6CB16DEC6FE2DED, 0x381F4DE7578E97A7 },
      { 0xB60A6474CA442A21, 0xC21D2EB332D76A72, 0x0C0DDB9F5E6B2D78, 0x5CC6C9F2E2630FA8 },
    },
    {
      { 0xE1C3BE306973F1F9, 0x1D9A5550184145D8, 0x941F1373B9CF789C, 0x34CE4E48016182BB },
      { 0x8E25E8B399F12470, 0x5ECF09438ADF852F, 0xEA1FC678508581BB, 0x69D84DAEEF8C8D89 },
      { 0xF9835391ACA378E6, 0xC90B8C5AE672ECBE, 0x9466E923C0DA74BA, 0x28E5798637E6EC83 },
    },
    {
      { 0xB9BD7CCD0C562A5A, 0xC819BC6E628E5987, 0x15C4DE19A6708663, 0x495714E0C4FC74CD },
      { 0xD305D3A13B3A7005, 0x318742B850BD3DF9, 0x1BAC2B1EE7999266, 0x2A82551491C1FED5 },
      { 0x54CF60658F8680DE, 0x06E8F7E61D1A7BD7, 0x2AE53A90E84E2711, 0x6FE8A7F4AC75D2F5 },
    },
    {
      { 0x9E217F2F5FC9E5C7, 0x5F6FD4289B6A2B2F, 0x707842CF44211074, 0x3EEB9FCB0392E894 },
      { 0xDAEE16EF9422D596, 0x034A48D8853FF4C9, 0xA6D579EB200171A3, 0x049FF9372C323A68 },
      { 0xD886927F3402CC0B, 0xBAB983396DD791F2, 0x09B3929D5A2BD614, 0x57ACDE5E435A3852 },
    },
    {
      { 0x3820EAB05B48E177, 0xC2900D9FD6EBF38F, 0x8B6170B18899AAC2, 0x5552AF1E80841458 },
      { 0xC1C23EA14C8B89A6, 0xD68DC88FAB3E81EB, 0x3F3BA46CDF3BD568, 0x6A2C5A171728D7A5 },
      { 0x76A8C9730603A21E, 0x2162A716BA210E88, 0x1A95A6417EDAE432, 0x49F48025453B4332 },
    },
  },
  { // base[28][0] - base[28][7]
    {
      { 0xDD9700648C9DC3B6, 0xF671D448706E2835, 0x7679DBE01B6F324F, 0x6FA302044C6F4F0D },
      { 0xB9CEC422C197E764, 0xFD1C6B64D735FC31, 0x195E06E55BA7FF7B, 0x79C9BF440A5E722F },
      { 0x5C81B88B14BDFA3D, 0x54E89D916A78341A, 0xF0CB8AABA34B5EE3, 0x567C527448EAF41A },
    },
    {
      { 0x965AA43F0BE835D1, 0x12691C0E1CA80CB5, 0x6BB2CE40C2CBC518, 0x3D37BF9448EB527B },
      { 0xB6C9FEC3838887AA, 0x6EA604E9BE1C5AA4, 0xBE58B5266139B543, 0x5024DEB72C129504 },
      { 0xBE84FEC6B54D779B, 0x7877305F2D9EDC85, 0xA610DA9628CF5FEA, 0x7008D56FE2A2CA0E },
    },
    {
      { 0x31EF68640165FC48, 0xFDE84C6506FFB555, 0x3126857EDCDB76CC, 0x34E70C5BBC9A058F },
      { 0xAFB3C936E7CAFC1E, 0xBE3F42B0B8944838, 0x6B50F3F9CCC7BD2D, 0x37E2E60D85B17CF0 },
      { 0x68F7AA94594C3CDC, 0x6E8D571E6A3F4849, 0x893F02210C4F91A9, 0x4AACB59D72B880F8 },
    },
    {
      { 0x84FF988B3F289265, 0x85F20AC5ECE3A2EF, 0x1495BE913F7AAB72, 0x1D527120D7DB689C },
      { 0xA3139CE91FC8BCD1, 0x7FC18918CBEB6EE7, 0x0983EF5AA9F90A45, 0x06911BB43DAFA6ED },
      { 0x96609C5F951982F1, 0xA42C1FC7D1749361, 0x041D58D12AA4E975, 0x597E55372343D1B6 },
    },
    {
      { 0x1903D23993C3E666, 0x0BF18C847909F1FC, 0xEAD9766BFB06CB27, 0x7A37B19B613005F4 },
      { 0x3DE2B746E2BECD71, 0x9F05E976CEF85EF9, 0x55600A6F1177F251, 0x6313F4E77F5EB52E },
      { 0x9FA579A2D5B64B8E, 0xF14475B94CA1B98F, 0x99ACB54501A20C36, 0x3DA48B803F6B3149 },
    },
    {
      { 0xAA1A0D946EA9D165, 0x550BC82DEFA17E1F, 0x87FAC96CA6E97C7A, 0x4E81B107F04669BE },
      { 0x16C2FDAE55C25832, 0x24BC086944F9DE0A, 0xA3B56E223D8AE706, 0x0915D1BB7C227EBD },
      { 0xE1011966D15A72AA, 0xCF687EA108AE8C3F, 0x1755DA5F3EEA3CEC, 0x016385FA95B47626 },
    },
    {
      { 0xD672995FA2C6B967, 0xADF3B4703E5B9E5A, 0xAB67BECA7745DEF3, 0x75834BF1FF5A1D01 },
      { 0x5143F625AE8597D4, 0x20AE2BC803A44165, 0x7022530F60E840AE, 0x5319A785204F7AF0 },
      { 0xDB6478256FB151E0, 0x9F78007783B6D22E, 0x4E4E5CAC24F86954, 0x0E61BFA1A20D97D7 },
    },
    {
      { 0x6BB32B78E4E75753, 0x0E2D8AFC0C72B2B5, 0x137394194226119A, 0x5D1A37BB978CB41C },
      { 0x2847AB2D2E63991E, 0xE830E26072835491, 0x7D8C55EDAE22D60C, 0x763404E081C018A5 },
      { 0xE96AA889716D3564, 0x74DE8198D8F428F6, 0x4B03A36EC7633931, 0x02C88DCFB77BE2E8 },
    },
  },
  { // base[29][0] - base[29][7]
    {
      { 0xA9A4D679A897756E, 0x9060A0E98E60E5A4, 0x920C0603D4CD3446, 0x6AAF7CE8E325968B },
      { 0x6099CCC1FAEF8754, 0xE079BAC7A8962656, 0x1678898505D9CC70, 0x759E767CD5138631 },
      { 0x1B53B85D1DB9DE84, 0x3125778451D7DF86, 0xAB9BFFCC5A04A379, 0x33AFA6F099C0BA4E },
    },
    {
      { 0x5097FB6C9DED5985, 0x8392EC07CEB296FF, 0x80CE27F9CD5F4A43, 0x707A285C41669E21 },
      { 0x07162901B58BC17E, 0x3240B640B656C1DA, 0x686E5EECB4B2039F, 0x09ECCEFB80B26290 },
      { 0x00A5AF6190F2E065, 0x27E5B4E4EB8CDFD1, 0x272F8E0C3A8011D0, 0x0E036F4F2AF6D640 },
    },
    {
      { 0xFFEA95A07FB6B4D5, 0x9A29858C6E2024DD, 0xAF252E17D1A6CC4B, 0x3C04B7C73E995D16 },
      { 0x4C87BD6B519F6271, 0xB3047C9C5F177047, 0xCE50A1E2B194228A, 0x2679C50FF616DB06 },
      { 0x5055553286FD2E6A, 0xFCE24F15245AA3C6, 0x2784E63EB801B92D, 0x5B9E569D13048D5F },
    },
    {
      { 0xF8DF547DC4EEDDCD, 0x9DE5965E30D75765, 0x47AC53DD8FCDC6CE, 0x50AA3F6D28895343 },
      { 0x594A31A6C164FABE, 0x0FE5FEF09249A29B, 0x4F4E26D9A0C66DAD, 0x002079ECA2A20CFE },
      { 0x6F7EF49CFEACC360, 0xFA139644A8851C06, 0x5B95D203DD988CDC, 0x71AA8519A512A6FA },
    },
    {
      { 0x4DDD2C22B92A3D9B, 0xF982E37D71620470, 0x30747D34E39A78C6, 0x5CD495D7D53EBB3D },
      { 0xB283EB76DAB4E792, 0x8BC214D806529770, 0xF53E004875B5EB2E, 0x5F753163ED7BC7A8 },
      { 0xCE926389F34993A7, 0x2FB1C2BBB46DE9BA, 0xB8292AB075C559CD, 0x3421B50332E4E266 },
    },
    {
      { 0x8F2B698BF410083E, 0x62933422420574B2, 0x60F050E31907FE3D, 0x29B9B34E48B08A39 },
      { 0x5E011AF410179F8A, 0x63AE0071327CBAD8, 0xC8D274EAD1C6120B, 0x4BDB73FB455BC4FC },
      { 0x2E940A55CD072922, 0x4847904264E559FE, 0xD7E57AC42D968F65, 0x4E5A5AD33C7C3E88 },
    },
    {
      { 0xB5BA5531515003EA, 0xF988F0587838E3D8, 0x2B83F21C0005F111, 0x56E07E96C177B756 },
      { 0xDB3260359CA57E36, 0x06446F124C6D2FD8, 0x27C6818B25E56C1A, 0x1985E53C6F3A1A51 },
      { 0x4C177FC2088CBA27, 0xFCB4AF4B1583A392, 0xE2C560A7DFE4A9FD, 0x1E18DF0D9B4F8E7A },
    },
    {
      { 0x60B45E49FCE5A139, 0x575C388FE3E76DCC, 0xBF8729551A5646BD, 0x20E76084BF3EAADD },
      { 0xD87A4326844A4577, 0x03EE0CF748DD0E37, 0xBCBAE6D7535650AC, 0x1985B089A1316A5D },
      { 0xFBCB0CEBD136C045, 0xDD8FEA4F0ECF0AB6, 0xB9EBE1B66DD0EF93, 0x135A266C8F943BBE },
    },
  },
  { // base[30][0] - base[30][7]
    {
      { 0x70DBF949B13981B8, 0x25A93CFFF5E54516, 0xED7E3BC55FEA098A, 0x7EF5880A4E393087 },
      { 0x28F82423C79C3A31, 0xD92EDE7A4E5F659E, 0x4D5589226CCF902A, 0x56384F360E0852EB },
      { 0x65B157B543B3773D, 0xB365F6022AA9E687, 0x2C40009C0785F25A, 0x0473474FFB167175 },
    },
    {
      { 0x979684EA855C7973, 0xD65C90C6E2AC91EF, 0xA5479A1339BB365C, 0x66589DEB9C7B8CFA },
      { 0x1A56A80525E0984D, 0x469C6DA49E85EA4E, 0xD12E1ECC280544DF, 0x578F9FC3F75D1D84 },
      { 0x7BC2463AF28ADB25, 0x52CA80DD6DA5481C, 0x6106989F9FBA8DA8, 0x0CD0F1A9E0571774 },
    },
    {
      { 0x5A10B966EACB5ED5, 0xC9F022A1CC777E20, 0x4FD8A9A3DA2084DA, 0x79B5E9CC8133571A },
      { 0xBE8E3AB05D7D02D8, 0xD9F0D0506372AF30, 0x71A94B8C606B3239, 0x60AA3588E1061A43 },
      { 0xC2A996A8499A59DA, 0xA37E88A5B040B2B9, 0x662FAF98212E41BA, 0x20914AD15C3FD5AE },
    },
    {
      { 0x9732A930F149F563, 0xC22D49019099D66D, 0x2304BAE5BC804CB5, 0x03B05DC68CAD6EC0 },
      { 0x8CE4CDC47ABF6B6B, 0xA9C9E59336FC6412, 0x2E771909D9856939, 0x4A70A9F5DA96971A },
      { 0x209F0D0BE6F340C5, 0xAB8AB6D4F6B4D042, 0x165F9347A3656658, 0x359A5F4DE19D62F9 },
    },
    {
      { 0x08FE34B2B2B8F960, 0xE364F422A98739BD, 0xF19D73D16A7F281A, 0x40DCE3DB1736E985 },
      { 0x79EF97B21D3C6059, 0xA61F4B8F7917013E, 0xF63E8E2F24E0DAD1, 0x100960C784916E96 },
      { 0xC405AAF2AD644E8B, 0x8A41920FA2D053B1, 0x1E9B77EFE173B60F, 0x4457ADBC2725D6F4 },
    },
    {
      { 0xF1398A6944E6161C, 0x25F25E88D1438BC6, 0x0C6A946B7D19B267, 0x7211E0EAD7ECC137 },
      { 0x141A4CEE440F9290, 0x4E8292ED3BC991DB, 0xC4BD6EFDB39A20FA, 0x595BCEB88B1D0B46 },
      { 0xE642FC6CF6FE59B5, 0x915E61479BA372FC, 0x724EF19C7CF2E9E6, 0x24052F7DE09F16E6 },
    },
    {
      { 0x5B30A67221671106, 0xB70CCEE626029C94, 0x331FDA526E0E5F01, 0x52598EA3B48E4703 },
      { 0x85A8F38580B1102F, 0x835A82E7C58ED7E2, 0x9635D830F7AD55E6, 0x63D513B0065BDF98 },
      { 0x152A0F76E00AFC58, 0x08D27F3F3E349FBE, 0x7857B309A7513C6B, 0x6A2DAC2E8A6ED04A },
    },
    {
      { 0xB102726871D990E7, 0x1DD531BD147F8F4A, 0x858667FEADCCDECF, 0x66916E1F32646838 },
      { 0x33DF93AF506A1CFE, 0x56F3471A044DF75F, 0x214489A66A3CF397, 0x47B17CE1995D2A2A },
      { 0xFE5A32C4EB1DAF93, 0xAE57351FABE5E7B0, 0x7F5D61694A9D7D02, 0x4E07D00D1B1B8A1B },
    },
  },
  { // base[31][0] - base[31][7]
    {
      { 0x348415874649B528, 0x7CE4D01AE0ED62DB, 0x37DB9F2A626FDB1A, 0x2002A0CD8E95E0A0 },
      { 0x6963025B114A1EF6, 0x5E465F672267D9D0, 0xAE92A7F9CBC0433C, 0x47D1B0A79D8E535F },
      { 0x5001FF5EDCDF4178, 0x1044E0D79D225648, 0x7C24CFC88CAA7D47, 0x0FDD10C577A055A1 },
    },
    {
      { 0xA79F2B821F3D80CA, 0xD40E9F77046D57BF, 0xE41CE355CCEE6F78, 0x3629AE89FFBBB0EA },
      { 0xD5AA4A247D647A9F, 0xC0FB744D3DD31BA0, 0x3A7EB63EB615AF00, 0x1C971D6554643721 },
      { 0x265E9A749F451ACE, 0x9703C0A22C43F40B, 0x8C98CE3B794D5855, 0x52F0BFF26A85609D },
    },
    {
      { 0xC8AFBFFABB7890CA, 0x61A5191397E6C3F1, 0xE65D17EF268DF293, 0x75DD4147C4B4C4CD },
      { 0x85144DEB8F027B3B, 0x90470E296B2107CA, 0xA8C36C581A348FD5, 0x12ABA8A2154FD9A8 },
      { 0xF168DE33487F1C77, 0x265AA50C504CBD6A, 0x4B8EB48A57C2306A, 0x6ACEA827BFB5BDF2 },
    },
    {
      { 0xCE2448F3FB693375, 0x83A250CD8183BC0D, 0x441C478EB030F11D, 0x491F53519AA5EA87 },
      { 0x53B9C1BC59F5AA61, 0x0EB4E9B352AA9E3E, 0x851367B17C94005D, 0x40D5896AC03F190B },
      { 0x08C68C4820F196CB, 0xDCF6F1E16C18AC24, 0x0F55A138EC1922EC, 0x251CB0F1648C50AA },
    },
    {
      { 0x0193EB2279919F8F, 0xA4CD13071A7E780B, 0xC1DAD38B796D5CBC, 0x34567756CDEA088F },
      { 0x38EE1DF07C735DD0, 0x6B677C1A3F7FF185, 0xD4C9212FF09D23B5, 0x166B5E71FD8EDBB1 },
      { 0x1C5A648779EBE108, 0x984737125BD68206, 0x9C306CF8DBF39F11, 0x2CABB06ADA847ACB },
    },
    {
      { 0x441565F5FE8113C8, 0x44D79982E208ABB0, 0xB2FA4971BE9A39FA, 0x165962EFAA28AD15 },
      { 0xB094DFF0825531C2, 0x47CB00045253FE65, 0x9FC5E044BEC84A2C, 0x784FD21F6E5948D4 },
      { 0xF5852EC631FE9565, 0x6915E0B31734A77F, 0x9391F9B77C65D81D, 0x387814F67864098F },
    },
    {
      { 0x953557D52F085853, 0xBC7851B877820D54, 0xBB9F7DBBD51D6B0F, 0x627652D153A5ECF0 },
      { 0xA30983D997769F10, 0x57021799D22AC0F3, 0x64A224E98CAF81B3, 0x45BEAEC53618A42C },
      { 0x92EA24193DCAEA98, 0xB869C180519A078E, 0x6F0E298E3070E295, 0x53911228963EF4F2 },
    },
    {
      { 0x5FBDDDC52154BAFE, 0x462E1CBBCB56D1AC, 0x713FE3B7E6F6D524, 0x0CB9AFEBFB5E1053 },
      { 0x0D5E4957A4E29A17, 0xFFF76C08D9737D68, 0xF7D14646FE642714, 0x08DAEF8C522662A1 },
      { 0x71D5C86821642133, 0xF5C24707BF8CAA3F, 0x9281D0E832D24BDC, 0x07F7C888C8EFC4AF },
    },
  },
};