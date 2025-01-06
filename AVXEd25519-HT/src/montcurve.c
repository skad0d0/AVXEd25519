#include "montcurve.h"
#include "tedarith.h"
// (P, Q) <- Ladderstep(P, Q, pk)
// only operates on x and z projective coordinate
void mon_ladder_step(ProPoint *p, ProPoint *q, const __m256i *xd)
{
    __m256i t1[NWORDS], t2[NWORDS];

    mpi29_gfp_add_avx2(t1, p->x, p->z);          // t1 = xP + zP
    mpi29_gfp_sbc_avx2(p->x, p->x, p->z);        // xP = xP - zP
    mpi29_gfp_add_avx2(t2, q->x, q->z);          // t2 = xQ + zQ
    mpi29_gfp_sub_avx2(q->x, q->x, q->z);
    mpi29_gfp_sqr_avx2(p->z, t1);                // zP = t1 ^ 2
    mpi29_gfp_mul_avx2(q->z, t2, p->x);          // zQ = t2 * xP
    mpi29_gfp_mul_avx2(t2, q->x, t1);            // t2 = xQ * t1
    mpi29_gfp_sqr_avx2(t1, p->x);                // t1 = xP ^ 2
    mpi29_gfp_mul_avx2(p->x, p->z, t1);          // xP = zP * t1
    mpi29_gfp_sub_avx2(t1, p->z, t1);
    mpi29_gfp_mul29_avx2(q->x, t1, 121665);       // xQ = t1 * (A-2)/4
    mpi29_gfp_add_avx2(q->x, q->x, p->z);        // xQ = xQ + zP
    mpi29_gfp_mul_avx2(p->z, q->x, t1);          // zP = xQ * t1
    mpi29_gfp_add_avx2(t1, t2, q->z);            // t1 = t2 + zQ
    mpi29_gfp_sqr_avx2(q->x, t1);                // xQ = t1 ^ 2
    mpi29_gfp_sbc_avx2(t1, t2, q->z);            // t1 = t2 - zQ
    mpi29_gfp_sqr_avx2(t2, t1);                  // t2 = t1 ^ 2
    mpi29_gfp_mul_avx2(q->z, t2, xd);            // zQ = t2 * xd
}

// swap (P, Q) if c = 1
static void mon_cswap(ProPoint *p, ProPoint *q, const __m256i b)
{
    const __m256i one = VSET164(1);
    __m256i c;
    c = VAND(b, one);
    mpi29_cswap_avx2(p->x, q->x, c);
    mpi29_cswap_avx2(p->z, q->z, c);
}

// xR = k * xP
// only computes the x-coordinate of R = k * P.
// Return (X1 Z1), (X2 Z2) after the ladder step, (Q1, Q2) <- MonLadder(P)
void mon_mul_varbase(ProPoint *q1, ProPoint *q2, const __m256i *k, const __m256i *x)
{
    ProPoint p1, p2;
    __m256i b, s = VZERO, kp[8];

    int i;
    const __m256i t0 = VSET164(0xFFFFFFF8UL);
    const __m256i t1 = VSET164(0x7FFFFFFFUL);
    const __m256i t2 = VSET164(0x40000000UL);
    // prune scalar k
    for(i = 0; i < 8; i++) kp[i] = k[i];
    kp[0] = VAND(kp[0], t0);
    kp[7] = VAND(kp[7], t1);
    kp[7] = VOR(kp[7], t2);

    // initialize ladder
    for(i = 0; i < NWORDS; i++)
    {
        p1.x[i] = p1.z[i] = p2.z[i] = VZERO;
        p2.x[i] = x[i];
    }

    p1.x[0] = p2.z[0] = VSET164(1);

    // ladder loop, from msb to lsb
    for (i = 254; i >= 0; i--)
    {
        b = kp[i >> 5];
        b = VSHR(b, i&31);
        s = VXOR(s, b);
        mon_cswap(&p1, &p2, s);
        mon_ladder_step(&p1, &p2, x);
        s = b;
    }
    mon_cswap(&p1, &p2, s);
    
    mpi29_copy_avx2(q1->x, p1.x);
    mpi29_copy_avx2(q1->z, p1.z);
    mpi29_copy_avx2(q2->x, p2.x);
    mpi29_copy_avx2(q2->z, p2.z);
}
