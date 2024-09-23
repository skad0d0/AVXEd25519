#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "gfparith.h"
#include "tedarith.h"
#include "montcurve.h"
#include "jsf.h"
#include "utils.h"
// #include "keygen.h"
extern uint64_t read_tsc(void);


void test_dsmul()
{
    uint32_t k1[8] = {0x10000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    uint32_t k2[8] = {0x00100000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
    uint32_t k3[8] = {0x000000f6, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000}; // k3 = 102
    uint32_t k4[8] = {0x00000035, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000}; // k4= 53

    uint32_t t0[8] = {0x00000000, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0x00000000, 0x00000000};
    uint32_t t1[8] = {0x00000000, 0xFFFFFFF7, 0xFFFFFFFF, 0xFFFFFFFF,0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000};


    // create point A
    ProPoint a;
    __m256i t[8];
    int i;
    for (i = 0; i < 8; i++) t[i] = VSET164(k1[i]);

    ted_mul_fixbase(&a, t);

    // create scalar vec 1 and 2
    __m256i s1[8], s2[8];

    for (i = 0; i < 8; i++)
    {
        s1[i] = VSET164(t0[i]);
        s2[i] = VSET164(k4[i]);
    }
    


    ProPoint r;
    ted_sep_double_scalar_mul(&r, &a, s1, s2);

    AffPoint h;

    ted_pro_to_aff(&h, &r);
    mpi29_gfp_canonic_avx2(h.x);
    
    uint32_t x1[NWORDS], x2[NWORDS], x3[NWORDS], x4[NWORDS];

    uint32_t x_1[8], x_2[8], x_3[8], x_4[8];

    for (i = 0; i < NWORDS; i++)
    {
        x1[i] = VEXTR32(h.x[i], 0);
        x2[i] = VEXTR32(h.x[i], 2);
        x3[i] = VEXTR32(h.x[i], 4);
        x4[i] = VEXTR32(h.x[i], 6);
    }

    mpi29_conv_29to32(x_1, 8, x1, NWORDS);
    mpi29_conv_29to32(x_2, 8, x2, NWORDS);
    mpi29_conv_29to32(x_3, 8, x3, NWORDS);
    mpi29_conv_29to32(x_4, 8, x4, NWORDS);

    mpi29_print("x1 = ", x_1, 8);
    mpi29_print("x2 = ", x_2, 8);
    mpi29_print("x3 = ", x_3, 8);
    mpi29_print("x4 = ", x_4, 8);
    
    // Simultaneous
    ted_sim_double_scalar_mul_v2(&r, &a, s1, s2);
    ted_pro_to_aff(&h, &r);
    mpi29_gfp_canonic_avx2(h.x);
    for (i = 0; i < NWORDS; i++)
    {
        x1[i] = VEXTR32(h.x[i], 0);
        x2[i] = VEXTR32(h.x[i], 2);
        x3[i] = VEXTR32(h.x[i], 4);
        x4[i] = VEXTR32(h.x[i], 6);
    }

    mpi29_conv_29to32(x_1, 8, x1, NWORDS);
    mpi29_conv_29to32(x_2, 8, x2, NWORDS);
    mpi29_conv_29to32(x_3, 8, x3, NWORDS);
    mpi29_conv_29to32(x_4, 8, x4, NWORDS);

    mpi29_print("x1 = ", x_1, 8);
    mpi29_print("x2 = ", x_2, 8);
    mpi29_print("x3 = ", x_3, 8);
    mpi29_print("x4 = ", x_4, 8);
    



}


void timing_fp_arith()
{
        __m256i a[NWORDS], b[NWORDS], r[NWORDS];
    int i, seed;

    // initialize random generator
    seed = (int) time(NULL);
        srand(seed);

    // randomize the input
    for (i = 0; i < NWORDS; i++) {
        a[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        b[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        r[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
    }

    // benchmark part
    unsigned long long start_cycles, end_cycles, diff_cycles;
    int iterations = 10000;

    puts("");

    // load cache
    for (i = 0; i < iterations; i++) mpi29_gfp_add_avx2(r, r, b);
    // measure timing
    start_cycles = read_tsc();
    for (i = 0; i < iterations; i++) {
        mpi29_gfp_add_avx2(r, r, b);
        mpi29_gfp_add_avx2(r, r, b);
        mpi29_gfp_add_avx2(r, r, b);
        mpi29_gfp_add_avx2(r, r, b);
        mpi29_gfp_add_avx2(r, r, b);
        mpi29_gfp_add_avx2(r, r, b);
        mpi29_gfp_add_avx2(r, r, b);
        mpi29_gfp_add_avx2(r, r, b);
        mpi29_gfp_add_avx2(r, r, b);
        mpi29_gfp_add_avx2(r, r, b);
    }
    end_cycles = read_tsc();
    diff_cycles = (end_cycles-start_cycles)/(10*iterations);
    printf("* 4-Way ADD: %lld\n", diff_cycles);

    // load cache
    for (i = 0; i < iterations; i++) mpi29_gfp_sbc_avx2(r, r, b);
    // measure timing
    start_cycles = read_tsc();
    for (i = 0; i < iterations; i++) {
        mpi29_gfp_sbc_avx2(r, r, b);
        mpi29_gfp_sbc_avx2(r, r, b);
        mpi29_gfp_sbc_avx2(r, r, b);
        mpi29_gfp_sbc_avx2(r, r, b);
        mpi29_gfp_sbc_avx2(r, r, b);
        mpi29_gfp_sbc_avx2(r, r, b);
        mpi29_gfp_sbc_avx2(r, r, b);
        mpi29_gfp_sbc_avx2(r, r, b);
        mpi29_gfp_sbc_avx2(r, r, b);
        mpi29_gfp_sbc_avx2(r, r, b);
    }
    end_cycles = read_tsc();
    diff_cycles = (end_cycles-start_cycles)/(10*iterations);
    printf("* 4-Way SBC: %lld\n", diff_cycles);

    // load cache
    for (i = 0; i < iterations; i++) mpi29_gfp_mul_avx2(r, r, a);
    // measure timing
    start_cycles = read_tsc();
    for (i = 0; i < iterations; i++) {
        mpi29_gfp_mul_avx2(r, r, a);
        mpi29_gfp_mul_avx2(r, r, a);
        mpi29_gfp_mul_avx2(r, r, a);
        mpi29_gfp_mul_avx2(r, r, a);
        mpi29_gfp_mul_avx2(r, r, a);
        mpi29_gfp_mul_avx2(r, r, a);
        mpi29_gfp_mul_avx2(r, r, a);
        mpi29_gfp_mul_avx2(r, r, a);
        mpi29_gfp_mul_avx2(r, r, a);
        mpi29_gfp_mul_avx2(r, r, a);
    }
    end_cycles = read_tsc();
    diff_cycles = (end_cycles-start_cycles)/(10*iterations);
    printf("* 4-Way MUL: %lld\n", diff_cycles);

    // load cache
    for (i = 0; i < iterations; i++) mpi29_gfp_sqr_avx2(r, r);
    // measure timing 
    start_cycles = read_tsc();
    for (i = 0; i < iterations; i++) {
        mpi29_gfp_sqr_avx2(r, r);
        mpi29_gfp_sqr_avx2(r, r);
        mpi29_gfp_sqr_avx2(r, r);
        mpi29_gfp_sqr_avx2(r, r);
        mpi29_gfp_sqr_avx2(r, r);
        mpi29_gfp_sqr_avx2(r, r);
        mpi29_gfp_sqr_avx2(r, r);
        mpi29_gfp_sqr_avx2(r, r);
        mpi29_gfp_sqr_avx2(r, r);
        mpi29_gfp_sqr_avx2(r, r);
    }
    end_cycles = read_tsc();
    diff_cycles = (end_cycles-start_cycles)/(10*iterations);
    printf("* 4-Way SQR: %lld\n", diff_cycles);

    // load cache
    for (i = 0; i < iterations; i++) mpi29_gfp_inv_avx2(r, r);
    // measure timing 
    start_cycles = read_tsc();
    for (i = 0; i < iterations; i++)
    {
        mpi29_gfp_inv_avx2(r, r);
    }
    end_cycles = read_tsc();
    diff_cycles = (end_cycles-start_cycles)/(1*iterations);
    printf("* 4-Way INV: %lld\n", diff_cycles);

}

void timing_point_arith()
{
    ProPoint p, q, h;
    ExtPoint r, a;
    AffPoint f;
    __m256i t[NWORDS], s[NWORDS];
    int i, seed;

    // initialize random generator
    seed = (int) time(NULL);
    srand(seed);

    // randomize the input
    for (i = 0; i < NWORDS; i++) {
        t[i]   = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        s[i]   = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        p.x[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        p.y[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        p.z[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        q.x[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        q.y[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        q.z[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        h.x[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        h.y[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        h.z[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        r.x[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        r.y[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        r.z[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        r.e[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        r.h[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        a.x[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        a.y[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        a.z[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        a.e[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        a.h[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        f.x[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
        f.y[i] = VSET64(mpi29_rand32(), mpi29_rand32(), mpi29_rand32(), mpi29_rand32());
    }
        // benchmark part
        unsigned long long start_cycles, end_cycles, diff_cycles;
        int iterations = 10000;

        

        puts("\nTwisted Edwards curve:");

       

        // // load cache
        // for (i = 0; i < iterations; i++) ted_sep_double_scalar_mul(&h, &p, s, t);
        // start_cycles = read_tsc();
        // for (i = 0; i < iterations; i++)
        // {
        //     ted_sep_double_scalar_mul(&h, &p, s, t);
        //     ted_sep_double_scalar_mul(&h, &p, s, t);
        //     ted_sep_double_scalar_mul(&h, &p, s, t);
        //     ted_sep_double_scalar_mul(&h, &p, s, t);
        //     ted_sep_double_scalar_mul(&h, &p, s, t);
        //     ted_sep_double_scalar_mul(&h, &p, s, t);
        //     ted_sep_double_scalar_mul(&h, &p, s, t);
        //     ted_sep_double_scalar_mul(&h, &p, s, t);
        //     ted_sep_double_scalar_mul(&h, &p, s, t);
        //     ted_sep_double_scalar_mul(&h, &p, s, t);
        // }
        // end_cycles = read_tsc();
        // diff_cycles = (end_cycles-start_cycles)/(10*iterations);
        // printf("* 4-Way Separate double-scalar Point multiplication: %lld\n", diff_cycles);
        // printf("* single Separate double-scalar Point multiplication: %lld\n", diff_cycles/4);

        // load cache
        for (i = 0; i < iterations; i++) ted_sep_double_scalar_mul_v2(&h, &p, s, t);
        start_cycles = read_tsc();
        for (i = 0; i < iterations; i++)
        {
            ted_sep_double_scalar_mul_v2(&h, &p, s, t);
            ted_sep_double_scalar_mul_v2(&h, &p, s, t);
            ted_sep_double_scalar_mul_v2(&h, &p, s, t);
            ted_sep_double_scalar_mul_v2(&h, &p, s, t);
            ted_sep_double_scalar_mul_v2(&h, &p, s, t);
            ted_sep_double_scalar_mul_v2(&h, &p, s, t);
            ted_sep_double_scalar_mul_v2(&h, &p, s, t);
            ted_sep_double_scalar_mul_v2(&h, &p, s, t);
            ted_sep_double_scalar_mul_v2(&h, &p, s, t);
            ted_sep_double_scalar_mul_v2(&h, &p, s, t);
        }
        end_cycles = read_tsc();
        diff_cycles = (end_cycles-start_cycles)/(10*iterations);
        printf("* 4-Way Separate double-scalar Point multiplication (load table once): %lld\n", diff_cycles);
        printf("* single Separate double-scalar Point multiplication (load table once): %lld\n", diff_cycles/4);

        for (i = 0; i < iterations; i++) ted_sim_double_scalar_mul_v2(&h, &p, s, t);
        start_cycles = read_tsc();
        for (i = 0; i < iterations; i++)
        {
            ted_sim_double_scalar_mul_v2(&h, &p, s, t);
            ted_sim_double_scalar_mul_v2(&h, &p, s, t);
            ted_sim_double_scalar_mul_v2(&h, &p, s, t);
            ted_sim_double_scalar_mul_v2(&h, &p, s, t);
            ted_sim_double_scalar_mul_v2(&h, &p, s, t);
            ted_sim_double_scalar_mul_v2(&h, &p, s, t);
            ted_sim_double_scalar_mul_v2(&h, &p, s, t);
            ted_sim_double_scalar_mul_v2(&h, &p, s, t);
            ted_sim_double_scalar_mul_v2(&h, &p, s, t);
            ted_sim_double_scalar_mul_v2(&h, &p, s, t);
        }
        end_cycles = read_tsc();
        diff_cycles = (end_cycles-start_cycles)/(10*iterations);
        printf("* 4-Way Simultaneous double-scalar Point multiplication (load table once): %lld\n", diff_cycles);
        printf("* single Simultaneous double-scalar Point multiplication (load table once): %lld\n", diff_cycles/4);
}

void test_fix_mul()
{
    uint32_t t1[8] = {0x00000000, 0xFFFFFFF7, 0xFFFFFFFF, 0xFFFFFFFF,0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000000};


    // create point A
    ProPoint a;
    __m256i t[8];
    int i;
    for (i = 0; i < 8; i++) t[i] = VSET164(t1[i]);

    ted_mul_fixbase(&a, t);
    AffPoint h;

    ted_pro_to_aff(&h, &a);
    mpi29_gfp_canonic_avx2(h.x);
    
    uint32_t x1[NWORDS], x2[NWORDS], x3[NWORDS], x4[NWORDS];

    uint32_t x_1[8], x_2[8], x_3[8], x_4[8];

    for (i = 0; i < NWORDS; i++)
    {
        x1[i] = VEXTR32(h.x[i], 0);
        x2[i] = VEXTR32(h.x[i], 2);
        x3[i] = VEXTR32(h.x[i], 4);
        x4[i] = VEXTR32(h.x[i], 6);
    }

    mpi29_conv_29to32(x_1, 8, x1, NWORDS);
    mpi29_conv_29to32(x_2, 8, x2, NWORDS);
    mpi29_conv_29to32(x_3, 8, x3, NWORDS);
    mpi29_conv_29to32(x_4, 8, x4, NWORDS);

    mpi29_print("x1 = ", x_1, 8);
    mpi29_print("x2 = ", x_2, 8);
    mpi29_print("x3 = ", x_3, 8);
    mpi29_print("x4 = ", x_4, 8);

    // non-cache attack
    printf("------- non cache attack ---------- \n");
    ted_mul_fixbase_v2(&a, t);
    ted_pro_to_aff(&h, &a);
    mpi29_gfp_canonic_avx2(h.x);
    for (i = 0; i < NWORDS; i++)
    {
        x1[i] = VEXTR32(h.x[i], 0);
        x2[i] = VEXTR32(h.x[i], 2);
        x3[i] = VEXTR32(h.x[i], 4);
        x4[i] = VEXTR32(h.x[i], 6);
    }

    mpi29_conv_29to32(x_1, 8, x1, NWORDS);
    mpi29_conv_29to32(x_2, 8, x2, NWORDS);
    mpi29_conv_29to32(x_3, 8, x3, NWORDS);
    mpi29_conv_29to32(x_4, 8, x4, NWORDS);

    mpi29_print("x1 = ", x_1, 8);
    mpi29_print("x2 = ", x_2, 8);
    mpi29_print("x3 = ", x_3, 8);
    mpi29_print("x4 = ", x_4, 8);

}


int main(){
    // test_dsmul();
    timing_point_arith();
    return 0;
}
