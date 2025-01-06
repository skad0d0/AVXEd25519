#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include "gfparith.h"
#include "tedarith.h"
#include "montcurve.h"
#include "jsf.h"
#include "utils.h"
#include "wnaf.h"
// #include "keygen.h"
extern uint64_t read_tsc(void);


void test_dsmul()
{
    uint32_t k1[8] = {0x10000000, 0x00000200, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000};
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
    AffPoint h;
    uint32_t x1[NWORDS], x2[NWORDS], x3[NWORDS], x4[NWORDS];
    uint32_t x_1[8], x_2[8], x_3[8], x_4[8];

    // Separate double scalar multiplication
    ted_sep_double_scalar_mul(&r, &a, s1, s2);
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
    printf(" -------------------- separate double scalar multiplication -------------------\n");
    mpi29_print("x1 = ", x_1, 8);

    // NAF double scalar multiplication
    ted_naf_double_scalar_mul(&r, &a, s1, s2);
    ted_pro_to_aff(&h, &r);
    mpi29_gfp_canonic_avx2(h.x);
    for (i = 0; i < NWORDS; i++)
    {
        x1[i] = VEXTR32(h.x[i], 0);
        x2[i] = VEXTR32(h.x[i], 2);
        x3[i] = VEXTR32(h.x[i], 4);
        x4[i] = VEXTR32(h.x[i], 6);
    }
    mpi29_conv_29to32(x_2, 8, x2, NWORDS);
    printf(" -------------------- naf double scalar multiplication -------------------\n");
    mpi29_print("x2 = ", x_2, 8);

    // JSF double scalar multiplication
    ted_jsf_double_scalar_mul(&r, &a, s1, s2);
    ted_pro_to_aff(&h, &r);
    mpi29_gfp_canonic_avx2(h.x);
    for (i = 0; i < NWORDS; i++)
    {
        x1[i] = VEXTR32(h.x[i], 0);
        x2[i] = VEXTR32(h.x[i], 2);
        x3[i] = VEXTR32(h.x[i], 4);
        x4[i] = VEXTR32(h.x[i], 6);
    }
    mpi29_conv_29to32(x_3, 8, x3, NWORDS);
    printf(" -------------------- jsf double scalar multiplication -------------------\n");
    mpi29_print("x3 = ", x_3, 8);
    // printf("test\n");
    // Check if results are equal
    if (memcmp(x_1, x_2, sizeof(x_1)) == 0 && memcmp(x_1, x_3, sizeof(x_1)) == 0)
    {
        printf("\nThe results of three dsm methods are equal!\n");
    }
    else
    {
        printf("\nThe results of three dsm methods are not equal!\n");
    }
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
    int iterations = 1000000;

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
    for (i = 0; i < iterations; i++) mpi29_gfp_sub_avx2(r, r, b);
    // measure timing
    start_cycles = read_tsc();
    for (i = 0; i < iterations; i++) {
        mpi29_gfp_sub_avx2(r, r, b);
        mpi29_gfp_sub_avx2(r, r, b);
        mpi29_gfp_sub_avx2(r, r, b);
        mpi29_gfp_sub_avx2(r, r, b);
        mpi29_gfp_sub_avx2(r, r, b);
        mpi29_gfp_sub_avx2(r, r, b);
        mpi29_gfp_sub_avx2(r, r, b);
        mpi29_gfp_sub_avx2(r, r, b);
        mpi29_gfp_sub_avx2(r, r, b);
        mpi29_gfp_sub_avx2(r, r, b);
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

void timing_table_query()
{
    ProPoint p, q, h;
    __m256i t[NWORDS], s[NWORDS];
    int i, seed;

    // initialize random generator
    seed = (int) time(NULL);
    srand(seed);
    int pos = 1 + rand() % 31;
    uint32_t index = 1 + rand()% 7;
    __m256i b = VSET64(index, index, index, index);
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
    }
    // benchmark part
    unsigned long long start_cycles, end_cycles, diff_cycles;
    int iterations = 10000;
    puts("\nTable query:");
    // load cache
    for (i = 0; i < iterations; i++) ted_table_query(&p, pos, b);
    start_cycles = read_tsc();
    for (i = 0; i < iterations; i++)
    {
        ted_table_query(&p, pos, b);
        p.x[2] = VXOR(p.z[2], p.y[2]);
    }
    end_cycles = read_tsc();
    diff_cycles = ((end_cycles-start_cycles) - iterations )/(iterations);
    printf("* Batched table query in sep method: %lld\n", diff_cycles);

    // load cache
    for (i = 0; i < iterations; i++) jsf_query(&p, &q, b);
    start_cycles = read_tsc();
    for (i = 0; i < iterations; i++)
    {
        jsf_query(&p, &q, b);
        p.x[2] = VXOR(p.z[2], p.y[2]);
    }
    end_cycles = read_tsc();
    diff_cycles = ((end_cycles-start_cycles) - iterations )/(iterations);
    printf("* Batched table query in JSF method: %lld\n", diff_cycles);

    // load cache
    for (i = 0; i < iterations; i++) table_query_wB(&p, b);
    start_cycles = read_tsc();
    for (i = 0; i < iterations; i++)
    {
        table_query_wB(&p, b);
        p.x[2] = VXOR(p.z[2], p.y[2]);
    }
    end_cycles = read_tsc();
    diff_cycles = ((end_cycles-start_cycles) - iterations )/(iterations);
    printf("* Batched table query B (fixed point) in NAF method: %lld\n", diff_cycles);

    // load cache
    for (i = 0; i < iterations; i++) table_query_wA(&p, &q, b);
    start_cycles = read_tsc();
    for (i = 0; i < iterations; i++)
    {
        table_query_wA(&p, &q, b);
        p.x[2] = VXOR(p.z[2], p.y[2]);
    }
    end_cycles = read_tsc();
    diff_cycles = ((end_cycles-start_cycles) - iterations )/(iterations);
    printf("* Batched table query A (variable point) in NAF method: %lld\n", diff_cycles);


    // printf("* single Separate double-scalar Point multiplication: %lld\n", diff_cycles/4);


}

void timing_point_arith()
{
    ProPoint p, q, h;
    ExtPoint r, a;
    AffPoint f;
    __m256i t[NWORDS], s[NWORDS];
    int i, seed;
    clock_t start_time, end_time;
    double tp;
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

        puts("\nMontgomery curve:");

        for (i = 0; i < iterations; i++) mon_ladder_step(&p, &q, t);
        start_cycles = read_tsc();
        for (i = 0; i < iterations; i++)
        {
            mon_ladder_step(&p, &q, t);
            mon_ladder_step(&p, &q, t);
            mon_ladder_step(&p, &q, t);
            mon_ladder_step(&p, &q, t);
            mon_ladder_step(&p, &q, t);
            mon_ladder_step(&p, &q, t);
            mon_ladder_step(&p, &q, t);
            mon_ladder_step(&p, &q, t);
            mon_ladder_step(&p, &q, t);
            mon_ladder_step(&p, &q, t);
        }

        end_cycles = read_tsc();
        diff_cycles = (end_cycles-start_cycles)/(10*iterations);
        printf("* 4-Way Ladder step: %lld\n", diff_cycles);

        puts("\nTwisted Edwards curve:");

        for (i = 0; i < iterations; i++)ted_add(&r, &r, &p);
        start_cycles = read_tsc();
        for (i = 0; i < iterations; i++)
        {
            ted_add(&r, &r, &p);
            ted_add(&r, &r, &p);
            ted_add(&r, &r, &p);
            ted_add(&r, &r, &p);
            ted_add(&r, &r, &p);
            ted_add(&r, &r, &p);
            ted_add(&r, &r, &p);
            ted_add(&r, &r, &p);
            ted_add(&r, &r, &p);
            ted_add(&r, &r, &p);
        }
        end_cycles = read_tsc();
        diff_cycles = (end_cycles-start_cycles)/(10*iterations);
        printf("* 4-Way Point Addition: %lld\n", diff_cycles);

        for (i = 0; i < iterations; i++)ted_dbl(&r, &r);
        start_cycles = read_tsc();
        for (i = 0; i < iterations; i++)
        {
            ted_dbl(&r, &r);
            ted_dbl(&r, &r);
            ted_dbl(&r, &r);
            ted_dbl(&r, &r);
            ted_dbl(&r, &r);
            ted_dbl(&r, &r);
            ted_dbl(&r, &r);
            ted_dbl(&r, &r);
            ted_dbl(&r, &r);
            ted_dbl(&r, &r);
        }
        end_cycles = read_tsc();
        diff_cycles = (end_cycles-start_cycles)/(10*iterations);
        printf("* 4-Way Point doubling: %lld\n", diff_cycles);


        // load cache
        for (i = 0; i < iterations; i++) ted_sep_double_scalar_mul(&h, &p, s, t);
        start_cycles = read_tsc();
        start_time = clock();
        for (i = 0; i < iterations; i++)
        {
            ted_sep_double_scalar_mul(&h, &p, s, t);
            ted_sep_double_scalar_mul(&h, &p, s, t);
            ted_sep_double_scalar_mul(&h, &p, s, t);
            ted_sep_double_scalar_mul(&h, &p, s, t);
            ted_sep_double_scalar_mul(&h, &p, s, t);
            ted_sep_double_scalar_mul(&h, &p, s, t);
            ted_sep_double_scalar_mul(&h, &p, s, t);
            ted_sep_double_scalar_mul(&h, &p, s, t);
            ted_sep_double_scalar_mul(&h, &p, s, t);
            ted_sep_double_scalar_mul(&h, &p, s, t);
        }
        end_time = clock();
        end_cycles = read_tsc();
        diff_cycles = (end_cycles-start_cycles)/(10*iterations);
        printf("* 4-Way Separate double-scalar Point multiplication: %lld\n", diff_cycles);
        printf("* single Separate double-scalar Point multiplication: %lld\n", diff_cycles/4);
        tp = 1e6*4*10*iterations / (double)(end_time-start_time);
        printf("  - Throughput: %8.1f op/sec\n", tp);

        for (i = 0; i < iterations; i++) ted_jsf_double_scalar_mul(&h, &p, s, t);
        start_cycles = read_tsc();
        start_time = clock();
        for (i = 0; i < iterations; i++)
        {
            ted_jsf_double_scalar_mul(&h, &p, s, t);
            ted_jsf_double_scalar_mul(&h, &p, s, t);
            ted_jsf_double_scalar_mul(&h, &p, s, t);
            ted_jsf_double_scalar_mul(&h, &p, s, t);
            ted_jsf_double_scalar_mul(&h, &p, s, t);
            ted_jsf_double_scalar_mul(&h, &p, s, t);
            ted_jsf_double_scalar_mul(&h, &p, s, t);
            ted_jsf_double_scalar_mul(&h, &p, s, t);
            ted_jsf_double_scalar_mul(&h, &p, s, t);
            ted_jsf_double_scalar_mul(&h, &p, s, t);
        }
        end_time = clock();
        end_cycles = read_tsc();
        diff_cycles = (end_cycles-start_cycles)/(10*iterations);
        printf("* 4-Way JSF Simultaneous double-scalar Point multiplication: %lld\n", diff_cycles);
        printf("* single JSF Simultaneous double-scalar Point multiplication: %lld\n", diff_cycles/4);
        tp = 1e6*4*10*iterations / (double)(end_time-start_time);
        printf("  - Throughput: %8.1f op/sec\n", tp);


        for (i = 0; i < iterations; i++) ted_naf_double_scalar_mul(&h, &p, s, t);
        start_cycles = read_tsc();
        start_time = clock();
        for (i = 0; i < iterations; i++)
        {
            ted_naf_double_scalar_mul(&h, &p, s, t);
            ted_naf_double_scalar_mul(&h, &p, s, t);
            ted_naf_double_scalar_mul(&h, &p, s, t);
            ted_naf_double_scalar_mul(&h, &p, s, t);
            ted_naf_double_scalar_mul(&h, &p, s, t);
            ted_naf_double_scalar_mul(&h, &p, s, t);
            ted_naf_double_scalar_mul(&h, &p, s, t);
            ted_naf_double_scalar_mul(&h, &p, s, t);
            ted_naf_double_scalar_mul(&h, &p, s, t);
            ted_naf_double_scalar_mul(&h, &p, s, t);
        }
        end_cycles = read_tsc();
        end_time = clock();
        diff_cycles = (end_cycles-start_cycles)/(10*iterations);
        printf("* 4-Way NAF Simultaneous double-scalar Point multiplication: %lld\n", diff_cycles);
        printf("* single NAF Simultaneous double-scalar Point multiplication: %lld\n", diff_cycles/4);
        tp = 1e6*4*10*iterations / (double)(end_time-start_time);
        printf("  - Throughput: %8.1f op/sec\n", tp);


}


void timing_all()
{
    puts("\n\n*******************************************************************");
    puts("TIMING OF SOFTWARE (clock cycles):");
    puts("-------------------------------------------------------------------");
    puts("Field operations:");
    timing_fp_arith();
    puts("-------------------------------------------------------------------");
    puts("Point operations:");
    timing_table_query();
    timing_point_arith();
    puts("*******************************************************************");
}


int main()
{
    test_dsmul();
    timing_all();
    return 0;
}
