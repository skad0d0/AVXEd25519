/**
 ******************************************************************************
 * @file gfparith.h
 * @version 1.0.0
 * @date 2024-10-01
 * @copyright Copyright © 2024 by University of Luxembourg.
 * @author Developed at SnT APSIA by: Bowen Zhang.
 *
 * @brief Header file for AVX2-optimized arithmetic operations on field elements using radix-2^29 representation.
 *
 * @details
 * This file contains function declarations and macros for performing arithmetic operations on field elements
 * using AVX2 intrinsics. The field elements are represented in a radix-2^29 format.
 *
 * Macros:
 * - NWORDS: Number of words in the field element representation.
 * - BITS29: Number of bits in each word (29 bits).
 * - BITS23: Number of bits in the last word (23 bits).
 * - MASK29: Mask for 29-bit words.
 * - CONST2A: Constant value 2A.
 * - CONSTC: Constant value C.
 * - MASK23: Mask for 23-bit words.
 * - LSWP29: Least significant 29-bit word of p.
 ******************************************************************************
 */
#ifndef _GFPARITH_H
#define _GFPARITH_H

#include "intrin.h"
#include <stdint.h>

// we use a radix-2^29 representation for the field elements
#define NWORDS 9
#define BITS29 29
#define BITS23 23
#define MASK29 0x1FFFFFFFUL
#define CONST2A 973324 // 2A = 2*A
// #define CONST2B 973328 // 2B = -(A+2)*2
#define CONSTC 1216
#define MASK23 0x007FFFFFUL

// least significant 29-bit word of p = 64*(2^255 - 19) = 2^261 - 1216
#define LSWP29 0x1FFFFB40UL


void mpi29_gfp_add_avx2(__m256i *r, const __m256i *a, const __m256i *b);
void mpi29_gfp_sub_avx2(__m256i *r, const __m256i *a, const __m256i *b);
void mpi29_gfp_sbc_avx2(__m256i *r, const __m256i *a, const __m256i *b);
void mpi29_gfp_mul_avx2(__m256i *r, const __m256i *a, const __m256i *b);
void mpi29_gfp_mul29_avx2(__m256i *r, const __m256i *a, const uint32_t b);
void mpi29_gfp_sqr_avx2(__m256i *r, const __m256i *a);
void mpi29_gfp_inv_avx2(__m256i *r, const __m256i *a);
void mpi29_gfp_neg_avx2(__m256i *r);
void mpi29_cswap_avx2(__m256i *r, __m256i *a, const __m256i b);
void mpi29_copy_avx2(__m256i *r, const __m256i *a);
void mpi29_gfp_canonic_avx2(__m256i *a);
void mpi29_ini_to_one_avx2(__m256i *r);
void mpi29_ini_to_zero_avx2(__m256i *r);
int is_all_zero(__m256i r);
#endif