/**
 ******************************************************************************
 * @file intrin.h
 * @version 1.0.0
 * @date 2024-10-01
 * @copyright Copyright © 2024 by University of Luxembourg.
 * @author Developed at SnT APSIA by: Bowen Zhang.
 *
 * @brief Header file for AVX2 intrinsics and macros for 4-way arithmetic and logical operations on packed 64-bit integers.
 *
 * @details
 * This header file provides a set of macros for performing various arithmetic, logical, and other operations
 * on packed 64-bit integers using AVX2 intrinsics.
 *
 * Macros:
 * - VADD(X, Y): Adds packed 64-bit integers in X and Y.
 * - VSUB(X, Y): Subtracts packed 64-bit integers in Y from X.
 * - VMUL(X, Y): Multiplies packed 32-bit integers in X and Y, producing packed 64-bit integers.
 * - VMAC(Z, X, Y): Multiplies packed 32-bit integers in X and Y, then adds the result to Z.
 * - VABS8(X): Computes the absolute value of packed 8-bit integers in X.
 * - VABS32(X): Computes the absolute value of packed 32-bit integers in X.
 * - VXOR(X, Y): Computes the bitwise XOR of packed 64-bit integers in X and Y.
 * - VAND(X, Y): Computes the bitwise AND of packed 64-bit integers in X and Y.
 * - VOR(X, Y): Computes the bitwise OR of packed 64-bit integers in X and Y.
 * - VSHR(X, Y): Shifts packed 64-bit integers in X right by Y bits.
 * - VSHL(X, Y): Shifts packed 64-bit integers in X left by Y bits.
 * - VLOAD128(X): Loads 128-bit integer from memory address X.
 * - VSET164(X): Sets all packed 64-bit integers to X.
 * - VSET64(W, X, Y, Z): Sets packed 64-bit integers to W, X, Y, and Z.
 * - VZERO: Sets all packed 64-bit integers to zero.
 * - VEXTR32(X, Y): Extracts the 32-bit integer from X at index Y.
 * - VSHUF32(X, Y): Shuffles the 32-bit integers in X according to the control in Y.
 * - VBROAD64(X): Broadcasts the 64-bit integer X to all elements.
 * - VPERM64(X, Y): Permutes the 64-bit integers in X according to the control in Y.
 ******************************************************************************
 */
#ifndef INTRIN_H
#define INTRIN_H

// AVX2 header file 
#include <immintrin.h>

// 4-way arithmetic operations on packed 64-bit integers
#define VADD(X, Y)         _mm256_add_epi64(X, Y)
#define VSUB(X, Y)         _mm256_sub_epi64(X, Y)
#define VMUL(X, Y)         _mm256_mul_epu32(X, Y)
#define VMAC(Z, X, Y)      VADD(Z, VMUL(X, Y))
#define VABS8(X)           _mm256_abs_epi8(X)
#define VABS32(X)          _mm256_abs_epi32(X)
// 4-way logical operations on packed 64-bit integers
#define VXOR(X, Y)         _mm256_xor_si256(X, Y)
#define VAND(X, Y)         _mm256_and_si256(X, Y)
#define VOR(X, Y)          _mm256_or_si256(X, Y)
#define VSHR(X, Y)         _mm256_srli_epi64(X, Y)
#define VSHL(X, Y)         _mm256_slli_epi64(X, Y)
// other 4-way operations on packed 64-bit integers
#define VLOAD128(X)        _mm_load_si128((__m128i*)X)
#define VSET164(X)         _mm256_set1_epi64x(X)
#define VSET64(W, X, Y, Z) _mm256_set_epi64x(W, X, Y, Z)
#define VZERO              _mm256_setzero_si256()
#define VEXTR32(X, Y)      _mm256_extract_epi32(X, Y)
#define VSHUF32(X, Y)      _mm256_shuffle_epi32(X, Y)
#define VBROAD64(X)        _mm256_broadcastq_epi64(X)
#define VPERM64(X, Y)      _mm256_permute4x64_epi64(X, Y)
#endif
