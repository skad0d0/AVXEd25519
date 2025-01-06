/**
 ******************************************************************************
 * @file wnaf.h
 * @version 1.0.0
 * @date 2024-10-01
 * @copyright Copyright © 2024 by University of Luxembourg.
 * @author Developed at SnT APSIA by: Bowen Zhang.
 *
 * @brief Header file for Window Non-Adjacent Form (WNAF) conversion functions and structures.
 *
 * @details
 * This file contains the definitions and function prototypes for converting integers
 * to their Window Non-Adjacent Form (WNAF) representation, which is used in elliptic
 * curve cryptography to optimize scalar multiplication.
 ******************************************************************************
 */
#ifndef _WNAF_H
#define _WNAF_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "gfparith.h"
#define A_WINDOW 5
#define B_WINDOW 7

typedef struct
{
    uint32_t k[256];
    int length;
} NAFResult;

typedef struct
{
    __m256i k0[256];
    __m256i k1[256];
    int max_length;
} NAFResult_avx2;

void sc25519_slide(signed char *r_signed, const uint32_t *s, int swindowsize);
void conv_char_to_NAF(NAFResult *r, const signed char *r_signed);
int find_max(const int *arr, int size);
void NAF_conv(NAFResult_avx2 *r, const __m256i *s, const __m256i *h);
#endif