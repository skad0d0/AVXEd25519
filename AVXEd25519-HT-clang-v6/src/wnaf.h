#ifndef _WNAF_H
#define _WNAF_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "gfparith.h"
#define A_WINDOW 7
#define B_WINDOW 4

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