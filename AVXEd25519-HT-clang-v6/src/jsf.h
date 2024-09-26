#ifndef _JSF_H
#define _JSF_H

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "gfparith.h"
typedef struct jsf_result{
    uint32_t k0[256]; 
    uint32_t k1[256];
    int length;
} JSFResult;

typedef struct
{
    __m256i k0[256];
    __m256i k1[256];
    int length;
}JSFResult_avx2;


// void right_shift(uint32_t *num);

void JSF(JSFResult *r, const uint32_t *a, const uint32_t *b);
void JSF_conv(JSFResult_avx2 *r, const __m256i *s, const __m256i *h);


#endif