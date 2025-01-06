#include "jsf.h"
// r = r >> 1
static void right_shift(uint32_t *r) {
    uint32_t carry = 0;
    uint32_t temp;
    int i;

    for (i = 7; i >= 0; i--) {
        temp =(uint32_t) r[i] & 1;
        r[i] = (r[i] >> 1) | (carry << 31);
        carry = temp;
    }
}

// if r = 0, return 1 else return 0
static int is_zero(uint32_t *r) {
    int i;

    for (i = 0; i < 8; i++) {
        if (r[i] != 0) {
            return 0;
        }
    }
    return 1;
}

// compute the Joint Sparse Form of a and b
void JSF(JSFResult *r, const uint32_t *a, const uint32_t *b)
{
    uint32_t k0[8], k1[8];
    
    r->length = 0;

    int i;
    uint8_t d[2] = {0x00, 0x00};
    uint8_t l[2] = {0x00, 0x00};
    uint8_t u;
    uint8_t t;
    int is_3, is_2;
    for (i = 0; i < 8; i++)
    {
        k0[i] = a[i];
        k1[i] = b[i];
    }

    while (!is_zero(k0) || !is_zero(k1) || d[0] > 0 || d[1] > 0){
        l[0] = d[0] + k0[0];
        l[1] = d[1] + k1[0];
        for (i = 0; i < 2; i++) {
            is_3 = 0;
            is_2 = 0;
            if ((l[i] & 1) == 0) { // l[i] % 2 == 0
                u = 0;
            } else {
                u = l[i] & 3; // l[i] % 4
                if (u == 3) u = 0xff;
                if (u == 0xfd) u = 0x01;
                if ((l[i] & 7) == 3 || (l[i] & 7) == 5) is_3 = 1; // l[i] % 8 == 3 or l[i] % 8 == 5
                if ((l[1 - i] & 3) == 2) is_2 = 1; // l[1-i] % 4 == 2
                if (is_3 && is_2) u = -u;
            }
            if (i == 0) r->k0[r->length] =(uint32_t) u;
            else r->k1[r->length] =(uint32_t) u;
        }
        for (i = 0; i < 2; i++) {
            if (i == 0) {
                t = 1 + r->k0[r->length];
                if (2 * d[0] == t) {
                    d[0] = 1 - d[0];
                }
                right_shift(k0);
            } else {
                t = 1 + r->k1[r->length];
                if (2 * d[1] == t) {
                    d[1] = 1 - d[1];
                }
                right_shift(k1);
            }
        }
        r->length++;
    }
}

void JSF_conv(JSFResult_avx2 *r, const __m256i *s, const __m256i *h)
{
    JSFResult r1, r2, r3, r4;
    uint32_t s1[8], s2[8], s3[8], s4[8], h1[8], h2[8], h3[8], h4[8];
    int i;
    
    for (i = 0; i < 8; i++)
    {
        s1[i] = VEXTR32(s[i], 0);
        s2[i] = VEXTR32(s[i], 2);
        s3[i] = VEXTR32(s[i], 4);
        s4[i] = VEXTR32(s[i], 6);

        h1[i] = VEXTR32(h[i], 0);
        h2[i] = VEXTR32(h[i], 2);
        h3[i] = VEXTR32(h[i], 4);
        h4[i] = VEXTR32(h[i], 6);        
    }

    JSF(&r1, s1, h1); JSF(&r2, s2, h2);
    JSF(&r3, s3, h3); JSF(&r4, s4, h4);

    for (i = 0; i < 256; i++)
    {
        r->k0[i] = VSET64(r4.k0[i], r3.k0[i], r2.k0[i], r1.k0[i]);
        r->k1[i] = VSET64(r4.k1[i], r3.k1[i], r2.k1[i], r1.k1[i]);
    }

    r->length = 254; //wrost case
}
