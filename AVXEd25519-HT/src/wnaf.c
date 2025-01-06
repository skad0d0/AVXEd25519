#include "wnaf.h"
#include "utils.h"

// convert char to uint32_t
void conv_char_to_NAF(NAFResult *r, const signed char *r_signed)
{
    r->length = 0;

    // Traverse from index 255 down to 0
    for (int i = 255; i >= 0; i--) {
        r->k[i] = (uint32_t)((int32_t)r_signed[i]);
        if (r->length == 0 && r->k[i] != 0)
            r->length = i + 1;
    }
}

// return the max. length among all representations
int find_max(const int *arr, int size)
{
  int max = arr[0];
  int i;

  for (i = 1; i < size; i++)
  {
    if (arr[i] > max) max = arr[i];
  }
  return max;
}


// convert an unsigned integer to its w-naf representation
void sc25519_slide(signed char *r_signed, const uint32_t *s, int swindowsize) {
    int i, j, k, b;
    signed char m = (1 << (swindowsize - 1)) - 1;
    const int soplen = 256;

    uint32_t sv0 = s[0];
    uint32_t sv1 = s[1];
    uint32_t sv2 = s[2];
    uint32_t sv3 = s[3];
    uint32_t sv4 = s[4];
    uint32_t sv5 = s[5];
    uint32_t sv6 = s[6];
    uint32_t sv7 = s[7];

    for (i = 0; i < 32; i++) {
        r_signed[i]       = (sv0 & 1);
        r_signed[i + 32]  = (sv1 & 1);
        r_signed[i + 64]  = (sv2 & 1);
        r_signed[i + 96]  = (sv3 & 1);
        r_signed[i + 128] = (sv4 & 1);
        r_signed[i + 160] = (sv5 & 1);
        r_signed[i + 192] = (sv6 & 1);
        r_signed[i + 224] = (sv7 & 1);

        sv0 >>= 1;
        sv1 >>= 1;
        sv2 >>= 1;
        sv3 >>= 1;
        sv4 >>= 1;
        sv5 >>= 1;
        sv6 >>= 1;
        sv7 >>= 1;
    }

    for (j = 0; j < soplen; ++j) {
        if (r_signed[j] != 0) {
            for (b = 1; b < soplen - j && b <= swindowsize; ++b) {
                if (r_signed[j] + (r_signed[j + b] << b) <= m) {
                    r_signed[j] += r_signed[j + b] << b;
                    r_signed[j + b] = 0;
                }
                else if ((r_signed[j] - (r_signed[j + b] << b)) >= -m) {
                    r_signed[j] -= r_signed[j + b] << b;
                    for (k = j + b; k < soplen; ++k) {
                        if (r_signed[k] == 0) {
                            r_signed[k] = 1;
                            break;
                        }
                        r_signed[k] = 0;
                    }
                }
                else if (r_signed[j + b] != 0) {
                    break;
                }
            }
        }
    }
}

// batch w-naf conversion
void NAF_conv(NAFResult_avx2 *r, const __m256i *s, const __m256i *h)
{
    uint32_t s1[8], s2[8], s3[8], s4[8], h1[8], h2[8], h3[8], h4[8];
    signed char r1[256], r2[256], r3[256], r4[256], r5[256], r6[256], r7[256], r8[256];
    int i;
    NAFResult rs1, rs2, rs3, rs4, rh1, rh2, rh3, rh4;
    for (i = 0; i < 8; i++)
    {
        s1[i] = VEXTR32(s[i], 0); s2[i] = VEXTR32(s[i], 2);
        s3[i] = VEXTR32(s[i], 4); s4[i] = VEXTR32(s[i], 6);

        h1[i] = VEXTR32(h[i], 0); h2[i] = VEXTR32(h[i], 2);
        h3[i] = VEXTR32(h[i], 4); h4[i] = VEXTR32(h[i], 6);        
    }

    sc25519_slide(r1, s1, B_WINDOW); sc25519_slide(r2, s2, B_WINDOW); sc25519_slide(r3, s3, B_WINDOW); sc25519_slide(r4, s4, B_WINDOW);
    sc25519_slide(r5, h1, A_WINDOW); sc25519_slide(r6, h2, A_WINDOW); sc25519_slide(r7, h3, A_WINDOW); sc25519_slide(r8, h4, A_WINDOW);

    conv_char_to_NAF(&rs1, r1); conv_char_to_NAF(&rs2, r2); conv_char_to_NAF(&rs3, r3); conv_char_to_NAF(&rs4, r4);
    conv_char_to_NAF(&rh1, r5); conv_char_to_NAF(&rh2, r6); conv_char_to_NAF(&rh3, r7); conv_char_to_NAF(&rh4, r8); 


    int length_arr[8] = {rs1.length, rs2.length, rs3.length, rs4.length, rh1.length, rh2.length, rh3.length, rh4.length};

    int max = find_max(length_arr, 8);
    for (i = 0; i < 256; i++)
    {
        r->k0[i] = VSET64(rs4.k[i], rs3.k[i], rs2.k[i], rs1.k[i]);
        r->k1[i] = VSET64(rh4.k[i], rh3.k[i], rh2.k[i], rh1.k[i]);
    }

    r->max_length = max;
}

