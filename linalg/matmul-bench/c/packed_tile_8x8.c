

#include <stddef.h>

void c_packed_tile_8x8(size_t m, size_t k, size_t n, float *a, float *b, float *c) {
    for(size_t row = 0 ; row < m / 8 ; row++) {
        for(size_t col = 0 ; col < n / 8 ; col++) {
            float sum00 = 0.0;
            float sum01 = 0.0;
            float sum02 = 0.0;
            float sum03 = 0.0;
            float sum04 = 0.0;
            float sum05 = 0.0;
            float sum06 = 0.0;
            float sum07 = 0.0;
            float sum10 = 0.0;
            float sum11 = 0.0;
            float sum12 = 0.0;
            float sum13 = 0.0;
            float sum14 = 0.0;
            float sum15 = 0.0;
            float sum16 = 0.0;
            float sum17 = 0.0;
            float sum20 = 0.0;
            float sum21 = 0.0;
            float sum22 = 0.0;
            float sum23 = 0.0;
            float sum24 = 0.0;
            float sum25 = 0.0;
            float sum26 = 0.0;
            float sum27 = 0.0;
            float sum30 = 0.0;
            float sum31 = 0.0;
            float sum32 = 0.0;
            float sum33 = 0.0;
            float sum34 = 0.0;
            float sum35 = 0.0;
            float sum36 = 0.0;
            float sum37 = 0.0;
            float sum40 = 0.0;
            float sum41 = 0.0;
            float sum42 = 0.0;
            float sum43 = 0.0;
            float sum44 = 0.0;
            float sum45 = 0.0;
            float sum46 = 0.0;
            float sum47 = 0.0;
            float sum50 = 0.0;
            float sum51 = 0.0;
            float sum52 = 0.0;
            float sum53 = 0.0;
            float sum54 = 0.0;
            float sum55 = 0.0;
            float sum56 = 0.0;
            float sum57 = 0.0;
            float sum60 = 0.0;
            float sum61 = 0.0;
            float sum62 = 0.0;
            float sum63 = 0.0;
            float sum64 = 0.0;
            float sum65 = 0.0;
            float sum66 = 0.0;
            float sum67 = 0.0;
            float sum70 = 0.0;
            float sum71 = 0.0;
            float sum72 = 0.0;
            float sum73 = 0.0;
            float sum74 = 0.0;
            float sum75 = 0.0;
            float sum76 = 0.0;
            float sum77 = 0.0;
            float *pa = a + row * k * 8;
            float *pb = b + col * k * 8;
            for(size_t i = 0 ; i < k ; i++) {
                float a0 = a[0];
                float a1 = a[1];
                float a2 = a[2];
                float a3 = a[3];
                float a4 = a[4];
                float a5 = a[5];
                float a6 = a[6];
                float a7 = a[7];
                float b0 = b[0];
                float b1 = b[1];
                float b2 = b[2];
                float b3 = b[3];
                float b4 = b[4];
                float b5 = b[5];
                float b6 = b[6];
                float b7 = b[7];
                pa += 8;
                pb += 8;
                sum00 += a0 * b0;
                sum01 += a0 * b1;
                sum02 += a0 * b2;
                sum03 += a0 * b3;
                sum04 += a0 * b4;
                sum05 += a0 * b5;
                sum06 += a0 * b6;
                sum07 += a0 * b7;
                sum10 += a1 * b0;
                sum11 += a1 * b1;
                sum12 += a1 * b2;
                sum13 += a1 * b3;
                sum14 += a1 * b4;
                sum15 += a1 * b5;
                sum16 += a1 * b6;
                sum17 += a1 * b7;
                sum20 += a2 * b0;
                sum21 += a2 * b1;
                sum22 += a2 * b2;
                sum23 += a2 * b3;
                sum24 += a2 * b4;
                sum25 += a2 * b5;
                sum26 += a2 * b6;
                sum27 += a2 * b7;
                sum30 += a3 * b0;
                sum31 += a3 * b1;
                sum32 += a3 * b2;
                sum33 += a3 * b3;
                sum34 += a3 * b4;
                sum35 += a3 * b5;
                sum36 += a3 * b6;
                sum37 += a3 * b7;
                sum40 += a4 * b0;
                sum41 += a4 * b1;
                sum42 += a4 * b2;
                sum43 += a4 * b3;
                sum44 += a4 * b4;
                sum45 += a4 * b5;
                sum46 += a4 * b6;
                sum47 += a4 * b7;
                sum50 += a5 * b0;
                sum51 += a5 * b1;
                sum52 += a5 * b2;
                sum53 += a5 * b3;
                sum54 += a5 * b4;
                sum55 += a5 * b5;
                sum56 += a5 * b6;
                sum57 += a5 * b7;
                sum60 += a6 * b0;
                sum61 += a6 * b1;
                sum62 += a6 * b2;
                sum63 += a6 * b3;
                sum64 += a6 * b4;
                sum65 += a6 * b5;
                sum66 += a6 * b6;
                sum67 += a6 * b7;
                sum70 += a7 * b0;
                sum71 += a7 * b1;
                sum72 += a7 * b2;
                sum73 += a7 * b3;
                sum74 += a7 * b4;
                sum75 += a7 * b5;
                sum76 += a7 * b6;
                sum77 += a7 * b7;
            }
            c[(row * 8 + 0) * n + col * 8] = sum00;
            c[(row * 8 + 0) * n + col * 8 + 1] = sum01;
            c[(row * 8 + 0) * n + col * 8 + 2] = sum02;
            c[(row * 8 + 0) * n + col * 8 + 3] = sum03;
            c[(row * 8 + 0) * n + col * 8 + 4] = sum04;
            c[(row * 8 + 0) * n + col * 8 + 5] = sum05;
            c[(row * 8 + 0) * n + col * 8 + 6] = sum06;
            c[(row * 8 + 0) * n + col * 8 + 7] = sum07;
            c[(row * 8 + 1) * n + col * 8] = sum10;
            c[(row * 8 + 1) * n + col * 8 + 1] = sum11;
            c[(row * 8 + 1) * n + col * 8 + 2] = sum12;
            c[(row * 8 + 1) * n + col * 8 + 3] = sum13;
            c[(row * 8 + 1) * n + col * 8 + 4] = sum14;
            c[(row * 8 + 1) * n + col * 8 + 5] = sum15;
            c[(row * 8 + 1) * n + col * 8 + 6] = sum16;
            c[(row * 8 + 1) * n + col * 8 + 7] = sum17;
            c[(row * 8 + 2) * n + col * 8] = sum20;
            c[(row * 8 + 2) * n + col * 8 + 1] = sum21;
            c[(row * 8 + 2) * n + col * 8 + 2] = sum22;
            c[(row * 8 + 2) * n + col * 8 + 3] = sum23;
            c[(row * 8 + 2) * n + col * 8 + 4] = sum24;
            c[(row * 8 + 2) * n + col * 8 + 5] = sum25;
            c[(row * 8 + 2) * n + col * 8 + 6] = sum26;
            c[(row * 8 + 2) * n + col * 8 + 7] = sum27;
            c[(row * 8 + 3) * n + col * 8] = sum30;
            c[(row * 8 + 3) * n + col * 8 + 1] = sum31;
            c[(row * 8 + 3) * n + col * 8 + 2] = sum32;
            c[(row * 8 + 3) * n + col * 8 + 3] = sum33;
            c[(row * 8 + 3) * n + col * 8 + 4] = sum34;
            c[(row * 8 + 3) * n + col * 8 + 5] = sum35;
            c[(row * 8 + 3) * n + col * 8 + 6] = sum36;
            c[(row * 8 + 3) * n + col * 8 + 7] = sum37;
            c[(row * 8 + 4) * n + col * 8] = sum40;
            c[(row * 8 + 4) * n + col * 8 + 1] = sum41;
            c[(row * 8 + 4) * n + col * 8 + 2] = sum42;
            c[(row * 8 + 4) * n + col * 8 + 3] = sum43;
            c[(row * 8 + 4) * n + col * 8 + 4] = sum44;
            c[(row * 8 + 4) * n + col * 8 + 5] = sum45;
            c[(row * 8 + 4) * n + col * 8 + 6] = sum46;
            c[(row * 8 + 4) * n + col * 8 + 7] = sum47;
            c[(row * 8 + 5) * n + col * 8] = sum50;
            c[(row * 8 + 5) * n + col * 8 + 1] = sum51;
            c[(row * 8 + 5) * n + col * 8 + 2] = sum52;
            c[(row * 8 + 5) * n + col * 8 + 3] = sum53;
            c[(row * 8 + 5) * n + col * 8 + 4] = sum54;
            c[(row * 8 + 5) * n + col * 8 + 5] = sum55;
            c[(row * 8 + 5) * n + col * 8 + 6] = sum56;
            c[(row * 8 + 5) * n + col * 8 + 7] = sum57;
            c[(row * 8 + 6) * n + col * 8] = sum60;
            c[(row * 8 + 6) * n + col * 8 + 1] = sum61;
            c[(row * 8 + 6) * n + col * 8 + 2] = sum62;
            c[(row * 8 + 6) * n + col * 8 + 3] = sum63;
            c[(row * 8 + 6) * n + col * 8 + 4] = sum64;
            c[(row * 8 + 6) * n + col * 8 + 5] = sum65;
            c[(row * 8 + 6) * n + col * 8 + 6] = sum66;
            c[(row * 8 + 6) * n + col * 8 + 7] = sum67;
            c[(row * 8 + 7) * n + col * 8] = sum70;
            c[(row * 8 + 7) * n + col * 8 + 1] = sum71;
            c[(row * 8 + 7) * n + col * 8 + 2] = sum72;
            c[(row * 8 + 7) * n + col * 8 + 3] = sum73;
            c[(row * 8 + 7) * n + col * 8 + 4] = sum74;
            c[(row * 8 + 7) * n + col * 8 + 5] = sum75;
            c[(row * 8 + 7) * n + col * 8 + 6] = sum76;
            c[(row * 8 + 7) * n + col * 8 + 7] = sum77;
        }
    }
}
