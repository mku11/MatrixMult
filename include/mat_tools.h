/*
MIT License

Copyright (c) 2024 Max Kas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef __MAT_TOOLS_H
#define __MAT_TOOLS_H
#include <time.h>

#include <stdbool.h>

#define FLOAT_FORMAT "%.2f"
#define PARTIAL_DISPLAY true
#define DISPLAY_INT false
#define MAX_DISPLAY_LEN 8

typedef struct TileParams
{
    int BM;
    int BN;
    int BK;
    int WIM;
    int WIN;
} TileParams;

typedef struct MatMultDims
{
    int m;
    int k;
    int n;
} MatMultDims;

typedef struct MatTransposeDims
{
    int m;
    int n;
    int tm; // transposed m
    int tn; // transposed n
} MatTransposeDims;

float *create(int sizeA, int sizeB, float val);
void transpose(MatTransposeDims dims, float *mat, float *mat2);
void copy_mat(int sizeA1, int sizeB1, float *mat1, int sizeA2, int sizeB2, float *mat2, int lengthA, int lengthB);
void assert_mat_equal(int sizeA, int sizeB, float *mat1, float *mat2);
void print_matrix(const char *header, float *m, int rows, int cols);
void validate_tiling(TileParams tile_params, int max_local_size);
void set_default_tiling_params(TileParams *tile_params);
void set_pref_tiling_params(MatMultDims dims, long max_local_size, TileParams *tile_params);
time_t gettime();
#endif // __MAT_TOOLS_H