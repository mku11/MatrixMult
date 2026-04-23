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

#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#include "mat_tools.h"

float *create(int sizeA, int sizeB, float val)
{
	float *mat = (float *)malloc(sizeB * sizeA * sizeof(float));
	memset(mat, val, sizeA * sizeB * sizeof(float));
	return mat;
}

void transpose(MatTransposeDims dims, float *mat, float *mat2)
{
	// printf("tr: %dx%d => %dx%d\n", dims.m, dims.n, dims.tm, dims.tn);
	for (int i = 0; i < dims.m; i++)
	{
		for (int j = 0; j < dims.n; j++)
		{
			// printf("%d => %d\n", i*sizeB + j, j*sizeA2 + i);
			*(mat2 + j * dims.tn + i) = *(mat + i * dims.tm + j);
		}
	}
}

void copy_mat(int sizeA1, int sizeB1, float *mat1, int sizeA2, int sizeB2, float *mat2, int lengthA, int lengthB)
{
	for (int i = 0; i < lengthA; i++)
	{
		for (int j = 0; j < lengthB; j++)
		{
			// printf("copy [%d] = %.2f => [%d] = %.2f\n", i*sizeB1 + j, *(mat1 + i*sizeB1 + j), i*sizeB2 + j, *(mat2 + i*sizeB2 + j));
			*(mat2 + i * sizeB2 + j) = *(mat1 + i * sizeB1 + j);
		}
	}
}

time_t gettime()
{
	struct timespec tp;
	timespec_get(&tp, TIME_UTC);
	return tp.tv_sec * 1e9 + tp.tv_nsec;
}

void assert_mat_equal(int sizeA, int sizeB, float *mat1, float *mat2)
{
	for (int i = 0; i < sizeA; i++)
	{
		for (int j = 0; j < sizeB; j++)
		{
			assert(*(mat1 + i * sizeB + j) == *(mat2 + i * sizeB + j) && "not equal");
		}
	}
}

void print_matrix(const char *header, float *m, int rows, int cols)
{
	printf("%s %dx%d\r\n", header, rows, cols);
	for (int i = 0; i < rows; i++)
	{
		if (PARTIAL_DISPLAY && i > MAX_DISPLAY_LEN)
		{
			break;
		}
		else if (PARTIAL_DISPLAY && i == MAX_DISPLAY_LEN)
		{
			printf("...\n");
			break;
		}

		for (int j = 0; j < cols; j++)
		{
			if (PARTIAL_DISPLAY && j > MAX_DISPLAY_LEN)
			{
				break;
			}
			else if (PARTIAL_DISPLAY && j == MAX_DISPLAY_LEN)
			{
				printf("...");
				continue;
			}
			if (DISPLAY_INT)
				printf("%d ", (int)*(m + cols * i + j));
			else
				printf(FLOAT_FORMAT " ", *(m + cols * i + j));
		}
		printf("\n");
	}
	if (PARTIAL_DISPLAY)
	{
		// printf("last element [%d][%d] = "FLOAT_FORMAT"\r\n", rows-1, cols-1, *(m + cols*(rows-1)+(cols-1)));
	}
	printf("\n");
}

bool is_power_two(int val)
{
	return (int)(val & (val - 1)) == 0;
}

void validate_tiling(TileParams tile_params, int max_local_size)
{
	if (!is_power_two(tile_params.BM) || !is_power_two(tile_params.BK) || !is_power_two(tile_params.BN) || !is_power_two(tile_params.WIM) || !is_power_two(tile_params.WIN))
	{
		printf("Error: BM,BN,BK,WIM,WIN should be powers of two");
		exit(1);
	}

	int BMBK = tile_params.BM * tile_params.BK;											  // tile size for first dim
	int BNBK = tile_params.BN * tile_params.BK;											  // tile size for second dim
	int WIPG = ((tile_params.BM * tile_params.BN) / (tile_params.WIM * tile_params.WIN)); // work items per workgroup
	int BMWIM = tile_params.BM / tile_params.WIM;
	int BNWIN = tile_params.BN / tile_params.WIN;

	if (BMBK % WIPG != 0)
	{
		printf("Error: BM * BK should be a multiple of the number of work items per group: (BM * BN) / (WIM * WIN) (number of work items)\r\n");
		printf("BM * BK = %d * %d = %d\n", tile_params.BM, tile_params.BK, BMBK);
		printf("(BM * BN) / (WIM * WIN) = %d * %d / (%d * %d) = %d\n",
			   tile_params.BM, tile_params.BN, tile_params.WIM, tile_params.WIN, WIPG);
		exit(1);
	}

	if (BNBK % WIPG != 0)
	{
		printf("Error: BN * BK should be a multiple of the number of work items per group: (BM * BN) / (WIM * WIN) (number of work items)\r\n");
		printf("%d * %d = %d, %d * %d / (%d * %d) = %d\r\n",
			   tile_params.BN, tile_params.BK, tile_params.BN * tile_params.BK, tile_params.BM, tile_params.BN, tile_params.WIM, tile_params.WIN, WIPG);
		exit(1);
	}

	if (BMWIM > max_local_size)
	{
		printf("Error: BM / WIM should be less or equal to the max local size %d \r\n", max_local_size);
		printf("%d / %d = %d > %d\r\n", tile_params.BM, tile_params.WIM, BMWIM, max_local_size);
		exit(1);
	}
	if (BNWIN > max_local_size)
	{
		printf("Error: BN / WIN should be less or equal to the max local size %d \r\n", max_local_size);
		printf("%d / %d = %d > %d\r\n", tile_params.BN, tile_params.WIN, BNWIN, max_local_size);
		exit(1);
	}
}

void set_default_tiling_params(TileParams *tile_params)
{
	// tiling params
	// note: these values are per thread
	tile_params->BM = 128; // block size for dimension M
	tile_params->BN = 128; // block size for dimension N
	tile_params->BK = 16;  // block size for dimension K
	tile_params->WIM = 8;  // work items/elements for dimension M
	tile_params->WIN = 8;  // work items/elements for dimension N
}

void set_pref_tiling_params(MatMultDims dims, long max_local_size, TileParams *tile_params)
{
	// for now these preferred values seems to yield fast results
	if (dims.m <= 16 && dims.n <= 16 && dims.k <= 16)
	{
		tile_params->BM = 8;
		tile_params->BN = 8;
		tile_params->BK = 8;
		tile_params->WIM = 4;
		tile_params->WIN = 4;
	}
	else if (dims.m <= 256 && dims.n <= 256 && dims.k <= 128)
	{
		tile_params->BM = 16;
		tile_params->BN = 16;
		tile_params->BK = 8;
		tile_params->WIM = 4;
		tile_params->WIN = 4;
	}
	else if (dims.m <= 512 && dims.n <= 512 && dims.k <= 512)
	{
		tile_params->BM = 64;
		tile_params->BN = 64;
		tile_params->BK = 8;
		tile_params->WIM = 4;
		tile_params->WIN = 4;
	}
	else
	{
		tile_params->BM = 128;
		tile_params->BN = 128;
		tile_params->BK = 16;
		tile_params->WIM = 8;
		tile_params->WIN = 8;
	}
	// also these may be fast for large matrices with DTYPE=half for high end devices
	// 256, 256, 4, 16, 16
	// 256, 256, 8, 16, 16

	printf("setting preferred tiling params\n");
}