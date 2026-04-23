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

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "matmult.h"
#include "mat_tools.h"
#include "test_tools.h"

#include "opencl_matmult.h"
#include "opencl_tools.h"

#define INFO 1
#define DEBUG true

void testTrials();
void printUsage(char *exename);

const enum GenType GEN_TYPE = GEN_INCR;

// Default Dimensions
int M = 512*10;
int K = 512*10;
int N = 512*10;
// Or use random sizes
bool USE_RAND_SIZES = false;
const int RAND_MIN_SIZE = 128;
const int RAND_MAX_SIZE = 4096;
// trials
const int TRIALS = 1;

// validate with correct results
bool validate_results = false;
// bool validate_results = true;

bool use_simple_matmult = false;
// bool use_simple_matmult = true;

bool print_mat = false;
bool enable_log = false;


int main(int argc, char *argv[])
{
	if (argc == 2 && strcmp(argv[1], "--help") == 0)
	{
		printUsage(argv[0]);
		return 0;
	}
	else if (argc == 2 && strcmp(argv[1], "--list-gpu") == 0)
	{
		displayPlatforms();
		return 0;
	}

	init_opencl();
	testTrials();
}

void run_matmult(MatMultDims dims, float *a, float *b, float *c)
{
	float *res_mat = NULL;
	time_t start, end, total_time;

	if (validate_results)
	{
		start = time(NULL);
		printf("\nrunning cpu matmult\n");
		// simple mat mult using CPU for validating results
		mult(dims.m, dims.k, dims.n, a, b, c);
		end = time(NULL);
		// printf("time for mult (secs): %.3lf\n", (double)(end - start));
		res_mat = create(dims.m, dims.n, 0);
		copy_mat(dims.m, dims.n, c, dims.m, dims.n, res_mat, dims.m, dims.n);
		if (print_mat)
			print_matrix("mult c", res_mat, dims.m, dims.n);
		memset(c, 0, sizeof(float) * dims.m * dims.n);
	}

	// opencl simple mat mult
	if (use_simple_matmult)
	{
		printf("\nrunning simple opencl matmult\n");
		openclMatMult(dims, a, b, c, MatMultSimple);
		if (print_mat)
		{
			print_matrix("opencl matmult c", c, dims.m, dims.n);
		}
		if (validate_results)
		{
			assert_mat_equal(dims.m, dims.n, c, res_mat);
		}
		memset(c, 0, sizeof(float) * dims.m * dims.n);
	}

	// opencl no transpose with tiling (fast)
	printf("\nrunning opencl matmult w/ tiling\n");
	openclMatMult(dims, a, b, c, MatMultTiling);
	if (print_mat)
	{
		print_matrix("opencl matmult w/ tiling c", c, dims.m, dims.n);
	}
	if (validate_results)
	{
		assert_mat_equal(dims.m, dims.n, c, res_mat);
	}
	memset(c, 0, sizeof(float) * dims.m * dims.n);

	// opencl col major a (transpose) with tiling (faster for small matrices)
	printf("\nrunning opencl matmult w/ tiling and col major c\n");
	openclMatMult(dims, a, b, c, MatMultTilingColMaj);
	if (print_mat)
	{
		print_matrix("opencl matmult w/ tiling and col major c", c, dims.m, dims.n);
	}
	if (validate_results)
	{
		assert_mat_equal(dims.m, dims.n, c, res_mat);
	}
	memset(c, 0, sizeof(float) * dims.m * dims.n);

	// opencl col major a with tiling (fastest) all matrices need to be padded
	// so dims can be multiple of M, N, K in expense of even more memory
	printf("\nrunning opencl matmult w/ tiling and col major c and padded\n");
	openclMatMult(dims, a, b, c, MatMultTilingColMajPadded);
	if (print_mat)
	{
		print_matrix("opencl matmult tiling col major and padding c", c, dims.m, dims.n);
	}
	if (validate_results)
	{
		assert_mat_equal(dims.m, dims.n, c, res_mat);
	}
	memset(c, 0, sizeof(float) * dims.m * dims.n);

	if (validate_results)
	{
		free(res_mat);
	}
}

void testTrials()
{
	srand(time(NULL));

	for (int i = 0; i < TRIALS; i++)
	{
		if (enable_log)
		{
			FILE *f = freopen("out.txt", "w+", stdout);
		}

		MatMultDims dims;
		// randomize sizes
		if (USE_RAND_SIZES)
		{
			dims.m = rand_int(RAND_MIN_SIZE, RAND_MAX_SIZE);
			dims.n = rand_int(RAND_MIN_SIZE, RAND_MAX_SIZE);
			dims.k = rand_int(RAND_MIN_SIZE, RAND_MAX_SIZE);
		}
		else
		{
			dims.m = M;
			dims.n = N;
			dims.k = K;
		}

		printf("\niteration: %d\n", i + 1);
		printf("dimensions: M: %d, K: %d, N: %d\n", dims.m, dims.k, dims.n);
		float *a = create(dims.m, dims.k, 0);
		float *b = create(dims.k, dims.n, 0);
		float *c = create(dims.m, dims.n, 0);

		gen(GEN_TYPE, a, dims.m, dims.k);
		gen(GEN_TYPE, b, dims.k, dims.n);
		if (print_mat)
		{
			print_matrix("a", a, dims.m, dims.k);
			print_matrix("b", b, dims.k, dims.n);
		}

		run_matmult(dims, a, b, c);

		free(a);
		free(b);
		free(c);
	}
}

void printUsage(char *exename)
{
	printf("%s [--help | --list-gpu]\r\n", exename);
	printf("--help: show help\r\n");
	printf("--list-gpu]: display gpu info\r\n");
}