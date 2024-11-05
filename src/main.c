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

#if USE_OPENCL
#include "opencl_matmult.h"
#include "opencl_tools.h"
#endif

#define INFO 1
#define DEBUG true

void testTrials();
void printUsage(char* exename);

const enum GenType GEN_TYPE = GEN_INCR;

// Default Dimensions
int M = 4096;
int K = 4096;
int N = 4096;

// use random sizes for testing
bool USE_RAND_SIZES = false;
const int RAND_MIN_SIZE = 128;
const int RAND_MAX_SIZE = 4096;

// validate with correct results
bool validate_results = false;

bool print_mat = false;
bool enable_log = false;
const int TRIALS = 1;

int main(int argc, char* argv[])
{
	if(argc == 2 && strcmp(argv[1], "--help")==0) {
		printUsage(argv[0]);
		return 0;
	} else if (argc == 2 && strcmp(argv[1], "--list-gpu")==0) {
		#ifdef USE_OPENCL
		displayPlatforms();
		#endif
		return 0;
	}
	
	init_opencl();
	testTrials();
}

void run_matmult(int M, int K, int N, float* a, float* b, float* c) {
	float* res_mat = NULL;
	time_t start, end, total_time;
		
	if(validate_results) {
		start = time(NULL);
		printf("running on CPU\n");
		// simple mat mult using CPU for validating results
		mult(M, K, N, a, b, c);
		end = time(NULL);
		// printf("time for mult (secs): %.3lf\n", (double)(end - start));
		res_mat = create(M, N, 0);
		copy_mat(M, N, c, M, N, res_mat, M, N);
		if(print_mat)
			print_matrix("mult c", res_mat, M, N);
		memset(c, 0, sizeof(float) * M*N);
	}
		
	// opencl no transpose with tiling (fast)
	openclMatMultBlock(M, K, N, a, b, c);
	if(print_mat) {
		print_matrix("opencl matmult tiling c", c, M, N);	
	}
	if(validate_results) {
		assert_mat_equal(M, N, c, res_mat);
	}
	memset(c, 0, sizeof(float) * M*N);
	
	// opencl col major a (transpose) with tiling (faster for small matrices)
	openclMatMultTilingColMajor(M, K, N, a, b, c);
	if(print_mat) {
		print_matrix("opencl matmult tiling col major c", c, M, N);	
	}
	if(validate_results) {
		assert_mat_equal(M, N, c, res_mat);
	}
	memset(c, 0, sizeof(float) * M*N);
	
	// opencl col major a with tiling (fastest) all matrices need to be padded
	// so dims can be multiple of M, N, K in expense of even more memory
	openclMatMultTilingColMajorPadded(M, K, N, a, b, c);
	if(print_mat) {
		print_matrix("opencl matmult tiling col major c", c, M, N);	
	}
	if(validate_results) {
		assert_mat_equal(M, N, c, res_mat);
	}
	memset(c, 0, sizeof(float) * M*N);
	
	if(validate_results) {
		free(res_mat);
	}
}

void testTrials() {
	srand(time(NULL));
	
	for(int i=0; i<TRIALS; i++) {
		if(enable_log) {
			FILE *f = freopen("out.txt", "w+", stdout); 
		}
		
		// randomize sizes
		if(USE_RAND_SIZES) {
			M = rand_int(RAND_MIN_SIZE, RAND_MAX_SIZE);
			N = rand_int(RAND_MIN_SIZE, RAND_MAX_SIZE);
			K = rand_int(RAND_MIN_SIZE, RAND_MAX_SIZE);
		}
		
		printf("\niteration: %d\n", i+1);
		printf("dimensions: M: %d, K: %d, N: %d\n", M, K, N);
		float* a = create(M, K, 0);
		float* b = create(K, N, 0);
		float* c = create(M, N, 0);
		
		gen(GEN_TYPE, a, M, K);
		gen(GEN_TYPE, b, K, N);
		if(print_mat) {
			print_matrix("a", a, M, K);
			print_matrix("b", b, K, N);
		}
		
		run_matmult(M, K, N, a, b, c);
		
		free(a);
		free(b);
		free(c);
	}
}

void printUsage(char* exename) {
	printf("%s [--help | --list-gpu]\r\n", exename);
	printf("--help: show help\r\n");
	printf("--list-gpu]: display gpu info\r\n");
}