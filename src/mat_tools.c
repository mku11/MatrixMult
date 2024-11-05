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

#include "mat_tools.h"

float* create(int sizeA, int sizeB, float val) {
	float* mat = (float*) malloc(sizeB * sizeA * sizeof(float));
	memset(mat, val, sizeA * sizeB * sizeof(float));
	return mat;
}

void transpose(int sizeA, int sizeB, float* mat, int sizeA2, int sizeB2, float* mat2) {
	// printf("tr: %dx%d => %dx%d\n", sizeA, sizeB, sizeA2, sizeB2);
	for(int i=0; i<sizeA; i++) {
		for(int j=0; j<sizeB; j++) {
			// printf("%d => %d\n", i*sizeB + j, j*sizeA2 + i);
			*(mat2 + j*sizeB2 + i) = *(mat + i*sizeB + j);
		}
	}
}

void copy_mat(int sizeA1, int sizeB1, float* mat1, int sizeA2, int sizeB2, float *mat2, int lengthA, int lengthB) {
	for(int i=0; i<lengthA; i++) {
		for(int j=0; j<lengthB; j++) {
			// printf("copy [%d] = %.2f => [%d] = %.2f\n", i*sizeB1 + j, *(mat1 + i*sizeB1 + j), i*sizeB2 + j, *(mat2 + i*sizeB2 + j));
			*(mat2 + i*sizeB2 + j) = *(mat1 + i*sizeB1 + j);
		}
	}
}

void assert_mat_equal(int sizeA, int sizeB, float* mat1, float* mat2) {
	for(int i=0; i<sizeA; i++) {
		for(int j=0; j<sizeB; j++) {
			if( *(mat1 + i*sizeB + j) != *(mat2 + i*sizeB + j) ) {
				printf("not equal: mat1[%d][%d](%.2f) != mat2[%d][%d](%.2f)\n", 
					i, j, *(mat1 + i*sizeB + j), i, j, *(mat2 + i*sizeB + j));
				exit(1);
			}
		}
	}
}

void print_matrix(const char* header, float* m, int rows, int cols) {	
	printf("%s %dx%d\r\n", header, rows, cols);
	for(int i=0; i<rows; i++) {
		if(PARTIAL_DISPLAY && i > MAX_DISPLAY_LEN) {
			break;
		} else if(PARTIAL_DISPLAY && i == MAX_DISPLAY_LEN) {
			printf("...\n");
			break;
		}
		
		for(int j=0; j<cols; j++) {
			if(PARTIAL_DISPLAY && j > MAX_DISPLAY_LEN) {
				break;
			} else if(PARTIAL_DISPLAY && j == MAX_DISPLAY_LEN) {
				printf("...");
				continue;
			}
			if(DISPLAY_INT)
				printf("%d ", (int) *(m + cols*i+j));
			else
				printf(FLOAT_FORMAT" ", *(m + cols*i+j));
		}
		printf("\n");
	}
	if(PARTIAL_DISPLAY) {
		// printf("last element [%d][%d] = "FLOAT_FORMAT"\r\n", rows-1, cols-1, *(m + cols*(rows-1)+(cols-1)));
	}
	printf("\n");
}


void validate_tiling(int BM, int BN, int BK, int WIM, int WIN, int max_local_size) {
	if( (int) (BM&(BM-1)) != 0 || (int) (BK&(BK-1)) != 0 || (int) (BN&(BN-1)) != 0 || (int) (WIM&(WIM-1)) != 0 || (int) (WIN&(WIN-1)) != 0) {
		printf("Error: BM,BN,BK,WIM,WIN should be powers of two");
		exit(1);
	}
	
	// printf("%d * %d = %d, %d * %d / (%d * %d) = %d\r\n", 
			// BM, BK, BM * BK, BM, BN, WIM, WIN, BM * BN / (WIM * WIN));
			
	if((BM * BK) % ((BM * BN) / (WIM * WIN)) != 0) {
		printf("Error: BM * BK should be a multiple of the number of work items per group: (BM * BN) / (WIM * WIN) (number of work items)\r\n");
		printf("%d * %d = %d, %d * %d / (%d * %d) = %d\r\n", 
			BM, BK, BM * BK, BM, BN, WIM, WIN, BM * BN / (WIM * WIN));
		exit(1);
	}
	
	if((BN * BK) % ((BM * BN) / (WIM * WIN)) != 0) {
		printf("Error: BN * BK should be a multiple of the number of work items per group: (BM * BN) / (WIM * WIN) (number of work items)\r\n");
		printf("%d * %d = %d, %d * %d / (%d * %d) = %d\r\n", 
			BN, BK, BN * BK, BM, BN, WIM, WIN, BM * BN / (WIM * WIN));
		exit(1);
	}
	
	if(BM/WIM > max_local_size) {
		printf("Error: BM / WIM should be less or equal to the max local size %d \r\n", max_local_size);
		printf("%d / %d = %d > %d\r\n", BM, WIM, BM/WIM, max_local_size);
		exit(1);
	}
	if(BN/WIN > max_local_size) {
		printf("Error: BN / WIN should be less or equal to the max local size %d \r\n", max_local_size);
		printf("%d / %d = %d > %d\r\n", BN, WIN, BM/WIN, max_local_size);
		exit(1);
	}
}

int rand_int(int min, int max) {
	if(min == max)
		return min;
	return min + rand() % (max + 1 - min);
}

void gen(GenType genType, float* mat, int M, int N) {
	int k =0;
	for(int i=0; i<M; i++) {
		for(int j=0; j<N; j++) {
			if(genType == GEN_RAND)
				*(mat + i*N + j) = rand() % 10;
			else if(genType == GEN_CONSTANT)
				*(mat + i*N + j) = j;
			else if(genType == GEN_INCR)
				*(mat + i*N + j) = k++;
		}
	}
}

time_t gettime() {
	struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
	return tp.tv_sec * 1e9 + tp.tv_nsec;
}