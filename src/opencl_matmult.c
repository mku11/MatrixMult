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
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include <CL/opencl.h>
#include "opencl_matmult.h"
#include "opencl_tools.h"
#include "mat_tools.h"

#define KERNEL_DIR "../kernels/"

int cl_mult(char *kernel_file, char *kernel_name,
			MatMultDims dims, float *a, float *b, float *c, cl_mem d_at,
			bool use_tiling, TileParams *tile_params);
int cl_transpose(char *kernel_file, char *kernel_name,
				 MatTransposeDims dims,
				 float *a, cl_mem d_at);
void printBuildError(cl_device_id device_id, cl_program program);

cl_platform_id cpPlatform;	   // OpenCL platform
cl_device_id device_id = NULL; // device ID
char device_name[MAX_CHARS];
int num_devices;
cl_context context;		// context
cl_command_queue queue; // command queue
cl_device_id device_ids[MAX_DEVICES];
extern cl_platform_id platforms[MAX_PLATFORMS];
long max_shared_mem;
long max_shared_mem_per_dim;
int default_local_size = 16;

int platform_index = 0;
int currentDevice = 0;
bool use_cl_transpose = true;
bool validate_transpose_results = false;
bool print_temp_mat = false;

// bool use_optimal_params = true;
bool use_optimal_params = false;
bool use_optimal_local_size = false;
bool validate_params = true;

void openclMatMultSimple(MatMultDims dims, float *a, float *b, float *c)
{
	time_t start, end;

	start = gettime();

	cl_mult(KERNEL_DIR "kernel_matmult.cl", "matmult_simple",
			dims,
			a, b, c,
			NULL,
			false, NULL);
	end = gettime();
	unsigned long long FLOPs = (long long)dims.m * (long long)dims.n * (long long)(2 * dims.k - 1);
	double dtime = difftime(end, start) / 1e9;
	printf("total estimated FLOPs: %llu\n", FLOPs);
	printf("total extra mem used: %ld\n", 0L);
	printf("time for openclMatMult (secs): %.3lf, total GFLOPS: %.2lf\n", dtime,
		   FLOPs * 1e-9 / dtime);
}

void openclMatMultBlock(MatMultDims dims, float *a, float *b, float *c)
{
	time_t start, end;

	start = gettime();

	TileParams tile_params;
	if (use_optimal_local_size) // we don't have a kernel to get the size so we use the default local size
		set_pref_tiling_params(dims, default_local_size, &tile_params);
	else
		set_default_tiling_params(&tile_params);

	cl_mult(KERNEL_DIR "kernel_matmult_tiling.cl", "matmult_block",
			dims,
			a, b, c,
			NULL,
			true,
			&tile_params);

	end = gettime();
	unsigned long long FLOPs = (long long)dims.m * (long long)dims.n * (long long)(2 * dims.k - 1);
	double dtime = difftime(end, start) / 1e9;
	printf("total estimated FLOPs: %llu\n", FLOPs);
	printf("total extra mem used: %ld\n", 0L);
	printf("total time for openclMatMultTiling (secs): %.3lf, total GFLOPS: %.2lf\n", dtime,
		   FLOPs * 1e-9 / dtime);
}

void openclMatMultTilingColMajor(MatMultDims dims, float *a, float *b, float *c)
{
	time_t start, end;

	start = gettime();
	time_t begint = gettime();
	float *at = create(dims.k, dims.m, 0);
	cl_mem d_at = NULL;
	MatTransposeDims transpose_dims;
	transpose_dims.m = dims.m;
	transpose_dims.n = dims.k;
	transpose_dims.tm = dims.k;
	transpose_dims.tn = dims.m;
	int transpose_size = dims.k * dims.m * sizeof(*at);
	if (use_cl_transpose)
	{
		d_at = clCreateBuffer(context, CL_MEM_WRITE_ONLY, transpose_size, NULL, NULL);
		cl_transpose(KERNEL_DIR "kernel_transpose.cl", "transpose",
					 transpose_dims, a, d_at);

		if (validate_transpose_results || print_temp_mat)
		{
			// Wait for the command queue to get serviced before reading back results
			clFinish(queue);
			// Read the results from the device
			cl_event event;
			clEnqueueReadBuffer(queue, d_at, CL_TRUE, 0, transpose_size, at, 0, NULL, &event);
			clWaitForEvents(1, &event);
			clFinish(queue);

			float *at_res = create(dims.k, dims.m, 0);
			transpose(transpose_dims, a, at_res);
			assert_mat_equal(dims.k, dims.m, at, at_res);
		}
	}
	else
	{
		transpose(transpose_dims, a, at);
	}
	time_t endt = gettime();
	double difft = difftime(endt, begint) / 1e9;
	if (print_temp_mat)
	{
		print_matrix("AT", at, dims.k, dims.m);
	}

	TileParams tile_params;
	if (use_optimal_local_size) // we don't have a kernel to get the size so we use the default local size
		set_pref_tiling_params(dims, default_local_size, &tile_params);
	else
		set_default_tiling_params(&tile_params);

	cl_mult(KERNEL_DIR "kernel_matmult_tiling_colmajor.cl", "matmult_block_colmajor",
			dims,
			at, b, c, d_at,
			true, &tile_params);
	free(at);

	end = gettime();
	unsigned long long FLOPs = (long long)dims.m * (long long)dims.n * (long long)(2 * dims.k - 1);
	double dtime = difftime(end, start) / 1e9;
	printf("total estimated FLOPs: %llu\n", FLOPs);
	printf("total time to transpose (secs): %.3lf\n", difft);
	printf("total extra mem used: %lld\n", dims.k * dims.m * sizeof(*a));
	printf("total time for openclMatMultTilingColMajor (secs): %.3lf, total GFLOPS: %.2lf\n", dtime,
		   FLOPs * 1e-9 / dtime);
}

void openclMatMultTilingColMajorPadded(MatMultDims dims, float *a, float *b, float *c)
{
	time_t start, end;

	start = gettime();

	time_t begint = gettime();
	TileParams tile_params;
	if (use_optimal_local_size) // we don't have a kernel to get the size so we use the default local size
		set_pref_tiling_params(dims, default_local_size, &tile_params);
	else
		set_default_tiling_params(&tile_params);

	int paddedm = ceil(dims.m / (float)tile_params.BM) * tile_params.BM;
	int paddedk = ceil(dims.k / (float)tile_params.BK) * tile_params.BK;
	int paddedn = ceil(dims.n / (float)tile_params.BN) * tile_params.BN;

	float *aTpadded = create(paddedk, paddedm, 0);
	cl_mem d_at = NULL;
	int padded_size = dims.k * dims.m * sizeof(*aTpadded);
	MatTransposeDims transpose_dims;
	transpose_dims.m = dims.m;
	transpose_dims.n = dims.k;
	transpose_dims.tm = paddedk;
	transpose_dims.tn = paddedm;
	if (use_cl_transpose)
	{
		d_at = clCreateBuffer(context, CL_MEM_WRITE_ONLY, padded_size, NULL, NULL);
		cl_transpose(KERNEL_DIR "kernel_transpose.cl", "transpose",
					 transpose_dims,
					 a, d_at);
		if (validate_transpose_results)
		{
			// Wait for the command queue to get serviced before reading back results
			clFinish(queue);
			// Read the results from the device
			cl_event event;
			int read_size = dims.k * dims.m * sizeof(*aTpadded);
			clEnqueueReadBuffer(queue, d_at, CL_TRUE, 0, read_size, aTpadded, 0, NULL, &event);
			clWaitForEvents(1, &event);
			clFinish(queue);

			float *at_res = create(paddedk, paddedm, 0);
			transpose(transpose_dims, a, at_res);
			assert_mat_equal(paddedk, paddedm, aTpadded, at_res);
		}
	}
	else
	{
		transpose(transpose_dims, a, aTpadded);
	}

	float *bpadded = NULL;
	if (paddedk != dims.k || paddedn != dims.n)
	{
		bpadded = create(paddedk, paddedn, 0);
		copy_mat(dims.k, dims.n, b, paddedk, paddedn, bpadded, dims.k, dims.n);
	}
	float *cpadded = NULL;
	if (paddedm != dims.m || paddedn != dims.n)
	{
		cpadded = create(paddedm, paddedn, 0);
	}
	time_t endt = gettime();
	double difft = difftime(endt, begint) / 1e9;
	if (print_temp_mat)
	{
		print_matrix("ATpadded", aTpadded, paddedk, paddedm);
		print_matrix("Bpadded", bpadded, paddedk, paddedn);
	}

	MatMultDims padded_dims;
	padded_dims.m = paddedm;
	padded_dims.k = paddedk;
	padded_dims.n = paddedn;
	cl_mult(KERNEL_DIR "kernel_matmult_tiling_colmajor_padded.cl", "matmult_block_colmajor_padded",
			padded_dims,
			aTpadded,
			bpadded ? bpadded : b,
			cpadded ? cpadded : c,
			d_at,
			true, &tile_params);

	if (cpadded != NULL)
	{
		if (print_temp_mat)
		{
			print_matrix("Cpadded", cpadded, paddedm, paddedn);
		}
		copy_mat(paddedm, paddedn, cpadded, dims.m, dims.n, c, dims.m, dims.n);
	}
	free(aTpadded);
	if (bpadded != NULL)
		free(bpadded);
	if (cpadded != NULL)
		free(cpadded);

	end = gettime();

	unsigned long long FLOPs = (long long)dims.m * (long long)dims.n * (long long)(2 * dims.k - 1);
	double dtime = difftime(end, start) / 1e9;
	printf("total estimated FLOPs: %llu\n", FLOPs);
	printf("total time to transpose and pad (secs): %.3lf\n", difft);
	printf("total extra mem used: %lld\n", (paddedk * paddedm + paddedk * paddedn + paddedm * paddedn) * sizeof(*a));
	printf("total time for openclMatMultTilingColMajorPadded (secs): %.3lf, total GFLOPS: %.2lf\n", dtime,
		   FLOPs * 1e-9 / dtime);
}

void openclMatMult(MatMultDims dims, float *a, float *b, float *c, int mult_type)
{
	switch (mult_type)
	{
	case MatMultSimple:
		openclMatMultSimple(dims, a, b, c);
		break;
	case MatMultTiling:
		openclMatMultBlock(dims, a, b, c);
		break;
	case MatMultTilingColMaj:
		openclMatMultTilingColMajor(dims, a, b, c);
		break;
	case MatMultTilingColMajPadded:
		openclMatMultTilingColMajorPadded(dims, a, b, c);
		break;
	}
}

// d_at is the transpose buffer if we have transposed the matrix a, otherwise we will use the buffer a
int cl_mult(char *kernel_file, char *kernel_name,
			MatMultDims dims, float *a, float *b, float *c, cl_mem d_at,
			bool use_tiling, TileParams *tile_params)
{

	// Device input buffers
	cl_mem d_a;
	cl_mem d_b;
	// Device output buffer
	cl_mem d_c;

	cl_program program; // program
	cl_kernel kernel;	// kernel

	cl_int err;
	size_t local[2], global[2];

	FILE *cl_code = fopen(kernel_file, "rb");
	if (cl_code == NULL)
	{
		printf("Could not open mult kernel file: %s\n", kernel_file);
		exit(1);
	}
	char *source_str = (char *)malloc(MAX_SOURCE_SIZE + 1);
	memset(source_str, 0, MAX_SOURCE_SIZE + 1);
	int res = fread(source_str, 1, MAX_SOURCE_SIZE, cl_code);
	fclose(cl_code);

	int local_size = default_local_size;
	if (use_optimal_local_size)
	{
		// TODO: this has overhead which is more noticable with small matrixes
		local_size = get_kernel_max_local_size(context, source_str, kernel_name, device_id, *tile_params);
	}

	if (use_tiling)
	{
		if (use_optimal_params)
		{
			set_pref_tiling_params(dims, local_size, tile_params);
		}
		add_kernel_defines(source_str, *tile_params);
	}
	// printf("mult kernel\r\n%s:", source_str);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Could not create mult program, code: %d\n", err);
		exit(1);
	}

	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Could not build mult program, code: %d\n", err);
		if (err == CL_BUILD_PROGRAM_FAILURE)
		{
			printBuildError(device_id, program);
		}
		exit(1);
	}

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, kernel_name, &err);
	if (err != CL_SUCCESS)
	{
		printf("Could not create mult kernel: %s, code: %d\n", kernel_name, err);
		exit(1);
	}

	int max_local_size = getMaxLocalSize(kernel, device_id, 2);
	printf("max_local_size: %d\n", max_local_size);
	int maxWorkGroupSize = getWorkgroupSize(kernel, device_id);
	printf("max workgroup size: %d\n", maxWorkGroupSize);
	int maxWorkGroupSizePerDim = (long)pow(maxWorkGroupSize, 1.0f / 2);
	printf("max workgroup size per dim: %d\n", maxWorkGroupSizePerDim);

	if (use_tiling && validate_params)
	{
		validate_tiling(*tile_params, local_size);
	}

	if (use_tiling)
	{
		local[0] = tile_params->BM / tile_params->WIM;
		local[1] = tile_params->BN / tile_params->WIN;
		global[0] = (size_t)(ceil(dims.m / (float)tile_params->BM) * tile_params->BM / tile_params->WIM);
		global[1] = (size_t)(ceil(dims.n / (float)tile_params->BN) * tile_params->BN / tile_params->WIN);
	}
	else
	{
		local[0] = local_size;
		local[1] = local_size;
		global[0] = dims.m;
		global[1] = dims.n;
	}

	// use the transpose if we have one
	printf("creating buffers\n");
	if (d_at)
		d_a = d_at;
	else
		d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, dims.m * dims.k * sizeof(*a), NULL, NULL);
	d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, dims.k * dims.n * sizeof(*b), NULL, NULL);
	d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dims.m * dims.n * sizeof(*c), NULL, NULL);

	printf("writing buffers\n");
	cl_event wevent, kevent, revent;
	cl_ulong time_start = 0;
	cl_ulong time_end = 0;
	double time_passed_write;
	// Write our data set into the input array in device memory
	if (!d_at)
		err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, dims.m * dims.k * sizeof(*a), a, 0, NULL, &wevent);
	err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, dims.k * dims.n * sizeof(*b), b, 0, NULL, &wevent);
	if (err != CL_SUCCESS)
	{
		printf("Could not enqueue mult buffers, code: %d\n", err);
		exit(1);
	}
	clWaitForEvents(1, &wevent);
	err = clGetEventProfilingInfo(wevent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	err |= clGetEventProfilingInfo(wevent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	err |= clReleaseEvent(wevent);
	time_passed_write = (time_end - time_start) / (double)1e9;
	printf("mult write time (sec): %f\n", time_passed_write);

	printf("setting kernel args\n");
	// Set the arguments to our compute kernel
	int param = 0;
	err = clSetKernelArg(kernel, param++, sizeof(int), (void *)&dims.m);
	err |= clSetKernelArg(kernel, param++, sizeof(int), (void *)&dims.k);
	err |= clSetKernelArg(kernel, param++, sizeof(int), (void *)&dims.n);
	err |= clSetKernelArg(kernel, param++, sizeof(cl_mem), (void *)&d_a);
	err |= clSetKernelArg(kernel, param++, sizeof(cl_mem), (void *)&d_b);
	err |= clSetKernelArg(kernel, param++, sizeof(cl_mem), (void *)&d_c);
	if (err != CL_SUCCESS)
	{
		printf("Could not set mult kernel args, code: %d\n", err);
		exit(1);
	}
	if (use_tiling)
	{
		printf("Block size for dim M, BM=%d\n", tile_params->BM);
		printf("Block size for dim N, BN=%d\n", tile_params->BN);
		printf("Block size for dim K, BK=%d\n", tile_params->BK);
		printf("Work items for dim M, WIM=%d\n", tile_params->WIM);
		printf("Work items for dim N, WIN=%d\n", tile_params->WIN);
	}

	printf("local_size: %lld:%lld, global_size: %lld:%lld\r\n", local[0], local[1], global[0], global[1]);
	printf("total workgroups to submit: %lld * %lld = %lld\n", global[0] / local[0], global[1] / local[1], global[0] / local[0] * global[1] / local[1]);

	printf("exec kernel\n");
	// printf("exec kernel: %s\r\n", kernel_name);
	fflush(stdout);
	// Execute the kernel over the entire range of the data set
	double time_passed_kernel;
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &kevent);
	if (err != CL_SUCCESS)
	{
		printf("Could not exec mult kernel, code: %d\n", err);
		exit(1);
	}
	clWaitForEvents(1, &kevent);
	clFinish(queue);
	err = clGetEventProfilingInfo(kevent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	err |= clGetEventProfilingInfo(kevent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	err |= clReleaseEvent(kevent);
	if (err != CL_SUCCESS)
	{
		printf("Could not get profiling mult kernel, code: %d\n", err);
		exit(1);
	}
	unsigned long long FLOPs = (long long)dims.m * (long long)dims.n * (long long)(2 * dims.k - 1);
	printf("mult estimated FLOPs: %llu\n", FLOPs);
	time_passed_kernel = (time_end - time_start) / (double)1e9;
	printf("mult kernel time (sec): %f\n", time_passed_kernel);
	printf("mult GFLOPS: %lf\n", FLOPs * 1e-9 / time_passed_kernel);

	// Wait for the command queue to get serviced before reading back results
	clFinish(queue);

	// Read the results from the device
	double time_passed_read;
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, dims.m * dims.n * sizeof(*c), c, 0, NULL, &revent);
	clWaitForEvents(1, &revent);
	err = clGetEventProfilingInfo(revent, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	err |= clGetEventProfilingInfo(revent, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	err |= clReleaseEvent(revent);
	time_passed_read = (time_end - time_start) / (double)1e9;
	printf("mult read time (sec): %f\n", time_passed_write);

	clFinish(queue);

	if (err != CL_SUCCESS)
	{
		printf("Could not release mult event, code: %d\n", err);
		exit(1);
	}

	err = clReleaseMemObject(d_a);
	err |= clReleaseMemObject(d_b);
	err |= clReleaseMemObject(d_c);
	if (err != CL_SUCCESS)
	{
		printf("Could not release mult resources, code: %d\n", err);
		exit(1);
	}

	err = clReleaseKernel(kernel);
	if (err != CL_SUCCESS)
	{
		printf("Could not release mult kernel, code: %d\n", err);
		exit(1);
	}

	err = clReleaseProgram(program);
	if (err != CL_SUCCESS)
	{
		printf("Could not release mult program, code: %d\n", err);
		exit(1);
	}

	free(source_str);
	fflush(stdout);
	return 0;
}

int cl_transpose(char *kernel_file, char *kernel_name,
				 MatTransposeDims dims,
				 float *a, cl_mem d_at)
{

	// Device input buffers
	cl_mem d_a;

	cl_program program; // program
	cl_kernel kernel;	// kernel
	cl_int err;

	FILE *cl_code = fopen(kernel_file, "rb");
	if (cl_code == NULL)
	{
		printf("Could not open transpose kernel file: %s\n", kernel_file);
		exit(1);
	}

	char *source_str = (char *)malloc(MAX_SOURCE_SIZE + 1);
	memset(source_str, 0, MAX_SOURCE_SIZE + 1);
	int res = fread(source_str, 1, MAX_SOURCE_SIZE, cl_code);
	fclose(cl_code);

	// printf("transpose kernel\r\n%s:", source_str);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Could not create transpose program, code: %d\n", err);
		exit(1);
	}

	// Build the program executable
	err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Could not build transpose program, code: %d\n", err);
		if (err == CL_BUILD_PROGRAM_FAILURE)
		{
			printBuildError(device_id, program);
		}
		exit(1);
	}

	// Create the compute kernel in the program we wish to run
	kernel = clCreateKernel(program, kernel_name, &err);
	if (err != CL_SUCCESS)
	{
		printf("Could not create transpose kernel, code: %d\n", err);
		exit(1);
	}

	int max_local_size = getMaxLocalSize(kernel, device_id, 2);
	printf("transpose max_local_size: %d\n", max_local_size);
	size_t t_local[2] = {max_local_size, max_local_size};
	size_t t_global[2] = {
		(size_t)(int)(ceil(dims.m / (float)t_local[0]) * t_local[0]),
		(size_t)(int)(ceil(dims.n / (float)t_local[1]) * t_local[1])};

	d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, dims.m * dims.n * sizeof(*a), NULL, NULL);

	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, d_a, CL_FALSE, 0, dims.m * dims.n * sizeof(*a), a, 0, NULL, NULL);

	// Set the arguments to our compute kernel
	int param = 0;
	err = clSetKernelArg(kernel, param++, sizeof(int), (void *)&dims.m);
	err |= clSetKernelArg(kernel, param++, sizeof(int), (void *)&dims.n);
	err |= clSetKernelArg(kernel, param++, sizeof(int), (void *)&dims.tm);
	err |= clSetKernelArg(kernel, param++, sizeof(int), (void *)&dims.tn);
	err |= clSetKernelArg(kernel, param++, sizeof(cl_mem), (void *)&d_a);
	err |= clSetKernelArg(kernel, param++, sizeof(cl_mem), (void *)&d_at);
	if (err != CL_SUCCESS)
	{
		printf("Could not set kernel args, code: %d\n", err);
		exit(1);
	}

	// printf("t_local_size: %ld:%ld, t_global_size: %ld:%ld\r\n", t_local[0], t_local[1], t_global[0], t_global[1]);

	cl_event event;
	cl_ulong time_start = 0;
	cl_ulong time_end = 0;
	double time_passed_kernel;

	// printf("exec kernel: %s\r\n", kernel_name);
	fflush(stdout);
	// Execute the kernel over the entire range of the data set
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, t_global, t_local, 0, NULL, &event);
	if (err != CL_SUCCESS)
	{
		printf("Could not exec transpose kernel, code: %d\n", err);
		exit(1);
	}

	clWaitForEvents(1, &event);
	clFinish(queue);
	err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	if (err != CL_SUCCESS)
	{
		printf("Could not get profiling transpose kernel, code: %d\n", err);
		exit(1);
	}
	// transpose kernel FLOPs estimated:
	unsigned long long FLOPs = (long long)dims.m * (long long)dims.n;
	printf("transpose estimated FLOPs: %llu\n", FLOPs);
	time_passed_kernel = (time_end - time_start) / (double)1e9;
	// printf("transpose kernel time (sec): %f\n", time_passed_kernel);
	printf("transpose GFLOPS: %lf\n", FLOPs * 1e-9 / time_passed_kernel);

	err = clReleaseEvent(event);
	if (err != CL_SUCCESS)
	{
		printf("Could not release transpose event, code: %d\n", err);
		exit(1);
	}

	err = clReleaseMemObject(d_a);
	// TODO: releasing the output buffer causes problems when it is read again by another kernel
	// err |= clReleaseMemObject(d_c);
	if (err != CL_SUCCESS)
	{
		printf("Could not release transpose memory, code: %d\n", err);
		exit(1);
	}

	err = clReleaseKernel(kernel);
	if (err != CL_SUCCESS)
	{
		printf("Could not release transpose kernel, code: %d\n", err);
		exit(1);
	}

	err = clReleaseProgram(program);
	if (err != CL_SUCCESS)
	{
		printf("Could not release transpose program, code: %d\n", err);
		exit(1);
	}

	free(source_str);
	fflush(stdout);
	return 0;
}

void printBuildError(cl_device_id device_id, cl_program program)
{
	// Determine the size of the log
	size_t log_size;
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

	// Allocate memory for the log
	char *log = (char *)malloc(log_size);

	// Get the log
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

	// Print the log
	printf("%s\n", log);

	free(log);
}

void init_opencl()
{
	size_t strSize = (sizeof(char) * MAX_CHARS);
	size_t retSize;
	cl_int err;

	if (device_id)
		return;

	initPlatforms();

	// choose the platform
	cpPlatform = platforms[platform_index];
	printf("using platform: %d\n", platform_index);

	// Get IDs for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, MAX_DEVICES, device_ids, &num_devices);

	device_id = device_ids[currentDevice];
	err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, strSize, (void *)device_name, &retSize);
	if (err != CL_SUCCESS)
	{
		printf("Could not get device name, code: %d\n", err);
		exit(1);
	}
	printf("using device: %d:%s\n", currentDevice, device_name);

	// Create a context
	context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if (err != CL_SUCCESS)
	{
		printf("Could not create context, code: %d\n", err);
		exit(1);
	}

	// Create a command queue
	queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if (err != CL_SUCCESS)
	{
		printf("Could not create command queue, code: %d\n", err);
		exit(1);
	}

	max_shared_mem = getMaxSharedMemSize(device_id);
	printf("max_shared_mem: %ld\n", max_shared_mem);

	max_shared_mem_per_dim = (long)pow(max_shared_mem, 1.0f / 2);
	printf("max_shared_mem per dim: %ld\n", max_shared_mem_per_dim);
}

void close_opencl()
{
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}