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
#include "opencl_tools.h"
#include "mat_tools.h"

#define KERNEL_DIR "../kernels/"

int cl_mult(char* kernel_file, char* kernel_name, 
	int m, int k, int n, float* a, float* b, float* c, cl_mem d_at,
	bool use_tiling);
int cl_transpose(char* kernel_file, char* kernel_name, 
	int m, int k, int k2, int m2,
	float* a, cl_mem d_at);
void printBuildError(cl_device_id device_id, cl_program program);
int get_kernel_max_local_size(cl_context context, char* source_str, char* kernel_name, cl_device_id device_id);
void set_tiling_params(int m, int k, int n, long max_local_size);

cl_platform_id cpPlatform;        // OpenCL platform
cl_device_id device_id = NULL;           // device ID
char device_name[MAX_CHARS];
int num_devices;
cl_context context;               // context
cl_command_queue queue;           // command queue
cl_device_id device_ids[MAX_DEVICES];
extern cl_platform_id platforms[MAX_PLATFORMS];
long max_shared_mem;
long max_shared_mem_per_dim;
int kernel_max_local_size = 16;

int platform_index = 0;
int currentDevice = 0;
bool use_cl_transpose = true;
bool validate_transpose_results = false;
bool print_temp_mat = false;

bool use_optimal_params = true;
bool validate_params = true;

int BM = 128;
int BN = 128;
int BK = 16;
int WIM= 8;
int WIN= 8;

void openclMatMult(int m, int k, int n, float* a, float* b, float* c) {
	time_t start, end;
	
	printf("\n");
	start = gettime();
	
	cl_mult(KERNEL_DIR"kernel_matmult.cl", "matmult_simple", 
		m, k, n, 
		a, b, c,
		NULL,
		false);
	end = gettime();
	unsigned long long FLOPs = (long) m * (long) n * (long) (2*k-1);
	double dtime = difftime(end, start)/1e9;
	printf("extra mem used: %ld\n", 0L);
	printf("time for openclMatMult (secs): %.3lf, total GFLOPS: %.2lf\n", dtime, 
		FLOPs * 1e-9 / dtime);
}
	
void openclMatMultBlock(int m, int k, int n, float* a, float* b, float* c) {	
	time_t start, end;
	
	printf("\n");
	start = gettime();
	
	cl_mult(KERNEL_DIR"kernel_matmult_tiling.cl", "matmult_block", 
		m, k, n, 
		a, b, c,
		NULL,
		true);
		
	end = gettime();
	unsigned long long FLOPs = (long) m * (long) n * (long) (2*k-1);
	double dtime = difftime(end, start)/1e9;
	printf("extra mem used: %ld\n", 0L);
	printf("time for openclMatMultTiling (secs): %.3lf, total GFLOPS: %.2lf\n", dtime, 
		FLOPs * 1e-9 / dtime);
}

void openclMatMultTilingColMajor(int m, int k, int n, float* a, float* b, float* c) {
	time_t start, end;
	
	printf("\n");
		
	start = gettime();
	time_t begint = gettime();
	float* at = create(k, m, 0);
	cl_mem d_at = NULL;
	if(use_cl_transpose) {
		d_at = clCreateBuffer(context, CL_MEM_WRITE_ONLY, k*m*sizeof(*at), NULL, NULL);
		cl_transpose(KERNEL_DIR"kernel_transpose.cl", "transpose", 
		m, k, k, m,
		a, d_at);
		
		if(validate_transpose_results || print_temp_mat) {
			// Wait for the command queue to get serviced before reading back results
			clFinish(queue);
			// Read the results from the device
			cl_event event;
			clEnqueueReadBuffer(queue, d_at, CL_TRUE, 0, k*m*sizeof(*at), at, 0, NULL, &event);
			clWaitForEvents(1 , &event);
			clFinish(queue);
			
			float* at_res = create(k, m, 0);
			transpose(m, k, a, k, m, at_res);
			assert_mat_equal(k, m, at, at_res);
		}
	} else {
		transpose(m, k, a, k, m, at);
	}
	time_t endt = gettime();
	double difft = difftime(endt, begint)/1e9;
	if(print_temp_mat) {
		print_matrix("AT", at, k, m);
	}
	
	cl_mult(KERNEL_DIR"kernel_matmult_tiling_colmajor.cl", "matmult_block_colmajor", 
		m, k, n,
		at, b, c, d_at,
		true);
	free(at);
	
	end = gettime();
	unsigned long long FLOPs = (long) m * (long) n * (long) (2*k-1);
	double dtime = difftime(end, start)/1e9;
	
	printf("time to transpose (secs): %.3lf\n", difft);
	printf("extra mem used: %ld\n", k*m*sizeof(*a));
	printf("time for openclMatMultTilingColMajor (secs): %.3lf, total GFLOPS: %.2lf\n", dtime, 
		FLOPs * 1e-9 / dtime);
}

void openclMatMultTilingColMajorPadded(int m, int k, int n, float* a, float* b, float* c) {
	time_t start, end;
	
	printf("\n");
	start = gettime();
	
	time_t begint = gettime();
	int paddedm = ceil(m / (float) BM) * BM;
	int paddedk = ceil(k / (float) BK) * BK;
	int paddedn = ceil(n / (float) BN) * BN;
	
	float* aTpadded = create(paddedk, paddedm, 0);
	cl_mem d_at = NULL;
	if(use_cl_transpose) {
		d_at = clCreateBuffer(context, CL_MEM_WRITE_ONLY, k*m*sizeof(*aTpadded), NULL, NULL);
		cl_transpose(KERNEL_DIR"kernel_transpose.cl", "transpose", 
		m, k, paddedk, paddedm,
		a, d_at);
		
		if(validate_transpose_results) {
			// Wait for the command queue to get serviced before reading back results
			clFinish(queue);
			// Read the results from the device
			cl_event event;
			clEnqueueReadBuffer(queue, d_at, CL_TRUE, 0, k*m*sizeof(*aTpadded), aTpadded, 0, NULL, &event);
			clWaitForEvents(1 , &event);
			clFinish(queue);
			
			float* at_res = create(paddedk, paddedm, 0);
			transpose(m, k, a, paddedk, paddedm, at_res);
			assert_mat_equal(paddedk, paddedm, aTpadded, at_res);
		}
	} else {
		transpose(m, k, a, paddedk, paddedm, aTpadded);
	}
	
	float* bpadded = NULL;
	if(paddedk != k || paddedn != n) {
		bpadded = create(paddedk, paddedn, 0);
		copy_mat(k, n, b, paddedk, paddedn, bpadded, k, n);
	}
	float* cpadded = NULL;
	if(paddedm != m || paddedn != n) {
		cpadded = create(paddedm, paddedn, 0);
	}
	time_t endt = gettime();
	double difft = difftime(endt, begint)/1e9;
	if(print_temp_mat) {
		print_matrix("ATpadded", aTpadded, paddedk, paddedm);
		print_matrix("Bpadded", bpadded, paddedk, paddedn);
	}
	
	cl_mult(KERNEL_DIR"kernel_matmult_tiling_colmajor_padded.cl", "matmult_block_colmajor_padded", 
		paddedm, paddedk, paddedn,
		aTpadded, bpadded?bpadded:b, cpadded?cpadded:c,
		d_at,
		true);
	
	if(cpadded != NULL) {
		if(print_temp_mat) {
			print_matrix("Cpadded", cpadded, paddedm, paddedn);	
		}
		copy_mat(paddedm, paddedn, cpadded, m, n, c, m, n);
	}
	free(aTpadded);
	if(bpadded != NULL)
		free(bpadded);
	if(cpadded != NULL)
		free(cpadded);
	
	end = gettime();
	
	unsigned long long FLOPs = (long) m * (long) n * (long) (2*k-1);
	double dtime = difftime(end, start)/1e9;
	
	printf("time to transpose and pad (secs): %.3lf\n", difft);
	printf("extra mem used: %ld\n", (paddedk*paddedm + paddedk*paddedn + paddedm*paddedn) *sizeof(*a));
	printf("time for openclMatMultTilingColMajorPadded (secs): %.3lf, total GFLOPS: %.2lf\n", dtime, 
		FLOPs * 1e-9 / dtime);
}

// d_at is the transpose buffer if we have transposed the matrix a, otherwise we will use the buffer a
int cl_mult(char* kernel_file, char* kernel_name, 
	int m, int k, int n, float* a, float* b, float* c, cl_mem d_at,
	bool use_tiling)
{
    
    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;

	cl_program program;               // program
    cl_kernel kernel;                 // kernel
	
	cl_int err;
	size_t local[2], global[2];
	
	FILE* cl_code = fopen(kernel_file, "rb");
    if (cl_code == NULL) { 
		printf("Could not open mult kernel file: %s\n", kernel_file); 
		exit(1);
	}
    char* source_str = (char *)malloc(MAX_SOURCE_SIZE + 1);
	memset(source_str, 0, MAX_SOURCE_SIZE + 1);
	int res = fread(source_str, 1, MAX_SOURCE_SIZE, cl_code);
	fclose(cl_code);
	
	if(use_tiling) {
		if(use_optimal_params) {
			if(kernel_max_local_size == 0) {
				// TODO: this has overhead which is more noticable with small matrixes
				// kernel_max_local_size = get_kernel_max_local_size(context, source_str, kernel_name, device_id);
			}
			set_tiling_params(m, k, n, kernel_max_local_size);
		}
		add_kernel_defines(source_str, BM, BN, BK, WIM, WIN);
	}
	
	// printf("mult kernel\r\n%s:", source_str);
		
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, NULL, &err);
	if(err != CL_SUCCESS) {
		printf("Could not create mult program, code: %d\n", err);
		exit(1);
	}
	
    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS) {
		printf("Could not build mult program, code: %d\n", err);
		if (err == CL_BUILD_PROGRAM_FAILURE) {
			printBuildError(device_id, program);
		}
		exit(1);
	}
	
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, kernel_name, &err);
	if(err != CL_SUCCESS) {
		printf("Could not create mult kernel: %s, code: %d\n",kernel_name, err);
		exit(1);
	}

	// printf("kernel_max_local_size: %d\n", kernel_max_local_size);
	
	if(validate_params) {
		validate_tiling(BM, BN, BK, WIM, WIN, kernel_max_local_size);
	}
	
	if(use_tiling) {
		local[0] = BM/WIM; 
		local[1] = BN/WIN;
		global[0] = (size_t)(ceil(m/(float)BM) * BM / WIM);
		global[1] = (size_t)(ceil(n/(float)BN) * BN / WIN);
	} else {
		local[0] = kernel_max_local_size;
		local[1] = kernel_max_local_size;
		global[0] = m;
		global[1] = n;
	}
	
	// use the transpose if we have one
	if(d_at)
		d_a = d_at;
	else
		d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, m*k*sizeof(*a), NULL, NULL);
	d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, k*n*sizeof(*b), NULL, NULL);
	d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, m*n*sizeof(*c), NULL, NULL);
			
	// Write our data set into the input array in device memory
	if(!d_at)
		err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, m*k*sizeof(*a), a, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, k*n*sizeof(*b), b, 0, NULL, NULL);
	err |= clEnqueueWriteBuffer(queue, d_c, CL_TRUE, 0, m*n*sizeof(*c), c, 0, NULL, NULL);
	if(err != CL_SUCCESS) {
		printf("Could not enqueue mult buffers, code: %d\n", err);
		exit(1);
	}
	
	// Set the arguments to our compute kernel
	int param = 0;
	err = clSetKernelArg(kernel, param++, sizeof(int), (void*)&m);
	err |= clSetKernelArg(kernel, param++, sizeof(int), (void*)&k);
	err |= clSetKernelArg(kernel, param++, sizeof(int), (void*)&n);
	err |= clSetKernelArg(kernel, param++, sizeof(cl_mem), (void*)&d_a);
	err |= clSetKernelArg(kernel, param++, sizeof(cl_mem), (void*)&d_b);
	err |= clSetKernelArg(kernel, param++, sizeof(cl_mem), (void*)&d_c);
	if(err != CL_SUCCESS) {
		printf("Could not set mult kernel args, code: %d\n", err);
		exit(1);
	}
	printf("BM: %d, BN: %d, BK: %d, WIM: %d, WIN: %d\n", BM, BN, BK, WIM, WIN);
	printf("local_size: %ld:%ld, global_size: %ld:%ld\r\n", local[0], local[1], global[0], global[1]);
	
	cl_event event;
	cl_ulong time_start = 0;
	cl_ulong time_end = 0;
	double time_passed_kernel;

	// printf("exec kernel: %s\r\n", kernel_name);
	fflush(stdout);
	// Execute the kernel over the entire range of the data set 
	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
	if(err != CL_SUCCESS) {
		printf("Could not exec mult kernel, code: %d\n", err);
		exit(1);
	}
 
	clWaitForEvents(1 , &event);
	clFinish(queue);	
	err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	if(err != CL_SUCCESS) {
		printf("Could not get profiling mult kernel, code: %d\n", err);
		exit(1);
	}
	
	// kernel FLOPs estimated:
	unsigned long long FLOPs = (long) m * (long) n * (long) (2*k-1);
	// printf("estimated FLOPs: %llu\n", FLOPs);
	time_passed_kernel = (time_end-time_start)/(double)1e9;
	// printf("kernel time (sec): %f\n", time_passed_kernel);
	printf("mult GFLOPS: %lf\n", FLOPs * 1e-9 / time_passed_kernel);

    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
    // Read the results from the device
	clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, m*n*sizeof(*c), c, 0, NULL, &event);
	clWaitForEvents(1 , &event);
	clFinish(queue);	
	
	err = clReleaseEvent(event);
	if(err != CL_SUCCESS) {
		printf("Could not release mult event, code: %d\n", err);
		exit(1);
	}
	
	err = clReleaseMemObject(d_a);
	err |= clReleaseMemObject(d_b);
	err |= clReleaseMemObject(d_c);
	if(err != CL_SUCCESS) {
		printf("Could not release mult resources, code: %d\n", err);
		exit(1);
	}
	
    err = clReleaseKernel(kernel);
	if(err != CL_SUCCESS) {
		printf("Could not release mult kernel, code: %d\n", err);
		exit(1);
	}
	
	err = clReleaseProgram(program);
	if(err != CL_SUCCESS) {
		printf("Could not release mult program, code: %d\n", err);
		exit(1);
	}
	
    free(source_str);
	fflush(stdout);
    return 0;
}


int cl_transpose(char* kernel_file, char* kernel_name, 
	int m, int k, int k2, int m2,
	float* a, cl_mem d_at)
{
    
    // Device input buffers
    cl_mem d_a;

    cl_program program;               // program
    cl_kernel kernel;                 // kernel
	cl_int err;
	
	FILE* cl_code = fopen(kernel_file, "rb");
    if (cl_code == NULL) { 
		printf("Could not open transpose kernel file: %s\n", kernel_file); 
		exit(1);
	}
	
	char* source_str = (char *)malloc(MAX_SOURCE_SIZE + 1);
	memset(source_str, 0, MAX_SOURCE_SIZE + 1);
	int res = fread(source_str, 1, MAX_SOURCE_SIZE, cl_code);
	fclose(cl_code);
	
	// printf("transpose kernel\r\n%s:", source_str);
		
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char**)&source_str, NULL, &err);
	if(err != CL_SUCCESS) {
		printf("Could not create transpose program, code: %d\n", err);
		exit(1);
	}

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS) {
		printf("Could not build transpose program, code: %d\n", err);
		if (err == CL_BUILD_PROGRAM_FAILURE) {
			printBuildError(device_id, program);
		}
		exit(1);
	}

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, kernel_name, &err);
	if(err != CL_SUCCESS) {
		printf("Could not create transpose kernel, code: %d\n", err);
		exit(1);
	}
	
	int max_local_size = getMaxLocalSize(kernel, device_id, 2);
	// printf("transpose max_local_size: %d\n", max_local_size);
	size_t t_local[2] = { max_local_size, max_local_size };
	size_t t_global[2] = {
		(size_t) (int) (ceil(m/(float)t_local[0])*t_local[0]), 
		(size_t) (int) (ceil(k/(float)t_local[1])*t_local[1])
	};

	d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, m*k*sizeof(*a), NULL, NULL);
			
	// Write our data set into the input array in device memory
	err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, m*k*sizeof(*a), a, 0, NULL, NULL);
	// err = clEnqueueWriteBuffer(queue, d_c, CL_TRUE, 0, k2*k2*sizeof(*c), c, 0, NULL, NULL);

	// Set the arguments to our compute kernel
	int param = 0;
	err = clSetKernelArg(kernel, param++, sizeof(int), (void*)&m);
	err |= clSetKernelArg(kernel, param++, sizeof(int), (void*)&k);
	err |= clSetKernelArg(kernel, param++, sizeof(int), (void*)&k2);
	err |= clSetKernelArg(kernel, param++, sizeof(int), (void*)&m2);
	err |= clSetKernelArg(kernel, param++, sizeof(cl_mem), (void*)&d_a);
	err |= clSetKernelArg(kernel, param++, sizeof(cl_mem), (void*)&d_at);
	if(err != CL_SUCCESS) {
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
	if(err != CL_SUCCESS) {
		printf("Could not exec transpose kernel, code: %d\n", err);
		exit(1);
	}
 
	clWaitForEvents(1 , &event);
	clFinish(queue);
	err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	err |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	if(err != CL_SUCCESS) {
		printf("Could not get profiling transpose kernel, code: %d\n", err);
		exit(1);
	}
	// kernel FLOPs estimated:
	unsigned long long FLOPs = (long) m * (long) k;
	// printf("transpose estimated FLOPs: %llu\n", FLOPs);
	time_passed_kernel = (time_end-time_start)/(double)1e9;
	// printf("transpose kernel time (sec): %f\n", time_passed_kernel);
	printf("transpose GFLOPS: %lf\n", FLOPs * 1e-9 / time_passed_kernel);
	
	err = clReleaseEvent(event);
	if(err != CL_SUCCESS) {
		printf("Could not release transpose event, code: %d\n", err);
		exit(1);
	}
	
	err = clReleaseMemObject(d_a);
	// TODO: releasing the output buffer causes problems when it is read again by another kernel
	// err |= clReleaseMemObject(d_c);
	if(err != CL_SUCCESS) {
		printf("Could not release transpose memory, code: %d\n", err);
		exit(1);
	}
	
    err = clReleaseKernel(kernel);
	if(err != CL_SUCCESS) {
		printf("Could not release transpose kernel, code: %d\n", err);
		exit(1);
	}
	
	err = clReleaseProgram(program);
	if(err != CL_SUCCESS) {
		printf("Could not release transpose program, code: %d\n", err);
		exit(1);
	}
	
    free(source_str);
	fflush(stdout);
    return 0;
}

void printBuildError(cl_device_id device_id, cl_program program) {
	// Determine the size of the log
	size_t log_size;
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

	// Allocate memory for the log
	char *log = (char *) malloc(log_size);

	// Get the log
	clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

	// Print the log
	printf("%s\n", log);
	
	free(log);
}

void init_opencl() {
	size_t strSize = (sizeof(char) * MAX_CHARS);
	size_t retSize;
	cl_int err;
	
	if(device_id)
		return;
	
	initPlatforms();
	
    // choose the platform
	cpPlatform = platforms[platform_index];
	printf("using platform: %d\n", platform_index);

    // Get IDs for the device
	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, MAX_DEVICES, device_ids, &num_devices);
	
	device_id = device_ids[currentDevice];
	err = clGetDeviceInfo(device_id,CL_DEVICE_NAME, strSize,(void*)device_name,&retSize);
    if(err != CL_SUCCESS) {
		printf("Could not get device name, code: %d\n", err);
		exit(1);
	}
	printf("using device: %d:%s\n", currentDevice, device_name);
	
	// Create a context 
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
	if(err != CL_SUCCESS) {
		printf("Could not create context, code: %d\n", err);
		exit(1);
	}

    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
	if(err != CL_SUCCESS) {
		printf("Could not create command queue, code: %d\n", err);
		exit(1);
	}
	
	max_shared_mem = getMaxSharedMemSize(device_id);
	printf("max_shared_mem: %ld\n", max_shared_mem);
	
	max_shared_mem_per_dim = (long) pow(max_shared_mem, 1.0f/2);
}

void close_opencl() {
	clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

int get_kernel_max_local_size(cl_context context, char* source_str, char* kernel_name, cl_device_id device_id) {
	cl_program program;
	cl_int err;
	char *kernel_src = (char *)malloc(4*1024*sizeof(char));
	
	// printf("mult kernel\r\n%s:", source_str);
	
	strcpy(kernel_src, source_str);
	strcat(kernel_src, "\0");
	add_kernel_defines(kernel_src, BM, BN, BK, WIM, WIN);
	
	// printf("tuning mult kernel\r\n%s\r\n:", kernel_src);
		
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_src, NULL, &err);
	if(err != CL_SUCCESS) {
		printf("Could not create mult program, code: %d\n", err);
		exit(1);
	}
	
    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(err != CL_SUCCESS) {
		printf("Could not build mult program, code: %d\n", err);
		if (err == CL_BUILD_PROGRAM_FAILURE) {
			printBuildError(device_id, program);
		}
		exit(1);
	}
	
    // Create the compute kernel in the program we wish to run
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
	if(err != CL_SUCCESS) {
		printf("Could not create mult kernel: %s, code: %d\n",kernel_name, err);
		exit(1);
	}
	// printf("ck\n");
		
	int max_local_size = getMaxLocalSize(kernel, device_id, 2);
	// printf("max_local_size: %d\n", max_local_size);
		
    err = clReleaseKernel(kernel);
	if(err != CL_SUCCESS) {
		printf("Could not release mult kernel, code: %d\n", err);
		exit(1);
	}
	
	err = clReleaseProgram(program);
	if(err != CL_SUCCESS) {
		printf("Could not release mult program, code: %d\n", err);
		exit(1);
	}
	free(kernel_src);
	return max_local_size;
}

void set_tiling_params(int m, int k, int n, long max_local_size) {
		
  // for now these default values seems to yield fast results
  if (m <= 16 && n <= 16 && k <= 16) {
    BM=8; BN=8; BK=8; WIM=4; WIN=4;
  }
  else if (m <= 256 && n <= 256 && k <= 128) {
    BM=16; BN=16; BK=8; WIM=4; WIN=4;
  }
  else if (m <= 512 && n <= 512 && k <= 512) {
    BM=64; BN=64; BK=8; WIM=4; WIN=4;
  }
  else {
	BM=128; BN=128; BK=16; WIM=8; WIN=8;
  }
  // also these may be fast for large matrices with DTYPE=half for high end devices
  // 256, 256, 4, 16, 16
  // 256, 256, 8, 16, 16
  
	printf("optimal BM: %d, BN: %d, BK: %d, WIM: %d, WIN: %d\n",
		BM, BN, BK, WIM, WIN);
}