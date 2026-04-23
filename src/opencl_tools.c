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
#include <string.h>

#include <CL/opencl.h>
#include "mat_tools.h"
#include "opencl_tools.h"

int num_platforms;
cl_platform_id platforms[MAX_PLATFORMS];

void displayDevice(cl_device_id device_id)
{
	char device_vendor[MAX_CHARS];
	char device_name[MAX_CHARS];
	cl_device_exec_capabilities execCaps;
	cl_device_svm_capabilities svmCaps;
	size_t max_work_group_size;
	size_t max_compute_units;
	size_t strSize = (sizeof(char) * MAX_CHARS);
	size_t retSize;
	cl_int err;

	printf("-------------------\n");
	printf("device_id: %lld\n", (long long)device_id);

	char device_version[MAX_CHARS];
	clGetDeviceInfo(device_id, CL_DEVICE_VERSION,
					strSize, &device_version, &retSize);
	printf("device_version: %s\n", device_version);

	char device_cl_version[MAX_CHARS];
	clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION,
					strSize, &device_cl_version, &retSize);
	printf("device_cl_version: %s\n", device_cl_version);

	char device_driver_version[MAX_CHARS];
	clGetDeviceInfo(device_id, CL_DRIVER_VERSION,
					strSize, &device_driver_version, &retSize);
	printf("device_driver_version: %s\n", device_driver_version);

	err = clGetDeviceInfo(device_id, CL_DEVICE_NAME,
						  strSize, (void *)device_name, &retSize);
	printf("device name: %s\n", device_name);

	err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR,
						  strSize, (void *)device_vendor, &retSize);
	printf("device vendor: %s\n", device_vendor);

	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE,
						  sizeof(max_work_group_size), &max_work_group_size, 0);
	printf("max work group size: %ld\n", (long)max_work_group_size);

	err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS,
						  sizeof(max_compute_units), &max_compute_units, 0);
	printf("max compute units: %d\n", (int)max_compute_units);

	err = clGetDeviceInfo(device_id, CL_DEVICE_EXECUTION_CAPABILITIES,
						  sizeof(cl_device_exec_capabilities), &execCaps, 0);
	printf("CL_DEVICE_EXECUTION_CAPABILITIES err: %d\n", err);
	if (err == CL_SUCCESS)
	{
		if (execCaps & CL_EXEC_KERNEL)
			printf("CL_EXEC_KERNEL: yes\n");
		if (execCaps & CL_EXEC_NATIVE_KERNEL)
			;
		printf("CL_EXEC_NATIVE_KERNEL: yes\n");
	}
	else if (err == CL_INVALID_VALUE)
	{
		printf("CL_DEVICE_EXECUTION_CAPABILITIES CL_INVALID_VALUE\n");
	}

	printf("cl_device_svm_capabilities size: %lld\n", sizeof(cl_device_svm_capabilities));
	err = clGetDeviceInfo(device_id, CL_DEVICE_SVM_CAPABILITIES,
						  sizeof(cl_device_svm_capabilities), &svmCaps, 0);
	printf("CL_DEVICE_SVM_CAPABILITIES err: %d\n", err);
	if (err == CL_SUCCESS)
	{
		printf("svmCaps: %d\n", (int)svmCaps);
		if (svmCaps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)
			printf("CL_DEVICE_SVM_COARSE_GRAIN_BUFFER: yes\n");
		if (svmCaps & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)
			printf("CL_DEVICE_SVM_FINE_GRAIN_BUFFER: yes\n");
		if (svmCaps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)
			printf("CL_DEVICE_SVM_FINE_GRAIN_SYSTEM: yes\n");
		if (svmCaps & CL_DEVICE_SVM_ATOMICS)
			printf("CL_DEVICE_SVM_ATOMICS: yes\n");
	}
	else if (err == CL_INVALID_VALUE)
	{
		printf("CL_DEVICE_SVM_CAPABILITIES CL_INVALID_VALUE\n");
	}
	printf("-------------------\n");
}

void displayDevices(cl_platform_id cpPlatform)
{
	int num_devices;
	cl_device_id device_ids[MAX_DEVICES];
	cl_int err;
	char platform_vendor[MAX_CHARS];
	char platform_name[MAX_CHARS];
	char platform_version[MAX_CHARS];

	size_t strSize = (sizeof(char) * MAX_CHARS);
	size_t retSize;

	printf("cl_platform_id: %lld\n", (long long)cpPlatform);

	err = clGetPlatformInfo(cpPlatform, CL_PLATFORM_NAME,
							strSize, (void *)platform_name, &retSize);
	printf("platform name: %s\n", platform_name);

	err = clGetPlatformInfo(cpPlatform, CL_PLATFORM_VERSION,
							strSize, (void *)platform_version, &retSize);
	printf("platform version: %s\n", platform_version);

	err = clGetPlatformInfo(cpPlatform, CL_PLATFORM_VENDOR,
							strSize, (void *)platform_vendor, &retSize);
	printf("platform vendor: %s\n", platform_vendor);

	err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, MAX_DEVICES, device_ids, &num_devices);
	printf("platform devices found: %d\n", num_devices);
	for (int i = 0; i < num_devices; i++)
	{
		displayDevice(device_ids[i]);
	}
}

void displayPlatforms()
{
	cl_int err;
	char driver_version[MAX_CHARS];
	clGetDeviceInfo(0, CL_DRIVER_VERSION, sizeof(char *), &driver_version, NULL);
	err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &num_platforms);

	printf("OpenCL driver version: %s\n", driver_version);
	printf("platforms found: %d\n", num_platforms);

	for (int i = 0; i < num_platforms; i++)
	{
		printf("\n-------------------\n");
		displayDevices(platforms[i]);
	}
}

void initPlatforms()
{
	cl_int err;
	err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &num_platforms);
}

int getWorkgroupSize(cl_kernel kernel, cl_device_id device_id)
{
	cl_int err;
	size_t workgroup_size;

	err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE,
								   sizeof(workgroup_size), &workgroup_size, NULL);

	return workgroup_size;
}

long getMaxSharedMemSize(cl_device_id device_id)
{
	cl_int err;
	size_t max_shared_mem;

	err = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE,
						  sizeof(max_shared_mem), &max_shared_mem, 0);
	return (long)max_shared_mem;
}

int getMaxLocalSize(cl_kernel kernel, cl_device_id device_id, int dims)
{
	int maxWorkGroupSize = getWorkgroupSize(kernel, device_id);
	return (int)pow(maxWorkGroupSize, 1.0f / dims);
}

void add_kernel_defines(char *source_str, TileParams tile_params)
{
	char *source_defines_str = (char *)malloc(4 * 1024 * sizeof(char));

	// size per work item for each submatrice: (size of submatrice tile) / (total size of a work group)
	// total size of a work group: (size of output matrix) / (size of work item)
	const int WIA_SIZE = tile_params.BK * tile_params.WIM * tile_params.WIN / tile_params.BN; // (BM * BK) / ((BM * BN) / (WIM * WIN));
	const int WIB_SIZE = tile_params.BK * tile_params.WIM * tile_params.WIN / tile_params.BM; // (BK * BN) / ((BM * BN) / (WIM * WIN));

	// printf("BM: %d, BN: %d, BK: %d, WIM: %d, WIN: %d, WIA_SiZE: %d, WIB_SIZE: %d\n",
	// BM, BN, BK, WIM, WIN, WIA_SIZE, WIB_SIZE);

	sprintf(source_defines_str,
			"#define BM %d // block a height\r\n"
			"#define BN %d // block b width\r\n"
			"#define BK %d // block a width, b height\r\n"
			"#define WIM %d // Work item height\r\n"
			"#define WIN %d // Work item width\r\n"
			"#define WIA_SIZE %d // Work item a size\r\n"
			"#define WIB_SIZE %d // Work item b size\r\n"
			"\r\n",
			tile_params.BM, tile_params.BN, tile_params.BK, tile_params.WIM, tile_params.WIN, WIA_SIZE, WIB_SIZE);
	size_t len = strlen(source_defines_str);
	memmove(source_str + len, source_str, strlen(source_str) + 1);
	memcpy(source_str, source_defines_str, len);
	strcat(source_str, "\0");
	free(source_defines_str);
}

int get_kernel_max_local_size(cl_context context, char *source_str, char *kernel_name, cl_device_id device_id, TileParams tile_params)
{
	cl_program program;
	cl_int err;
	char *kernel_src = (char *)malloc(4 * 1024 * sizeof(char));

	// printf("mult kernel\r\n%s:", source_str);

	strcpy(kernel_src, source_str);
	strcat(kernel_src, "\0");
	add_kernel_defines(kernel_src, tile_params);

	// printf("tuning mult kernel\r\n%s\r\n:", kernel_src);

	// Create the compute program from the source buffer
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src, NULL, &err);
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
	cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
	if (err != CL_SUCCESS)
	{
		printf("Could not create mult kernel: %s, code: %d\n", kernel_name, err);
		exit(1);
	}

	int max_local_size = getMaxLocalSize(kernel, device_id, 2);
	printf("max_local_size: %d\n", max_local_size);

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
	free(kernel_src);
	return max_local_size;
}