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

#ifndef __OPENCL_TOOLS_H
#define __OPENCL_TOOLS_H

#include <CL/opencl.h>
#include "mat_tools.h"

#define MAX_PLATFORMS 8
#define MAX_DEVICES 8
#define MAX_CHARS 1024
#define MAX_SOURCE_SIZE (0x100000)

void displayDevice(cl_device_id device_id);
void displayDevices(cl_platform_id cpPlatform);
void displayPlatforms();
void initPlatforms();
int getWorkgroupSize(cl_kernel kernel, cl_device_id device_id);
int getMaxLocalSize(cl_kernel kernel, cl_device_id device_id, int dims);
long getMaxSharedMemSize();
void add_kernel_defines(char *source_str, TileParams tile_params);
void add_kernel_transpose_defines(char *source_str, int TRANSPOSEX, int TRANSPOSEY);
int get_kernel_max_local_size(cl_context context, char *source_str, char *kernel_name, cl_device_id device_id, TileParams tile_params);

#endif // __OPENCL_TOOLS_H