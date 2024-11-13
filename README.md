Fast matrix multiplication in OpenCL using blocks inspired by:  
https://en.wikipedia.org/wiki/Block_matrix  
https://wrigstad.com/upmarc-ss14/simon3.pdf  
https://cnugteren.github.io/tutorial/pages/page4.html  
  
The current implementation differs from the above tutorials/designs:  
1. read patterns by work item are more "sequential"  
2. there is no bank conflict resolution  
3. kernels support non-square matrices (dim sizes M, K, N can differ)  
4. kernel 2 and 3 transpose matrix A instead of B for better memory coalescence in expense of more memory  
5. kernel 3 uses padding to fit all blocks eliminating branch divergence in expense of even more memory  
  
  
```
// Sample code

// dimension sizes
int M=128, K=128, N=128;

// allocate buffers
float* a = (float*) malloc(M * K * sizeof(float));
float* b = (float*) malloc(K * N * sizeof(float));
float* c = (float*) malloc(M * N * sizeof(float));

// populate the buffers somewhere here

// matrix multiplication using opencl with tiling (fast)
openclMatMultBlock(M, K, N, a, b, c);

// matrix multiplication using opencl uses transpose internally with tiling
openclMatMultTilingColMajor(M, K, N, a, b, c);

// matrix multiplication using opencl uses transpose internally with tiling
// also uses padding so M, N, K will be multiples of tile dimensions
openclMatMultTilingColMajorPadded(M, K, N, a, b, c);

// free your buffers when not needed
	
```