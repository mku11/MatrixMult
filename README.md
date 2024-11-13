Very fast matrix multiplication in OpenCL using blocks (aka tiling) inspired by:  
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

// allocate input buffers
float* a = (float*) malloc(M * K * sizeof(float));
float* b = (float*) malloc(K * N * sizeof(float));
// allocate output buffer
float* c = (float*) malloc(M * N * sizeof(float));

// populate the buffers somewhere here

// There are 3 kernels implemented to choose from

// Kernel 1: plain blocks
// pros: faster than generic matrix multiplication
// cons: slower GPU processing due to branch divergence
openclMatMultBlock(M, K, N, a, b, c);

// Kernel 2: blocks with transposing matrix A
// pros: often faster GPU processing, transposed matrices yield better mem coalescence (depends on shape)
// cons: extra memory for transposing
// cons: more GPU time for transposing
// cons: slower GPU processing due to branch divergence
openclMatMultTilingColMajor(M, K, N, a, b, c);

// Kernel 3: blocks with transposing mat A and padding all matrices
// pros: often faster GPU processing, transposed matrices yield better mem coalescence (depends on shape)
// pros: faster GPU processing because no branch divergence since all blocks fit to matrices
// cons: extra memory for transposing and padding
// cons: more GPU time for transposing
openclMatMultTilingColMajorPadded(M, K, N, a, b, c);

// free your buffers when not needed
	
```