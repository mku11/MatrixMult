matrix multiplication in OpenCL inspired by:  
https://en.wikipedia.org/wiki/Block_matrix  
https://wrigstad.com/upmarc-ss14/simon3.pdf  
https://cnugteren.github.io/tutorial/pages/page4.html  
  
The current implementation differs from the above tutorials/designs:  
1. read patterns by work item are more "sequential"  
2. there is no bank conflict resolution  
3. kernels also support non-square matrices (dim sizes M, K, N can differ)  
4. kernel 2 and 3 transposes matrix A instead of B  
5. kernel 3 uses padding to fit all blocks  
  