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

// simple
// matrix a needs to be in row major format (M*K)
// matrix b needs to be in row major format (K*N)
// matrix c will be in row major format (M*N)
__kernel void matmult_simple(const int M, const int K, const int N,
                      const __global float* a,
                      const __global float* b,
                      __global float* c) {{
    const int row = get_global_id(0);
    const int col = get_global_id(1);
	if(row >= M || col >= N)
		return;
    float C = 0.0f;
    for (int ik=0; ik<K; ik++) {{
        C += a[row*K + ik] * b[ik*N + col];
    }}
    c[row*N + col] = C;
}}