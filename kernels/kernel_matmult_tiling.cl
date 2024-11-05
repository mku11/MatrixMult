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

// tiling
// matrix a needs to be in row major format (M*K)
// matrix b needs to be in row major format (K*N)
// matrix c will be in row major format (M*N)
// block BA will be transposed in col major format (BK*BM)
// block BB will be in row major format (BK*BN)
// block BC will be in row major format (BM*BN)
__kernel void matmult_block(const int M, const int K, const int N,
					const __global float* a,
					const __global float* b,
					__global float* c) {{

    const int lclId0 = get_local_id(0);
    const int lclId1 = get_local_id(1);
	
	// offset
    const int offsetm = BM*get_group_id(0);
    const int offsetn = BN*get_group_id(1);
    const int tiles = ceil(K/(float)BK);
	
	// work item for the current work group
	const int witem = lclId1*get_local_size(1) + lclId0;
	
	// offsets for sub matrices
	const int offsetA = witem*WIA_SIZE;	
	const int offsetB = witem*WIB_SIZE;
	
	// submatrices
    __local float BA[BK][BM];
	__local float BB[BK][BN];
	float BC[WIM][WIN];
	#pragma unroll
    for (int row=0; row<WIM; row++) {{
        #pragma unroll
        for (int col=0; col<WIN; col++) {{
            BC[row][col] = 0.0f;
        }}
    }}
	
    for(int tile=0; tile<tiles; tile++) {{
		
		int offseta = offsetm*K + BK*tile;
		int row = offsetA / BK, col;
		#pragma unroll
		for(int idx=0; idx<WIA_SIZE; idx++) {{
			col = (offsetA + idx) % BK;
			if(idx>0 && col == 0) {{
				row++;
			}}
			if(offseta + K*row + col >= K*M)
				break;
			BA[col][row] = a[offseta + K*row + col];
		}}
		
		int offsetb = offsetn + BK*tile*N;
		row = offsetB / BN;
		int offsetbb = offsetb + N*row;
		#pragma unroll
		for(int idx=0; idx<WIB_SIZE;idx++) {{
			col = (offsetB + idx) % BN;
			if(idx>0 && col == 0) {{ 
				row++;
				offsetbb = offsetb + N*row;
			}}
			if(offsetbb + col >= K*N) {{
				break;
			}}
			BB[row][col] = b[offsetbb + col];
		}}

        barrier(CLK_LOCAL_MEM_FENCE);

		// partial writes
		const int maxK = K - BK*tile < BK ? K - BK*tile : BK;
		
		for(int ik=0; ik<BK; ik++) {{
			#pragma unroll
			for(int row=0; row<WIM; row++) {{
				#pragma unroll	
				for(int col=0; col<WIN; col++) {{
					if(ik < maxK)
						BC[row][col] += BA[ik][row + WIM*lclId0] * BB[ik][col + WIN*lclId1];
				}}
			}}
		}}

        barrier(CLK_LOCAL_MEM_FENCE);
    }}
    
	
    const int cOffsetRow = offsetm + WIM*lclId0;
	const int cOffsetCol = offsetn + WIN*lclId1;
	
	int idx = cOffsetRow*N + cOffsetCol;
	if(cOffsetCol < N && cOffsetRow < M) {{
		#pragma unroll
		for(int row=0; row<WIM; row++) {{
			#pragma unroll
			for(int col=0; col<WIN; col++) {{
				if((idx + row*N) % N + col >= N)
					continue;
				c[idx + row*N + col] = BC[row][col];
			}}
		}}
	}}
}}