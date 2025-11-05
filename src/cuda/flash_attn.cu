#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#include "flash_attn.h"

// implement scaled dot product attention (softmax(Q @ K^T * softmax_scale) @ V)

// const int Bc = min(ceil(prop.sharedMemPerBlock/sizeof(float)/(4*d)), (float)N);
'''
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // const int Bc = std::ceil(prop.sharedMemPerBlock/4*d);
    const int Bc = min(ceil(prop.sharedMemPerBlock/sizeof(float)/(4*d)), (float)N);
    const int Br = std::min(Bc,d);
'''

// __global__ void flash_attn_forward_kernel(
//     const float *, const float *, const float *, 
//     const int, const int, const int, 
//     const int, const int, const int, 
//     const float, float*, float*, float*);


__global__ void flash_attn_forward_kernel( 
    const float *Q,
    const float *K,
    const float *V,
    const int N,
    const int d,
    const int Tc,
    const int Tr,
    const int Bc,
    const int Br,
    const float softmax_scale,
    float* l,
    float* m,
    float* O,
) {
    int tx; int bx; int by; int qkv_offset; int lm_offset; 
    int tile_size; float* Qi; float* Kj; float* Vj; float* S;
    float row_m_prev; float row_l_prev; 
    float row_m; float row_l;
    float row_m_new; float row_l_new; 
    int y; int x; float sum; float pv;

    tx = threadIdx.x; bx = blockIdx.x; by = blockIdx.y;

    // find offset into Q,K,V,O,l,m
    qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    lm_offset  = (bx * gridDim.y * N)     + (by * N);

    // sram for Q,K,V,S
    extern __shared__ float sram[]; // QUESTION:    what does extern do?
    // above should be on SM shared memory
    tile_size = Bc * d;
    Qi = sram;
    Kj = &sram[tile_size];
    Vj = &sram[tile_size * 2];
    S  = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++){
        // load Kj, Vj into sram
        for (x = 0; x < d; x++){
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();
        for (int i = 0; i < Tr; i++){
            // load Qi to shared mem, li, mi to registers
            for (x = 0; x < d; x++)
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            row_m_prev = m[lm_offset + (Br * i) + tx];
            row_l_prev = l[lm_offset + (Br * i) + tx];

            // compute Sij = softmax_scale * Qi * Kj^T
            // m_tilde_ij = rowmax(Sij)
            row_m = -INFINITY;
            for (y = 0; y < Bc; y++){
                sum = 0;
                for (x = 0; x < d; x++)
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;
                if (sum > row_m) row_m = sum; // pain point?
            }

            // 12. compute
            // P_tilde_ij = exp(Sij_masked - m_tilde_ij)
            // l_tilde_ij = rowsum(P_tilde_ij)
            row_l = 0;
            for (y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y]; // sram access for above calc... store intermed val in register ?
            }

            // 13. compute mi_new, li_new
            row_m_new = max(row_m_prev, row_m);
            row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // 15. Write Oi to global mem
            for (x = 0; x < d; x++){
                pv = 0;
                for (y = 0; y < Bc; y++)
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }

            // 16. Write li <- li_new, mi <- mi_new to global mem (HBM)
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads(); 
        // reasoning QUESTION -> why does/not this need to be here?
        // Kj Vj do they rely on li and mi?
    }
}

"""
    TODO:
        -> finish and refine flash attn
            -> dynamic block sizes for different GPU sram specs (defined in paper)
            -> get rid of thread-per-row simplification
            -> speed up matmul (run on tensorcore?)
            -> Q,K,V make float16 (can I ?)
        -> backwards pass
        -> plug into karpathy gpt model...?
"""

float* flash_forward(
        float* Q, float* K, float* V, 
        int B, int nh, int N, int d
    ) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // const int Bc = std::ceil(prop.sharedMemPerBlock/4*d);
    const int Bc = min(ceil(prop.sharedMemPerBlock/sizeof(float)/(4*d)), (float)N);
    const int Br = min(Bc,d);

    const int Tc = ceil((float)N / Bc); 
    const int Tr = ceil((float)N / Br);
    const float softmax_scale = 1.0f / sqrtf((float)d);

    // Initialize O, l, m on GPU
    float *O, *l, *m;
    size_t Q_size = B * nh * N * d * sizeof(float);
    size_t l_m_size = B * nh * N * sizeof(float);
    
    // Allocate GPU memory
    cudaMalloc(&O, Q_size);
    cudaMalloc(&l, l_m_size);
    cudaMalloc(&m, l_m_size);
    
    // Initialize O to zeros, l to zeros, m to -INFINITY
    cudaMemset(O, 0, Q_size);
    cudaMemset(l, 0, l_m_size);
    cudaMemset(m, 0, l_m_size); // We'll set to -INF later
    
    // Set m to -INFINITY using a simple kernel or cudaMemcpy
    float* m_host = new float[B * nh * N];
    for (int i = 0; i < B * nh * N; i++) {
        m_host[i] = -INFINITY;
    }
    cudaMemcpy(m, m_host, l_m_size, cudaMemcpyHostToDevice);
    delete[] m_host;

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d\n", max_sram_size, sram_size);

    // Set up grid and block dimensions
    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);    // Bc threads per block

    // Launch kernel
    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q, K, V, N, d, Tc, Tr, Bc, Br, softmax_scale, l, m, O
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    // Synchronize to make sure kernel completes
    cudaDeviceSynchronize();
    
    cudaFree(l);
    cudaFree(m);
    
    return O;
}