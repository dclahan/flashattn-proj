#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <torch/types.h>

// TODO: implement scaled dot product attention (softmax(Q @ K^T * softmax_scale) @ V)

// const int Bc = min(ceil(prop.sharedMemPerBlock/sizeof(float)/(4*d)), (float)N);
'''
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    // const int Bc = std::ceil(prop.sharedMemPerBlock/4*d);
    const int Bc = min(ceil(prop.sharedMemPerBlock/sizeof(float)/(4*d)), (float)N);
    const int Br = std::min(Bc,d);
'''

// TODO: move to like a `flash_attn.h` file?
__global__ void flash_attn_forward_kernel(
    const float *, const float *, const float *, 
    const int, const int, const int, 
    const int, const int, const int, 
    const float, float*, float*, float*);
// __global__ void flash_attn_backward_kernel();


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
        for (int x = 0; x < d; x++){
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();
        for (int i = 0; i < Tr; i++){
            // load Qi, Oi (? why), li, mi to sram

            // compute Sij = softmax_scale * Qi * Kj^T

            // compute Sij_masked = mask(Sij)

            // 12. compute
            // m_tilde_ij = rowmax(Sij_masked)
            // P_tilde_ij = exp(Sij_masked - m_tilde_ij)
            // l_tilde_ij = rowsum(P_tilde_ij)

            // 13. compute mi_new, li_new

            // 14. compute P_tilde_ij_dropped

            // 15. Write Oi to global mem
            // Oi := 

            // 16. Write li <- li_new, mi <- mi_new to global mem (HBM)
        }
        __syncthreads(); 
        // reasoning QUESTION -> why does this need to be here?
        // Kj Vj dont rely on li and mi do they?
    }
}