#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "naive_attn.h"

// ****************************
// GLOBALS
// ****************************
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y)) 

const int N = 4096;
const int M = 4096;
const int d = 1024;

// ****************************
// FUNCTION DECLARATIONS
// ****************************
__global__ void matrix_transpose(const float*, float* , int, int);
__global__ void matrix_multiply(const float*, const float*, float*, int, int, int);
__global__ void array_divide(float*, float, int)
__global__ void matrix_softmax(float*, int, int)
void naive_attention(const float*, const float*, const float*, float*, int, int, int)

// ****************************
// DRIVER FUNCTION
// TODO: 
// move this main to main.cpp or main.c 
// so can call naive attn and flash from same main and benchmark
// THIS FILE should just be declarations and definitions
// ****************************
// int main(int argc, char* argv[]) {
//     float *Q, *K, *V, *O;
//     // ****************************************************************************************
//     float *Qd, *Kd, *Vd, *Od;
//     int QO = N * d * sizeof(float)
//     int KV = M * d * sizeof(float)
//     Q = (float *)malloc(QO);
//     if(!Q){
//         printf("Cannot allocate matrix a with %d elements\n", QO);
//         exit(1);	
//     }
//     K = (float *)malloc(KV);
//     if(!K){
//         printf("Cannot allocate matrix a with %d elements\n", KV);
//         exit(1);	
//     }
//     V = (float *)malloc(KV);
//     if(!V){
//         printf("Cannot allocate matrix a with %d elements\n", KV);
//         exit(1);	
//     }
//     O = (float *)malloc(QO);
//     if(!O){
//         printf("Cannot allocate matrix a with %d elements\n", QO);
//         exit(1);	
//     }
//     cudaMalloc((void **)&Qd, QO);
//     if(!Qd){
//         printf("Cannot CuAllocate matrix a with %d elements\n", QO);
//         exit(1);	
//     }
//     cudaMalloc((void **)&Kd, KV);
//     if(!Kd){
//         printf("Cannot CuAllocate matrix a with %d elements\n", KV);
//         exit(1);	
//     }
//     cudaMalloc((void **)&Vd, KV);
//     if(!Vd){
//         printf("Cannot CuAllocate matrix a with %d elements\n", KV);
//         exit(1);	
//     }
//     cudaMalloc((void **)&Od, QO);
//     if(!Od){
//         printf("Cannot CuAllocate matrix a with %d elements\n", QO);
//         exit(1);	
//     }
//     // ****************************************************************************************
//     // cudaMallocManaged(&Q, N * d * sizeof(float));
//     // cudaMallocManaged(&K, M * d * sizeof(float));
//     // cudaMallocManaged(&V, M * d * sizeof(float));
//     // cudaMallocManaged(&O, N * d * sizeof(float));
    
//     // initialize pseudo-random test matrices
//     for (int row = 0; row < N; ++row) {
//         for (int col = 0; col < d; ++col) {
//             Q[row * d + col] = ((row * 7 + col * 13) % 31) * 0.1f - 1.5f;
//         }
//     }
//     for (int row = 0; row < M; ++row) {
//         for (int col = 0; col < d; ++col) {
//             K[row * d + col] = ((row * 11 + col * 17) % 29) * 0.1f - 1.3f;
//             V[row * d + col] = ((row * 5 + col * 19) % 37) * 0.1f - 1.1f;
//         }
//     }
    
//     // prefetch matrices to gpu
//     // cudaMemLocation loc;
//     // loc.type = cudaMemLocationTypeDevice;
//     // loc.id = 0;
//     // cudaMemPrefetchAsync(Q, N * d * sizeof(float), loc, 0);
//     // cudaMemPrefetchAsync(K, M * d * sizeof(float), loc, 0);
//     // cudaMemPrefetchAsync(V, M * d * sizeof(float), loc, 0);
//     // cudaMemPrefetchAsync(O, N * d * sizeof(float), loc, 0);
    
//     cudaMemcpy(Qd, Q, QO, cudaMemcpyHostToDevice);
//     cudaMemcpy(Kd, K, KV, cudaMemcpyHostToDevice);
//     cudaMemcpy(Vd, V, KV, cudaMemcpyHostToDevice);
    
//     naive_attention(Qd, Kd, Vd, Od, N, M, d);
    
//     cudaMemcpy(O, Od, QO, cudaMemcpyDeviceToHost);
    
//     // // write output to file
//     if (argc > 1) {
//         std::ofstream file(argv[1]);
//         if (!file) {
//             std::cerr << "Error opening file: " << argv[1] << std::endl;
//         }
//         for (int i = 0; i < N * d; i++) {
//             file << std::fixed << std::setprecision(6) << O[i];
//             if (i < N * d - 1) {
//                 file << " ";
//             }
//         }
//         file.close();
//         std::cout << "Output written to file: " << argv[1] << std::endl;
//     }
    
//     cudaFree(Qd);
//     cudaFree(Kd);
//     cudaFree(Vd);
//     cudaFree(Od);
// }

// ****************************
// function definitions
// ****************************

// matrix transpose
__global__ void matrix_transpose(
    const float* src,
    float* dst,
    int N, int M
) {
    int row = blockID.x * blockDim.x + threadIdx.x;
    int col = blockID.y * blockDim.y + threadIdx.y;
    if (row >= N || col > M) return;

    dst[col*N+row] = src[row*M + col];
}

// naive matmul
// TODO: change to Cuda optimized "hello world" cuMatMul
__global__ void matrix_multiply(
    const float* A,
    const float* B,
    float* C,
    int N, int M, int d
) {
    int row = blockID.x * blockDim.x + threadIdx.x;
    int col = blockID.y * blockDim.y + threadIdx.y;
    if (row >= N || col > M) return;
    
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        sum += A[row * d + i] * B[i * M + col];
    }
    C[row * M + col] = sum;
}

// divide elems of array by value in place
__global__ void array_divide(
    float* array,
    float value,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        array[idx] /= value;
    }
}

// compute safe softmax
__global__ void matrix_softmax(
    float* matrix,
    int N, int M
) {
    int row = blockID.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // find m
    float max = -CUDART_INF_F;
    for (int col = 0; col < M; ++col) {
        max = fmaxf(max, matrix[row * M + col]);
    }

    float sum_exp = 0.0f;
    for (int col = 0; col < M; ++col) {
        int idx = row * M + col;
        // subtract m for numerical stability
        matrix[idx] = expf(matrix[idx] - max);
        sum_exp += matrix[idx];
    }

    for (int col = 0; col < M; ++col) {
        matrix[row * M + col] /= sum_exp;
    }
}

// 1 naive_attention pass
void naive_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N, int M, int d
) {
    float *K_T, *scores;
    cudaMalloc(&K_T, d * M * sizeof(float));
    cudaMalloc(&scores, N * M * sizeof(float));

    dim3 blockDim1(32, 32);
    dim3 gridDim1(CEIL_DIV(M, blockDim1.x), CEIL_DIV(d, blockDim1.y));
    matrix_transpose<<<gridDim1, blockDim1>>>(K, K_T, M, d);
    cudaDeviceSynchronize();

    dim3 blockDim2(32, 32);
    dim3 gridDim2(CEIL_DIV(N, blockDim2.x), CEIL_DIV(M, blockDim2.y));
    matrix_multiply<<<gridDim2, blockDim2>>>(Q, K_T, scores, N, M, d);
    cudaDeviceSynchronize();

    dim3 blockDim3(1024);
    dim3 gridDim3(CEIL_DIV(N * M, blockDim3.x));
    array_divide<<<gridDim3, blockDim3>>>(scores, sqrtf((float)d), N * M);
    cudaDeviceSynchronize();

    dim3 blockDim4(1024);
    dim3 gridDim4(CEIL_DIV(N, blockDim4.x));
    matrix_softmax<<<gridDim4, blockDim4>>>(scores, N, M);
    cudaDeviceSynchronize();

    dim3 blockDim5(32, 32);
    dim3 gridDim5(CEIL_DIV(N, blockDim5.x), CEIL_DIV(d, blockDim5.y));
    matrix_multiply<<<gridDim5, blockDim5>>>(scores, V, O, N, d, M);
    cudaDeviceSynchronize();

    cudaFree(K_T);
    cudaFree(scores);
}
