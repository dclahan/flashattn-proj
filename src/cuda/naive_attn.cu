#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math_constants.h>


#include "naive_attn.h"

// ****************************
// GLOBALS
// ****************************
#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y)) 

// ****************************
// FUNCTION DECLARATIONS
// ****************************
__global__ void matrix_transpose(const float*, float* , int, int);
__global__ void matrix_multiply(const float*, const float*, float*, int, int, int);
__global__ void array_divide(float*, float, int);
__global__ void matrix_softmax(float*, int, int);
int naive_attention(const float*, const float*, const float*, float*, int, int, int);

// matrix transpose
__global__ void matrix_transpose(
    const float* src,
    float* dst,
    int N, int M
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
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
int naive_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N, int M, int d
) {
    float *K_T, *scores;
    cudaMalloc(&K_T, d * M * sizeof(float));
    cudaMalloc(&scores, N * M * sizeof(float));

    printf("Starting Naive Attention\n");

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
    return 0;
}
