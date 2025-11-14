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
    int Nna, int Mna
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= Nna || col > Mna) return;

    dst[col * Nna + row] = src[row * Mna + col];
}

// naive matmul
// TODO: There is out of bounds error occurring, need to fix! and then test
__global__ void matrix_multiply(
    const float* A,
    const float* B,
    float* C,
    int rowsA, int colsB, int common
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= rowsA || col >= colsB) return;
    
    float sum = 0.0f;
    for (int i = 0; i < common; ++i) {
        sum += A[row * common + i] * B[i * colsB + col];
    }
    C[row * colsB + col] = sum;
}

// divide elems of array by value in place
__global__ void array_divide(
    float* array,
    float value,
    int Nna
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nna) {
        array[idx] /= value;
    }
}

// compute safe softmax
__global__ void matrix_softmax(
    float* matrix,
    int Nna, int Mna
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= Nna) return;

    // find Mna
    float max = -CUDART_INF_F;
    for (int col = 0; col < Mna; ++col) {
        max = fmaxf(max, matrix[row * Mna + col]);
    }

    float sum_exp = 0.0f;
    for (int col = 0; col < Mna; ++col) {
        int idx = row * Mna + col;
        // subtract Mna for numerical stability
        matrix[idx] = expf(matrix[idx] - max);
        sum_exp += matrix[idx];
    }

    for (int col = 0; col < Mna; ++col) {
        matrix[row * Mna + col] /= sum_exp;
    }
}

// 1 naive_attention pass
int naive_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int Nna, int Mna, int d
) {
    float *K_T, *scores;
    cudaMalloc(&K_T, d * Mna * sizeof(float));
    cudaMalloc(&scores, Nna * Mna * sizeof(float));

    printf("Starting Naive Attention\n");
    printf("Nna = %d, Mna = %d, d = %d\n", Nna, Mna, d);


    dim3 blockDim1(32, 32);
    dim3 gridDim1(CEIL_DIV(Mna, blockDim1.x), CEIL_DIV(d, blockDim1.y));
    matrix_transpose<<<gridDim1, blockDim1>>>(K, K_T, Mna, d);
    cudaDeviceSynchronize();


    printf("matmul 1\n");
    dim3 blockDim2(32, 32);
    dim3 gridDim2(CEIL_DIV(Mna, blockDim2.x), CEIL_DIV(Nna, blockDim2.y));
    matrix_multiply<<<gridDim2, blockDim2>>>(Q, K_T, scores, Nna, Mna, d);
    cudaDeviceSynchronize();
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    //     exit(-1);
    // }

    dim3 blockDim3(1024);
    dim3 gridDim3(CEIL_DIV(Nna * Mna, blockDim3.x));
    array_divide<<<gridDim3, blockDim3>>>(scores, sqrtf((float)d), Nna * Mna);
    cudaDeviceSynchronize();


    dim3 blockDim4(1024);
    dim3 gridDim4(CEIL_DIV(Nna, blockDim4.x));
    matrix_softmax<<<gridDim4, blockDim4>>>(scores, Nna, Mna);
    cudaDeviceSynchronize();


    printf("matmul 2\n");
    dim3 blockDim5(32, 32);
    dim3 gridDim5(CEIL_DIV(d, blockDim5.x), CEIL_DIV(Nna, blockDim5.y));
    matrix_multiply<<<gridDim5, blockDim5>>>(scores, V, O, Nna, d, Mna);
    cudaDeviceSynchronize();
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    //     exit(-1);
    // }

    cudaFree(K_T);
    cudaFree(scores);
    return 0;
}
