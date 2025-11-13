#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <math_constants.h>

#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y)) 

__global__ void matrix_transpose(const float*, float* , int, int);
__global__ void matrix_multiply(const float*, const float*, float*, int, int, int);
__global__ void array_divide(float*, float, int);
__global__ void matrix_softmax(float*, int, int);
float* flash_forward(float* , float* , float* , int , int , int , int );
int naive_attention(const float* , const float* , const float* , float* , int , int , int );

const int B = 16;  // batch size
const int nh = 12; // number of heads
const int N = 64;  // sequence length
const int d = 64;  // head dimension

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
    float* O
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
    }
}

float* flash_forward(
        float* Q, float* K, float* V, 
        int B, int nh, int N, int d
    ) {
    // cudaDeviceProp prop;
    // cudaGetDeviceProperties(&prop, 0);
    // // const int Bc = std::ceil(prop.sharedMemPerBlock/4*d);
    // const int Bc = min(ceil(prop.sharedMemPerBlock/sizeof(float)/(4*d)), (float)N);
    // const int Br = min(Bc,d);
    const int Bc = 32;
    const int Br = 32;

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
    flash_attn_forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q, K, V, N, d, Tc, Tr, Bc, Br, softmax_scale, l, m, O
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    
    // Synchronize to make sure kernel completes
    cudaDeviceSynchronize();
    
    cudaFree(l);
    cudaFree(m);
    
    return O;
}

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


void print_gpu_info() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "=== GPU Information ===" << std::endl;
    std::cout << "Name: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Registers per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "Warp Size: " << prop.warpSize << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Number of SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "=================================" << std::endl << std::endl;
}

void initialize_matrices(float* Q, float* K, float* V, int size_Q, int size_KV) {
    for (int i = 0; i < size_Q; ++i) {
        Q[i] = ((i * 7) % 31) * 0.1f - 1.5f;
    }
    for (int i = 0; i < size_KV; ++i) {
        K[i] = ((i * 11) % 29) * 0.1f - 1.3f;
        V[i] = ((i * 5) % 37) * 0.1f - 1.1f;
    }
}

bool validate_results(float* result1, float* result2, int size, float tolerance = 1e-4f) {
    for (int i = 0; i < size; i++) {
        if (std::abs(result1[i] - result2[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << result1[i] << " vs " << result2[i] << std::endl;
            return false;
        }
    }
    return true;
}

int benchmark_attention() {
    std::cout << "=== Attention Mechanism Benchmark ===" << std::endl;
    std::cout << "Configuration: B=" << B << ", nh=" << nh << ", N=" << N << ", d=" << d << std::endl;
    std::cout << "Total Q size: " << (B * nh * N * d * sizeof(float)) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << std::endl;

    // Allocate host memory
    size_t Q_size = B * nh * N * d;
    size_t KV_size = B * nh * N * d;  // Assuming same sequence length for K, V
    
    float *h_Q = new float[Q_size];
    float *h_K = new float[KV_size];
    float *h_V = new float[KV_size];
    float *h_O_naive = new float[Q_size];
    float *h_O_flash = new float[Q_size];

    // Initialize matrices
    initialize_matrices(h_Q, h_K, h_V, Q_size, KV_size);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_O_naive, *d_O_flash;
    cudaMalloc(&d_Q, Q_size * sizeof(float));
    cudaMalloc(&d_K, KV_size * sizeof(float));
    cudaMalloc(&d_V, KV_size * sizeof(float));
    cudaMalloc(&d_O_naive, Q_size * sizeof(float));
    cudaMalloc(&d_O_flash, Q_size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_Q, h_Q, Q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, KV_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, KV_size * sizeof(float), cudaMemcpyHostToDevice);

    // Warm-up runs
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 3; ++i) {
        naive_attention(d_Q, d_K, d_V, d_O_naive, B * nh * N, B * nh * N, d);
        float* flash_result = flash_forward(d_Q, d_K, d_V, B, nh, N, d);
        if (i == 0) cudaFree(flash_result);  // Free after first warm-up
    }
    cudaDeviceSynchronize();

    // Benchmark Naive Attention
    std::cout << "Benchmarking Naive Attention..." << std::endl;
    int num_runs = 10;
    std::vector<float> naive_times;
    
    for (int i = 0; i < num_runs; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        naive_attention(d_Q, d_K, d_V, d_O_naive, B * nh * N, B * nh * N, d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        naive_times.push_back(milliseconds);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Copy naive result back for validation
    cudaMemcpy(h_O_naive, d_O_naive, Q_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Benchmark Flash Attention
    std::cout << "Benchmarking Flash Attention..." << std::endl;
    std::vector<float> flash_times;
    
    for (int i = 0; i < num_runs; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        float* flash_result = flash_forward(d_Q, d_K, d_V, B, nh, N, d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        flash_times.push_back(milliseconds);
        
        // Copy result on last run for validation
        if (i == num_runs - 1) {
            cudaMemcpy(h_O_flash, flash_result, Q_size * sizeof(float), cudaMemcpyDeviceToHost);
        }
        
        cudaFree(flash_result);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Calculate statistics
    auto calculate_stats = [](const std::vector<float>& times) {
        float min_time = *std::min_element(times.begin(), times.end());
        float max_time = *std::max_element(times.begin(), times.end());
        float avg_time = 0;
        for (float t : times) avg_time += t;
        avg_time /= times.size();
        return std::make_tuple(min_time, max_time, avg_time);
    };

    auto [naive_min, naive_max, naive_avg] = calculate_stats(naive_times);
    auto [flash_min, flash_max, flash_avg] = calculate_stats(flash_times);

    // Print results
    std::cout << std::endl;
    std::cout << "=== RESULTS ===" << std::endl;
    std::cout << "Naive Attention:" << std::endl;
    std::cout << "  Min: " << std::fixed << std::setprecision(3) << naive_min << " ms" << std::endl;
    std::cout << "  Max: " << naive_max << " ms" << std::endl;
    std::cout << "  Avg: " << naive_avg << " ms" << std::endl;
    
    std::cout << "Flash Attention:" << std::endl;
    std::cout << "  Min: " << flash_min << " ms" << std::endl;
    std::cout << "  Max: " << flash_max << " ms" << std::endl;
    std::cout << "  Avg: " << flash_avg << " ms" << std::endl;
    
    std::cout << std::endl;
    std::cout << "Speedup: " << std::setprecision(2) << (naive_avg / flash_avg) << "x" << std::endl;

    // Validate results
    std::cout << std::endl;
    std::cout << "=== VALIDATION ===" << std::endl;
    if (validate_results(h_O_naive, h_O_flash, std::min(1000, (int)Q_size), 1e-3f)) {
        std::cout << "✓ Results match within tolerance" << std::endl;
    } else {
        std::cout << "✗ Results don't match!" << std::endl;
    }

    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O_naive;
    delete[] h_O_flash;
    
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O_naive);
    
    std::cout << std::endl;
    std::cout << "=== PROFILING WITH NSIGHT ===" << std::endl;
    std::cout << "To profile with Nsight Systems:" << std::endl;
    std::cout << "  nsys profile --stats=true ./benchmark_attention" << std::endl;
    std::cout << std::endl;
    std::cout << "To profile with Nsight Compute:" << std::endl;
    std::cout << "  nv-nsight-cu-cli --kernel-id ::flash_attn_forward_kernel:1 ./benchmark_attention" << std::endl;
    std::cout << "  nv-nsight-cu-cli --kernel-id ::matrix_multiply:1 ./benchmark_attention" << std::endl;
    return 0;
}

int main(int argc, char* argv[]) {
    std::cout << "CUDA Attention Benchmarking Tool" << std::endl;

    print_gpu_info();
    
    benchmark_attention();

    return 0;
}
