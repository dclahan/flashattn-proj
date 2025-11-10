#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

#include "cuda/flash_attn.h"
#include "cuda/naive_attn.h"

const int B = 1;  // batch size
const int nh = 8; // number of heads
const int N = 4096; // sequence length
const int d = 1024; // head dimension

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
    for (int i = 0; i < size; ++i) {
        if (abs(result1[i] - result2[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << result1[i] << " vs " << result2[i] << std::endl;
            return false;
        }
    }
    return true;
}

void benchmark_attention() {
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
}

int main(int argc, char* argv[]) {
    std::cout << "CUDA Attention Benchmarking Tool" << std::endl;

    print_gpu_info();
    
    benchmark_attention();

    return 0;
}