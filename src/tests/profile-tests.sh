# run nsight or nsys profiler for best speedup configs, and normal regular config

ncu --kernel-id ::flash_attn_forward_kernel:4 ./benchmark_attn 32 16 64 64 1