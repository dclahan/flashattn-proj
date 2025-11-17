# run nsight or nsys profiler for best speedup configs, and normal regular config

# everyone's best speedup was B=32 nh=16 N=64 d=64
ncu --kernel-id ::flash_attn_forward_kernel:4 ./benchmark_attn 32 16 64 64 1