NVCC = nvcc
CXX = g++
TARGET = benchmark_attn

# Source files
MAIN_SOURCE = benchmark_attn.cu
CUDA_SOURCES = cuda/flash_attn.cu cuda/naive_attn.cu
ALL_SOURCES = $(MAIN_SOURCE) $(CUDA_SOURCES)

# Object files
MAIN_OBJ = $(MAIN_SOURCE:.cu=.o)
CUDA_OBJS = $(CUDA_SOURCES:.cu=.o)
ALL_OBJS = $(MAIN_OBJ) $(CUDA_OBJS)

# Compiler flags
NVCC_FLAGS = -std=c++14 -O3 -arch=sm_70
CXX_FLAGS = -std=c++14 -O3

# Include paths
INCLUDES = -I.

# Nsight compute flags
NSIGHT_FLAGS = -lineinfo -src-in-ptx

# Default target
all: $(TARGET)

# Link all object files
$(TARGET): $(ALL_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(ALL_OBJS)

# Compile main benchmark file
benchmark_attn.o: benchmark_attn.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA kernel files
cuda/%.o: cuda/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Debug build
debug: NVCC_FLAGS += -g -G
debug: $(TARGET)

# Profile build
profile: NVCC_FLAGS += -pg
profile: $(TARGET)

# Clean
clean:
	rm -f $(TARGET) $(ALL_OBJS)

# Run benchmark
run: $(TARGET)
	./$(TARGET)

# Profile with Nsight Systems
profile-nsys: $(TARGET)
	nsys profile --stats=true ./$(TARGET)

# Profile with Nsight Compute
profile-ncu: $(TARGET)
	nv-nsight-cu-cli --kernel-id ::flash_attn_forward_kernel:1 ./$(TARGET)

.PHONY: all clean run profile-nsys profile-ncu