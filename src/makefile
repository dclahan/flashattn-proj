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

# Default architecture (will be overridden by detect-gpu)
ARCH ?= sm_70

# Compiler flags
NVCC_FLAGS = -std=c++14 -O3 -arch=$(ARCH)
CXX_FLAGS = -std=c++14 -O3

# Include paths
INCLUDES = -I.

# Nsight compute flags
NSIGHT_FLAGS = -lineinfo -src-in-ptx

# GPU Architecture mapping
GPU_ARCHS = \
	cuda1:sm_35 \
	cuda2:sm_75 \
	cuda3:sm_70 \
	cuda4:sm_52 \
	cuda5:sm_89

# Default target
all: detect-gpu $(TARGET)

# Detect GPU architecture based on hostname
detect-gpu:
	@hostname | grep -q "cims.nyu.edu" && \
	HOST=$$(hostname) || \
	HOST=local; \
	ARCH=$$(echo "$(GPU_ARCHS)" | grep "$$HOST:" | cut -d: -f2); \
	if [ -z "$$ARCH" ]; then \
		echo "Unknown host $$HOST, using default arch sm_70"; \
		ARCH=sm_70; \
	else \
		echo "Detected $$HOST, using arch $$ARCH"; \
	fi; \
	$(MAKE) $(TARGET) ARCH=$$ARCH

# Link all object files
$(TARGET): $(ALL_OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(ALL_OBJS)

# Compile main benchmark file
benchmark_attn.o: benchmark_attn.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA kernel files
cuda/%.o: cuda/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Individual GPU targets
cuda1: NVCC_FLAGS += -arch=sm_35
cuda1: $(TARGET)

cuda2: NVCC_FLAGS += -arch=sm_75
cuda2: $(TARGET)

cuda3: NVCC_FLAGS += -arch=sm_70
cuda3: $(TARGET)

cuda4: NVCC_FLAGS += -arch=sm_52
cuda4: $(TARGET)

cuda5: NVCC_FLAGS += -arch=sm_89
cuda5: $(TARGET)

# Debug build
debug: NVCC_FLAGS += -g -G
debug: $(TARGET)

# Clean
clean:
	rm -f $(TARGET) $(ALL_OBJS)

# Run benchmark
run: $(TARGET)
	./$(TARGET)

# Profile with basic CUDA events (works on all GPUs)
profile-basic: $(TARGET)
	./$(TARGET)

# Profile with Nsight Systems (only on supported GPUs)
profile-nsys: $(TARGET)
	@if command -v nsys > /dev/null 2>&1; then \
		nsys profile --stats=true ./$(TARGET); \
	else \
		echo "Nsight Systems not available, using basic profiling"; \
		./$(TARGET); \
	fi

# Profile with Nsight Compute (only on supported GPUs)
profile-ncu: $(TARGET)
	@if command -v nv-nsight-cu-cli > /dev/null 2>&1; then \
		nv-nsight-cu-cli --kernel-id ::flash_attn_forward_kernel:1 ./$(TARGET); \
	else \
		echo "Nsight Compute not available"; \
	fi

.PHONY: all clean run profile-nsys profile-ncu detect-gpu cuda1 cuda2 cuda3 cuda4 cuda5 batch-run