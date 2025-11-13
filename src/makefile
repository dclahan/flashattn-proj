TARGET = benchmark_attn
MAIN_SOURCE = benchmark_attn.cu
CUDA_SOURCES = cuda/flash_attn.cu cuda/naive_attn.cu
SOURCES = $(MAIN_SOURCE) $(CUDA_SOURCES)


all:
	@echo "usage: make cudaX "
	@echo "        1 <= X <= 5"
	@echo "       run"
	@echo "       run_nsight // only on compatable arch"

use_tmp_dir:
	mkdir ~/tmp_dir
	export TMPDIR=~/tmp_dir

cuda1:
	make use_tmp_dir
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_35 -code=sm_35

cuda2:
	make use_tmp_dir
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_75 -code=sm_75

cuda3:
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_70 -code=sm_70

cuda4:
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_52 -code=sm_52

cuda5:
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_89 -code=sm_89

run:
	./benchmark_attn

run_nsight:
	ncu ./benchmark_attn

clean:
	rm -f ~/tmp_dir 

# 	nvcc -arch=compute_75 -code=sm_75  ./benchmark_attn.cu ./cuda/flash_attn.cu ./cuda/naive_attn.cu -o benchmark_attn


# NVCC = nvcc
# CXX = g++
# TARGET = benchmark_attn

# # Source files
# MAIN_SOURCE = benchmark_attn.cu
# CUDA_SOURCES = cuda/flash_attn.cu cuda/naive_attn.cu
# ALL_SOURCES = $(MAIN_SOURCE) $(CUDA_SOURCES)

# # Object files
# MAIN_OBJ = $(MAIN_SOURCE:.cu=.o)
# CUDA_OBJS = $(CUDA_SOURCES:.cu=.o)
# ALL_OBJS = $(MAIN_OBJ) $(CUDA_OBJS)

# # Default architecture (will be overridden by detect-gpu)
# ARCH ?= sm_70

# # Compiler flags
# NVCC_FLAGS = -std=c++14 -O3 -arch=$(ARCH)
# CXX_FLAGS = -std=c++14 -O3

# # Include paths
# INCLUDES = -I.

# # Nsight compute flags
# NSIGHT_FLAGS = -lineinfo -src-in-ptx

# # GPU Architecture mapping
# GPU_ARCHS = \
# 	cuda1.cims.nyu.edu:sm_35 \
# 	cuda2.cims.nyu.edu:sm_75 \
# 	cuda3.cims.nyu.edu:sm_70 \
# 	cuda4.cims.nyu.edu:sm_52 \
# 	cuda5.cims.nyu.edu:sm_89

# # Default target
# all: detect-gpu $(TARGET)

# # Detect GPU architecture based on hostname
# detect-gpu:
# 	@hostname | grep -q "cims.nyu.edu" && \
# 	HOST=$$(hostname) || \
# 	HOST=local; \
# 	ARCH=$$(echo "$(GPU_ARCHS)" | grep "$$HOST:" | cut -d: -f2); \
# 	if [ -z "$$ARCH" ]; then \
# 		echo "Unknown host $$HOST, using default arch sm_70"; \
# 		ARCH=sm_70; \
# 	else \
# 		echo "Detected $$HOST, using arch $$ARCH"; \
# 	fi; \
# 	$(MAKE) $(TARGET) ARCH=$$ARCH

# # Link all object files
# $(TARGET): $(ALL_OBJS)
# 	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(ALL_OBJS)

# # Compile main benchmark file
# benchmark_attn.o: benchmark_attn.cu
# 	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# # Compile CUDA kernel files
# cuda/%.o: cuda/%.cu
# 	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# # Individual GPU targets
# cuda1.cims.nyu.edu: NVCC_FLAGS += -arch=compute_35 -code=sm_35
# cuda1.cims.nyu.edu: $(TARGET)

# cuda2.cims.nyu.edu: NVCC_FLAGS += -arch=compute_75 -code=sm_75
# cuda2.cims.nyu.edu: $(TARGET)

# cuda3.cims.nyu.edu: NVCC_FLAGS += -arch=compute_70 -code=sm_70
# cuda3.cims.nyu.edu: $(TARGET)

# cuda4.cims.nyu.edu: NVCC_FLAGS += -arch=compute_52 -code=sm_52
# cuda4.cims.nyu.edu: $(TARGET)

# cuda5.cims.nyu.edu: NVCC_FLAGS += -arch=compute_89 -code=sm_89
# cuda5.cims.nyu.edu: $(TARGET)

# # Debug build
# debug: NVCC_FLAGS += -g -G
# debug: $(TARGET)

# # Clean
# clean:
# 	rm -f $(TARGET) $(ALL_OBJS)

# # Run benchmark
# run: $(TARGET)
# 	./$(TARGET)

# # Profile with basic CUDA events (works on all GPUs)
# profile-basic: $(TARGET)
# 	./$(TARGET)

# # Profile with Nsight Systems (only on supported GPUs)
# profile-nsys: $(TARGET)
# 	@if command -v nsys > /dev/null 2>&1; then \
# 		nsys profile --stats=true ./$(TARGET); \
# 	else \
# 		echo "Nsight Systems not available, using basic profiling"; \
# 		./$(TARGET); \
# 	fi

# # Profile with Nsight Compute (only on supported GPUs)
# profile-ncu: $(TARGET)
# 	@if command -v nv-nsight-cu-cli > /dev/null 2>&1; then \
# 		nv-nsight-cu-cli --kernel-id ::flash_attn_forward_kernel:1 ./$(TARGET); \
# 	else \
# 		echo "Nsight Compute not available"; \
# 	fi

# .PHONY: all clean run profile-nsys profile-ncu detect-gpu cuda1.cims.nyu.edu cuda2 cuda3 cuda4 cuda5 batch-run
