TARGET = benchmark_attn
MAIN_SOURCE = benchmark_attn.cu
CUDA_SOURCES = cuda/flash_attn.cu cuda/naive_attn.cu
SOURCES = $(MAIN_SOURCE) $(CUDA_SOURCES)


all:
	@echo "usage: make cudaX "
	@echo "        1 <= X <= 5"
	@echo "       run"
	@echo "       run_nsight // only on compatable arch"

cuda1:
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_35 -code=sm_35

cuda2:
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