TARGET = benchmark_attn
MAIN_SOURCE = benchmark_attn.cu
CUDA_SOURCES = cuda/flash_attn.cu cuda/naive_attn.cu
SOURCES = $(MAIN_SOURCE) $(CUDA_SOURCES)

all:
	@echo "usage: make cudaX "
	@echo "        1 <= X <= 5"
	@echo "       make run (K2=K2 DB=DB)"
	@echo "       run_nsight // only on compatable arch"

use_tmp_dir:
	@mkdir ~/tmp_dir
	@export TMPDIR=~/tmp_dir

cuda1:
	@make use_tmp_dir
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_35 -code=sm_35

cuda2:
	@make use_tmp_dir
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_75 -code=sm_75

cuda3:
	@make use_tmp_dir
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_70 -code=sm_70

cuda4:
	@make use_tmp_dir
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_52 -code=sm_52

cuda5:
	@make use_tmp_dir
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_89 -code=sm_89

run:
	./benchmark_attn $(K2) $(DB)

run_nsight:
	ncu ./benchmark_attn

clean:
	rm -rf ~/tmp_dir $(TARGET)

