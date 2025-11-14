TARGET = benchmark_attn
MAIN_SOURCE = benchmark_attn.cu
CUDA_SOURCES = cuda/flash_attn.cu cuda/naive_attn.cu
SOURCES = $(MAIN_SOURCE) $(CUDA_SOURCES)

all:
	@echo "usage: make cudaX "
	@echo "         1 <= X <= 5"
	@echo "       make run (B=B nh=nh N=N d=d)"
	@echo "           B  = batch size"
	@echo "           nh = number of heads"
	@echo "           N  = sequence length"
	@echo "           d  = head dimension"
	@echo "       run_nsys profile"
	@echo "       run_nsight // only on compatable arch"

use_tmp_dir:
	@mkdir ~/tmp_dir
	@export TMPDIR=~/tmp_dir

cuda1:
	@make use_tmp_dir
	nvcc -o $(TARGET) $(SOURCES)

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
	./$(TARGET) $(B) $(nh) $(N) $(d)
 
run_nsys:
	nsys profile --stats=true ./$(TARGET)

run_nsight:
	ncu --kernel-id ::flash_attn_forward_kernel:4 ./$(TARGET)

clean:
	rm -rf ~/tmp_dir $(TARGET)

clean-all:
	rm -rf ~/tmp_dir $(TARGET) *.sqlite *.nsys-rep
