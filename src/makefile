TARGET = benchmark_attn
MAIN_SOURCE = benchmark_attn.cu
CUDA_SOURCES = cuda/flash_attn.cu cuda/naive_attn.cu
SOURCES = $(MAIN_SOURCE) $(CUDA_SOURCES)

all:
	@echo "usage: make cudaX "
	@echo "         2 <= X <= 5"
	@echo "       make run (B=B nh=nh N=N d=d)"
	@echo "           B  = batch size"
	@echo "           nh = number of heads"
	@echo "           N  = sequence length"
	@echo "           d  = head dimension"
	@echo "       make run_nsys"

use_tmp_dir:
	@mkdir ~/tmp_dir
	@export TMPDIR=~/tmp_dir

#nvcc V11.4
cuda1:
	@echo "on cuda1, use a newer GPU"

# nvcc V12.6
cuda2:
	@make use_tmp_dir
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_75 -code=sm_75

# nvcc V12.6
cuda3:
	@make use_tmp_dir
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_70 -code=sm_70

# nvcc V12.6
cuda4:
	@make use_tmp_dir
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_52 -code=sm_52

#nvcc V13.0
cuda5:
	@make use_tmp_dir
	nvcc -o $(TARGET) $(SOURCES) -arch=compute_89 -code=sm_89

run:
	./$(TARGET) $(B) $(nh) $(N) $(d)
 
run_nsys:
	nsys profile --stats=true ./$(TARGET)

clean:
	rm -rf ~/tmp_dir $(TARGET)

clean-all:
	rm -rf ~/tmp_dir $(TARGET) *.sqlite *.nsys-rep
