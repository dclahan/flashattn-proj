all:
	@echo "usage: make cudaX "
	@echo "        1 <= X <= 5"
	@echo "       run"
	@echo "       run_nsight // only on compatable arch"

cuda1:

cuda2:

cuda3:

cuda4:

cuda5:
	@echo argument is $(argument)

run:
	./benchmark_attn

run_nsight:
	ncu ./benchmark_attn