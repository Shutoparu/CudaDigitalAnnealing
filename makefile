default: compileLib
	python3 main.py

compileLib: cudaDigitalAnnealing.cu
	nvcc --compiler-options -fPIC -shared -o ./lib/cudaDA.so cudaDigitalAnnealing.cu

debug: cudaDigitalAnnealing.cu
	nvcc -g -G -o coreDump cudaDigitalAnnealing.cu
	cuda-gdb ./coreDump

