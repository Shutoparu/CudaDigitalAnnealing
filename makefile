default: compileLib
	python3 main.py
	
compileLib: cudaDigitalAnnealing.cu
	nvcc --compiler-options -fPIC -shared -o ./lib/cudaDA.so cudaDigitalAnnealing.cu

normal: cudaDigitalAnnealing.cu
	nvcc -o ./bin/cudaDA.o cudaDigitalAnnealing.cu
	./bin/cudaDA.o

debug: cudaDigitalAnnealing.cu
	nvcc -g -G -o coreDump cudaDigitalAnnealing.cu
	cuda-gdb ./coreDump
	rm ./coreDump

