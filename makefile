default: compileLib
	echo ""
	python3 main.py
	echo ""
	
compileLib: cudaDigitalAnnealing.cu
	nvcc --compiler-options -fPIC -shared -o ./lib/cudaDA.so cudaDigitalAnnealing.cu

debug: cudaDigitalAnnealing.cu
	nvcc -g -G -o coreDump cudaDigitalAnnealing.cu
	cuda-gdb ./coreDump

