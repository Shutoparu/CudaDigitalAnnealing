import os
import numpy as np
from ctypes import c_void_p, c_double, c_int, cdll

dim = 1500
sweeps = 100000
qubo = 2*np.random.rand(dim*dim)-1

os.system("nvcc --compiler-options -fPIC -shared -o ./lib/cudaDA.so cudaDigitalAnnealing.cu")

cudaDA = cdll.LoadLibrary("./lib/cudaDA.so")

main = cudaDA.pythonEntry

main()
