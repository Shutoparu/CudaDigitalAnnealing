import os
import numpy as np
import numpy.ctypeslib as ctplib
from ctypes import c_void_p, c_double, c_int, cdll, POINTER


dim = 1500
sweeps = 100000
qubo = 2*np.random.rand(dim*dim)-1
binary = np.ones(dim, dtype = np.int32)

binary = ctplib.as_ctypes(binary)
qubo = ctplib.as_ctypes(qubo)

os.system("nvcc --compiler-options -fPIC -shared -o ./lib/cudaDA.so cudaDigitalAnnealing.cu")
cudaDA = cdll.LoadLibrary("./lib/cudaDA.so")

main = cudaDA.pythonEntry

main.argtypes = [POINTER(c_int), POINTER(c_double), c_int, c_int]
main(binary, qubo, dim, sweeps)

