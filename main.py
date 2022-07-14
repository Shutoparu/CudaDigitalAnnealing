import numpy as np
import numpy.ctypeslib as ctplib
from ctypes import c_float, c_int, cdll, POINTER
import time

np.random.seed(1)

dim = 1050
sweeps = 100000
qubo = 2 * np.random.rand(dim,dim).astype(np.float32) - 1
qubo = (qubo + qubo.T) / 2 
qubo = qubo.flatten()
binary = np.ones(dim, dtype=np.int32)

# # test code
# dim = 5
# sweeps=10000
# qubo = np.array([[-.1,.2,-.3,.4,-.5],[.2,.3,-.4,.5,.6],[-.3,-.4,-.5,-.6,-.7],[.4,.5,-.6,.7,.8],[-.5,.6,-.7,.8,-.9]]).astype(np.float32)
# # qubo = qubo.flatten()
# binary = np.array([-1,-1,1,-1,-1])
# # test code

binary = ctplib.as_ctypes(binary)
qubo = ctplib.as_ctypes(qubo)

cudaDA = cdll.LoadLibrary("./lib/cudaDA.so")

main = cudaDA.digitalAnnealingPy

main.argtypes = [POINTER(c_int), POINTER(c_float), c_int, c_int]
main.restype = c_float

start = time.time()
energy = main(binary, qubo, dim, sweeps)
end = time.time()

binary = ctplib.as_array(binary)

print(energy)
print(binary)
print("spent time: ", end-start)

# # test code
# binary = np.expand_dims(binary, axis=1)
# print( - binary.T @ qubo @ binary)
# # test code