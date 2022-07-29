from main import DA
import numpy as np

fileName = "./data/size5/size5_rt0.44_0099.txt"
dim = 1136
qubo = np.ndarray((dim, dim))
with open(fileName, 'r') as f:
    for line in f:
        id1, id2, val = line.split(' ')
        qubo[int(id1)][int(id2)] = float(val)
        qubo[int(id2)][int(id1)] = float(val)

da = DA(qubo)
da.run()
print(da.energy)
print(da.time)
