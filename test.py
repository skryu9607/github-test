import numpy as np
row = 4
col = 5
g = [[1,1] * col for _ in range(row)]
g_array = np.asarray(g)
g = g.reshape(-1,5)
print(g)
print(g_array.shape) 


