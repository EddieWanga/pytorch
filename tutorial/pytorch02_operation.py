import torch
import numpy as np

data = [-1, -2, -3, 1]
arr1 = np.array(data, dtype=np.float32)
tr1 = torch.from_numpy(arr1)

# abs
print("numpy abs:", np.abs(arr1))
print("torch abs:", torch.abs(tr1))

# mean
print("numpy mean:", np.mean(arr1))
print("torch mean:", torch.mean(tr1))

# linespace
print("numpy linespace:", np.linspace(-1, 1, 10))
print("torch linespace:", torch.linspace(-1, 1, 10))

#=============================================
data = [[1,2], [3,4]]
arr1 = np.array(data, dtype=np.float32)
tr1 = torch.from_numpy(arr1)

# matirx multiplication
print("numpy matmul:", np.matmul(arr1, arr1))
print("torch matmul:", torch.mm(tr1, tr1))

