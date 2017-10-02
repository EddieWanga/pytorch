### list v.s numpy v.s pytorch-tensor v.s pytorch-variable
#
#   --------> numpy
#   |           ^
# list          |         
#   |           V               
#   -----> pytorch-tensor <---> pytorch-variable
#========================================
import torch
import numpy as np

list1 = [[3,2,1],[4,5,6]]

#========================================
### list -> numpy
###      -> pytorch-tensor
#----------------------------------------
# list > numpy:
arr1 = np.array(list1)
print(arr1)
# list > pytorch-tensor
tr1 = torch.FloatTensor(list1)
print(tr1)

#----------------------------------------
# numpy > pytorch-tensor
tr2 = torch.from_numpy(arr1) # dtype same as numpy
print(tr2)
# pytorch-tensor > numpy
arr2 = tr1.numpy()
print(arr2)

#----------------------------------------
# pytorch-tensor > pytorch-variable
from torch.autograd import Variable

var1 = Variable(tr1, requires_grad=True) # only variable can realize gradient, update.
print(var1)                              # e.g. weights

v_out = torch.mean(var1*var1) # var1^2
print(v_out)

v_out.backward()
print(var1.grad)

# pytorch-variable > pytorch-tensor
print(var1.data)
# pytorch-tensor > numpy
print(var1.data.numpy())