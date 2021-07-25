import torch
from torch._C import device
from utils import get_device
import numpy as np
device = get_device()

shape = (2,5)
data = np.random.random(10).reshape(shape)
tensor =  torch.tensor(data, dtype=torch.float32, requires_grad=False, device=device )

print("data\n", data)
print( "create tensor:\n", torch.rand(10, device=device))
print( "create 2d tensor:\n", torch.tensor(data, dtype=torch.float32, requires_grad=False, device=device ))
print( "create tensor frm numpy array:\n", torch.from_numpy(np.array( data )))
print( "new tensor from another tensor:\n", torch.ones_like(tensor))
print( "random tensor from another tensor:\n", torch.rand_like( tensor, dtype=torch.float16))
print( "random tensor with given shape:\n", torch.rand( shape))
print( "one tensor with given shape:\n", torch.ones( shape))
print( "zero tensor with given shape:\n", torch.zeros( shape))

# Operation on tensors
ones = torch.rand(shape, device=device)
print("data:\n ", ones)
print( "first row:\n", ones[0])
print( "second columns:\n", ones[:, 1])
print( "last column:\n", ones[..., -1])
cat_tensor = torch.cat([ones, ones, ones], dim=0)
print(" concat multiple tensors row wise:\n", cat_tensor)
print(" concat tensor shape:\n", cat_tensor.shape)

# arithmetic operations
matrix = torch.tensor( np.array([[1,2], [3,4]]), dtype=torch.float16, device=device)
print( "matrix:\n", matrix)
mat_mul = matrix.mul(matrix)
print( "element-wise product\n", mat_mul)
mat_mul = matrix.matmul(matrix)
print( "matrix multiplication between two tensor\n", mat_mul)
agg = mat_mul.sum()
single_element = agg.item()
print( single_element )
print( "matrix inplace opertion:\n", matrix.add_(2))

