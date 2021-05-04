import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# initialize tensor directly on some list of lists
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device, requires_grad=True)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# other initialize tensor methods
X = torch.empty(size=(3, 3))
print(X)

X = torch.zeros(size=(3, 3))
print(X)

X = torch.rand(size=(3, 3))
print(X)

X = torch.ones(size=(3, 3))
print(X)

X = torch.eye(3, 5)
print(X)

X = torch.arange(start=0, end=5, step=1)
print(X)

X = torch.linspace(start=0.1, end=1, steps=10)
print(X)

X = torch.empty(size=(2, 5)).normal_(mean=0, std=1)
print(X)

X = torch.empty(size=(2, 5)).uniform_(0, 1)
print(X)

X = torch.diag(torch.ones(3))
print(X)

# initialize tensors and convert to other dtypes
tensor = torch.arange(4)
print(tensor.dtype)
print(tensor.bool())
print(tensor.short())
print(tensor.long())
print(tensor.half())
print(tensor.float())
print(tensor.double())

# conversion tensor and numpy arrays

np_array = np.zeros((3, 3))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()
print(np_array == np_array_back)
