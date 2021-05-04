import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)
print(z1.shape)

z2 = torch.add(x, y)
print(z2)

z3 = x + y
print(z3)

# subtraction
z4 = x - y
print(z4)

# division
z5 = torch.true_divide(x, y)
print(z5)

# multiplication
z = x * y
print(z)

# inplace operations
t = torch.zeros(3)
t.add_(x)
print(t)

# exponentiation
z = x.pow(2)
print(z)

z = x ** 2
print(z)

# simple conparison
z = x > 0
print(z)

# matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
print(x3)

x3 = x1.mm(x2)
print(x3)

# matrix exponentiation
matrix_exp = torch.rand((5, 5))
matrix_exp.matrix_power(3)
print(matrix_exp)

# dot product
z = torch.dot(x, y)
print(z)

# batch matrix multiplication
batch = 32
n = 10
m = 20
p = 30
tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)
print(out_bmm.shape)

# broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))
z = x1 - x2
print(z)

z = x1 ** x2
print(z)

# other useful operations
sum_x = torch.sum(x, dim=0)
print(sum_x)

# values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
print(x)
print(values, indices)

abs_x = torch.abs(x)
print(abs_x)

z = torch.argmax(x, dim=0)
print(z)
z = torch.argmin(x, dim=0)
print(z)

mean_x = torch.mean(x.float(), dim=0)
print(mean_x)

z = torch.eq(x, y)
print(z)

sorted_y, indices = torch.sort(y, dim=0, descending=False)
print(sorted_y, indices)

z = torch.clamp(x, min=2, max=10)
print(z)

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
z = torch.any(x)
print(z)

z = torch.all(x)
print(z)
