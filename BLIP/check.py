import torch

num_i2t = torch.rand(2,4,4)
den_i2t = torch.rand(2,4,4)
print(num_i2t)
print("-----------")
print(den_i2t)
print(torch.exp(den_i2t))
a = torch.sum(torch.exp(den_i2t), dim=-1)

print(a.size())
print(a)
b = torch.log(torch.exp(num_i2t)/a.unsqueeze(-1))
print(b)
print(b.size())