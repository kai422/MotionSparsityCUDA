import torch
import x2_cpp

a = torch.rand((2, 2))
print(a)
print(x2_cpp.x2(a))
