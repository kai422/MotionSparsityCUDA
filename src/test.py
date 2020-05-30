import torch
import add_cpp

a = torch.rand(2)
b = torch.rand(2)
c = torch.rand(2)

out = add_cpp.AddCPU(a, b, c)
add_cpp.printCPU(out)