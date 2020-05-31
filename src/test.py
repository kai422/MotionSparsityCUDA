import torch
import add_cpp
import MSBackend


b = torch.rand(2)
a = torch.zeros(2,2,256,256)

quadtree = MSBackend.CreateFromDense(a)
print(quadtree)
print(type(quadtree))