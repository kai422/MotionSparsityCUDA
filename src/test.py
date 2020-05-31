import torch
import add_cpp
import MotionSparsityBackend


b = torch.rand(2)
a = torch.zeros(2,2,256,256)

quadtree = MotionSparsityBackend.CreateFromDense(b)