import torch
from torch import nn,optim
import numpy as np

input_dim = 6
cells = []
lsize1 = 4 #4 hidden nodes
lsize2 = 5 #5 hidden nodes

cell1 = nn.LSTMCell(input_dim,lsize1)
cell2 = nn.LSTMCell(lsize1,lsize2)
cells.append([cell1,cell2])

sample_input = torch.tensor([[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],
                             [[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]],dtype=torch.float32)
print(sample_input.shape)
h1 = torch.zeros((sample_input.shape[0],lsize1))
h2 = torch.zeros((sample_input.shape[0],lsize2))
for i in sample_input:
    y1,h1 = cell1(i)
    y2,h2 = cell2(y1)
    print(y2)
