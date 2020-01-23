import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets as dsets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

class RBM(nn.Module):
    def __init__(self,n_visible=784,n_hidden=500,k=5):
        super(RBM,self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden,n_visible))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.k = k

    def sample_from_p(self,p):
        #return torch.relu(torch.sign(p - Variable(torch.rand(p.size()))))
        return torch.relu(p - Variable(torch.rand(p.size())))

    def v_to_h(self,v):
        p_h = torch.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h

    def h_to_v(self,h):
        p_v = torch.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v

    def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)

        h_ = h1
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)

        #return original visible input and final pass visible probability distribution sample
        return v,v_

    def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

batch_size = 64
trainset = dsets.MNIST('./root',train=True,transform=transforms.ToTensor(),download=True)
valset = dsets.MNIST('./root',train=False,transform=transforms.ToTensor(),download=True)
trainloader = DataLoader(trainset,batch_size)
valloader = DataLoader(valset,batch_size)

model = RBM(k=1)
optimizer = optim.SGD(model.parameters(),lr=0.00002)
epochs = 20

for epoch in range(epochs):
    LOSS = []
    for x,y in trainloader:
        x = Variable(x.view(-1,784))
        sample_data = x.bernoulli()

        v1,v2 = model(sample_data)
        #print(v1,v2)
        l = model.free_energy(v1) - model.free_energy(v2)
        LOSS.append(l.item())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    plt.imshow(v1.view(32,1,28,28).data[0][0])
    plt.show()
    plt.imshow(v2.view(32, 1, 28, 28).data[0][0])
    plt.show()
    print('loss: {}'.format(np.abs(np.mean(LOSS))))

