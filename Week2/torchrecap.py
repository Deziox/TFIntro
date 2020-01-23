import torch
from torch import nn,optim
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self,layers):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(layers[0],layers[1],kernel_size=5,padding=2)
        self.conv1_bn = nn.BatchNorm2d(layers[1])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(layers[1],layers[2],kernel_size=5,padding=2)
        self.conv2_bn = nn.BatchNorm2d(layers[2])
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.fc1 = nn.Linear(layers[2] * 4 * 4,layers[3])
        self.fc1_bn = nn.BatchNorm1d(layers[3])

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = torch.relu(x)
        x1 = self.pool1(x)

        x = self.conv2(x1)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x2 = self.pool2(x)
        x = x2.view(x2.size(0),-1)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        return x,x1[0],x2[0]

composed = transforms.Compose([transforms.Resize((16,16)),transforms.ToTensor()])
trainset = dsets.MNIST(root='./data', train=True, download=True, transform=composed)
valset = dsets.MNIST(root='./data', train=False, download=True, transform=composed)
trainloader = DataLoader(trainset,batch_size=100)
valloader = DataLoader(valset,batch_size=5000)


def trainer_brock(model,trainloader,valloader,optimizer,criterion,epochs=4):
    ACC = []
    LOSS = []
    for epoch in range(epochs):
        CONE = []
        CTWO = []
        print('epoch',epoch+1,'/',epochs)
        model.train()
        total = 0
        for x,y in trainloader:
            print(x.shape)
            yhat,c1,c2 = model(x)
            print(yhat.shape,y.shape)
            CONE.append(c1)
            CTWO.append(c2)
            l = criterion(yhat,y)
            total += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        print('\tLoss:',total)
        LOSS.append(total)


        correct = 0
        for x,y in valloader:
            z = model(x)[0]
            _,labels = torch.max(z,1)
            correct += (labels == y).sum().item()
        accuracy = correct/len(valset)
        print('\tAccuracy:',accuracy)
        ACC.append(accuracy)

        '''for i,j in enumerate(CONE[0],1):
            plt.imshow(j.detach())
            plt.title(('Conv1 ' + str(i)))
            plt.show()'''

    return ACC,LOSS

model = CNN([1,16,32,10])
optimizer = optim.Adam(model.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()
A,L = trainer_brock(model,trainloader,valloader,optimizer,criterion,epochs=10)
plt.plot(A)
plt.plot(L)
plt.show()

