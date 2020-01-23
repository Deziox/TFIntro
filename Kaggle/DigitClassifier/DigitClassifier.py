import torch
from torch import nn,optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sub = pd.read_csv('MNISTsubmission.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = []
y_train = []
X_test = []
for i,row in tqdm(train.iterrows(),total=train.shape[0]):
    temp = row.to_numpy()
    X_train.append(temp[1:])
    y_train.append(temp[0])
X_train = torch.tensor(X_train,dtype=torch.float32).reshape(-1,1,28,28)
y_train = torch.tensor(y_train,dtype=torch.long)

for i,row in tqdm(test.iterrows(),total=test.shape[0]):
    temp = row.to_numpy()
    X_test.append(temp)
X_test = torch.tensor(X_test,dtype=torch.float32).reshape(-1,1,28,28)

trainloader = DataLoader(list(zip(X_train,y_train)),batch_size=1000)
# O = (W - K + 2*P)/S + 1
# 28-6+1 = 23 conv1
# 23 - 6 + 1 = 18 pool1

# 18 - 6 + 1 = 13 conv2
# 13 - 4 + 2*2 + 1 = 14 pool2

# (14 - 4)/2 + 1 = 6 conv3
# (6 - 3) + 1 pool3


class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.drop = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1,16,kernel_size=6)
        self.conv2 = nn.Conv2d(16,32,kernel_size=6)
        self.conv3 = nn.Conv2d(32,64,kernel_size=4,stride=2)
        self.fc4 = nn.Linear(64*4*4,32)
        self.fc5 = nn.Linear(32,10)

        self.pool1 = nn.AvgPool2d(kernel_size=6,padding=0,stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=4,padding=2,stride=1)
        self.pool3 = nn.AvgPool2d(kernel_size=3,padding=0,stride=1)

    def forward(self,x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.drop(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.drop(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.drop(x)

        x = self.fc4(x.view(x.size(0),-1))
        x = torch.relu(x)
        x = self.fc5(x)

        return x

def trainer_ash(model,trainloader,criterion,optimizer,epochs=10,valloader=None):
    L = []
    for epoch in range(1,epochs+1):
        LOSS = []
        print(f"epoch {epoch}/{epochs}")
        for x,y in tqdm(trainloader,total=len(trainloader)):
            yhat = model.forward(x)
            l = criterion(yhat,y)
            LOSS.append(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        print(f"\tAvg Loss: {sum(LOSS)/len(LOSS)}")
        L.append(sum(LOSS)/len(LOSS))

    return L
try:
    model = torch.load('model1.h5')
except FileNotFoundError:
    model = Network()
    optimizer = optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    losses = trainer_ash(model,trainloader,criterion,optimizer,epochs=10)
    print(sum(losses)/len(losses))
    torch.save(model,'model1.h5')
    plt.plot(losses)
    plt.show()

"""plt.ion()"""
testloader = DataLoader(X_test,batch_size=1)
for i,x in enumerate(tqdm(testloader,total=len(testloader))):
    yhat = torch.argmax(model(x))
    sub["Label"][i] = yhat.detach()

print(sub)

"""plt.imshow(x.reshape(28,28))
    plt.title(f"{torch.argmax(yhat)}")
    plt.draw()
    plt.pause(0.001)"""

"""plt.ioff()
plt.show()"""
pd.DataFrame.to_csv(sub,'Submission1.csv',index=False)