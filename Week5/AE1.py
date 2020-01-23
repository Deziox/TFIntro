import torch
from torch import nn,optim
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


class AE(nn.Module):
    def __init__(self,i=(28*28),n_features=3,o=(28*28)):
        super(AE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(i,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, 72),
            nn.ReLU(),
            nn.Linear(72,n_features)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_features, 72),
            nn.ReLU(),
            nn.Linear(72,128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,o),
            nn.Sigmoid()#compress to range (0,1)
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded

# Hyper Parameters
epochs = 30
BATCH_SIZE = 64
LR = 0.0005
DOWNLOAD_MNIST = True
N_TEST_IMG = 5

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,                        # download it if you don't have it
)

# plot one example
print(train_data.data.size())     # (60000, 28, 28)
print(train_data.targets.size())   # (60000)
'''plt.imshow(train_data.data[2].numpy(), cmap='gray')
plt.title('%i' % train_data.targets[2])
plt.show()'''

x = train_data.data

# Data Loader for easy mini-batch return in training, the image batch shape will be (64, 1, 28, 28)
trainloader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

model = AE(n_features=16)
optimizer = optim.Adam(model.parameters(),LR)
criterion = nn.BCELoss()

plt.ion()

width=10
height=10
fig = plt.figure(figsize=(8,8))
for epoch in range(epochs):
    print('epoch {}/{}'.format(epoch+1,epochs))
    for j,(x,y) in enumerate(trainloader):
        xi = x.view(-1,28*28)
        xo = x.view(-1,28*28)
        encoded,decoded = model.forward(xi)

        l = criterion(decoded,xo)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        if j % 100 == 0:
            print('\tLoss: {}'.format(l.item()))
            print('\tFeatures: {}'.format(encoded[0].detach().numpy().reshape(-1,4,4)))

            fig.add_subplot(1,2,1)
            plt.imshow(decoded[0].detach().view(28,28))
            fig.add_subplot(1,2,2)
            plt.imshow(encoded[0].detach().view(4,4))
            plt.title('{}'.format(str(y[0].item())))
            plt.draw();

            plt.pause(0.001)

plt.ioff()
plt.show()