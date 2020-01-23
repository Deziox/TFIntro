import torch
from torch import nn,optim
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


class AE(nn.Module):
    def __init__(self,i=(28*28),n_features=3,o=(28*28)):
        super(AE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(i,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16,n_features)
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,o),
            nn.Sigmoid() #compress to range (0,1)
        )

    def reparameterize(self,mu,logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self,x):
        encoded = self.encoder(x)
        mu = encoded
        logvar = encoded
        z = self.reparameterize(mu,logvar)
        decoded = self.decoder(z)
        return mu,logvar,decoded

def loss_function(decoded,x,mu,logvar):
    BCE = F.binary_cross_entropy(decoded,x.view(-1,784),reduction='sum')
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Hyper Parameters
epochs = 30
BATCH_SIZE = 64
LR = 0.001
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

# Data Loader for easy mini-batch return in training, the image batch shape will be (64, 1, 28, 28)
trainloader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

model = AE(n_features=12)
optimizer = optim.Adam(model.parameters(),LR)
criterion = nn.BCELoss()

plt.ion()

width=10
height=10
fig = plt.figure(figsize=(8,8))
for epoch in range(epochs):
    total = 0
    print('epoch {}/{}'.format(epoch+1,epochs))
    for j,(x,y) in enumerate(trainloader):
        xi = x.view(-1,28*28)
        xo = x.view(-1,28*28)
        mu,logvar,decoded = model.forward(xi)

        l = loss_function(decoded,x,mu,logvar)
        optimizer.zero_grad()
        l.backward()
        total += l.item()
        optimizer.step()

        if j % 100 == 0:
            print('\tLoss: {}'.format(l.item()))
            #print('\tFeatures: {}'.format(encoded[0].detach().numpy().reshape(-1,8,8)))
            for i in range(1,8*8+1):
                fig.add_subplot(8,8,i)
                plt.imshow(decoded[i-1].detach().view(28,28))
            plt.title('{}'.format(str(y[0].item())))
            plt.draw();
            plt.pause(0.001)
    print('Avg loss: {}'.format(total/len(trainloader.dataset)))

plt.ioff()
plt.show()