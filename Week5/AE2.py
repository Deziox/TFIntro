import torch
from torch import nn,optim
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm,image
import numpy as np
import PIL
from PIL import Image


class AE(nn.Module):
    def __init__(self,i=(28*28),n_features=3,o=(28*28)):
        super(AE,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(i, 784),
            nn.ReLU(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_features),

        )
        self.decoder = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.ReLU(),
            nn.Linear(784, o),
            nn.Sigmoid()  # compress to range (0,1)
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded

# Hyper Parameters
epochs = 200
BATCH_SIZE = 10
LR = 0.0001
DOWNLOAD_MNIST = True
N_TEST_IMG = 5

#128x128x4
img = Image.open('/Users/deziox/PycharmProjects/TFIntro/Week5/flower_images/0001.png')
images = []
for i in range(1,210):
    data = image.imread('/Users/deziox/PycharmProjects/TFIntro/Week5/flower_images/{}.png'.format(str(i).zfill(4))).reshape(4,128,128)
    data = torch.tensor(data.reshape(4,(128*128)),dtype=torch.float32)
    images.append(data)
# plot one example
#print(train_data.data.size())     # (60000, 28, 28)
#print(train_data.targets.size())   # (60000)
'''plt.imshow(train_data.data[2].numpy(), cmap='gray')
plt.title('%i' % train_data.targets[2])
plt.show()'''

trainloader = Data.DataLoader(dataset=images, batch_size=BATCH_SIZE, shuffle=True)
for x in trainloader:
    print(x.shape)

model = AE(i=(128*128),n_features=3,o=(128*128))
optimizer = optim.Adam(model.parameters(),LR)
criterion = nn.MSELoss()
#encoded,decoded = model.forward(data)
#plt.imshow(decoded.reshape(128,128,-1).detach())
#plt.show()
plt.ion()

for epoch in range(epochs):
    print('epoch {}/{}'.format(epoch+1,epochs))
    for j,x in enumerate(trainloader):
        xi = x.view(-1,4,128*128)
        xo = x.view(-1,4,128*128)
        encoded,decoded = model.forward(xi)

        l = criterion(decoded,xo)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        if j % 10 == 0:
            print('\tLoss: {}'.format(l.item()))
            #print('\tFeatures: {}'.format(encoded[0].detach().numpy()))
            #print(decoded.shape)
            plt.imshow(decoded[0].detach().view(128,128,-1))
            plt.title('\tLoss: {}'.format(l.item()))
            plt.draw();
            plt.pause(0.0001)

plt.ioff()
plt.show()