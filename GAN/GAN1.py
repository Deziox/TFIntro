import torch
from torch import nn,optim
from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision import datasets as dsets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n

class Discriminator(nn.Module):
    def __init__(self,drop_prob=0.3):
        super(Discriminator,self).__init__()
        n_features = 784
        n_out = 1
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features,1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(drop_prob)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(drop_prob)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(drop_prob)
        )
        self.out = nn.Sequential(
            nn.Linear(256,n_out),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        n_features = 100
        n_out = 784
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
        )
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

def trainer_brock(model,optimizer,real_data,fake_data,criterion):
    N = real_data.size(0)
    optimizer.zero_grad()
    prediction_real = model(real_data)
    prediction_fake = model(fake_data)

    error_real = criterion(prediction_real,Variable(torch.ones(N,1)))
    error_fake = criterion(prediction_fake,Variable(torch.zeros(N,1)))
    error_real.backward()
    error_fake.backward()

    optimizer.step()
    return error_fake + error_real,prediction_real,prediction_fake

def trainer_ash(model,discriminator,optimizer,fake_data,criterion):
    N = fake_data.size(0)
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    loss = criterion(prediction,Variable(torch.ones(N,1)))
    loss.backward()
    optimizer.step()
    return loss

composed = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((.5,),(.5,))
])
data = dsets.MNIST(root='./root',train=True,transform=composed,download=True)
trainloader = DataLoader(data,batch_size=100,shuffle=True)
num_batches = len(trainloader)
discriminator = Discriminator()
generator = Generator()

d_optimizer = optim.Adam(discriminator.parameters(),lr=0.0001)
g_optimizer = optim.Adam(generator.parameters(),lr=0.0001)
criterion = nn.BCELoss()

test_noise = noise(16)
print(test_noise)

epochs = 100
plt.ion()
for epoch in range(epochs):
    print('epoch {}/{}'.format(epoch+1,epochs))
    for i,(x,y) in enumerate(trainloader):
        N = x.size(0)

        #Training Discriminator
        real_data = Variable(x.view(N,784))
        fake_data = generator(noise(N)).detach()
        d_err,d_pred_real,d_pred_fake = trainer_brock(discriminator,d_optimizer,real_data,fake_data,criterion)

        #Training Generator
        fake_data = generator(noise(N))
        g_err = trainer_ash(generator,discriminator,g_optimizer,fake_data,criterion)

        if(i%10==0):
            test = generator(test_noise).view(-1,1,28,28).data
            plt.imshow(test[0][0])
            plt.draw();
            plt.pause(0.0001)

plt.ioff()
plt.show()

