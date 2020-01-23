import gym
import random
import numpy as np
import torch
from torch import nn,optim
from torch.nn import functional as F

lr = 1e-5
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if(done):
                break

#some_random_games_first()

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation,reward,done,info = env.step(action)

            if(len(prev_observation) > 0):
                game_memory.append([prev_observation,action])

            prev_observation = observation
            score += reward
            if(done):
                break

        if(score >= score_requirement):
            accepted_scores.append(score)
            for data in game_memory:
                if(data[1] == 0):
                    output = [1,0]
                elif(data[1] == 1):
                    output = [0,1]
                training_data.append([data[0],output])
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy',training_data)
    print('Average accepted score: {}'.format(np.mean(accepted_scores)))
    print(accepted_scores)
    return training_data

class Network(nn.Module):
    def __init__(self,i):
        super(Network,self).__init__()
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(i,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8,2)

    def forward(self,x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.fc4(x)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.fc5(x)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.fc6(x)
        #x = nn.Softmax(dim=1)(x)
        return x

def trainer_may(traindata,model,optimizer,criterion,epochs):
    X = torch.tensor(np.array([i[0] for i in traindata]),dtype=torch.float32).view(-1,1,len(traindata[0][0]))
    Y = torch.tensor([i[1] for i in traindata])

    for epoch in range(epochs):
        print("epoch {}/{}".format(epoch+1,epochs))
        LOSS = []
        total = 0
        for x,y in list(zip(X,Y)):
            print(x)
            yhat = model.forward(x).view(-1,2)
            y = torch.argmax(torch.tensor(y,dtype=torch.float32),dim=0).view(1)
            print(yhat,y,yhat.shape,y.shape)
            l = criterion(yhat,y)
            print(l)
            LOSS.append(l)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()


traindata = initial_population()

model = Network(len(traindata[0][0]))
optimizer = optim.Adam(model.parameters(),lr)
criterion = nn.CrossEntropyLoss()
trainer_may(traindata,model,optimizer,criterion,100)