import math
import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('CartPole-v0').unwrapped

plt.ion()

Transition = namedtuple('Transition',('state','action','next_state','reward'))

class ReplayMemory(object):

    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self,*args):
        """Saves a transition"""
        if(len(self.memory) < self.capacity):
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        #loop around to the front of the memory list
        self.position = (self.position + 1) % self.capacity

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)
