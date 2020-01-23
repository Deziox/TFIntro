from PIL import Image
from matplotlib.pyplot import imshow
import pandas
import matplotlib.pyplot as plt
import os
import glob
import tensorflow as tf
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torch import optim

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])
    plt.show()

negPath = './Week5/Capstone/concrete_crack_images_for_classification/Negative'
posPath = './Week5/Capstone/concrete_crack_images_for_classification/Positive'
negative = [os.path.join(negPath,file) for file in os.listdir(negPath) if file.endswith('.jpg')]
positive = [os.path.join(posPath,file) for file in os.listdir(posPath) if file.endswith('.jpg')]
negative.sort()
positive.sort()
number_of_samples = (len(positive) + len(negative))

Y = torch.zeros([number_of_samples],dtype=torch.float)
Y[::2] = 1
all_files = []
for i in tqdm(range(int(number_of_samples/2)),desc="all pics"):
    all_files.append(positive[i])
    all_files.append(negative[i])