import torch
from torch import nn,optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Activation,AvgPool2D,Flatten
tf.disable_v2_behavior()

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

X_train = (np.asarray(X_train,dtype=np.float)/255.0).reshape(-1,28,28,1)
y_train = np.asarray(y_train,dtype=np.long)

for i,row in tqdm(test.iterrows(),total=test.shape[0]):
    temp = row.to_numpy()
    X_test.append(temp)
X_test = (np.asarray(X_test,dtype=np.float)/255.0).reshape(-1,28,28,1)

# O = (W - K + 2*P)/S + 1
# 28-6+1 = 23 conv1
# 23 - 6 + 1 = 18 pool1

# 18 - 6 + 1 = 13 conv2
# 13 pool2

# (13 - 3)/2 + 1 = 6 conv3
# (6 - 3) + 1 pool3

model = Sequential()
conv1 = Sequential([
    Conv2D(16,kernel_size=3,input_shape=X_test.shape[1:],activation='relu'),
    AvgPool2D(pool_size=(2,2)),
    Dropout(0.2)
])

conv2 = Sequential([
    Conv2D(32,kernel_size=2,padding='same',activation='relu'),
    AvgPool2D(pool_size=(2,2),padding='same'),
    Dropout(0.2)
])

conv3 = Sequential([
    Conv2D(64,kernel_size=2,activation='relu'),
    AvgPool2D(pool_size=(3,3)),
    Dropout(0.2)
])

model.add(conv1)
model.add(conv2)
model.add(conv3)
model.add(Flatten())
model.add(Dense(64*2*2,activation='relu'))
model.add(Dense(10,activation='softmax'))
#print(model.summary())
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=100,epochs=10)