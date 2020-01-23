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

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(16,kernel_size=(16,16),input_shape=(128,128,3)))
model.add(tf.keras.layers.Conv2D(32,kernel_size=(16,16)))
model.add(tf.keras.layers.Conv2D(64,kernel_size=(32,32)))
model.add(tf.keras.layers.Conv2D(64,kernel_size=(32,32)))
model.add(tf.keras.layers.Conv2D(32,kernel_size=(32,32)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dense(2,activation='softmax'))
model.compile(optimizer='adam',loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])
print(model.summary())

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])

negPath = './Week5/Capstone/concrete_crack_images_for_classification/Negative'
posPath = './Week5/Capstone/concrete_crack_images_for_classification/Positive'
negative = [os.path.join(negPath,file) for file in os.listdir(negPath) if file.endswith('.jpg')]
positive = [os.path.join(posPath,file) for file in os.listdir(posPath) if file.endswith('.jpg')]
negative.sort()
positive.sort()

negative = [(n,0) for n in negative]
positive = [(p,1) for p in positive]

trainset = []
testset = []

dataset = [item for sublist in zip(negative,positive) for item in sublist]
random.shuffle(dataset)

for i in tqdm(range(len(dataset)),desc="dataset img formatting",total=40000):
    image = Image.open(dataset[i][0])
    image.thumbnail((128,128))
    dataset[i] = (np.array(image),dataset[i][1])
    #print(dataset[i][0].shape)

trainset,valset = train_test_split(dataset,train_size=0.75)
#print(trainset[0])
#print(valset[0])
xtrain = []
ytrain = []
for x,y in trainset:
    xtrain.append(x)
    ytrain.append(y)

xtrain = np.asarray(xtrain)
ytrain = np.asarray(ytrain)

print(type(xtrain))
print(type(ytrain))
print(xtrain.shape)
print(ytrain.shape)
model.fit(xtrain,ytrain,batch_size=100,epochs=5)