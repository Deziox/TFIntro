import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/deziox/PycharmProjects/TFIntro/Week1/FuelConsumptionCo2.csv')
print(df.keys())
X = np.asanyarray(df[['ENGINESIZE']])
y = np.asanyarray(df[['CO2EMISSIONS']])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

X_train = tf.constant(X_train,dtype=tf.float32)
y_train = tf.constant(y_train,dtype=tf.float32)
X_test = tf.constant(X_train,dtype=tf.float32)
y_test = tf.constant(y_train,dtype=tf.float32)

weights = tf.random.normal((1,1))
bias = 0

w = tf.Variable(20.0)
b = tf.Variable(30.2)

@tf.function
def f(x):
    return (w * x) + b

@tf.function
def loss(yhat,ytarget):
    return tf.reduce_mean(tf.square(yhat-ytarget))

@tf.function
def train(dataset,optimizer,batch_size=1):
    epoch_loss = []
    for i in range(len(dataset)):
        x = dataset[i][0]
        y = dataset[i][1]

        with tf.GradientTape() as tape:
            yhat = f(x)
            l = loss(yhat,y)

        dJ_dW = tape.gradient(l,w)
        dJ_dB = tape.gradient(l, b)
        tf.print(dJ_dW,dJ_dB)



optimizer = tf.optimizers.Adam(0.05)
loss_values = []
train_data = []
epochs = 100
trainset = tf.stack([X_train,y_train],axis=1).numpy()
train(trainset,optimizer=optimizer)