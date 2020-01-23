import tensorflow as tf

import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data[:], iris.target[:]
y = pd.get_dummies(y).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

numFeatures = X_train.shape[1]
numLabels = y_train.shape[1]

epochs = 700

weights = tf.Variable(tf.random.normal([numFeatures, numLabels], mean=0, stddev=0.01, name='weights'),dtype=tf.float32)
biases = tf.Variable(tf.random.normal([numLabels], mean=0, stddev=0.01, name="biases"),dtype=tf.float32)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16,input_shape=(4,),activation='relu'),
    tf.keras.layers.Dense(3,activation='softmax')
])
batch_size = 10
epochs = 200

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test,y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

print(tf.argmax(model.predict(X_test),axis=1) == tf.argmax(y_test,axis=1))
