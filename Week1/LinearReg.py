import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import matplotlib.patches as mpatches
tf.disable_v2_behavior()

plt.rcParams['figure.figsize'] = (10, 6)
X = np.arange(0.0, 5.0, 0.1)
w = 1
b = 0
y = w * X + b
'''plt.plot(X, Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()'''

df = pd.read_csv('Week1/FuelConsumptionCo2.csv')
X_train = np.asanyarray(df[['ENGINESIZE']])
y_train = np.asanyarray(df[['CO2EMISSIONS']])

graph = tf.Graph()
with graph.as_default():
    w = tf.Variable(20.0)
    b = tf.Variable(30.2)
    y = w * X_train + b

    loss = tf.reduce_mean(tf.square(y - y_train))
    optimizer = tf.train.GradientDescentOptimizer(0.05)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

with tf.Session(graph=graph) as sesh:
    sesh.run(init)
    loss_val = []
    train_data = []

    for epoch in range(100):
        _, l, w_param,b_param = sesh.run([train,loss,w,b])
        loss_val.append(l)
        print(epoch, l, w_param, b_param)
        train_data.append([w_param, b_param])

plt.plot(loss_val)
plt.show()

cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(X_train)
    line = plt.plot(X_train, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(X_train, y_train, 'ro')


green_line = mpatches.Patch(color='red', label='Data Points')

plt.legend(handles=[green_line])

plt.show()