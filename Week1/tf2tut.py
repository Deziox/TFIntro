import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist
(X_train,y_train),(X_test,y_test) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover','Dress', 'Coat', 'Sandal','Shirt','Sneaker','Bag','Ankle boot']

X_train = X_train/255.0
X_test = X_test/255.0

plt.imshow(X_train[0])
plt.show()

#784 128 10
