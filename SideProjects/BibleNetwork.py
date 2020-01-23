from gensim.models import Word2Vec
import pandas as pd
import multiprocessing
import string
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Masking,Embedding
from keras_preprocessing.text import Tokenizer,one_hot



df = pd.read_csv('t_asv.csv')
df['t'] = [i.translate(str.maketrans('', '', string.punctuation)).split() for i in df['t']]

#print(df['t'].tolist())

t = Tokenizer(lower=False)
t.fit_on_texts(df['t'].tolist())

word2idx = t.word_index
idx2word = {v:k for k,v in word2idx.items()}

sequences = t.texts_to_sequences(df['t'].tolist())
print(sequences[0])
print([idx2word[i] for i in sequences[0]])

features = []
labels = []

training_length = 3

# Iterate through the sequences of tokens
for seq in sequences:

    # Create multiple training examples from each sequence
    for i in range(training_length, len(seq)):
        # Extract the features and label
        extract = seq[i - training_length:i + 1]

        # Set the features and label
        features.append(extract[:-1])
        labels.append(extract[-1])

features = np.array(features)

print(features[0:3])
print(labels[0:3])

label_array = np.zeros((len(features),len(word2idx)),dtype=np.int)
print(label_array)

for i,word_idx in enumerate(labels):
    label_array[i,word_idx-1] = 1



EMB_DIM = 300
w2v = Word2Vec(df['t'].tolist(),size=EMB_DIM,window=3,min_count=1,negative=15,iter=10,workers=multiprocessing.cpu_count())
word_vectors = w2v.wv
vocab = list(word_vectors.vocab.keys())

model = Sequential()
model.add(Embedding(input_dim=EMB_DIM))
model.add(LSTM(64,input_shape=features.shape[1:]))
