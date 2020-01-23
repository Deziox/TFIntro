import torch
from torch import nn,optim
import torch.nn.functional as F
import numpy as np
import tqdm
import os

def one_hot_encode(arr,n_labels):
    one_hot = np.zeros((np.multiply(*arr.shape),n_labels),dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]),arr.flatten()] = 1
    one_hot = one_hot.reshape((*arr.shape,n_labels))
    return one_hot

def get_batches(arr,batch_size,seq_length):
    #N batches by M sequences
    batch_size_total = batch_size*seq_length
    #K total number of batches
    n_batches = len(arr)//batch_size_total
    #keep enough batches to make full batches
    arr = arr[:n_batches*batch_size_total]
    #reshape data to batch_size of rows
    arr = arr.reshape((batch_size,-1))

    for i in range(0,arr.shape[1],seq_length):
        x = arr[:,i:i+seq_length]
        y = np.zeros_like(x)
        try:
            y[:,:-1] ,y[:,-1] = x[:,1:],arr[:,i+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x,y

class CBLSTM(nn.Module):
    def __init__(self,tokens,n_hidden=612,n_layers=4,drop_prob=0.5,lr=0.001):
        super(CBLSTM,self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        #creating token dictionaries
        self.chars = chars
        self.idx_to_char = dict(enumerate(self.chars))
        self.char_to_idx = {char:val for val,char in self.idx_to_char.items()}

        #lstms
        self.lstm1 = nn.LSTM(len(self.chars),n_hidden,n_layers,dropout=drop_prob,batch_first=True)
        self.dropout = nn.Dropout(self.drop_prob)
        self.fc1 = nn.Linear(n_hidden,len(self.chars))

    def forward(self,x,h):
        y,h = self.lstm1(x,h)
        y = self.dropout(y)

        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        y = y.contiguous().view(-1,self.n_hidden)
        y = self.fc1(y)
        return y,h

    def init_hidden(self,batch_size):
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        w = next(self.parameters()).data
        h = (w.new(self.n_layers,batch_size,self.n_hidden).zero_(),
             w.new(self.n_layers,batch_size,self.n_hidden).zero_())
        return h

def trainer_brock(model,data,epochs=10,batch_size=10,seq_length=50,lr=0.001,clip=5,val_frac=0.1,print_every=10):
    '''
    clip: gradient clipping
    val_frac: Fraction of data to hold out for validation
    print_every: Number of steps for printing training and validation loss
    '''
    model.train()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()

    validex = int(len(data)*(1-val_frac))
    data,valdata = data[:validex],data[validex:]

    counter = 0
    n_chars = len(model.chars)
    for epoch in range(epochs):
        #init hidden states
        h = model.init_hidden(batch_size)
        print('epoch {}/{}'.format(epoch+1,epochs))
        for x,y in tqdm.tqdm(get_batches(data,batch_size,seq_length)):
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x,n_chars)
            inputs,targets = torch.from_numpy(x),torch.from_numpy(y)

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([i.data for i in h])
            #model.zero_grad()
            optimizer.zero_grad()

            yhat,h = model(inputs,h)
            l = criterion(yhat,targets.view(batch_size*seq_length))
            l.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(),clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for x, y in get_batches(valdata, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y

                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size * seq_length))

                    val_losses.append(val_loss.item())

                model.train()  # reset to train mode after iterating through validation data

                print("Epoch: {}/{}...".format(epoch + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(l.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))

with open('../ptb.train.txt','r') as file:
    text = file.read()

chars = tuple(set(text))
idx_to_char = dict(enumerate(chars))
char_to_idx = {char:idx for idx,char in idx_to_char.items()}
encoded = np.array([char_to_idx[i] for i in text])
print(encoded)

n_hidden=512
n_layers=4

model = CBLSTM(chars,n_hidden,n_layers)
#print(model)

batch_size = 100
seq_length = 100
n_epochs = 10

trainer_brock(model,encoded,n_epochs,batch_size,seq_length,lr=0.001)
torch.save(model.state_dict(),'cblstm1.plk')