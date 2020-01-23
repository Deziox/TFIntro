import torch
from torch import nn,optim
from gensim.models import Word2Vec
import tokenize

file = open('ptb.train.txt','r')
train = ""
for i in file:
    train += i

train = train.split()

#tokenization later
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
seq_size = 4
training = [([train[i:i+seq_size]],train[i+seq_size]) for i in range(len(train)-seq_size)]
print(training[:3])
vocab = set(train)
word_to_idx = {word: i for i,word in enumerate(vocab)}
idx_to_word = {value:key for key,value in word_to_idx.items()}
print(idx_to_word)
print(word_to_idx)

class NLP(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,context_size):
        super(NLP,self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)

        self.lstm1 = nn.LSTM(embedding_dim,hidden_dim[0])
        self.lstm2 = nn.LSTM(hidden_dim[0],hidden_dim[1])

        self.linear1 = nn.Linear(hidden_dim[1],context_size)

    def forward(self,x):
        embeds = self.embeddings(x)
        #print(embeds.shape)
        lstm_out1,_ = self.lstm1(embeds.view(len(x),1,-1))
        #print(lstm_out1.shape)
        lstm_out2,_ = self.lstm2(lstm_out1)
        #print(lstm_out2.shape)
        fc1 = self.linear1(lstm_out2.view(len(x),-1))
        soft_max = torch.log_softmax(fc1,dim=1)
        #print(soft_max.shape)
        return soft_max

model = NLP(10000,200,[256,128],10000)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)
epochs = 15
for context,target in training:
    context_idxs = torch.tensor([word_to_idx[w] for w in context[0]],dtype=torch.long)
    test = torch.argmax(model(context_idxs),dim=1)
    yhat = torch.tensor(test[seq_size-1].reshape(-1,1),dtype=torch.float32,requires_grad=True)
    l = criterion(yhat,torch.tensor([word_to_idx[target]],dtype=torch.float32))
    optimizer.zero_grad()
    l.backward()
    nn.utils.clip_grad_norm_(model.parameters(),5)
    optimizer.step()
    print('loss:',l.item())
