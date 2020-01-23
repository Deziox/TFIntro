import numpy as np
import torch
from torch import nn,optim

LSTM_CELL_SIZE = 6

lstm_cell = nn.LSTMCell(LSTM_CELL_SIZE,LSTM_CELL_SIZE)
state = (torch.zeros([1,LSTM_CELL_SIZE],dtype=torch.float32),)*2

sample_input = torch.tensor([[3,2,2,2,2,2]],dtype=torch.float32)
ynew,hnew = lstm_cell(sample_input,state)
