#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:37:17 2019

@author: carlysle
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def SeriesGen(N):
    x = torch.arange(0, 6.28*N, 6.28/24)
    return torch.sin(x)
 
def trainDataGen(seq, k, pre_next):
    dat = list()
    L = len(seq)
    for i in range(1, L-1-pre_next):
        indat = seq[i-1:i+k]
        outdat = seq[i+pre_next:i+pre_next+1]
        dat.append((indat,outdat))
    return dat
 
def ToVariable(x):
    tmp = torch.FloatTensor(x)
    return Variable(tmp)

N = 7
y = SeriesGen(N)
plt.scatter(np.arange(0, len(y)), y.numpy())
dat = trainDataGen(y.numpy(),2, 24)


class LSTMpred(nn.Module):
 
    def __init__(self,input_size,hidden_dim):
        super(LSTMpred,self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_size, hidden_dim)
        self.hidden2out = nn.Linear(hidden_dim, 1)
        self.hidden = self.init_hidden()
 
    def init_hidden(self):
        return (Variable(torch.zeros(1, 1, self.hidden_dim)),
                Variable(torch.zeros(1, 1, self.hidden_dim)))
 
    def forward(self,seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)
        outdat = self.hidden2out(lstm_out.view(len(seq),-1))
        return outdat
 

model = LSTMpred(1,8)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    print("Epoch: {}".format(epoch))
    for seq, outs in dat[0:24*(N-1)]:
        seq = ToVariable(seq)
        outs = ToVariable(outs)
        optimizer.zero_grad()
        
        model.hidden = model.init_hidden()
 
        modout = model(seq)
        
        #out = modout.data.numpy()
        #plt.plot(out)
        
        loss = loss_function(modout, outs)
        # print(loss)
        loss.backward()
        optimizer.step()
    #plt.show()
predDat = []
# predDat.append(0)
TrueVal = []
for seq, out in dat[0:25]:
    seq = ToVariable(seq)
    out = ToVariable(out)
    #print(model(seq).size())
    Ff = model(seq)[-1].data.numpy()
    Tt = out.data.numpy()
    #plt.plot(Ff)
    #print(Ff)
    print(Tt)
    predDat.append(Ff[0])
    TrueVal.append(Tt[0])
    

fig = plt.figure()
plt.plot(TrueVal)
# plt.show()
plt.plot(predDat)
plt.show()
