# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import os, sys
import glob
import string
import unicodedata
import random
import time
import math
import re
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, accuracy_score
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#----------------------------------------------------------------

def cudify(x, cuda):
    return x.cuda() if cuda else x

def init_tensor(t, *size):
    if t==1:
        return torch.randn(*size).float()
    x = torch.zeros(*size).float()
    if t==0:
        return x
    elif t==2:
        torch.nn.init.xavier_normal_(x)
    elif t==3:
        torch.nn.init.xavier_uniform_(x)
    return x

class CharacterEmbedding(nn.Module):
    def __init__(self, embedding_dim, vocab, batch_first=True, cuda=True):
        super(CharacterEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab = vocab
        self.batch_first = batch_first
        self._cuda = cuda
        self.embed = cudify(nn.Embedding(len(self.vocab), embedding_dim, padding_idx=0), cuda)
        self.cos = nn.CosineSimilarity(dim=2)

    def forward(self, X, L):
        # utils.rnn lets you give (B,L,D) tensors where B is the batch size, L is the maxlength, 
        # if you use batch_first=True.  Otherwise, give (L,B,D) tensors
        if not self.batch_first:
            X = X.transpose(0, 1) # (B,L,D) -> (L,B,D)

        # embed your sequences
        X = self.embed(X)

        # pack them up nicely
        X = pack_padded_sequence(X, L, batch_first=self.batch_first)

        return X
    
    def unpackToSequence(self, packed_output):
        output, _ = pad_packed_sequence(packed_output, batch_first=self.batch_first)
        words = self.unembed(output)
        return words

#----------------------------------------------------------------

def masked_softmax(x, mask, dim=1, epsilon=1e-14):
    exps = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps/masked_sums)

class Attention(nn.Module):
    def __init__(self, p, cuda=True):
        super(Attention, self).__init__()
        self._cuda = cuda
        self.attn_size =  next(x for x in p.out if x > 1)
        self.linear = cudify(nn.Linear(p.hidden_size * p.directions, self.attn_size, bias=False), cuda)
        self.q = nn.Parameter(cudify(self.new_parameter(self.attn_size, 1), cuda))
    
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def new_parameter(self, *size):
        out = torch.FloatTensor(*size)
        torch.nn.init.xavier_normal_(out)
        return out

    def forward(self, X, L):
        batch_size = X.shape[0]
        input_size = X.shape[1]
        hidden_size = X.shape[2]

        K = self.linear(X)
        K = F.relu(K)

        beta = torch.matmul(K, self.q).squeeze(dim=2)
        # beta = F.relu(beta)
        beta = torch.tanh(beta)

        mask = torch.sum(abs(X), dim=2)>0
        alpha = masked_softmax(beta, mask, dim=1)

        output = torch.sum(X * alpha.view(batch_size, input_size, 1), dim=1)
        return output

#----------------------------------------------------------------

class BaseRNN(nn.Module):
    def __init__(self, rnn, p, cuda=True):
        super(BaseRNN, self).__init__()
        self.p = p
        self.input_size = p.embedding_dim
        self.hidden_size = p.hidden_size
        self.nb_layers = p.nb_layers
        self.dropout = p.dropout
        self.directions = p.directions
        self.act = p.act
        self.out = p.out
        self.hinit = p.hinit
        self.output_size = self.hidden_size * self.directions * len(self.out)
        self._cuda = cuda
        
        self.rnn = rnn(self.input_size, self.hidden_size,
                       num_layers=self.nb_layers, 
                       batch_first=True,
                       dropout= self.dropout if self.nb_layers>1 else 0,
                       bidirectional= self.directions>1
                       )
        self.rnn = cudify(self.rnn, cuda)
        if any([x > 1 for x in self.out]):
            self.attn = Attention(p, cuda)
    
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def init_tensor(self, batch_size):
        return cudify(init_tensor(self.hinit, self.nb_layers * self.directions, batch_size, self.hidden_size), self.is_cuda())

    def init_hidden(self, batch_size):
        return self.init_tensor(batch_size)

    # https://github.com/ranihorev/fastai/commit/5a67283a2594c789bfa321fb259f4dff473f5d49
    def last_output(self, X, L):
        idx = torch.arange(L.size(0))
        fw = X[idx, L-1, :self.hidden_size]
        if self.directions > 1:
            bw = X[idx, 0, self.hidden_size:]
            return torch.cat([fw, bw], 1)
        return fw

    def mean_pool(self, X, L):
        return torch.div(torch.sum(X, dim=1).permute(1, 0), L.float()).permute(1, 0)
    
    def max_pool(self, X, L):
        return torch.max(X, dim=1)[0]

    def select_pool(self, X, L, i):
        if i == -1:
            return self.last_output(X, L)
        elif i == 0:
            return self.mean_pool(X, L)
        elif i == 1:
            return self.max_pool(X, L)
        else:
            return self.attn(X, L)

    def pool(self, X, L):
        M = [self.select_pool(X, L, i) for i in self.out]
        return torch.cat(M, 1)

    def forward(self, X, L):
        H = self.init_hidden(L.size(0))
        X, H = self.rnn(X, H)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        # return X, H
        X = self.pool(X, L)
        return X

class MyGRU(BaseRNN):
    def __init__(self, p, cuda=True):
        super(MyGRU, self).__init__(nn.GRU, p, cuda)

class MyLSTM(BaseRNN):
    def __init__(self, p, cuda=True):
        super(MyLSTM, self).__init__(nn.LSTM, p, cuda)

    def init_hidden(self, batch_size):
        return (self.init_tensor(batch_size), 
                self.init_tensor(batch_size)
        )

class MyModel(nn.Module):
    def __init__(self, p, cuda=True):
        super(MyModel, self).__init__()
        self.p = p
        self._cuda = cuda
        self.dropout = p.dropout
        self.act = p.act
        self.hidden_size = p.hidden_size
        self.output_size = 2
        self.directions = p.directions

        self.embed = CharacterEmbedding(p.embedding_dim, p.vocab, cuda=cuda)
        self.rnn = p.rnn(p, cuda=cuda)
        self.drop_layer = nn.Dropout(p=self.dropout)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.rnn_output_size = self.rnn.output_size

        # Dense layers
        in_size = self.rnn_output_size
        self.lin_a = None

        ## add internal dense layers...?
        if p.dense_size > 0:
            out_size = p.dense_size
            self.lin_a = cudify(nn.Linear(in_size, out_size), cuda)
            in_size = out_size

        # final dense layer...
        self.lin_z = cudify(nn.Linear(in_size, self.output_size), cuda)
    
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def drop(self, X):
        return self.drop_layer(X) if self.dropout>0 else X

    def forward(self, X, L):
        # embedding layer
        X = self.embed(X, L)

        # # RNN layer
        X = self.rnn(X, L)
        X = self.drop(X)

        # extra linear layers
        if self.lin_a is not None:
            X = self.lin_a(X)
            X = F.relu(X)
            # X = torch.tanh(X)
            X = self.drop(X)
        
        # final linear --> output layer
        X = self.lin_z(X)

        # final activation
        X = self.act(X)
        X = self.logsoftmax(X)

        return X
