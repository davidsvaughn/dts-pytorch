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

import models as mod

device = torch.device("cuda:0")

#----------------------------------------------------------------

group = 'rs' # rs | bw | fw

data_dir = '../data/insufficient'
model_dir = '../chkpt'

fn_train = '{}/{}/train.txt'.format(data_dir, group)
fn_test = '{}/{}/test.txt'.format(data_dir, group)
fn_val = '{}/{}/valid.txt'.format(data_dir, group)

fn_roc_val = '{}/roc_{}_val.txt'.format(model_dir, group)
fn_roc_test = '{}/roc_{}_test.txt'.format(model_dir, group)


#----------------------------------------------------------------
vocab = [chr(i) for i in range(32, 127)]
letters = ''.join(vocab)
# PAD = '#'
# letters = string.printable.replace(PAD,'')[:-4]
# vocab = [PAD] + list(letters)

print(letters)
# print(len(letters))
# sys.exit()

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if c in letters
        # and unicodedata.category(c) != 'Mn'
    )
print(unicode_to_ascii('Slusàrski'))

cleanreg = re.compile('<.*?>')
def clean_html(s):
    return re.sub(cleanreg, '', s)

def clean(s):
    s = unicode_to_ascii(s)
    if group is 'fw': s = clean_html(s)
    s = s[:500]
    return s

# Read a file and split into lines
def read_lines(fn):
    df = pd.read_csv(fn, sep="\t", header=None, quoting=csv.QUOTE_NONE)
    i = df[0].values
    y = df[1].values
    t = np.array([clean(line) for line in df[2].values.astype(np.str)])
    return i,y,t

def timeSince(start):
    now = time.time()
    s = now - start
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %fs' % (m, s)

def sortrows(x, col=0, asc=True):
    n = 1 if asc else -1
    x[:,col] = x[:,col] * n
    x = x[x[:,col].argsort()]
    x[:,col] = x[:,col] * n
    return x

def save_roc(fn, t, p, id=None):
    if id is not None:
        id = np.array(id, np.str)
    t = np.array(t, np.float64)
    p = np.array(p, np.float64)
    # A = [t, p]
    # if id is not None: A.append(np.array(id))
    # x = np.vstack(A)
    #x = sortrows(x.transpose(), 1, False)
    with open(fn, 'w') as f:
        for i in range(len(t)): f.write('{0}\t{1:0.12g}\t{2}\n'.format(t[i], p[i], id[i]))
        # if id is None:
        #     for i in range(x.shape[0]): f.write('{0}\t{1:0.12g}\n'.format(x[i,0], x[i,1]))
        # else:
        #     for i in range(x.shape[0]): f.write('{0}\t{1:0.12g}\t{2}\n'.format(x[i,0], x[i,1], x[i,2]))

start = time.time()
_, y_train, x_train = read_lines(fn_train)
id_test, y_test, x_test = read_lines(fn_test)
id_val, y_val, x_val = read_lines(fn_val)
print(timeSince(start))

lens = list(map(len, x_val))
lens.sort()
# print(lens)

class adict(dict):
    def __init__(self, *av, **kav):
        dict.__init__(self, *av, **kav)
        self.__dict__ = self

def cudify(x, cuda):
    return x.cuda() if cuda else x

def to_mode(x, mode=None):
    if mode is None:
        return x
    if mode == 0:
        return x.cpu()
    if mode == 1:
        return x.cuda()
    return x

def vectorize_seqs(seqs, labels, ids=None, vocab=vocab, cuda=True):
    vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]
    seq_lens = torch.LongTensor(list(map(len, vectorized_seqs)))

    # right padding
    seq_tensor = Variable(torch.zeros((len(vectorized_seqs), seq_lens.max()))).long()

    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lens)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    # SORT YOUR TENSORS BY LENGTH! (for packing)
    seq_lens, perm_idx = seq_lens.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    n = len(seq_lens[seq_lens > 0])
    seq_tensor = seq_tensor[:n]
    seq_lens = seq_lens[:n]

    seq_labels = labels[perm_idx.cpu()]
    seq_labels = seq_labels[:n]

    if ids is not None:
        ids = ids[perm_idx.cpu().data.numpy()]
        ids = ids[:n]

    return cudify(seq_tensor, cuda), cudify(seq_lens, cuda), seq_labels, ids


batch_size = 100     # 50 if group is 'fw' else 25

p = adict({})
p.rnn = mod.MyGRU       # MyGRU , MyLSTM
p.embedding_dim = 20
p.hidden_size = 250
p.dense_size = 0
p.nb_layers = 1
p.directions = 1
p.out = [-1]  # [-1, 0, 1, 100] == [last, mean, max, attn]
p.act = F.relu      # F.relu , torch.tanh
p.hinit = 0         # [0, 1, 2, 3] == [zeros, rand, xavier_norm, xavier_uni]
p.opt = 'adam'      # adam adagrad sgd
p.dropout = 0.75
p.lr = 0.001
p.clip = 5.0
p.vocab = vocab

epochs = 50
test_p = 0.1 if group is 'fw' else 0.5

#----------------------------------------------------------------

CUDA = True
model = mod.MyModel(p, cuda=CUDA)

for i in model.parameters():
    print(i.shape)

## sanity check...
# x, y = x_train[:batch_size], y_train[:batch_size]
# x, seq_lens, y, _ = vectorize_seqs(x, y, vocab=vocab)
# print(model(x, seq_lens).shape)

## JIT
x, y = x_train[:batch_size], y_train[:batch_size]
x, seq_lens, y, _ = vectorize_seqs(x, y, vocab=vocab, cuda=CUDA)
x_samp, len_samp = x[:1], seq_lens[:1]

print('SAMPLE SHAPE: {}'.format(x_samp.shape))
print('LENGTH SHAPE: {}'.format(len_samp.shape))

def save_model(epoch, in_mode=None, save_mode=0):
    global model, x_samp, len_samp
    model.train(False)
    model.eval()
    if in_mode is None:
        in_mode = 1 if model.is_cuda() else 0
    if in_mode != save_mode:
        to_mode(model, save_mode)
        x_samp = to_mode(x_samp, save_mode)
        len_samp = to_mode(len_samp, save_mode)
    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, (x_samp, len_samp))
    jit_file = '{}/model_epoch-{}.pt'.format(model_dir, epoch)
    traced_script_module.save(jit_file)
    if in_mode != save_mode:
        to_mode(model, in_mode)
    model.train(True)

save_model(0)

# sys.exit()

# ## https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
# x = x.to('cuda')
# x = x.to('cpu', torch.double) # ``.to`` can also change dtype together!

## OPTIMIZER & LOSS
if p.opt == 'adam':
    optim = torch.optim.Adam(model.parameters(), lr=p.lr)
elif p.opt == 'adabound':
    optim = adabound.AdaBound(model.parameters(), lr=p.lr, final_lr=0.1)
elif p.opt == 'adagrad':
    optim = torch.optim.Adagrad(model.parameters(), lr=0.01)
else:# sgd
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

criterion = nn.NLLLoss()

## EVALUATION
def eval_batch(batch_x, batch_y, batch_id=None):
    x, seq_len, batch_y, batch_id = vectorize_seqs(batch_x, batch_y, batch_id, vocab=vocab, cuda=CUDA)
    batch_pred = model(x, seq_len)
    batch_pred = np.exp(batch_pred.cpu().data.numpy()[:, 1]) # .argmax(1)
    return batch_pred, batch_y, batch_id

def eval_set(x, y, ids=None, p=1., batch_size=500, fn=None):
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x = x[idx]
    y = y[idx]
    ids = None if ids is None else ids[idx]
    n = p * x.shape[0]
    current_idx = 0
    P, T, I = [], [], []
    with torch.no_grad():
        model.train(False)
        for b in range(x.shape[0] // batch_size):
            batch_x = x[current_idx: current_idx + batch_size]
            batch_y = y[current_idx: current_idx + batch_size]
            batch_id = None if ids is None else ids[current_idx: current_idx + batch_size]
            current_idx += batch_size
            if current_idx > n: break
            py, ty, id = eval_batch(batch_x, batch_y, batch_id)
            P.append(py)
            T.append(ty)
            I.append(id)
        model.train(True)
    P, T, I = np.hstack(P), np.hstack(T), None if ids is None else np.hstack(I)
    if p == 1. and fn is not None:
        save_roc(fn, T, P, I)
    return (P>0.5).astype(np.int32), T

def eval_test(p=1.): return eval_set(x_test, y_test, id_test, p=p, fn=fn_roc_test)
def eval_valid(p=1.): return eval_set(x_val, y_val, id_val, p=p, fn=fn_roc_val)

## sanity check...
# print(np.dstack(list(eval_test(0.01))).squeeze())

## TRAINING LOOP....
idx = np.arange(x_train.shape[0])
for epoch in range(epochs):
    start_epoch = start = time.time()
    print('epoch', epoch)

    # if epoch==1: batch_size*=2
    # if epoch==3: batch_size*=2

    print_iter = (20000 if group is 'fw' else 10000) // batch_size

    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]

    P, T = [], []
    current_idx = 0
    for b in range(x_train.shape[0] // batch_size):
        batch_x = x_train[current_idx: current_idx + batch_size]
        batch_y = y_train[current_idx: current_idx + batch_size]
        current_idx += batch_size
        if len(batch_x) == 0: continue
        #############################################
        ''' critical training section '''

        # get input tensors
        X, L, ty, _ = vectorize_seqs(batch_x, batch_y, vocab=vocab, cuda=CUDA)

        # forward pass...
        py = model(X, L)
        loss = criterion(py, torch.tensor(ty, dtype=torch.long).cuda())

        # backward pass...
        optim.zero_grad()
        loss.backward()

        # clip grad norm...?
        if p.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), p.clip)
        
        # optimizer step...
        optim.step()

        #############################################
        P.append(py.cpu().data.numpy().argmax(1))
        T.append(ty)#.cpu().data.numpy())

        if b % print_iter == 0 and b > 0:
            eps = (print_iter * batch_size)/(time.time() - start)
            per = 100 * float(current_idx)/x_train.shape[0]
            yp, yt = eval_test(p=test_p)
            test_acc = accuracy_score(yt, yp)
            # f1 = f1_score(yt, yp, average='weighted')
            # precision = precision_score(yt, yp, average='weighted')
            # print(loss.item(), '\tb:', b, '\tepoch', epoch, 'f1', f1)
            print('\t{0}%  eps:{1}\tacc:{2:0.4f}'.format(int(per), int(eps), test_acc))
            start = time.time()

    train_acc = accuracy_score(np.hstack(P), np.hstack(T))
    valid_acc = accuracy_score(*eval_valid())
    test_acc = accuracy_score(*eval_test())
    
    print('\ttrain:\t{0:0.4f}'.format(train_acc))
    print('\tvalid:\t{0:0.4f}'.format(valid_acc))
    print('\ttest:\t{0:0.4f}'.format(test_acc))
    print('\t{0}'.format(timeSince(start_epoch)))

    save_model(epoch)
