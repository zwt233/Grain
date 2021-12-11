import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import random
import copy
import os
import sys
import time
import argparse
import json
import numpy as np
import numpy.linalg as la
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
from scipy.sparse import csgraph
from torch.backends import cudnn
from torch.optim import lr_scheduler
from utils import *
from graphConvolution import *


os.environ["CUDA_VISIBLE_DEVICES"] = '3'
cudnn.benchmark = False            
cudnn.deterministic = True
num_coreset = 140
hidden_size = 128

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid,bias=True)
        self.gc2 = GraphConvolution(nhid, nclass,bias=True)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def train(epoch, model,record):

    model.train()
    optimizer.zero_grad()
    output = model(features_GCN, adj)
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    model.eval()
    output = model(features_GCN, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    early_stopping(loss_val)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    record[acc_val.item()] = acc_test.item()
    


adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset="cora")

adj = aug_normalized_adjacency(adj)
adj = sparse_mx_to_torch_sparse_tensor(adj).float().cuda()
idx_val = list(idx_val.cpu())
idx_test = list(idx_test.cpu())
with open('GRAIN(ball-D)_r0.05_cora_selected_nodes.json','r')as f:
    idx_train_all = json.load(f)
idx_train = idx_train_all[:num_coreset]
features_GCN = copy.deepcopy(features)
features_GCN = torch.FloatTensor(features_GCN).cuda()
labels = labels.cuda()

idx_train = torch.LongTensor(idx_train).cuda()
idx_val = torch.LongTensor(idx_val).cuda()
idx_test = torch.LongTensor(idx_test).cuda()

print('xxxxxxxxxx Evaluation begin xxxxxxxxxx')
t_total = time.time()
record = {}
for i in range(500):
    model = GCN(nfeat=features_GCN.shape[1],
            nhid=hidden_size,
            nclass=labels.max().item() + 1,
            dropout=0.85)
    model.cuda()
    early_stopping = EarlyStopping(patience = 10)
    optimizer = optim.Adam(model.parameters(),
                           lr=0.05, weight_decay=5e-4)
    for epoch in range(400):
        train(epoch,model,record)
        if early_stopping.early_stop==True:
            break

bit_list = sorted(record.keys())
bit_list.reverse()
for key in bit_list[:10]:
    value = record[key]
    print(round(key,3),round(value,3))
print('xxxxxxxxxx Evaluation end xxxxxxxxxx')