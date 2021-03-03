import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import torch
import random
import copy
import sys
import os
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

#hyperparameters
num_node = 2708
num_coreset = 140
dmax = np.ones(num_node)
gamma = 1
    
def get_receptive_fields_dense(cur_neighbors, selected_node, weighted_score): 
    receptive_vector=((cur_neighbors+adj_matrix2[selected_node])!=0)+0
    count=weighted_score.dot(receptive_vector)
    return count

def get_current_neighbors_dense(cur_nodes):
    if np.array(cur_nodes).shape[0]==0:
        return 0
    neighbors=(adj_matrix2[list(cur_nodes)].sum(axis=0)!=0)+0
    return neighbors

def get_max_nnd_node_dense(idx_used,high_score_nodes,min_distance): 
    max_receptive_node = 0
    max_total_score = 0
    cur_neighbors=get_current_neighbors_dense(idx_used)
    for node in high_score_nodes:
        receptive_field=get_receptive_fields_dense(cur_neighbors,node,num_ones)
        node_distance = distance_aax[node,:]
        node_distance = np.where(node_distance<min_distance,node_distance,min_distance)
        node_distance = dmax - node_distance
        distance_score = node_distance.dot(num_ones)
        total_score = receptive_field/num_node+gamma*distance_score/num_node
        if total_score > max_total_score:
            max_total_score = total_score
            max_receptive_node = node        
    return max_receptive_node

def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def compute_distance(_i,_j):
    return la.norm(features_aax[_i,:]-features_aax[_j,:])
#read dataset
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset="cora")

num_zeros = np.zeros(num_node)
num_ones = np.ones(num_node)
idx_val = list(idx_val.cpu())
idx_test = list(idx_test.cpu())
idx_avaliable = list()
for i in range(num_node):
    if i not in idx_val and i not in idx_test:
        idx_avaliable.append(i)

#compute normalized distance
adj = aug_normalized_adjacency(adj)
adj_matrix = torch.FloatTensor(adj.todense()).cuda()
adj_matrix2 = torch.mm(adj_matrix,adj_matrix).cuda()
features = features.cuda()
features_aax = np.array(torch.mm(adj_matrix2,features).cpu())
adj_matrix2 = np.array(adj_matrix2.cpu())

distance_aax = np.zeros((num_node,num_node))
for i in range(num_node-1):
    for j in range(i+1,num_node):
        distance_aax[i][j] = compute_distance(i,j)
        distance_aax[j][i] = distance_aax[i][j]
dis_range = np.max(distance_aax) - np.min(distance_aax)
distance_aax = (distance_aax - np.min(distance_aax))/dis_range

#chooose node
min_distance = np.ones(num_node)
idx_train = []
idx_avaliable_temp = copy.deepcopy(idx_avaliable)
count = 0
while True:
    t1 = time.time()
    max_reception_node = get_max_nnd_node_dense(idx_train,idx_avaliable_temp,min_distance) 
    idx_train.append(max_reception_node) 
    idx_avaliable.remove(max_reception_node)
    idx_avaliable_temp.remove(max_reception_node)
    count += 1
    print('the number '+str(count)+' node is selected')
    print("select time elapsed: {:.4f}s".format(time.time() - t1))
    max_node_distance = distance_aax[max_reception_node,:]
    min_distance = np.where(min_distance<max_node_distance,min_distance,max_node_distance)
    if count >= num_coreset:
        break

print('node selection finished')
with open('GRAIN(NN-D)_cora_select_nodes.json','w')as f:
    json.dump(idx_train,f)
