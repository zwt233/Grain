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
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#hyperparameters
num_node = 2708
num_coreset = 140
radium = 0.05

def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def get_current_neighbors_dense(cur_nodes):
    if np.array(cur_nodes).shape[0]==0:
        return num_zeros
    neighbors=(adj_matrix2[list(cur_nodes)].sum(axis=0)!=0)+0
    return neighbors

def compute_distance(_i,_j):
    return la.norm(features_aax[_i,:]-features_aax[_j,:])

#load data
adj, features, labels, idx_train, idx_val, idx_test = load_data(dataset="cora")

idx_val = list(idx_val.cpu())
idx_test = list(idx_test.cpu())
idx_avaliable = list()
for i in range(num_node):
	if i not in idx_val and i not in idx_test:
		idx_avaliable.append(i)

#compute and store normalized distance in A*A*X
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


#compute the balls
balls = np.zeros((num_node,num_node))
num_zeros = np.ones(num_node)
balls_dict=dict()
covered_balls = set()
for i in range(num_node):
	for j in range(num_node):
		if distance_aax[i][j] <= radium:
			balls[i][j]=1

idx_avaliable_tmp = copy.deepcopy(idx_avaliable)
for node in idx_avaliable_tmp:
	neighbors_tmp = get_current_neighbors_dense([node])
	neighbors_tmp = neighbors_tmp[:,np.newaxis]
	dot_result = np.matmul(balls,neighbors_tmp).T
	tmp_set = set()
	for i in range(num_node):
		if dot_result[0,i]!=0:
			tmp_set.add(i)
	balls_dict[node]=tmp_set

#choose the node
count = 0
idx_train = []
while True:	
	ball_num_max = 0
	node_max = 0
	for node in idx_avaliable_tmp:
		tmp_num = len(covered_balls.union(balls_dict[node]))
		if tmp_num > ball_num_max:
			ball_num_max = tmp_num
			node_max = node
	res_ball_num = num_node - ball_num_max
	count+=1
	print('the number '+str(count)+' is selected, with the balls '+str(ball_num_max-len(covered_balls))+' covered and the rest balls is '+str(res_ball_num))	
	idx_train.append(node_max)
	idx_avaliable_tmp.remove(node_max)
	covered_balls = covered_balls.union(balls_dict[node_max])
	if count >= num_coreset or res_ball_num==0:
		break
with open('GRAIN(ball-D)_r'+str(radium)+'_cora_selected_nodes.json','w')as f:
	json.dump(idx_train,f)
print('node selection finished')
