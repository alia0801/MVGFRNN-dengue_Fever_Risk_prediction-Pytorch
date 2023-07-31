#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchsummary import summary
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from fastdtw import fastdtw
import time
import matplotlib.pyplot as plt
import hdbscan
import math
import logging
import datetime
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import numpy as np
from config import *
from util import *

# %%

if DATASET =='dengue':
    meo_col = ['測站氣壓(hPa)', '氣溫(℃)', '相對溼度(%)', '風速(m/s)', '降水量(mm)','測站最高氣壓(hPa)', 
                  '最高氣溫(℃)', '最大陣風(m/s)', '測站最低氣壓(hPa)','最低氣溫(℃)', '最小相對溼度(%)']
    st_col = ['watersupply_hole','well', 'sewage_hole', 'underwater_con', 'pumping',
           'watersupply_others', 'watersupply_value', 'food_poi', 'rainwater_hole',
           'river', 'drainname', 'sewage_well', 'gaugingstation', 'underpass', 'watersupply_firehydrant']

if DATASET=='aqi':
    meo_col = ['temperature','pressure', 'humidity', 'wind_speed/kph',
                'A', 'B', 'C', 'D', 'E', 'F','G', 'H']
    st_col = ['Hotel','Food','Education','Culture','Financial','Shopping','Medical','Entertainment','Transportation Spots','Company',
                'Vehicle Service','Sport','Daily Life','Institution','primary','secondary','pedestrian','highway','water','industrial','green','residential']

st_col_all = st_col.copy()
# if DATASET =='dengue':
dirs = ['B', 'T', 'L', 'R', 'RB', 'RT', 'LB', 'LT']
for grid_dir in dirs:
    col_name = pd.Series(st_col.copy()) + '_'+grid_dir
    col_name = col_name.tolist()
    st_col_all+=col_name

# %%
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# In[14]:


# https://github.com/Diego999/pyGAT/blob/master/layers.py
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# In[15]:


class Attention_layer(nn.Module) :
    def __init__(self, num_station, unlabel_size, label_size, hidden_size) :#10,64,64,16
        super(Attention_layer, self).__init__()
        self.num = num_station
        self.un_emb = unlabel_size
        self.emb = label_size
        self.hidden = hidden_size
        self.linear_1 = nn.Linear(self.un_emb+self.emb, self.hidden)
        self.linear_2 = nn.Linear(self.hidden, 1)
        
    def forward(self, unlabel_emb, label_emb, dis_lab) :
        # unlabel_emb torch.Size([128, 64])
        # label_emb torch.Size([128, 64]) *10
        # dis_lab torch.Size([128, 10])
        
#         label_1 = []
        label_2 = []
        for k in label_emb : 
#             label_1.append(nn.ReLU()(self.linear_1(torch.cat((k, unlabel_emb), 1))))
            tmp = nn.ReLU()(self.linear_1(torch.cat((k, unlabel_emb), 1))) # torch.Size([128, 64])+torch.Size([128, 64]) -> torch.Size([128, 128]) -> torch.Size([128, 16])
            label_2.append(self.linear_2(tmp)) # torch.Size([128, 16]) -> torch.Size([128, 1])
#         for k in label_1 : 
#             label_2.append(self.linear_2(k))
        attention_out_ori = torch.stack(label_2).squeeze().permute(1,0) #torch.Size([128, 1])->torch.Size([128, 10])
#         print('attention_out_ori',attention_out_ori.shape)#torch.Size([128, 10])
#         print('dis_lab',dis_lab.shape)#torch.Size([128, 10])
        attention_out = attention_out_ori * dis_lab #torch.Size([128, 10]) * torch.Size([128, 10]) -> torch.Size([128, 10])
#         print('attention_out',attention_out.shape)#torch.Size([128, 10])
        attention_score = nn.Softmax(dim=1)(attention_out) #torch.Size([128, 10])
        return attention_score #, nn.Softmax(dim=1)(attention_out_ori)

# In[21]:
def nodelist2indexlist(node_list,node_id):
    index_list = [ np.where(node_id==node)[0][0] for node in node_list]
    return index_list
# nodelist2indexlist(node_list,node_id)

# In[22]:

def read_fusion_graph(t,path='dataset_processed/'+DATASET+'/graph_data/'):
    sp_dist_adj = np.load(path+'adj_spatial_dist/'+str(t)+'.npy')
    sp_cluster_adj = np.load(path+'adj_spatial_cluster/'+str(t)+'.npy')
    tmep_adj = np.load(path+'adj_temporal_new/'+str(t)+'.npy')
    # if VIEW_NUM==1:
    #     if graph_type=='sp_dist':
    #         adj=[torch.tensor(sp_dist_adj, device=DEVICE).double()]
    #     elif graph_type=='sp_clus':
    #         adj=[torch.tensor(sp_cluster_adj, device=DEVICE).double()]
    #     elif graph_type=='temp':
    #         adj=[torch.tensor(tmep_adj, device=DEVICE).double()]
    # el
    if VIEW_NUM==2:
        adj=[torch.tensor(sp_dist_adj, device=DEVICE).double(), torch.tensor(tmep_adj, device=DEVICE).double()]
    elif VIEW_NUM==3:
        # adj=[torch.Tensor(sp_dist_adj).to(device), torch.Tensor(sp_cluster_adj).to(device), torch.Tensor(tmep_adj).to(device)]
        adj=[torch.tensor(sp_dist_adj, device=DEVICE).double(), torch.tensor(sp_cluster_adj, device=DEVICE).double(), torch.tensor(tmep_adj, device=DEVICE).double()]
    elif VIEW_NUM==4:
        if fuse_adj_method=='add':
            adj_fuse = sp_dist_adj+sp_cluster_adj+tmep_adj
        elif fuse_adj_method=='cat':
            adj_fuse = np.load(path+'1view_4type/'+str(t)+'.npy')
        adj=[torch.tensor(sp_dist_adj, device=DEVICE).double(), torch.tensor(sp_cluster_adj, device=DEVICE).double(), torch.tensor(tmep_adj, device=DEVICE).double(), torch.tensor(adj_fuse, device=DEVICE).double()]
    feat = np.load(path+'feat/'+str(t)+'.npy')
    nid = np.load(path+'all_node_id.npy')
    return adj, torch.Tensor(feat).to(DEVICE), nid
    # return adj, torch.tensor(feat, device=device).double(), nid

def read_fusion_graph_1view(t,path='dataset_processed/'+DATASET+'/graph_data/',type_num=3):
    feat = np.load(path+'feat/'+str(t)+'.npy')
    if graph_type=='sp_dist':
        adj = np.load(path+'adj_spatial_dist/'+str(t)+'.npy')
    elif graph_type=='sp_clus':
        adj = np.load(path+'adj_spatial_cluster/'+str(t)+'.npy')
    elif graph_type=='temp':
        adj = np.load(path+'adj_temporal_new/'+str(t)+'.npy')
    adj_fuse=torch.tensor(adj, device=DEVICE).double()
    nid = np.load(path+'all_node_id.npy')
    return torch.tensor(adj_fuse, device=DEVICE).double(), torch.Tensor(feat).to(DEVICE), nid

# In[23]:


class MultiView_GNN(nn.Module):
    def __init__(self, in_features, out_size, gat_hidden_size, num_heads=1, dropout=0.2, alpha_gat=0.2, alpha_fusion=0.8, beta_fusion=0.5 ) :
        super(MultiView_GNN, self).__init__()
        self.input_size = in_features
        self.out_size = out_size
        self.gat_hidden_size = gat_hidden_size
        self.alpha = alpha_fusion
        self.beta = beta_fusion
        
        self.GATLayer = GraphAttentionLayer(self.input_size, self.gat_hidden_size, dropout, alpha_gat)
        self.GCNLayer = GraphConvolution(self.input_size, self.gat_hidden_size)
        self.multihead_attn = MultiheadAttention(self.gat_hidden_size,self.gat_hidden_size, num_heads)
#         self.multihead_attn = nn.MultiheadAttention(self.gat_hidden_size, num_heads)
        self.linear_fusion = nn.Linear(self.gat_hidden_size, self.gat_hidden_size)
        
    def forward(self, all_adj, node_feat):
#         print('enter MultiView_GNN()')
#         print('node_feat', node_feat.shape) #torch.Size([1608, 146])
        node_num = round(node_feat.shape[0]/3)
#         print('node num',node_num) #536
        
        hidden_GAT = []
        for adj in all_adj:
#             print('adj',adj.shape) #adj torch.Size([1608, 1608])
            h = self.GATLayer(node_feat, adj)
            # h = self.GCNLayer(node_feat, adj)
            hidden_GAT.append(h)
#         print('shape after GAT',h.shape) #torch.Size([1608, 32])
            
        hidden_self_att = []
        for h in hidden_GAT:
            h = h.unsqueeze(0)
#             h = h.transpose(1,0).unsqueeze(2) #batch_size, seq_length, _ = x.size()
            attn_output = self.multihead_attn(h) #h torch.Size([1608, 32])
#             attn_output = self.multihead_attn(h, h, h)
#             attn_output, attn_output_weights = multihead_attn(query, key, value)
            hidden_self_att.append(attn_output)
#         print('shape after self-att',attn_output.shape) #torch.Size([1, 1608, 32])
        
        hidden_fuse1 = []
        for i in range(len(hidden_self_att)):
            h = self.alpha*hidden_self_att[i]+(1-self.alpha)*hidden_GAT[i]
            hidden_fuse1.append(h)
#         print('shape after fusion', h.shape) #torch.Size([1, 1608, 32])
            
        view_hidden_final = []
        fusion_hidden = torch.zeros(h.shape,device=DEVICE)#.to(device=device)
#         print('fusion_hidden',fusion_hidden.shape) #torch.Size([1, 1608, 32])
        for h in hidden_fuse1:
            w = nn.Sigmoid()(self.linear_fusion(h))
            hid_now_view = w*h
            fusion_hidden += hid_now_view
            h_new = self.beta*hid_now_view + (1-self.beta)*h
            view_hidden_final.append(h_new)
#         print('fusion_hidden',fusion_hidden.shape) #torch.Size([1, 1608, 32])
        fusion_hidden = fusion_hidden[0,node_num*2:,:]#.squeeze()
#         print('shape final', h_new.shape) #torch.Size([1, 1608, 32])
#         print('output shape of MultiView_GNN', fusion_hidden.shape)# torch.Size([ 536, 32])
        
        return fusion_hidden.to(DEVICE), view_hidden_final


# In[24]:


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
#         print('x.size()',x.size()) #torch.Size([32, 3, 1608, 32]) #torch.Size([1, 1608, 32])
#         seq_length , batch_size = x.size() #torch.Size([1608, 32])
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)
#         print('qkv size',qkv.size()) #torch.Size([1, 1608, 96])

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim) #(32, 1608, 4, 12)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o
        
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


# In[25]:


class MultiView_GNN_batch(nn.Module):
    def __init__(self, in_features, out_size, gat_hidden_size, num_heads=1, dropout=0.2, alpha_gat=0.2, alpha_fusion=0.8, beta_fusion=0.5 ) :
        super(MultiView_GNN_batch, self).__init__()
        self.input_size = in_features
        self.out_size = out_size
        self.gat_hidden_size = gat_hidden_size
        self.alpha = alpha_fusion
        self.beta = beta_fusion
        
#         self.GATLayer = GraphAttentionLayer(self.input_size, self.gat_hidden_size, dropout, alpha_gat)
        self.multihead_attn = MultiheadAttention(self.gat_hidden_size,self.gat_hidden_size, num_heads)
#         self.multihead_attn = nn.MultiheadAttention(self.gat_hidden_size, num_heads)
        self.linear_fusion = nn.Linear(self.gat_hidden_size, self.gat_hidden_size)
        
    def forward(self, hidden_GAT):

#         print('shape after GAT',hidden_GAT.shape) #torch.Size([32, 3, 1608, 32])
        batchsz, views, seq_len, embsz = hidden_GAT.shape
        node_num = round(seq_len/3)
        hidden = hidden_GAT.reshape((batchsz,views*seq_len, embsz)) #torch.Size([32, 4824, 32])

        hidden_self_att = self.multihead_attn(hidden)

        hidden_fuse1 = self.alpha*hidden_self_att+(1-self.alpha)*hidden
#         print('shape after fusion', hidden_fuse1.shape) #torch.Size([32, 4824, 32])
        hidden_fuse1 = hidden_fuse1.reshape(hidden_GAT.shape) #torch.Size([32, 3, 1608, 32])
        hidden_fuse1 = hidden_fuse1.permute(1,0,2,3) #torch.Size([3, 32, 1608, 32])
#         print('hidden_fuse1',hidden_fuse1.shape)
            
        view_hidden_final = []
        fusion_hidden = torch.zeros(hidden_fuse1[0].shape,device=DEVICE)#.to(device=device)
#         print('fusion_hidden',fusion_hidden.shape) #torch.Size([4824, 32])
        for h in hidden_fuse1:
            w = nn.Sigmoid()(self.linear_fusion(h))
            hid_now_view = w*h
            fusion_hidden += hid_now_view
#         print('fusion_hidden',fusion_hidden.shape) #torch.Size([32, 1608, 32])
        
        fusion_hidden = fusion_hidden[:,node_num*2:,:]#.squeeze()
#         print('shape final', h_new.shape) #torch.Size([1, 1608, 32])
#         print('output shape of MultiView_GNN', fusion_hidden.shape)# torch.Size([32, 536, 32])
        
        return fusion_hidden.to(DEVICE), view_hidden_final

# In[27]:

def get_certain_node_batch(label_data_stfgn,gid_idx_list):
#     print('label_data_stfgn',label_data_stfgn.shape) #torch.Size([32, 536, 32])
    embed = []
    for count in range(len(gid_idx_list)):
        gid_idx = gid_idx_list[count]
#         print(gid_idx)
        tmp=[]
        for idx in gid_idx:
            tmp.append(label_data_stfgn[count][idx].cpu().detach().numpy())
        embed.append(tmp)
#     print('embed[0]',embed[0].shape,embed[0]) #32
    embed_tensor = torch.tensor(embed, device=DEVICE)#.to(device)
#     print('embed_tensor',embed_tensor.shape) # torch.Size([32, 10, 32])
    return embed_tensor

# %%

class GRU_DE(nn.Module) :
    def __init__(self, num_steps, hidden_dim=32, num_layers=1) :
        super(GRU_DE, self).__init__()
        self.lstm = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, 1)
        self.num_steps = num_steps
    
    def forward(self, in_data, hidden):
        # print('in_data',in_data.shape)#(28,32)
        #batch_size, num_steps = outputs.shape
        #in_data = torch.tensor([[0.0]] * batch_size, dtype=torch.float).cuda()
        #in_data = in_data.unsqueeze(0) * batch_size
        in_data = in_data.unsqueeze(0)
        hidden = hidden.unsqueeze(0)
        result = []
        for i in range(self.num_steps):
            output, hidden = self.lstm(in_data, hidden)
            output = self.out(output[-1])
            result.append(output)
            in_data = output.unsqueeze(0)
        result = torch.stack(result).squeeze(2).permute(1,0)
        return result

# In[28]:


class MVGFRNN(nn.Module) :
    def __init__(self, num_station=10, output_size=1) :
        super(MVGFRNN, self).__init__()
        self.num_station = num_station
        self.prev_slot = PREV_SLOT
        self.hidden_lstm = 32
        self.hidden_linear = 32
        self.hidden_gru = 32
        self.hidden_gnn = 32
        self.output_size = output_size
        meo_len = len(meo_col)
        st_len = len(st_col)
        print(meo_len,st_len)
        
        self.unlabel_lstm_1 = nn.LSTM(meo_len, self.hidden_lstm) #v
        self.unlabel_linear_1 = nn.Linear(st_len*9, self.hidden_linear)#v
        self.unlabel_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm, self.hidden_linear*2)
        self.label_lstm_1 = nn.LSTM(meo_len+1, self.hidden_lstm)#v
        self.label_linear_1 = nn.Linear(st_len*9+1, self.hidden_linear)#v
        self.label_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm+self.hidden_gnn, self.hidden_linear*2)
        self.label_linear_3 = nn.Linear(self.hidden_gnn, self.hidden_linear*2)

        # self.unlabel_lstm_1 = nn.LSTM(11, self.hidden_lstm) #v
        # self.unlabel_linear_1 = nn.Linear(135, self.hidden_linear)#v
        # self.unlabel_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm, self.hidden_linear*2)
        # self.label_lstm_1 = nn.LSTM(12, self.hidden_lstm)#v
        # self.label_linear_1 = nn.Linear(136, self.hidden_linear)#v
        # self.label_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm+self.hidden_gnn, self.hidden_linear*2)
        # self.label_linear_3 = nn.Linear(self.hidden_gnn, self.hidden_linear*2)
        
#         self.stfgn = MultiView_GNN(in_features=146, out_size=self.hidden_gnn, gat_hidden_size=self.hidden_gnn)
        self.stfgn = MultiView_GNN_batch(in_features=(len(st_col)*9+len(meo_col)), out_size=self.hidden_gnn, gat_hidden_size=self.hidden_gnn, alpha_fusion=alpha_multiview_fusion)
        self.GATLayer = GraphAttentionLayer(len(st_col)*9+len(meo_col), self.hidden_gnn, 0.2, 0.2)
        self.GCNLayer = GraphConvolution(len(st_col)*9+len(meo_col), self.hidden_gnn)
        
        self.idw_attention = Attention_layer(num_station, self.hidden_linear*2, self.hidden_linear*2, 16)
        
        self.GRU = nn.GRUCell(self.hidden_gru+self.hidden_linear*2+self.hidden_linear*2 , self.hidden_gru, bias=True) #nn.GRUCell(input_size, hidden_size)
        # self.GRU_2 = nn.GRUCell(self.hidden_gru , output_size, bias=True)
        self.GRU_DE = GRU_DE(num_steps=output_size)
        # self.GRU_EN_DE = GRU_predictor(self.hidden_gru+self.hidden_linear*2+self.hidden_linear*2 , self.hidden_gru, output_size)
        self.liner_t = nn.Linear(self.hidden_gru, self.hidden_gru)
        self.output_fc = nn.Linear(self.hidden_gru ,output_size )
        
    def forward(self, meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label, dis_label, h_t, timestamp, label_id_list) :#timestamp
        
#         batch_adj = []
#         batch_feat = []
        batch_graph_h = []
        gid_idx_list = []
        for i in range(len(timestamp)):
            try:
                t = timestamp[i].item()
            except:
                t = timestamp[i]
                # print('timestamp',t)
            gids = label_id_list[i]
            multi_view_adj, node_feat, node_id = read_fusion_graph(t,graph_path)
            gid_idx = nodelist2indexlist(gids.tolist(),node_id)
            gid_idx_list.append(gid_idx)
            
            hidden_GAT = []
            for adj in multi_view_adj:
                h = self.GATLayer(node_feat, adj)
                # h = self.GCNLayer(node_feat.float(), adj.float())
                hidden_GAT.append(h.cpu().detach().numpy())
            batch_graph_h.append(hidden_GAT)
        batch_graph_h = torch.tensor(batch_graph_h, device=DEVICE)#.to(device)
#         print('batch_graph_h',batch_graph_h.shape)# torch.Size([32, 3, 1608, 32])
        if torch.isnan(batch_graph_h).any():
            batch_graph_h = torch.nan_to_num(batch_graph_h)
            # print('batch_graph_h',batch_graph_h)
            # print('batch_graph_h has nan')
            # print(batch_graph_h)

        label_data_stfgn_batch, _ = self.stfgn(batch_graph_h) 
        label_data_stfgn_batch = get_certain_node_batch(label_data_stfgn_batch,gid_idx_list) # torch.Size([32, 10, 32])
        label_data_stfgn_batch = label_data_stfgn_batch.permute(1,0,2) #[10,32,32]
#         print('label_data_stfgn_batch',label_data_stfgn_batch.shape) # torch.Size([32, 10, 32])
        # if torch.isnan(label_data_stfgn_batch).any():
        #     print('label_data_stfgn_batch',label_data_stfgn_batch)

        for j in range(self.prev_slot):
            temp_approximate = F.relu(self.liner_t(h_t)) 
#             print('temp_approximate',temp_approximate.shape) #torch.Size([128, 32])
            # if torch.isnan(temp_approximate).any():
            #     print('temp_approximate',temp_approximate)

            unlabel_time_data = meo_unlabel.permute(1,0,2) # torch.Size([128, 4, 11]) ->  torch.Size([4, 128, 11])
            unlabel_time_data, _ = self.unlabel_lstm_1(unlabel_time_data) # torch.Size([4, 128, 11]) ->  torch.Size([4, 128, 32])
            unlabel_time_data = unlabel_time_data.float()[-1] # torch.Size([4, 128, 32]) ->  torch.Size([128, 32])
#             print('unlabel_time_data',unlabel_time_data.shape) #torch.Size([32, 32])
            # if torch.isnan(unlabel_time_data).any():
            #     print('unlabel_time_data',unlabel_time_data)

            unlabel_fea_data = nn.ReLU()(self.unlabel_linear_1(feature_unlabel)) # torch.Size([128, 135]) -> torch.Size([128, 32])
            unlabel_data = torch.cat((unlabel_time_data, unlabel_fea_data), 1) # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64])
            unlabel_data = nn.ReLU()(self.unlabel_linear_2(unlabel_data)) # torch.Size([128, 64]) -> torch.Size([128, 64])
#             print('unlabel_data',unlabel_data.shape) #torch.Size([32, 64])
            # if torch.isnan(unlabel_data).any():
            #     print('unlabel_data',unlabel_data)

            if add_labeled_embed: 
                label_time = torch.cat((ovi_label.unsqueeze(3), meo_label), 3) #torch.Size([128, 10, 4, 1]) + torch.Size([128, 10, 4, 11]) -> torch.Size([128, 10, 4, 12])
                label_time_data = []
                label_feature = []
                label_data = [] 
                for i in range(self.num_station) :
                    lstm_tmp, _ = self.label_lstm_1(label_time[:,i,:,:].permute(1,0,2)) #torch.Size([128, 4, 12]) -> torch.Size([4, 128, 12]) -> torch.Size([4, 128, 32])
                    lstm_tmp = lstm_tmp.float()[-1] #torch.Size([4, 128, 32]) -> torch.Size([128, 32])
                    label_time_data.append(lstm_tmp)

                    label_feature.append(nn.ReLU()(self.label_linear_1(feature_label[:,i,:])))
                    #torch.Size([128, 10, 136]) -> torch.Size([128, 136]) -> torch.Size([128, 32])

                    label_data.append(nn.ReLU()(self.label_linear_2(torch.cat([label_time_data[i], label_feature[i],label_data_stfgn_batch[i]], 1))))
                    # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64]) -> torch.Size([128, 64])

                # print('lstm_tmp',lstm_tmp.shape)
                # if torch.isnan(lstm_tmp).any():
                #     print('lstm_tmp',lstm_tmp)
                # print('label_feature[0]',label_feature[0].shape)
                
                
    #             label_feature = []
    #             for i in range(self.num_station) : #torch.Size([128, 10, 136]) -> torch.Size([128, 136]) -> torch.Size([128, 32])
    #                 label_feature.append(nn.ReLU()(self.label_linear_1(feature_label[:,i,:])))
    # #             print('label_feature[0]',label_feature[0].shape)
    #             label_data = []    
    #             for i in range(self.num_station) : # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64]) -> torch.Size([128, 64])
    #                 label_data.append(nn.ReLU()(self.label_linear_2(torch.cat([label_time_data[i], label_feature[i],label_data_stfgn_batch[i]], 1))))
                    
    #             label_feature = []
    #             for i in range(self.num_station) : #torch.Size([128, 10, 136]) -> torch.Size([128, 136]) -> torch.Size([128, 32])
    #                 label_feature.append(nn.ReLU()(self.label_linear_1(feature_label[:,i,:])))
    # #             print('label_feature[0]',label_feature[0].shape)
    #             label_data = []    
    #             for i in range(self.num_station) : # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64]) -> torch.Size([128, 64])
    #                 label_data.append(nn.ReLU()(self.label_linear_2(torch.cat([label_time_data[i], label_feature[i],label_data_stfgn_batch[i]], 1))))
            
            else:
                label_data = []    
                for i in range(self.num_station) : # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64]) -> torch.Size([128, 64])
                    label_data.append(nn.ReLU()(self.label_linear_3( label_data_stfgn_batch[i] )))
            
            # torch.Size([128, 64]) * self.num_station   
#             print('label_data',label_data.shape)
            # if torch.isnan(label_data).any():
            #     print('label_data',label_data)

            attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
            # attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
            attention_out = []
            for n,i in enumerate(label_data) :
                attention_out.append(attention_score[:,n].unsqueeze(1)*i)
            attention_out = torch.sum(torch.stack(attention_out).permute(1,0,2), 1) #torch.Size([128, 64])
        
            sp_approximate = F.relu(attention_out) #torch.Size([128, 64])
#             print('sp_approximate',sp_approximate.shape) #torch.Size([128, 64])
            # if torch.isnan(sp_approximate).any():
            #     print('sp_approximate',sp_approximate)

            # torch.Size([128, 64]), torch.Size([128, 32]), torch.Size([128, 64])
            X_feat = torch.cat( [unlabel_data,temp_approximate,sp_approximate], dim=1 ) #torch.Size([128, 160])
#             print('X_feat',X_feat.shape) #torch.Size([128, 160])
            # if torch.isnan(X_feat).any():
            #     print('X_feat',X_feat)

            h_t = self.GRU(X_feat)
            # if torch.isnan(h_t).any():
            #     print('h_t',h_t)
#             print('h_t',h_t.shape) #torch.Size([128, 32])
        out = nn.ReLU()(self.output_fc(h_t))
        if torch.isnan(out).any():
            # print('out',out)
            print('out has nan')
            print(out)
        # out = self.GRU_DE(h_t,h_t)
#         print('out',out.shape) #torch.Size([128, 1])
        return out
        
# %%
# 
class MVGFRNN_no_fusion(nn.Module) :
    def __init__(self, num_station=10, output_size=1) :
        super(MVGFRNN_no_fusion, self).__init__()
        self.num_station = num_station
        self.prev_slot = PREV_SLOT
        self.hidden_lstm = 32
        self.hidden_linear = 32
        self.hidden_gru = 32
        self.hidden_gnn = 32
        self.output_size = output_size
        
        self.unlabel_lstm_1 = nn.LSTM(len(meo_col), self.hidden_lstm) #v
        self.unlabel_linear_1 = nn.Linear(len(st_col)*9, self.hidden_linear)#v
        self.unlabel_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm, self.hidden_linear*2)
        self.label_lstm_1 = nn.LSTM(len(meo_col)+1, self.hidden_lstm)#v
        self.label_linear_1 = nn.Linear(len(st_col)*9+1, self.hidden_linear)#v
        self.label_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm+self.hidden_gnn, self.hidden_linear*2)
        
#         self.stfgn = MultiView_GNN(in_features=146, out_size=self.hidden_gnn, gat_hidden_size=self.hidden_gnn)
        self.linear_cat_graph = nn.Linear(self.hidden_linear*VIEW_NUM, self.hidden_linear)
        self.stfgn = MultiView_GNN_batch(in_features=(len(st_col)*9+len(meo_col)), out_size=self.hidden_gnn, gat_hidden_size=self.hidden_gnn)
        self.GATLayer = GraphAttentionLayer(len(st_col)*9+len(meo_col), self.hidden_gnn, 0.2, 0.2)
        
        self.idw_attention = Attention_layer(num_station, self.hidden_linear*2, self.hidden_linear*2, 16)
        
        self.GRU = nn.GRUCell(self.hidden_gru+self.hidden_linear*2+self.hidden_linear*2 , self.hidden_gru, bias=True) #nn.GRUCell(input_size, hidden_size)
        self.liner_t = nn.Linear(self.hidden_gru, self.hidden_gru)
        self.output_fc = nn.Linear(self.hidden_gru ,output_size )
        
    def forward(self, meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label, dis_label, h_t, timestamp, label_id_list) :#timestamp
        
#         batch_adj = []
#         batch_feat = []
        batch_graph_h = []
        gid_idx_list = []
        for i in range(len(timestamp)):
            try:
                t = timestamp[i].item()
            except:
                t = timestamp[i]
#             print('timestamp',t)
            gids = label_id_list[i]
            multi_view_adj, node_feat, node_id = read_fusion_graph(t,graph_path)
            gid_idx = nodelist2indexlist(gids.tolist(),node_id)
            gid_idx_list.append(gid_idx)
            
            hidden_GAT = []
            for adj in multi_view_adj:
                h = self.GATLayer(node_feat, adj) #[1608, 32]
                hidden_GAT.append(h) #3*[1608, 32] #.cpu().detach().numpy()
            hidden_GAT_cat = torch.cat(hidden_GAT, 1) #[1608, 32*3]
            batch_graph_h.append(hidden_GAT_cat.cpu().detach().numpy()) # batch_size*[1608, 32*3]

        batch_graph_h = torch.tensor(batch_graph_h, device=DEVICE)#.to(device) # [batch_size, 1608, 32*3]
        # print('batch_graph_h',batch_graph_h.shape)# torch.Size([128, 1608, 64])
        if torch.isnan(batch_graph_h).any():
            batch_graph_h = torch.nan_to_num(batch_graph_h)
        
        label_data_stfgn_batch = self.linear_cat_graph(batch_graph_h) # [batch_size, 1608, 32]
        # print('label_data_stfgn_batch',label_data_stfgn_batch.shape) #torch.Size([128, 1608, 32])
        
        # label_data_stfgn_batch, _ = self.stfgn(batch_graph_h) 
        label_data_stfgn_batch = get_certain_node_batch(label_data_stfgn_batch,gid_idx_list) # torch.Size([batch_size, 10, 32])
        label_data_stfgn_batch = label_data_stfgn_batch.permute(1,0,2) #[10,batch_size,32]
        # print('label_data_stfgn_batch',label_data_stfgn_batch.shape) # torch.Size([10, 128, 32])

        for j in range(self.prev_slot):
            temp_approximate = F.relu(self.liner_t(h_t)) 
#             print('temp_approximate',temp_approximate.shape) #torch.Size([128, 32])

            unlabel_time_data = meo_unlabel.permute(1,0,2) # torch.Size([128, 4, 11]) ->  torch.Size([4, 128, 11])
            unlabel_time_data, _ = self.unlabel_lstm_1(unlabel_time_data) # torch.Size([4, 128, 11]) ->  torch.Size([4, 128, 32])
            unlabel_time_data = unlabel_time_data.float()[-1] # torch.Size([4, 128, 32]) ->  torch.Size([128, 32])
#             print('unlabel_time_data',unlabel_time_data.shape) #torch.Size([32, 32])
        
            unlabel_fea_data = nn.ReLU()(self.unlabel_linear_1(feature_unlabel)) # torch.Size([128, 135]) -> torch.Size([128, 32])
            unlabel_data = torch.cat((unlabel_time_data, unlabel_fea_data), 1) # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64])
            unlabel_data = nn.ReLU()(self.unlabel_linear_2(unlabel_data)) # torch.Size([128, 64]) -> torch.Size([128, 64])
#             print('unlabel_data',unlabel_data.shape) #torch.Size([32, 64])

            label_time = torch.cat((ovi_label.unsqueeze(3), meo_label), 3) #torch.Size([128, 10, 4, 1]) + torch.Size([128, 10, 4, 11]) -> torch.Size([128, 10, 4, 12])
            label_time_data = []
            for i in range(self.num_station) :
                lstm_tmp, _ = self.label_lstm_1(label_time[:,i,:,:].permute(1,0,2)) #torch.Size([128, 4, 12]) -> torch.Size([4, 128, 12]) -> torch.Size([4, 128, 32])
                lstm_tmp = lstm_tmp.float()[-1] #torch.Size([4, 128, 32]) -> torch.Size([128, 32])
                label_time_data.append(lstm_tmp)
#             print('lstm_tmp',lstm_tmp.shape)
            label_feature = []
            for i in range(self.num_station) : #torch.Size([128, 10, 136]) -> torch.Size([128, 136]) -> torch.Size([128, 32])
                label_feature.append(nn.ReLU()(self.label_linear_1(feature_label[:,i,:])))
#             print('label_feature[0]',label_feature[0].shape)
            label_data = []    
            for i in range(self.num_station) : # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64]) -> torch.Size([128, 64])
                label_data.append(nn.ReLU()(self.label_linear_2(torch.cat([label_time_data[i], label_feature[i],label_data_stfgn_batch[i]], 1))))
            # torch.Size([128, 64]) * self.num_station   
#             print('label_data',label_data.shape)

            attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
            # attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
            attention_out = []
            for n,i in enumerate(label_data) :
                attention_out.append(attention_score[:,n].unsqueeze(1)*i)
            attention_out = torch.sum(torch.stack(attention_out).permute(1,0,2), 1) #torch.Size([128, 64])
        
            sp_approximate = F.relu(attention_out) #torch.Size([128, 64])
#             print('sp_approximate',sp_approximate.shape) #torch.Size([128, 64])
            
            # torch.Size([128, 64]), torch.Size([128, 32]), torch.Size([128, 64])
            X_feat = torch.cat( [unlabel_data,temp_approximate,sp_approximate], dim=1 ) #torch.Size([128, 160])
#             print('X_feat',X_feat.shape) #torch.Size([128, 160])
            h_t = self.GRU(X_feat)
#             print('h_t',h_t.shape) #torch.Size([128, 32])
        out = self.output_fc(h_t)
        if torch.isnan(out).any():
            # print('out',out)
            print('out has nan')
            print(out)
#         print('out',out.shape) #torch.Size([128, 1])
        return out        
    

# %%
class MVGFRNN_no_graph(nn.Module) :
    def __init__(self, num_station=10, output_size=1) :
        super(MVGFRNN_no_graph, self).__init__()
        self.num_station = num_station
        self.prev_slot = PREV_SLOT
        self.hidden_lstm = 32
        self.hidden_linear = 32
        self.hidden_gru = 32
        self.output_size = output_size
        
        self.unlabel_lstm_1 = nn.LSTM(len(meo_col), self.hidden_lstm) #v
        self.unlabel_linear_1 = nn.Linear(len(st_col)*9, self.hidden_linear)#v
        self.unlabel_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm, self.hidden_linear*2)
        self.label_lstm_1 = nn.LSTM(len(meo_col)+1, self.hidden_lstm)#v
        self.label_linear_1 = nn.Linear(len(st_col)*9+1, self.hidden_linear)#v
        self.label_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm, self.hidden_linear*2)
        
#         self.stfgn = HeteGAT_multi()
        
        self.idw_attention = Attention_layer(num_station, self.hidden_linear*2, self.hidden_linear*2, 16)
#         self.out_linear_1 = nn.Linear(self.hidden_linear*2+self.hidden_linear*2, self.hidden_linear*2)
#         self.out_linear_2 = nn.Linear(self.hidden_linear*2, output_size)
#         self.out_pred = LSTM_predictor(self.hidden_linear*2, 32, output_size)
#         self.out_linear_3 = nn.Linear(2,1, bias=False)
        
        self.GRU = nn.GRUCell(self.hidden_gru+self.hidden_linear*2+self.hidden_linear*2 , self.hidden_gru, bias=True) #nn.GRUCell(input_size, hidden_size)
#         self.liner_t = nn.Linear(self.hidden_gru+self.hidden_linear*2+self.hidden_linear*2, self.hidden_gru)
        self.liner_t = nn.Linear(self.hidden_gru, self.hidden_gru)
        self.output_fc = nn.Linear(self.hidden_gru ,output_size )
        
    def forward(self, meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label, dis_label, h_t, timestamp, label_id_list) :#timestamp
        
        for j in range(self.prev_slot):
            temp_approximate = F.relu(self.liner_t(h_t)) 
#             print('temp_approximate',temp_approximate.shape) #torch.Size([128, 32])

            unlabel_time_data = meo_unlabel.permute(1,0,2) # torch.Size([128, 4, 11]) ->  torch.Size([4, 128, 11])
            unlabel_time_data, _ = self.unlabel_lstm_1(unlabel_time_data) # torch.Size([4, 128, 11]) ->  torch.Size([4, 128, 32])
            unlabel_time_data = unlabel_time_data.float()[-1] # torch.Size([4, 128, 32]) ->  torch.Size([128, 32])
    #         print('unlabel_time_data',unlabel_time_data.shape)
        
            unlabel_fea_data = nn.ReLU()(self.unlabel_linear_1(feature_unlabel)) # torch.Size([128, 135]) -> torch.Size([128, 32])
            unlabel_data = torch.cat((unlabel_time_data, unlabel_fea_data), 1) # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64])
            unlabel_data = nn.ReLU()(self.unlabel_linear_2(unlabel_data)) # torch.Size([128, 64]) -> torch.Size([128, 64])
#             print('unlabel_data',unlabel_data.shape) #torch.Size([128, 64])

            label_time = torch.cat((ovi_label.unsqueeze(3), meo_label), 3) #torch.Size([128, 10, 4, 1]) + torch.Size([128, 10, 4, 11]) -> torch.Size([128, 10, 4, 12])
            label_time_data = []
            for i in range(self.num_station) :
                lstm_tmp, _ = self.label_lstm_1(label_time[:,i,:,:].permute(1,0,2)) #torch.Size([128, 4, 12]) -> torch.Size([4, 128, 12]) -> torch.Size([4, 128, 32])
                lstm_tmp = lstm_tmp.float()[-1] #torch.Size([4, 128, 32]) -> torch.Size([128, 32])
                label_time_data.append(lstm_tmp)
            label_feature = []
            for i in range(self.num_station) : #torch.Size([128, 10, 136]) -> torch.Size([128, 136]) -> torch.Size([128, 32])
                label_feature.append(nn.ReLU()(self.label_linear_1(feature_label[:,i,:])))
            label_data = []    
            for i in range(self.num_station) : # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64]) -> torch.Size([128, 64])
                label_data.append(nn.ReLU()(self.label_linear_2(torch.cat((label_time_data[i], label_feature[i]), 1))))
            # torch.Size([128, 64]) * self.num_station   
#             print('label_data',label_data.shape)
        
            attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
            attention_out = []
            for n,i in enumerate(label_data) :
                attention_out.append(attention_score[:,n].unsqueeze(1)*i)
            attention_out = torch.sum(torch.stack(attention_out).permute(1,0,2), 1) #torch.Size([128, 64])
        
            sp_approximate = F.relu(attention_out) #torch.Size([128, 64])
#             print('sp_approximate',sp_approximate.shape) #torch.Size([128, 64])
            
            # torch.Size([128, 64]), torch.Size([128, 32]), torch.Size([128, 64])
            X_feat = torch.cat( [unlabel_data,temp_approximate,sp_approximate], dim=1 ) #torch.Size([128, 160])
#             print('X_feat',X_feat.shape) #torch.Size([128, 160])
            h_t = self.GRU(X_feat)
#             print('h_t',h_t.shape) #torch.Size([128, 32])
        out = self.output_fc(h_t)
        if torch.isnan(out).any():
            # print('out',out)
            print('out has nan')
            print(out)
#         print('out',out.shape) #torch.Size([128, 1])
        return out

# %%

class MVGFRNN_no_idw(nn.Module) :
    def __init__(self, num_station=10, output_size=1) :
        super(MVGFRNN_no_idw, self).__init__()
        self.num_station = num_station
        self.prev_slot = PREV_SLOT
        self.hidden_lstm = 32
        self.hidden_linear = 32
        self.hidden_gru = 32
        self.hidden_gnn = 32
        self.output_size = output_size
        
        self.unlabel_lstm_1 = nn.LSTM(len(meo_col), self.hidden_lstm) #v
        self.unlabel_linear_1 = nn.Linear(len(st_col)*9, self.hidden_linear)#v
        self.unlabel_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm, self.hidden_linear*2)
        self.label_lstm_1 = nn.LSTM(len(meo_col)+1, self.hidden_lstm)#v
        self.label_linear_1 = nn.Linear(len(st_col)*9+1, self.hidden_linear)#v
        self.label_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm+self.hidden_gnn, self.hidden_linear*2)
        self.label_linear_3 = nn.Linear(self.hidden_gnn, self.hidden_linear*2)
        
#         self.stfgn = MultiView_GNN(in_features=146, out_size=self.hidden_gnn, gat_hidden_size=self.hidden_gnn)
        self.stfgn = MultiView_GNN_batch(in_features=(len(st_col)*9+len(meo_col)), out_size=self.hidden_gnn, gat_hidden_size=self.hidden_gnn)
        self.GATLayer = GraphAttentionLayer(len(st_col)*9+len(meo_col), self.hidden_gnn, 0.2, 0.2)
        self.GCNLayer = GraphConvolution(len(st_col)*9+len(meo_col), self.hidden_gnn)
        
        self.idw_attention = Attention_layer(num_station, self.hidden_linear*2, self.hidden_linear*2, 16)
        self.no_idw_cat_linear = nn.Linear(self.hidden_linear*2*(num_station+1), self.hidden_linear*2 )
        
        self.GRU = nn.GRUCell(self.hidden_gru+self.hidden_linear*2+self.hidden_linear*2 , self.hidden_gru, bias=True) #nn.GRUCell(input_size, hidden_size)
        # self.GRU_2 = nn.GRUCell(self.hidden_gru , output_size, bias=True)
        self.GRU_DE = GRU_DE(num_steps=output_size)
        # self.GRU_EN_DE = GRU_predictor(self.hidden_gru+self.hidden_linear*2+self.hidden_linear*2 , self.hidden_gru, output_size)
        self.liner_t = nn.Linear(self.hidden_gru, self.hidden_gru)
        self.output_fc = nn.Linear(self.hidden_gru ,output_size )
        
    def forward(self, meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label, dis_label, h_t, timestamp, label_id_list) :#timestamp
        
#         batch_adj = []
#         batch_feat = []
        batch_graph_h = []
        gid_idx_list = []
        for i in range(len(timestamp)):
            try:
                t = timestamp[i].item()
            except:
                t = timestamp[i]
#             print('timestamp',t)
            gids = label_id_list[i]
            multi_view_adj, node_feat, node_id = read_fusion_graph(t,graph_path)
            gid_idx = nodelist2indexlist(gids.tolist(),node_id)
            gid_idx_list.append(gid_idx)
            
            hidden_GAT = []
            for adj in multi_view_adj:
                h = self.GATLayer(node_feat, adj)
                # h = self.GCNLayer(node_feat.float(), adj.float())
                hidden_GAT.append(h.cpu().detach().numpy())
            batch_graph_h.append(hidden_GAT)
        batch_graph_h = torch.tensor(batch_graph_h, device=DEVICE)#.to(device)
#         print('batch_graph_h',batch_graph_h.shape)# torch.Size([32, 3, 1608, 32])
        if torch.isnan(batch_graph_h).any():
            batch_graph_h = torch.nan_to_num(batch_graph_h)

        label_data_stfgn_batch, _ = self.stfgn(batch_graph_h) 
        label_data_stfgn_batch = get_certain_node_batch(label_data_stfgn_batch,gid_idx_list) # torch.Size([32, 10, 32])
        label_data_stfgn_batch = label_data_stfgn_batch.permute(1,0,2) #[10,32,32]
#         print('label_data_stfgn_batch',label_data_stfgn_batch.shape) # torch.Size([32, 10, 32])

        for j in range(self.prev_slot):
            temp_approximate = F.relu(self.liner_t(h_t)) 
#             print('temp_approximate',temp_approximate.shape) #torch.Size([128, 32])

            unlabel_time_data = meo_unlabel.permute(1,0,2) # torch.Size([128, 4, 11]) ->  torch.Size([4, 128, 11])
            unlabel_time_data, _ = self.unlabel_lstm_1(unlabel_time_data) # torch.Size([4, 128, 11]) ->  torch.Size([4, 128, 32])
            unlabel_time_data = unlabel_time_data.float()[-1] # torch.Size([4, 128, 32]) ->  torch.Size([128, 32])
#             print('unlabel_time_data',unlabel_time_data.shape) #torch.Size([32, 32])
        
            unlabel_fea_data = nn.ReLU()(self.unlabel_linear_1(feature_unlabel)) # torch.Size([128, 135]) -> torch.Size([128, 32])
            unlabel_data = torch.cat((unlabel_time_data, unlabel_fea_data), 1) # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64])
            unlabel_data = nn.ReLU()(self.unlabel_linear_2(unlabel_data)) # torch.Size([128, 64]) -> torch.Size([128, 64])
#             print('unlabel_data',unlabel_data.shape) #torch.Size([32, 64])

            if add_labeled_embed: 
                label_time = torch.cat((ovi_label.unsqueeze(3), meo_label), 3) #torch.Size([128, 10, 4, 1]) + torch.Size([128, 10, 4, 11]) -> torch.Size([128, 10, 4, 12])
                label_time_data = []
                for i in range(self.num_station) :
                    lstm_tmp, _ = self.label_lstm_1(label_time[:,i,:,:].permute(1,0,2)) #torch.Size([128, 4, 12]) -> torch.Size([4, 128, 12]) -> torch.Size([4, 128, 32])
                    lstm_tmp = lstm_tmp.float()[-1] #torch.Size([4, 128, 32]) -> torch.Size([128, 32])
                    label_time_data.append(lstm_tmp)
    #             print('lstm_tmp',lstm_tmp.shape)
                label_feature = []
                for i in range(self.num_station) : #torch.Size([128, 10, 136]) -> torch.Size([128, 136]) -> torch.Size([128, 32])
                    label_feature.append(nn.ReLU()(self.label_linear_1(feature_label[:,i,:])))
    #             print('label_feature[0]',label_feature[0].shape)
                label_data = []    
                for i in range(self.num_station) : # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64]) -> torch.Size([128, 64])
                    label_data.append(nn.ReLU()(self.label_linear_2(torch.cat([label_time_data[i], label_feature[i],label_data_stfgn_batch[i]], 1))))
            
            else:
                label_data = []    
                for i in range(self.num_station) : # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64]) -> torch.Size([128, 64])
                    label_data.append(nn.ReLU()(self.label_linear_3( label_data_stfgn_batch[i] )))
            
            # torch.Size([128, 64]) * self.num_station   
#             print('label_data',label_data.shape)

            attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
            # attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
            attention_out = []
            for n,i in enumerate(label_data) :
                attention_out.append(attention_score[:,n].unsqueeze(1)*i)
            attention_out = torch.sum(torch.stack(attention_out).permute(1,0,2), 1) #torch.Size([128, 64])

            cat_all_grid_embed = torch.cat(label_data+[unlabel_data], 1)  #torch.Size([batch_sz, 64*(k+1)])
            # print('cat_all_grid_embed',cat_all_grid_embed.shape) #([28, 704])
            attention_out = self.no_idw_cat_linear(cat_all_grid_embed)
            # print('attention_out',attention_out.shape) #torch.Size([batch_sz, 64])
        
            sp_approximate = F.relu(attention_out) #torch.Size([128, 64])
#             print('sp_approximate',sp_approximate.shape) #torch.Size([128, 64])
            
            # torch.Size([128, 64]), torch.Size([128, 32]), torch.Size([128, 64])
            X_feat = torch.cat( [unlabel_data,temp_approximate,sp_approximate], dim=1 ) #torch.Size([128, 160])
#             print('X_feat',X_feat.shape) #torch.Size([128, 160])
            h_t = self.GRU(X_feat)
#             print('h_t',h_t.shape) #torch.Size([128, 32])
        out = self.output_fc(h_t)
        if torch.isnan(out).any():
            # print('out',out)
            print('out has nan')
            print(out)
        # out = self.GRU_DE(h_t,h_t)
#         print('out',out.shape) #torch.Size([128, 1])
        return out

# %%

class MVGFRNN_no_residual(nn.Module) :
    def __init__(self, num_station=10, output_size=1) :
        super(MVGFRNN_no_residual, self).__init__()
        self.num_station = num_station
        self.prev_slot = PREV_SLOT
        self.hidden_lstm = 32
        self.hidden_linear = 32
        self.hidden_gru = 32
        self.hidden_gnn = 32
        self.output_size = output_size
        
        self.unlabel_lstm_1 = nn.LSTM(len(meo_col), self.hidden_lstm) #v
        self.unlabel_linear_1 = nn.Linear(len(st_col)*9, self.hidden_linear)#v
        self.unlabel_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm, self.hidden_linear*2)
        self.label_lstm_1 = nn.LSTM(len(meo_col)+1, self.hidden_lstm)#v
        self.label_linear_1 = nn.Linear(len(st_col)*9+1, self.hidden_linear)#v
        self.label_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm+self.hidden_gnn, self.hidden_linear*2)
        self.label_linear_3 = nn.Linear(self.hidden_gnn, self.hidden_linear*2)
        
#         self.stfgn = MultiView_GNN(in_features=146, out_size=self.hidden_gnn, gat_hidden_size=self.hidden_gnn)
        self.stfgn = MultiView_GNN_batch(in_features=len(st_col)*9+len(meo_col), out_size=self.hidden_gnn, gat_hidden_size=self.hidden_gnn)
        self.GATLayer = GraphAttentionLayer(len(st_col)*9+len(meo_col), self.hidden_gnn, 0.2, 0.2)
        self.GCNLayer = GraphConvolution(len(st_col)*9+len(meo_col), self.hidden_gnn)
        
        self.idw_attention = Attention_layer(num_station, self.hidden_linear*2, self.hidden_linear*2, 16)
        
        self.GRU = nn.GRUCell(self.hidden_gru+self.hidden_linear*2+self.hidden_linear*2 , self.hidden_gru, bias=True) #nn.GRUCell(input_size, hidden_size)
        self.GRU2 = nn.GRUCell(self.hidden_linear*2 , self.hidden_gru, bias=True) #nn.GRUCell(input_size, hidden_size)
        self.GRU_DE = GRU_DE(num_steps=output_size)
        # self.GRU_EN_DE = GRU_predictor(self.hidden_gru+self.hidden_linear*2+self.hidden_linear*2 , self.hidden_gru, output_size)
        self.liner_t = nn.Linear(self.hidden_gru, self.hidden_gru)
        self.output_fc = nn.Linear(self.hidden_gru ,output_size )
        
    def forward(self, meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label, dis_label, h_t, timestamp, label_id_list) :#timestamp
        
#         batch_adj = []
#         batch_feat = []
        batch_graph_h = []
        gid_idx_list = []
        for i in range(len(timestamp)):
            try:
                t = timestamp[i].item()
            except:
                t = timestamp[i]
#             print('timestamp',t)
            gids = label_id_list[i]
            multi_view_adj, node_feat, node_id = read_fusion_graph(t,graph_path)
            gid_idx = nodelist2indexlist(gids.tolist(),node_id)
            gid_idx_list.append(gid_idx)
            
            hidden_GAT = []
            for adj in multi_view_adj:
                h = self.GATLayer(node_feat, adj)
                # h = self.GCNLayer(node_feat.float(), adj.float())
                hidden_GAT.append(h.cpu().detach().numpy())
            batch_graph_h.append(hidden_GAT)
        batch_graph_h = torch.tensor(batch_graph_h, device=DEVICE)#.to(device)
#         print('batch_graph_h',batch_graph_h.shape)# torch.Size([32, 3, 1608, 32])
        if torch.isnan(batch_graph_h).any():
            batch_graph_h = torch.nan_to_num(batch_graph_h)

        label_data_stfgn_batch, _ = self.stfgn(batch_graph_h) 
        label_data_stfgn_batch = get_certain_node_batch(label_data_stfgn_batch,gid_idx_list) # torch.Size([32, 10, 32])
        label_data_stfgn_batch = label_data_stfgn_batch.permute(1,0,2) #[10,32,32]
#         print('label_data_stfgn_batch',label_data_stfgn_batch.shape) # torch.Size([32, 10, 32])

        for j in range(self.prev_slot):
            # temp_approximate = F.relu(self.liner_t(h_t)) 
#             print('temp_approximate',temp_approximate.shape) #torch.Size([128, 32])

            unlabel_time_data = meo_unlabel.permute(1,0,2) # torch.Size([128, 4, 11]) ->  torch.Size([4, 128, 11])
            unlabel_time_data, _ = self.unlabel_lstm_1(unlabel_time_data) # torch.Size([4, 128, 11]) ->  torch.Size([4, 128, 32])
            unlabel_time_data = unlabel_time_data.float()[-1] # torch.Size([4, 128, 32]) ->  torch.Size([128, 32])
#             print('unlabel_time_data',unlabel_time_data.shape) #torch.Size([32, 32])
        
            unlabel_fea_data = nn.ReLU()(self.unlabel_linear_1(feature_unlabel)) # torch.Size([128, 135]) -> torch.Size([128, 32])
            unlabel_data = torch.cat((unlabel_time_data, unlabel_fea_data), 1) # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64])
            unlabel_data = nn.ReLU()(self.unlabel_linear_2(unlabel_data)) # torch.Size([128, 64]) -> torch.Size([128, 64])
#             print('unlabel_data',unlabel_data.shape) #torch.Size([32, 64])

            if add_labeled_embed: 
                label_time = torch.cat((ovi_label.unsqueeze(3), meo_label), 3) #torch.Size([128, 10, 4, 1]) + torch.Size([128, 10, 4, 11]) -> torch.Size([128, 10, 4, 12])
                label_time_data = []
                for i in range(self.num_station) :
                    lstm_tmp, _ = self.label_lstm_1(label_time[:,i,:,:].permute(1,0,2)) #torch.Size([128, 4, 12]) -> torch.Size([4, 128, 12]) -> torch.Size([4, 128, 32])
                    lstm_tmp = lstm_tmp.float()[-1] #torch.Size([4, 128, 32]) -> torch.Size([128, 32])
                    label_time_data.append(lstm_tmp)
    #             print('lstm_tmp',lstm_tmp.shape)
                label_feature = []
                for i in range(self.num_station) : #torch.Size([128, 10, 136]) -> torch.Size([128, 136]) -> torch.Size([128, 32])
                    label_feature.append(nn.ReLU()(self.label_linear_1(feature_label[:,i,:])))
    #             print('label_feature[0]',label_feature[0].shape)
                label_data = []    
                for i in range(self.num_station) : # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64]) -> torch.Size([128, 64])
                    label_data.append(nn.ReLU()(self.label_linear_2(torch.cat([label_time_data[i], label_feature[i],label_data_stfgn_batch[i]], 1))))
            
            else:
                label_data = []    
                for i in range(self.num_station) : # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64]) -> torch.Size([128, 64])
                    label_data.append(nn.ReLU()(self.label_linear_3( label_data_stfgn_batch[i] )))
            
            # torch.Size([128, 64]) * self.num_station   
#             print('label_data',label_data.shape)

            attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
            # attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
            attention_out = []
            for n,i in enumerate(label_data) :
                attention_out.append(attention_score[:,n].unsqueeze(1)*i)
            attention_out = torch.sum(torch.stack(attention_out).permute(1,0,2), 1) #torch.Size([128, 64])
        
            sp_approximate = F.relu(attention_out) #torch.Size([128, 64])
#             print('sp_approximate',sp_approximate.shape) #torch.Size([128, 64])
            
            # torch.Size([128, 64]), torch.Size([128, 32]), torch.Size([128, 64])
            # X_feat = torch.cat( [unlabel_data,temp_approximate,sp_approximate], dim=1 ) #torch.Size([128, 160])
#             print('X_feat',X_feat.shape) #torch.Size([128, 160])
            h_t = self.GRU2(sp_approximate)
#             print('h_t',h_t.shape) #torch.Size([128, 32])
        out = self.output_fc(h_t)
        if torch.isnan(out).any():
            # print('out',out)
            print('out has nan')
            print(out)
        # out = self.GRU_DE(h_t,h_t)
#         print('out',out.shape) #torch.Size([128, 1])
        return out
        
# %%



class LSTM_EN(nn.Module) :
    def __init__(self, input_size, hidden_dim, num_layers=1) :
        super(LSTM_EN, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, num_layers = self.num_layers)
        self.hidden = None
       
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).cuda())
    
    def forward(self, in_data):
        in_data = in_data.permute(1,0,2)
        out, self.hidden = self.lstm(in_data, self.hidden)
        return out, self.hidden

class LSTM_DE(nn.Module) :
    def __init__(self, num_steps, hidden_dim, num_layers=1) :
        super(LSTM_DE, self).__init__()
        self.lstm = nn.LSTM(1, hidden_dim, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, 1)
        self.num_steps = num_steps
    
    def forward(self, in_data, batch_size, hidden):
        #batch_size, num_steps = outputs.shape
        #in_data = torch.tensor([[0.0]] * batch_size, dtype=torch.float).cuda()
        #in_data = in_data.unsqueeze(0) * batch_size
        in_data = in_data.unsqueeze(0)
        result = []
        for i in range(self.num_steps):
            output, hidden = self.lstm(in_data, hidden)
            output = self.out(output[-1])
            result.append(output)
            in_data = output.unsqueeze(0)
        result = torch.stack(result).squeeze(2).permute(1,0)
        return result
    
class LSTM_predictor(nn.Module) :
    def __init__(self, input_size, hidden_dim, num_steps, num_layers=1) :
        super(LSTM_predictor, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.linear = nn.Linear(self.hidden_dim, 1)
        self.EN = LSTM_EN(self.input_size, self.hidden_dim, self.num_layers)
        self.DE = LSTM_DE(self.num_steps, self.hidden_dim, self.num_layers)
        
    def forward(self, in_data) :
        in_data = in_data.unsqueeze(1).repeat(1,8,1)
        self.EN.hidden = self.EN.init_hidden(in_data.shape[0])
        init_out, hidden = self.EN(in_data)
        init_out = nn.ReLU()(self.linear(init_out[-1].float()))
        out = self.DE(init_out, in_data.shape[0], hidden)
        #return torch.cat((init_out, out), 1) 
        return out

# In[13]:


class Attention_layer_zt(nn.Module) :
    def __init__(self, num_station, unlabel_size, label_size, hidden_size) :#10,64,64,16
        super(Attention_layer_zt, self).__init__()
        self.num = num_station
        self.un_emb = unlabel_size
        self.emb = label_size
        self.hidden = hidden_size
        self.linear_1 = nn.Linear(self.un_emb+self.emb, self.hidden)
        self.linear_2 = nn.Linear(self.hidden, 1)
        
    def forward(self, unlabel_emb, label_emb, dis_lab) :
        # unlabel_emb torch.Size([128, 64])
        # label_emb torch.Size([128, 64]) *10
        # dis_lab torch.Size([128, 10])
        
#         label_1 = []
        label_2 = []
        for k in label_emb : 
#             label_1.append(nn.ReLU()(self.linear_1(torch.cat((k, unlabel_emb), 1))))
            tmp = nn.ReLU()(self.linear_1(torch.cat((k, unlabel_emb), 1))) # torch.Size([128, 64])+torch.Size([128, 64]) -> torch.Size([128, 128]) -> torch.Size([128, 16])
            label_2.append(self.linear_2(tmp)) # torch.Size([128, 16]) -> torch.Size([128, 1])
#         for k in label_1 : 
#             label_2.append(self.linear_2(k))
        attention_out_ori = torch.stack(label_2).squeeze().permute(1,0) #torch.Size([128, 1])->torch.Size([128, 10])
#         print('attention_out_ori',attention_out_ori.shape)#torch.Size([128, 10])
#         print('dis_lab',dis_lab.shape)#torch.Size([128, 10])
        attention_out = attention_out_ori * dis_lab #torch.Size([128, 10]) * torch.Size([128, 10]) -> torch.Size([128, 10])
#         print('attention_out',attention_out.shape)#torch.Size([128, 10])
        attention_score = nn.Softmax(dim=1)(attention_out) #torch.Size([128, 10])
        return attention_score #, nn.Softmax(dim=1)(attention_out_ori)
# In[14]:

class ZTYao(nn.Module) :
    def __init__(self, num_station=10, output_size=1,hidden_lstm=32,hidden_linear=32,hidden_att=16) :
        super(ZTYao, self).__init__()
        self.num_station = num_station
        self.hidden_lstm = hidden_lstm
        self.hidden_linear = hidden_linear
        self.hidden_att = hidden_att
        self.output_size = output_size
        meo_len = len(meo_col)
        st_len = len(st_col)
        print(meo_len,st_len,self.output_size)

        self.unlabel_lstm_1 = nn.LSTM(meo_len, self.hidden_lstm)
        self.unlabel_linear_1 = nn.Linear(st_len*9, self.hidden_linear)
        self.unlabel_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm, self.hidden_linear*2)
        self.label_lstm_1 = nn.LSTM(meo_len+1, self.hidden_lstm)
        self.label_linear_1 = nn.Linear(st_len*9+1, self.hidden_linear)
        self.label_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm, self.hidden_linear*2)
        self.attention = Attention_layer_zt(num_station, self.hidden_linear*2, self.hidden_linear*2, self.hidden_att)
        self.out_linear_1 = nn.Linear(self.hidden_linear*2+self.hidden_linear*2, self.hidden_linear*2)
        self.out_linear_2 = nn.Linear(self.hidden_linear*2, output_size)
        self.out_pred = LSTM_predictor(self.hidden_linear*2, self.hidden_lstm, output_size)
        self.out_linear_3 = nn.Linear(2,1, bias=False)
        
    def forward(self, meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label, dis_label, h_t, timestamp, label_id_list) :
        
#         ovi_target torch.Size([128, 1])
#         meo_unlabel torch.Size([128, 4, 11])
#         feature_unlabel torch.Size([128, 135])
#         ovi_label torch.Size([128, 10, 4])
#         meo_label torch.Size([128, 10, 4, 11])
#         feature_label torch.Size([128, 10, 136])
#         inv_dis_label torch.Size([128, 10])
        
        unlabel_time_data = meo_unlabel.permute(1,0,2) # torch.Size([128, 4, 11]) ->  torch.Size([4, 128, 11])
        unlabel_time_data, _ = self.unlabel_lstm_1(unlabel_time_data) # torch.Size([4, 128, 11]) ->  torch.Size([4, 128, 32])
        unlabel_time_data = unlabel_time_data.float()[-1] # torch.Size([4, 128, 32]) ->  torch.Size([128, 32])
#         print('unlabel_time_data',unlabel_time_data.shape)
        
        unlabel_fea_data = nn.ReLU()(self.unlabel_linear_1(feature_unlabel)) # torch.Size([128, 135]) -> torch.Size([128, 32])
        unlabel_data = torch.cat((unlabel_time_data, unlabel_fea_data), 1) # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64])
        unlabel_data = nn.ReLU()(self.unlabel_linear_2(unlabel_data)) # torch.Size([128, 64]) -> torch.Size([128, 64])
#         print('unlabel_data',unlabel_data.shape)
        
        label_time = torch.cat((ovi_label.unsqueeze(3), meo_label), 3) #torch.Size([128, 10, 4, 1]) + torch.Size([128, 10, 4, 11]) -> torch.Size([128, 10, 4, 12])
        label_time_data = []
        label_feature = []
        label_data = [] 
        for i in range(self.num_station) :
            lstm_tmp, _ = self.label_lstm_1(label_time[:,i,:,:].permute(1,0,2)) #torch.Size([128, 4, 12]) -> torch.Size([4, 128, 12]) -> torch.Size([4, 128, 32])
            lstm_tmp = lstm_tmp.float()[-1] #torch.Size([4, 128, 32]) -> torch.Size([128, 32])
            label_time_data.append(lstm_tmp)

            label_feature.append(nn.ReLU()(self.label_linear_1(feature_label[:,i,:]))) 
            #torch.Size([128, 10, 136]) -> torch.Size([128, 136]) -> torch.Size([128, 32])

            label_data.append(nn.ReLU()(self.label_linear_2(torch.cat((label_time_data[i], label_feature[i]), 1))))
            # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64]) -> torch.Size([128, 64])
        
        attention_score = self.attention(unlabel_data, label_data, dis_label) # torch.Size([10])
#         print('attention_score',attention_score.shape) #torch.Size([128, 10])
        attention_out = []
        for n,i in enumerate(label_data) :
            attention_out.append(attention_score[:,n].unsqueeze(1)*i)
        attention_out = torch.sum(torch.stack(attention_out).permute(1,0,2), 1)
#         print('attention_out',attention_out.shape) #torch.Size([128, 64])
        
        data_out = torch.cat((unlabel_data, attention_out), 1) #torch.Size([128, 64])+torch.Size([128, 64]) ->torch.Size([128, 128])
        data_out = nn.ReLU()(self.out_linear_1(data_out)) # torch.Size([128, 128]) ->torch.Size([128, 64])
        data_out1 = nn.ReLU()(self.out_pred(data_out)) #torch.Size([128, 64])->torch.Size([128, 1])
        data_out2 = nn.ReLU()(self.out_linear_2(data_out)) #torch.Size([128, 64])->torch.Size([128, 1])
        #out =  data_out1 + data_out2
        data_out1 = data_out1.unsqueeze(2) #torch.Size([128, 1])->torch.Size([128, 1, 1])
        data_out2 = data_out2.unsqueeze(2) #torch.Size([128, 1])->torch.Size([128, 1, 1])
        out = []
        for i in range(self.output_size) :
            tmp = self.out_linear_3(torch.cat((data_out1[:,i], data_out2[:,i]), 1))
            out.append(tmp)
        # print(tmp.shape)
        # print(torch.stack(out).permute(1,0,2).shape)
        out = torch.squeeze(torch.stack(out).permute(1,0,2), 2)
        # print('out',out.shape) #torch.Size([128, 1])
        return out



class OneView(nn.Module) :
    def __init__(self, num_station=10, output_size=1) :
        super(OneView, self).__init__()
        self.num_station = num_station
        self.prev_slot = PREV_SLOT
        self.hidden_lstm = 32
        self.hidden_linear = 32
        self.hidden_gru = 32
        self.hidden_gnn = 32
        self.output_size = output_size
        meo_len = len(meo_col)
        st_len = len(st_col)
        print(meo_len,st_len)
        
        self.unlabel_lstm_1 = nn.LSTM(meo_len, self.hidden_lstm) #v
        self.unlabel_linear_1 = nn.Linear(st_len*9, self.hidden_linear)#v
        self.unlabel_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm, self.hidden_linear*2)
        self.label_lstm_1 = nn.LSTM(meo_len+1, self.hidden_lstm)#v
        self.label_linear_1 = nn.Linear(st_len*9+1, self.hidden_linear)#v
        self.label_linear_2 = nn.Linear(self.hidden_linear+self.hidden_lstm+self.hidden_gnn, self.hidden_linear*2)
        self.label_linear_3 = nn.Linear(self.hidden_gnn, self.hidden_linear*2)

#         self.stfgn = MultiView_GNN(in_features=146, out_size=self.hidden_gnn, gat_hidden_size=self.hidden_gnn)
        # self.stfgn = MultiView_GNN_batch(in_features=146, out_size=self.hidden_gnn, gat_hidden_size=self.hidden_gnn)
        self.GATLayer = GraphAttentionLayer(len(st_col)*9+len(meo_col), self.hidden_gnn, 0.2, 0.2)
        
        self.idw_attention = Attention_layer(num_station, self.hidden_linear*2, self.hidden_linear*2, 16)
        
        self.GRU = nn.GRUCell(self.hidden_gru+self.hidden_linear*2+self.hidden_linear*2 , self.hidden_gru, bias=True) #nn.GRUCell(input_size, hidden_size)
        self.liner_t = nn.Linear(self.hidden_gru, self.hidden_gru)
        self.output_fc = nn.Linear(self.hidden_gru ,output_size )
        
    def forward(self, meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label, dis_label, h_t, timestamp, label_id_list) :#timestamp
        
#         batch_adj = []
#         batch_feat = []
        batch_graph_h = []
        gid_idx_list = []
        for i in range(len(timestamp)):
            try:
                t = timestamp[i].item()
            except:
                t = timestamp[i]
#             print('timestamp',t)
            gids = label_id_list[i]
            adj, node_feat, node_id = read_fusion_graph_1view(t,graph_path)
            # print('node_feat',node_feat.shape)#torch.Size([1608, 146])
            gid_idx = nodelist2indexlist(gids.tolist(),node_id)
            gid_idx_list.append(gid_idx)
            node_num = int(node_feat.shape[0]/3)
            # hidden_GAT = []
            # for adj in multi_view_adj:
            #     h = self.GATLayer(node_feat, adj)
            #     hidden_GAT.append(h.cpu().detach().numpy())
            hidden_GAT = self.GATLayer(node_feat, adj)#torch.Size([1608, 32])
            batch_graph_h.append(hidden_GAT.cpu().detach().numpy()[node_num:node_num*2])
        # print('hidden_GAT',hidden_GAT.shape,'batch_graph_h',len(batch_graph_h)) #torch.Size([1608, 32]) 48
        batch_graph_h = torch.tensor(batch_graph_h, device=DEVICE)#.to(device) #torch.Size([48, 1608, 32])
        # print('batch_graph_h',batch_graph_h.shape) #torch.Size([128, 1608, 32])
        if torch.isnan(batch_graph_h).any():
            batch_graph_h = torch.nan_to_num(batch_graph_h)

        # label_data_stfgn_batch, _ = self.stfgn(batch_graph_h) 
        label_data_stfgn_batch = batch_graph_h #torch.Size([48, 1608, 32])
        label_data_stfgn_batch = get_certain_node_batch(label_data_stfgn_batch,gid_idx_list) # torch.Size([48, 10, 32])
        # print('label_data_stfgn_batch',label_data_stfgn_batch.shape)
        label_data_stfgn_batch = label_data_stfgn_batch.permute(1,0,2) #[10,48,32]
        # print('label_data_stfgn_batch',label_data_stfgn_batch.shape) # torch.Size([10, 48, 32])

        for j in range(self.prev_slot):
            temp_approximate = F.relu(self.liner_t(h_t)) 
#             print('temp_approximate',temp_approximate.shape) #torch.Size([128, 32])

            unlabel_time_data = meo_unlabel.permute(1,0,2) # torch.Size([128, 4, 11]) ->  torch.Size([4, 128, 11])
            unlabel_time_data, _ = self.unlabel_lstm_1(unlabel_time_data) # torch.Size([4, 128, 11]) ->  torch.Size([4, 128, 32])
            unlabel_time_data = unlabel_time_data.float()[-1] # torch.Size([4, 128, 32]) ->  torch.Size([128, 32])
#             print('unlabel_time_data',unlabel_time_data.shape) #torch.Size([32, 32])
        
            unlabel_fea_data = nn.ReLU()(self.unlabel_linear_1(feature_unlabel)) # torch.Size([128, 135]) -> torch.Size([128, 32])
            unlabel_data = torch.cat((unlabel_time_data, unlabel_fea_data), 1) # torch.Size([128, 32])+torch.Size([128, 32]) -> torch.Size([128, 64])
            unlabel_data = nn.ReLU()(self.unlabel_linear_2(unlabel_data)) # torch.Size([128, 64]) -> torch.Size([128, 64])
#             print('unlabel_data',unlabel_data.shape) #torch.Size([32, 64])

            label_time = torch.cat((ovi_label.unsqueeze(3), meo_label), 3) #torch.Size([128, 10, 4, 1]) + torch.Size([128, 10, 4, 11]) -> torch.Size([128, 10, 4, 12])
            label_time_data = []
            label_feature = []
            label_data = []
            for i in range(self.num_station) :
                lstm_tmp, _ = self.label_lstm_1(label_time[:,i,:,:].permute(1,0,2)) #torch.Size([128, 4, 12]) -> torch.Size([4, 128, 12]) -> torch.Size([4, 128, 32])
                lstm_tmp = lstm_tmp.float()[-1] #torch.Size([4, 128, 32]) -> torch.Size([128, 32])
                label_time_data.append(lstm_tmp)
                label_feature.append(nn.ReLU()(self.label_linear_1(feature_label[:,i,:])))
                label_data.append(nn.ReLU()(self.label_linear_2(torch.cat([label_time_data[i], label_feature[i],label_data_stfgn_batch[i]], 1))))

#             print('lstm_tmp',lstm_tmp.shape)
#             print('label_feature[0]',label_feature[0].shape)
#             print('label_data',label_data.shape)


            attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
            # attention_score = self.idw_attention(unlabel_data, label_data, dis_label)
            attention_out = []
            for n,i in enumerate(label_data) :
                attention_out.append(attention_score[:,n].unsqueeze(1)*i)
            attention_out = torch.sum(torch.stack(attention_out).permute(1,0,2), 1) #torch.Size([128, 64])
        
            sp_approximate = F.relu(attention_out) #torch.Size([128, 64])
#             print('sp_approximate',sp_approximate.shape) #torch.Size([128, 64])
            
            # torch.Size([128, 64]), torch.Size([128, 32]), torch.Size([128, 64])
            X_feat = torch.cat( [unlabel_data,temp_approximate,sp_approximate], dim=1 ) #torch.Size([128, 160])
#             print('X_feat',X_feat.shape) #torch.Size([128, 160])
            h_t = self.GRU(X_feat)
#             print('h_t',h_t.shape) #torch.Size([128, 32])
        out = nn.ReLU()(self.output_fc(h_t))
#         print('out',out.shape) #torch.Size([128, 1])
        return out
