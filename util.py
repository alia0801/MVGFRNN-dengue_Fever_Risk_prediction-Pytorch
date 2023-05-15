# %%
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error,mean_absolute_error
from tqdm import tqdm
import numpy as np
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
from config import *
import wandb
# %%
# import argparse
# parser = argparse.ArgumentParser(description='Test')
# parser.add_argument('--cv_k', type=int, default=0, help='k of cross validation')
# args = parser.parse_args()
# print('cv_k=',args.cv_k)
# cv_k=args.cv_k

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
dirs = ['B', 'T', 'L', 'R', 'RB', 'RT', 'LB', 'LT']
for grid_dir in dirs:
    col_name = pd.Series(st_col.copy()) + '_'+grid_dir
    col_name = col_name.tolist()
    st_col_all+=col_name
# %%

def get_splited_data(cv_k):

    with open('dataset_processed/'+DATASET+'/'+mask_folder_name+'/label_id.txt','r') as f:
        lines = f.read().split('\n')[:-1]
    labeled_id = [int(x) for x in lines]
    
    k_fold = 10 if DATASET=='dengue' else 5
    # print('k_fold',k_fold)

    split_id_list = []
    for i in range(k_fold):
        with open('dataset_processed/'+DATASET+'/'+mask_folder_name+'/unlabeled_split_'+str(i)+'.txt','r') as f:
            lines = f.read().split('\n')[:-1]
        gid = [int(x) for x in lines]
        # print(i,gid)
        split_id_list.append(gid)
    # print(split_id_list)

    k_val = (cv_k+(k_fold-2))%k_fold
    k_test = (cv_k+(k_fold-1))%k_fold
    if cv_k>=2:
        tmp = split_id_list[cv_k:]+split_id_list[:(cv_k-2)%k_fold]
    else:
        tmp = split_id_list[cv_k:cv_k+(k_fold-2)]
    # print(len(tmp))
    train_id = list(np.concatenate(tmp).flat)
    valid_id = split_id_list[k_val]
    test_id = split_id_list[k_test]

    print('label',len(labeled_id))
    print('train',len(train_id))
    print('valid',len(valid_id))
    print('test',len(test_id))
    return labeled_id, train_id, valid_id, test_id

# In[9]:

class station_data_dengue(Dataset) :
    def __init__(self, mode='train',cv_k=0) : #pred_slot
#         self.target_unlabel = np.array(valid_index)[:,0].tolist()
#         self.target_label = np.array(valid_index)[:,1:].tolist()
        self.all_data = pd.read_csv('dataset_processed/'+DATASET+'/all_processed_data_9box_nexty.csv')
        self.grid_df_neighbor = pd.read_csv('dataset_processed/'+DATASET+'/'+mask_folder_name+'/unlabel_grid_100neighbor_dist.csv')
        self.all_static_features = self.all_data[['id']+st_col_all].drop_duplicates().reset_index()
        labeled_id, train_id, valid_id, test_id = get_splited_data(cv_k)
        
        self.label_id = labeled_id
        if mode=='train':
            self.unlabel_id = train_id
        elif mode=='test':
            self.unlabel_id = test_id
        else:
            self.unlabel_id = valid_id
            
        self.label_data = self.all_data[self.all_data['id'].isin(self.label_id)].reset_index()
        self.unlabel_data = self.all_data[self.all_data['id'].isin(self.unlabel_id)].reset_index()
        
        self.label_ovi_target = self.label_data.egg_num.values
        self.unlabel_ovi_target = self.unlabel_data.egg_num.values
        self.label_meo = self.label_data[['time','id']+meo_col]
        self.label_feature = self.all_static_features[self.all_static_features['id'].isin(self.label_id)].reset_index()
        self.unlabel_meo = self.unlabel_data[['time','id']+meo_col]
        self.unlabel_feature = self.all_static_features[self.all_static_features['id'].isin(self.unlabel_id)].reset_index()
        
        self.near_station = self.grid_df_neighbor[['grid_id']+nearest_col]
        self.near_station_dist = self.grid_df_neighbor[['grid_id']+near_dist_col]
        
        self.prev_slot = PREV_SLOT
        self.pred_slot = model_output_size #pred_slot
        
    def __getitem__(self, index) :
        unlabel_id = self.unlabel_data.id[index]
        label_id_list = self.near_station[self.near_station['grid_id']==unlabel_id][nearest_col].values[0]
        timestamp = self.unlabel_data.time[index]
        
        if timestamp>=(202206-self.pred_slot+1):
            # ovi_target = torch.from_numpy(np.zeros(self.pred_slot)).float().to(device)
            ovi_target = torch.zeros(self.pred_slot,device=DEVICE).float()
        else:
            ovi_target = torch.from_numpy(self.unlabel_ovi_target[index+1:index+self.pred_slot+1]).float().to(DEVICE)
            
        if timestamp<(201806+self.prev_slot):
            # meo_unlabel = torch.from_numpy(np.zeros((self.prev_slot,11))).float().to(device)
            meo_unlabel = torch.zeros( self.prev_slot,11 ,device=DEVICE).float()
        else:
            meo_unlabel = torch.from_numpy(self.unlabel_meo[index-self.prev_slot:index][meo_col].values).float().to(DEVICE)
        feature_unlabel = torch.from_numpy(self.unlabel_feature[self.unlabel_feature['id']==unlabel_id][st_col_all].values[0]).float().to(DEVICE)
    
        ovi_label, meo_label, feature_label = self.get_feats_label(label_id_list,timestamp)
        dis_label, inv_dis_label = self.get_dist(unlabel_id)
        feature_label_out = torch.cat((feature_label, dis_label.unsqueeze(1) ), 1)
        
        return ovi_target, meo_unlabel, feature_unlabel,    ovi_label,      meo_label,                             feature_label_out, inv_dis_label, torch.from_numpy(label_id_list).float().to(DEVICE), timestamp
     # shape: [pred_slot],[prev_slot,11],   [135], [k_neighbor,prev_slot], [k_neighbor,prev_slot,11], [k_neighbor,136], [k_neighbor], [k_neighbor]
     
    def get_feats_label(self, label_id_list,timestamp) :
        meo_out = []
        ovi_out = []
        feat_out = []
        for gid in label_id_list:
            if timestamp<201810:
                # try:
                idx = self.label_meo[(self.label_meo['time']==timestamp)&(self.label_meo['id']==gid)].index.values[0]
                # except:
                    # print(timestamp,gid)
                    # idx = 4
                tmp_len = self.prev_slot-(201810-timestamp)
                tmp_meo = (self.label_meo[idx-tmp_len:idx][meo_col].values)
                tmp_ovi = (self.label_ovi_target[idx-tmp_len:idx])
                meo = torch.from_numpy(np.concatenate((np.zeros(self.prev_slot-tmp_len), tmp_meo), axis=None)).float().to(DEVICE)
                ovi = torch.from_numpy(np.concatenate((np.zeros(self.prev_slot-tmp_len), tmp_ovi), axis=None)).float().to(DEVICE)
            else:
                idx = self.label_meo[(self.label_meo['time']==timestamp)&(self.label_meo['id']==gid)].index.values[0]
                meo = (self.label_meo[idx-self.prev_slot:idx][meo_col].values)
                ovi = (self.label_ovi_target[idx-self.prev_slot:idx])
            # try:
            feat = (self.label_feature[self.label_feature['id']==gid][st_col_all].values[0])
            # except:
                # print(gid)
            meo_out.append(meo)
            ovi_out.append(ovi)
            feat_out.append(feat)
#         print(timestamp,np.array(meo_out).shape,np.array(ovi_out).shape,np.array(feat_out).shape)
        try:
            # final_ovi = torch.from_numpy(np.array(ovi_out)).float().to(device)
            # final_meo = torch.from_numpy(np.array(meo_out)).float().to(device)
            final_ovi = torch.tensor(ovi_out,device=DEVICE).float()
            final_meo = torch.tensor(meo_out,device=DEVICE).float()
            
        except:
            # final_ovi = torch.from_numpy( np.zeros((k_neighbor,self.prev_slot)) ).float().to(device)
            # final_meo = torch.from_numpy( np.zeros((k_neighbor,self.prev_slot,11)) ).float().to(device)
            final_ovi = torch.zeros( k_neighbor,self.prev_slot ,device=DEVICE).float()
            final_meo = torch.zeros( k_neighbor,self.prev_slot,len(meo_col) ,device=DEVICE).float()
        return final_ovi, final_meo, torch.tensor(feat_out,device=DEVICE).float() #torch.from_numpy(np.array(feat_out)).float().to(device)
    
    def get_dist(self, unlabel_id):
        dist = self.near_station_dist[self.near_station_dist['grid_id']==unlabel_id][near_dist_col].values[0]
        inv_dist = 1./dist
        # return torch.from_numpy(np.array(dist)).float().to(device), torch.from_numpy(np.array(inv_dist)).float().to(device)
        return torch.tensor(dist,device=DEVICE).float(), torch.tensor(inv_dist,device=DEVICE).float()
        
    def __len__(self) :
        return len(self.unlabel_ovi_target)
# %% 
class station_data_AQI(Dataset) :
    def __init__(self, mode='train',cv_k=0) : #pred_slot
#         self.target_unlabel = np.array(valid_index)[:,0].tolist()
#         self.target_label = np.array(valid_index)[:,1:].tolist()
        self.all_data = pd.read_csv('dataset_processed/aqi/all_processed_data_9box_nexty.csv')
        self.grid_df_neighbor = pd.read_csv('dataset_processed/aqi/unlabel_grid_15neighbor_dist.csv')
        self.all_static_features = self.all_data[['id']+st_col_all].drop_duplicates().reset_index()
        labeled_id, train_id, valid_id, test_id = get_splited_data(cv_k)
        
        self.label_id = labeled_id
        if mode=='train':
            self.unlabel_id = train_id
        elif mode=='test':
            self.unlabel_id = test_id
        else:
            self.unlabel_id = valid_id
        # print(self.unlabel_id)
        self.label_data = self.all_data[self.all_data['id'].isin(self.label_id)].reset_index()
        self.unlabel_data = self.all_data[self.all_data['id'].isin(self.unlabel_id)].reset_index()
        # print(self.unlabel_data.shape)

        self.label_ovi_target = self.label_data['original_pm25_target'].values
        self.unlabel_ovi_target = self.unlabel_data['original_pm25_target'].values
        # self.label_ovi_target = self.label_data['PM2.5_target'].values
        # self.unlabel_ovi_target = self.unlabel_data['PM2.5_target'].values
        self.label_meo = self.label_data[['time','id']+meo_col]
        self.label_feature = self.all_static_features[self.all_static_features['id'].isin(self.label_id)].reset_index()
        self.unlabel_meo = self.unlabel_data[['time','id']+meo_col]
        self.unlabel_feature = self.all_static_features[self.all_static_features['id'].isin(self.unlabel_id)].reset_index()
        # print('label_ovi_target',self.label_ovi_target.shape)
        # print('unlabel_ovi_target',self.unlabel_ovi_target.shape)
        # print('label_meo',self.label_meo.shape)
        # print('label_feature',self.label_feature.shape)
        # print('unlabel_meo',self.unlabel_meo.shape)
        # print('unlabel_feature',self.unlabel_feature.shape)

        self.near_station = self.grid_df_neighbor[['grid_id']+nearest_col]
        self.near_station_dist = self.grid_df_neighbor[['grid_id']+near_dist_col]
        
        self.prev_slot = PREV_SLOT
        self.pred_slot = model_output_size #pred_slot
        print('self.prev_slot, self.pred_slot = ',self.prev_slot,self.pred_slot)
        
    def __getitem__(self, index) :
        unlabel_id = self.unlabel_data.id[index]
        label_id_list = self.near_station[self.near_station['grid_id']==unlabel_id][nearest_col].values[0]
        timestamp = self.unlabel_data.time[index]
        
        if timestamp>=('2018-03-31-03'):
            # ovi_target = torch.from_numpy(np.zeros(self.pred_slot)).float().to(device)
            ovi_target = torch.zeros(self.pred_slot,device=DEVICE).float()
        else:
            ovi_target = torch.from_numpy(self.unlabel_ovi_target[index:index+self.pred_slot]).float().to(DEVICE)
            
        if timestamp<('2017-01-02-02'):
            # meo_unlabel = torch.from_numpy(np.zeros((self.prev_slot,11))).float().to(device)
            meo_unlabel = torch.zeros( self.prev_slot,len(meo_col) ,device=DEVICE).float()
        else:
            meo_unlabel = torch.from_numpy(self.unlabel_meo[index-self.prev_slot:index][meo_col].values).float().to(DEVICE)
        if meo_unlabel.shape!=(24,12):
            meo_unlabel = torch.zeros( self.prev_slot,len(meo_col) ,device=DEVICE).float()
        feature_unlabel = torch.from_numpy(self.unlabel_feature[self.unlabel_feature['id']==unlabel_id][st_col_all].values[0]).float().to(DEVICE)
    
        ovi_label, meo_label, feature_label = self.get_feats_label(label_id_list,timestamp)
        dis_label, inv_dis_label = self.get_dist(unlabel_id)
        feature_label_out = torch.cat((feature_label, dis_label.unsqueeze(1) ), 1)
        
        return ovi_target, meo_unlabel, feature_unlabel,    ovi_label,      meo_label,                             feature_label_out, inv_dis_label, torch.from_numpy(label_id_list).float().to(DEVICE), timestamp
     # shape: [pred_slot],[prev_slot,11],   [135], [k_neighbor,prev_slot], [k_neighbor,prev_slot,11], [k_neighbor,136], [k_neighbor], [k_neighbor]
     
    def get_feats_label(self, label_id_list,timestamp) :
        meo_out = []
        ovi_out = []
        feat_out = []
        for gid in label_id_list:
            if timestamp<'2017-01-02-02':
                idx = self.label_meo[(self.label_meo['time']==timestamp)&(self.label_meo['id']==gid)].index.values[0]
                splited_tmp = timestamp.split('-')
                now_date,now_hour = int(splited_tmp[2]),int(splited_tmp[3])
                # tmp_len = self.prev_slot-(201810-timestamp)
                diff_time = (24-now_hour+2) if now_date==1 else 2-now_hour
                tmp_len = self.prev_slot-(diff_time)
                tmp_meo = (self.label_meo[idx-tmp_len:idx][meo_col].values)
                tmp_ovi = (self.label_ovi_target[idx-tmp_len:idx])
                meo = torch.from_numpy(np.concatenate((np.zeros(self.prev_slot-tmp_len), tmp_meo), axis=None)).float().to(DEVICE)
                ovi = torch.from_numpy(np.concatenate((np.zeros(self.prev_slot-tmp_len), tmp_ovi), axis=None)).float().to(DEVICE)
            else:
                idx = self.label_meo[(self.label_meo['time']==timestamp)&(self.label_meo['id']==gid)].index.values[0]
                meo = (self.label_meo[idx-self.prev_slot:idx][meo_col].values)
                ovi = (self.label_ovi_target[idx-self.prev_slot:idx])
            feat = (self.label_feature[self.label_feature['id']==gid][st_col_all].values[0])
            meo_out.append(meo)
            ovi_out.append(ovi)
            feat_out.append(feat)
#         print(timestamp,np.array(meo_out).shape,np.array(ovi_out).shape,np.array(feat_out).shape)
        try:
            # final_ovi = torch.from_numpy(np.array(ovi_out)).float().to(device)
            # final_meo = torch.from_numpy(np.array(meo_out)).float().to(device)
            final_ovi = torch.tensor(ovi_out,device=DEVICE).float()
            final_meo = torch.tensor(meo_out,device=DEVICE).float()
            
        except:
            # final_ovi = torch.from_numpy( np.zeros((k_neighbor,self.prev_slot)) ).float().to(device)
            # final_meo = torch.from_numpy( np.zeros((k_neighbor,self.prev_slot,11)) ).float().to(device)
            final_ovi = torch.zeros( k_neighbor,self.prev_slot ,device=DEVICE).float()
            final_meo = torch.zeros( k_neighbor,self.prev_slot,len(meo_col) ,device=DEVICE).float()
        return final_ovi, final_meo, torch.tensor(feat_out,device=DEVICE).float() #torch.from_numpy(np.array(feat_out)).float().to(device)
    
    def get_dist(self, unlabel_id):
        dist = self.near_station_dist[self.near_station_dist['grid_id']==unlabel_id][near_dist_col].values[0]
        inv_dist = 1./dist
        # return torch.from_numpy(np.array(dist)).float().to(device), torch.from_numpy(np.array(inv_dist)).float().to(device)
        return torch.tensor(dist,device=DEVICE).float(), torch.tensor(inv_dist,device=DEVICE).float()
        
    def __len__(self) :
        return len(self.unlabel_ovi_target)
# %%    
class station_data_AQI_old(Dataset) :
    def __init__(self, valid_index, AQData, AQTarget, AQID, Meo, Feature, pred_slot) :
        self.target_unlabel = np.array(valid_index)[:,0].tolist()
        self.target_label = np.array(valid_index)[:,1:].tolist()
        self.aq = AQData
        self.aqtarget = AQTarget
        self.id = AQID
        self.meo = Meo
        self.feature = Feature
        self.prev_slot = 23
        self.pred_slot = pred_slot
        
    def __getitem__(self, index) :
        unlabel_index = self.target_unlabel[index]
        unlabel_id = self.id[unlabel_index]
        label_index_list = self.target_label[index]
        label_id = self.id[label_index_list]
        aq_target = torch.from_numpy(self.aqtarget[unlabel_index:unlabel_index+pred_slot+1]).float().cuda()
        
        meo_unlabel = torch.from_numpy(self.meo[unlabel_index-self.prev_slot:unlabel_index+1]).float().cuda()
        feature_unlabel = torch.from_numpy(self.feature[unlabel_id-1]).float().cuda()
        
        aq_label, meo_label = self.get_time_label(label_index_list)
        feature_label = torch.from_numpy(self.feature[label_id-1]).float().cuda()
        dis_label, inv_dis_label = self.cal_dis(unlabel_id, label_id)
        feature_label_out = torch.cat((feature_label, dis_label.unsqueeze(1)), 1)
        return aq_target, meo_unlabel, feature_unlabel, aq_label, meo_label, feature_label_out, inv_dis_label
        
    def get_time_label(self, index_list) :
        meo_out = []
        aq_out = []
        for i in range(self.prev_slot, -1, -1) :
            tmp = list(map(lambda x:x-i, index_list))
            meo = self.meo[tmp]
            aq = self.aq[tmp]
            meo_out.append(meo)
            aq_out.append(aq)
            #out.append(np.concatenate((aq, meo), 0))
        meo_out = np.transpose(np.array(meo_out), (1,0,2))
        aq_out = np.transpose(np.array(aq_out), (1,0))
        return torch.from_numpy(np.array(aq_out)).float().cuda(), torch.from_numpy(np.array(meo_out)).float().cuda()
    
    def cal_dis(self, id_u, id_l) :
        max_dis = math.sqrt(60**2 + 51**2)
        id_ux = int((id_u-1)/61) + 1
        id_uy = int((id_u-1)%61) + 1
        out = []
        inverse_out = []
        for k in id_l :
            id_lx = int((k-1)/61) + 1
            id_ly = int((k-1)%61) + 1
            out.append(math.sqrt((id_lx-id_ux)**2 + (id_ly-id_uy)**2)/max_dis)
            inverse_out.append(1/(math.sqrt((id_lx-id_ux)**2 + (id_ly-id_uy)**2)))
        return torch.from_numpy(np.array(out)).float().cuda(), torch.from_numpy(np.array(inverse_out)).float().cuda()
    
    def __len__(self) :
        return len(self.target_unlabel)

# %%  
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))
    
class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()
        
    def forward(self,yhat,y):
        return self.mae(yhat,y)

class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss()
        
    def forward(self,yhat,y):
        return self.huber(yhat,y)
    
loss_func_rmse = RMSELoss()
loss_func_mae = MAELoss()
loss_func_huber = HuberLoss()

def avg_loss(loss_func, y_pred, y_true) :
    loss_1 = loss_func(y_pred, y_true)
    loss_2 = loss_func(y_pred[:,0:3], y_true[:,0:3])
    loss_3 = loss_func(y_pred[:,3:6], y_true[:,3:6])
    loss_4 = loss_func(y_pred[:,6:9], y_true[:,6:9])
    loss_5 = loss_func(y_pred[:,9:], y_true[:,9:])
    return (loss_1, loss_2, loss_3, loss_4, loss_5)

def testing(net, data_loader) :
    len_data = len(data_loader)
    net.eval()
    
    if DATASET=='dengue':
        loss_sum_rmse = 0.0
        loss_sum_mae = 0.0
        loss_sum_huber = 0.0
        with torch.no_grad() :
            for ovi_target, meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label_out, inv_dis_label,label_id_list, timestamp in tqdm(data_loader):
                ovi_target = torch.nan_to_num(ovi_target)
                meo_unlabel = torch.nan_to_num(meo_unlabel)
                feature_unlabel = torch.nan_to_num(feature_unlabel)
                ovi_label = torch.nan_to_num(ovi_label)
                meo_label = torch.nan_to_num(meo_label)
                feature_label_out = torch.nan_to_num(feature_label_out)

                h_t = torch.zeros(ovi_target.shape[0],32, device=DEVICE)#.to(device=device)
                output = net(meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label_out, inv_dis_label, h_t, timestamp, label_id_list)
                # loss = loss_func_huber(output, ovi_target)
                ovi_target = torch.nan_to_num(ovi_target)
                output = torch.nan_to_num(output)
                # print('output',output)
                # print('ovi_target',ovi_target)
                rmse = loss_func_rmse(output, ovi_target)
                mae = loss_func_mae(output, ovi_target)
                loss = (rmse+mae)/2
                loss_sum_huber += loss#.item()
                loss_sum_rmse += rmse#.item()
                loss_sum_mae += mae#.item()
            print('output',output)
            print('ovi_target',ovi_target)
            avg_huber = loss_sum_huber/len_data
            avg_rmse = loss_sum_rmse/len_data
            avg_mae = loss_sum_mae/len_data
        return avg_huber, avg_rmse,avg_mae
    
    else:
        loss_sum_rmse = 0.0
        loss_sum_mae = 0.0
        loss_sum_huber = 0.0
        avg_rmse = np.zeros((5))
        avg_mae = np.zeros((5))
        with torch.no_grad() :
            for ovi_target, meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label_out, inv_dis_label,label_id_list, timestamp in tqdm(data_loader):
                ovi_target = torch.nan_to_num(ovi_target)
                meo_unlabel = torch.nan_to_num(meo_unlabel)
                feature_unlabel = torch.nan_to_num(feature_unlabel)
                ovi_label = torch.nan_to_num(ovi_label)
                meo_label = torch.nan_to_num(meo_label)
                feature_label_out = torch.nan_to_num(feature_label_out)

                h_t = torch.zeros(ovi_target.shape[0],32, device=DEVICE)#.to(device=device)
                output = net(meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label_out, inv_dis_label, h_t, timestamp, label_id_list)
                # loss = loss_func_huber(output, ovi_target)
                ovi_target = torch.nan_to_num(ovi_target)
                output = torch.nan_to_num(output)
                # print('output',output)
                # print('ovi_target',ovi_target)
                # rmse = loss_func_rmse(output, ovi_target)
                # mae = loss_func_mae(output, ovi_target)
                loss_part_rmse = avg_loss(loss_func_rmse,output, ovi_target)
                loss_part_mae = avg_loss(loss_func_mae,output, ovi_target)
                for n, k in enumerate(loss_part_rmse) :
                    avg_rmse[n] += k
                for n, k in enumerate(loss_part_mae) :
                    avg_mae[n] += k
                loss = (loss_part_rmse[0]+loss_part_mae[0])/2
                loss_sum_huber += loss#.item()
                # loss_sum_rmse += loss_part_rmse[0]#.item()
                # loss_sum_mae += loss_part_mae[0]#.item()
            print('output min, max', torch.min(output), torch.max(output))
            print('target min, max', torch.min(ovi_target), torch.max(ovi_target))
            # print('output',output)
            # print('ovi_target',ovi_target)
            # avg_huber = loss_sum_huber/len_data
            # avg_rmse = loss_sum_rmse/len_data
            # avg_mae = loss_sum_mae/len_data
        return loss_sum_huber/len_data, avg_rmse/len_data,avg_mae/len_data


