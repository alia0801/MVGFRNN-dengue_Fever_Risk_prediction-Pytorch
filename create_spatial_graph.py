# %%
import pandas as pd
import numpy as np
import os
from fastdtw import fastdtw
import time
import matplotlib.pyplot as plt
import hdbscan
import math
from tqdm import tqdm
from util import *
from config import *
# mask_folder_name = 'mask05'
print(mask_folder_name)
if not os.path.exists('dataset_processed/'+DATASET+'/'+mask_folder_name+'/graph_data/v2'):
    os.makedirs('dataset_processed/'+DATASET+'/'+mask_folder_name+'/graph_data/v2')
    os.makedirs('dataset_processed/'+DATASET+'/'+mask_folder_name+'/graph_data/v2/adj_spatial_dist')
    os.makedirs('dataset_processed/'+DATASET+'/'+mask_folder_name+'/graph_data/v2/feat')
    os.makedirs('dataset_processed/'+DATASET+'/'+mask_folder_name+'/graph_data/v2/adj_spatial_cluster')
    
# %%
if DATASET =='dengue':
    meo_col = ['測站氣壓(hPa)', '氣溫(℃)', '相對溼度(%)', '風速(m/s)', '降水量(mm)','測站最高氣壓(hPa)', 
                  '最高氣溫(℃)', '最大陣風(m/s)', '測站最低氣壓(hPa)','最低氣溫(℃)', '最小相對溼度(%)']
    st_col = ['watersupply_hole','well', 'sewage_hole', 'underwater_con', 'pumping',
           'watersupply_others', 'watersupply_value', 'food_poi', 'rainwater_hole',
           'river', 'drainname', 'sewage_well', 'gaugingstation', 'underpass', 'watersupply_firehydrant']
    
if DATASET=='aqi':
    st_col = ['Hotel','Food','Education','Culture','Financial','Shopping','Medical','Entertainment','Transportation Spots','Company',
                'Vehicle Service','Sport','Daily Life','Institution','primary','secondary','pedestrian','highway','water','industrial','green','residential']
    meo_col = ['temperature','pressure','humidity','wind_speed/kph','A','B','C','D','E','F','G','H',]
st_col_all = st_col.copy()
dirs = ['B', 'T', 'L', 'R', 'RB', 'RT', 'LB', 'LT']
for grid_dir in dirs:
    col_name = pd.Series(st_col.copy()) + '_'+grid_dir
    col_name = col_name.tolist()
    st_col_all+=col_name
all_data = pd.read_csv('dataset_processed/'+DATASET+'/all_processed_data_9box_nexty.csv')
all_time = all_data.time.unique()
# %%
with open('dataset_processed/'+DATASET+'/'+mask_folder_name+'/label_id.txt','r') as f:
    lines = f.read().split('\n')[:-1]
labeled_id = [int(x) for x in lines]
with open('dataset_processed/'+DATASET+'/'+mask_folder_name+'/train_id.txt','r') as f:
    lines = f.read().split('\n')[:-1]
train_id = [int(x) for x in lines]
with open('dataset_processed/'+DATASET+'/'+mask_folder_name+'/valid_id.txt','r') as f:
    lines = f.read().split('\n')[:-1]
valid_id = [int(x) for x in lines]
with open('dataset_processed/'+DATASET+'/'+mask_folder_name+'/test_id.txt','r') as f:
    lines = f.read().split('\n')[:-1]
test_id = [int(x) for x in lines]

# print('label',len(labeled_id),labeled_id)
# print('train',len(train_id),train_id)
# print('valid',len(valid_id),valid_id)
# print('test',len(test_id),test_id)
print('label',len(labeled_id))
print('train',len(train_id))
print('valid',len(valid_id))
print('test',len(test_id))
# %%

def spatial_graph_dist(dist_thresh = 500):
    if DATASET=='aqi':
        nei_file_name = 'label_grid_15neighbor_dist.csv'
    if DATASET=='dengue':
        nei_file_name = 'label_grid_100neighbor_dist.csv'
    grid_df = pd.read_csv('dataset_processed/'+DATASET+'/'+mask_folder_name+'/'+nei_file_name)
    sub_df = grid_df[grid_df['grid_id'].isin(labeled_id)].reset_index()
    # gid2nid_dict = sub_df.grid_id.to_dict()
    # print(grid_df)
    # print(labeled_id)
    sub_df = sub_df[['grid_id','x_center','y_center']]
    sub_df['x_center'] *= 100000
    sub_df['y_center'] *= 111000
    # print(sub_df)
    
    adj = np.zeros((len(sub_df),len(sub_df))) # np.eye(len(sub_df))
    for i in range(len(sub_df)):
        for j in range(len(sub_df)):
            if i<j:
                x1 = sub_df['x_center'][i]
                x2 = sub_df['x_center'][j]
                y1 = sub_df['y_center'][i]
                y2 = sub_df['y_center'][j]
                dist = ((x1-x2)**2 + (y1-y2)**2)**0.5
                adj[i][j] = dist
                adj[j][i] = dist
    # df_adj_1dsort = np.sort(adj.copy(), axis=None)
    # dist_thresh = df_adj_1dsort[round(len(df_adj_1dsort)*(1-connect_ratio))]
    # print(dist_thresh)

    df_adj = pd.DataFrame(adj)
    df_adj_cp = df_adj.copy(deep=True)
    df_adj[df_adj_cp<=dist_thresh] = 1
    df_adj[df_adj_cp>dist_thresh] = 0
    return df_adj.values
# %%

def spatial_graph_cluster(timestamp,connect_ratio = 0.4):
    # all_data = pd.read_csv('dataset_processed/'+DATASET+'/'+mask_folder_name+'/all_processed_data_9box_nexty.csv')
    # all_time = all_data.time.unique()
    # print(all_time)
    
    time_idx = np.where(all_time == timestamp)[0][0]
    time_list = all_time[time_idx-1:time_idx+2]
    # print(time_list)
    
    label_feat_org = all_data[all_data['id'].isin(labeled_id)]#.reset_index()
    label_feat = label_feat_org[label_feat_org['time']==timestamp][['id']+st_col_all+meo_col].reset_index(drop=True)
    
    all_feat=[]
    if len(time_list)!=3:
        tmp_df = label_feat[st_col_all+meo_col].reset_index(drop=True)
        all_feat = [tmp_df, tmp_df, tmp_df]
    else:
        pre_feat = label_feat_org[label_feat_org['time']==time_list[0]][st_col_all+meo_col].reset_index(drop=True)
        tmp_df = label_feat[st_col_all+meo_col].reset_index(drop=True)
        next_feat = label_feat_org[label_feat_org['time']==time_list[2]][st_col_all+meo_col].reset_index(drop=True)
        all_feat = [pre_feat, tmp_df , next_feat]
    
    label_feat_to_cluster = label_feat.copy(deep=True)
    # print(label_feat_to_cluster,label_feat_to_cluster.shape)#(15, 211)
    del label_feat_to_cluster['id']
    try:
        label_feat_np_tmp = label_feat_to_cluster.values
        label_feat_np=[]
        for tmp in label_feat_np_tmp:
            mylist = [0 if math.isnan(x) else x for x in tmp]
            label_feat_np.append(mylist)
        # print(label_feat_np,len(label_feat_np))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=3, prediction_data=True).fit(label_feat_np)
        soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
        # print(soft_clusters)
        soft_clusters_df = pd.DataFrame(soft_clusters)
    #     soft_clusters_df[soft_clusters_df>0.15] = 1
    #     soft_clusters_df[soft_clusters_df<=0.15] = 0
    #     print(soft_clusters)
        # print(soft_clusters_df)

        adj = np.eye(len(soft_clusters_df)) #np.zeros((len(soft_clusters_df),len(soft_clusters_df)))
        df_adj_1dsort = np.sort(adj.copy(), axis=None)
    #     print(df_adj_1dsort)
        dist_thresh = df_adj_1dsort[round(len(df_adj_1dsort)*(1-connect_ratio))]
        # print(dist_thresh)
        for col in soft_clusters_df.columns:
            idx_list = soft_clusters_df[soft_clusters_df[col] > dist_thresh].index.tolist()
            for i in range(len(idx_list)):
                for j in range(len(idx_list)):
                    if i<=j:
                        adj[i][j] = 1
                        adj[j][i] = 1
                    else:
                        break
    # print(adj)
    except:
        adj = np.eye(len(label_feat_np))
    return adj , pd.concat(all_feat,axis=0), label_feat[['id']].reset_index(drop=True)

# _,_,_ = spatial_graph_cluster(201906)
# _,_,_ = spatial_graph_cluster('2017-01-05-03')

# %%
def temporal_graph_dtw(timestamp,dist_thresh=25,connect_ratio = 0.05,pre_slot=8):
    try:
        # all_data = pd.read_csv('dataset_processed/'+DATASET+'/'+mask_folder_name+'/all_processed_data_9box_nexty.csv')
        # all_time = all_data.time.unique()
    #     all_id = all_data.id.unique()
    #     print('cluster_size',cluster_size)
    #     timestamp = 201901
        time_idx = np.where(all_time == timestamp)[0][0]
        if time_idx>pre_slot:
            time_list = all_time[time_idx-pre_slot:time_idx]
        else:
            time_list = all_time[:time_idx]
        # print(time_list)
        if len(time_list)==0:
    #         print(all_data.id.unique())
            adj = np.eye(len(labeled_id))
    #         print(adj.shape)
            return adj
        label_feat = all_data[all_data['id'].isin(labeled_id)]#.reset_index()
        label_feat = label_feat[label_feat['time'].isin(time_list)][['id','PM2.5']].reset_index(drop=True)

    #     df_dict = {}
        ts_list = []
        for gid, sub_df in label_feat.groupby(by=['id']):
            del sub_df['id']
            sub_df_values = sub_df.values.reshape(-1)
    #         df_dict[gid] = sub_df_values
            ts_list.append(sub_df_values)

        adj = np.zeros((len(ts_list),len(ts_list))) #np.eye(len(ts_list))
        for i in range(len(ts_list)):
            for j in range(len(ts_list)):
                if i<j:
                    ts1 = ts_list[i]
                    ts2 = ts_list[j]
                    dist,_ = fastdtw(ts1, ts2)
                    adj[i][j] = dist
                    adj[j][i] = dist
        df_adj = pd.DataFrame(adj)
        df_adj_1dsort = np.sort(adj.copy(), axis=None)
    #     print(df_adj_1dsort)
        dist_thresh = df_adj_1dsort[round(len(df_adj_1dsort)*connect_ratio)]
        # print('DTW dist_thresh',dist_thresh)
    #     print('0.5',df_adj_1dsort[round(len(df_adj_1dsort)*0.5)]) #0 9
    #     print('0.55',df_adj_1dsort[round(len(df_adj_1dsort)*0.55)]) #12 12
    #     print('0.6',df_adj_1dsort[round(len(df_adj_1dsort)*0.6)]) #23 18 
    #     print('0.65',df_adj_1dsort[round(len(df_adj_1dsort)*0.65)]) #30 24 
    #     print('0.7',df_adj_1dsort[round(len(df_adj_1dsort)*0.7)]) #40 36
        df_adj_cp = df_adj.copy(deep=True)
        df_adj[df_adj_cp<=dist_thresh] = 1
        df_adj[df_adj_cp>dist_thresh] = 0
    except:
        print('error',timestamp)
        df_adj = np.zeros((len(labeled_id),len(labeled_id)))
    return df_adj.values
# temporal_graph_dtw(201810)
# temporal_graph_dtw('2017-01-05-03')
# %%
adj_spatial_local = spatial_graph_dist()
N = len(adj_spatial_local)
adj = np.zeros([N * 3] * 2)
adj_eye = np.eye(N)
adj_1 = adj.copy()
adj_1[0:N,0:N] = adj_spatial_local
adj_1[0:N,N:2*N] = adj_eye
# adj_1[0:N,2*N:3*N] = adj_eye
adj_1[N:2*N,0:N] = adj_eye
adj_1[N:2*N,N:2*N] = adj_spatial_local
adj_1[N:2*N,2*N:3*N] = adj_eye
# adj_1[2*N:3*N,0:N] = adj_eye
adj_1[2*N:3*N,N:2*N] = adj_eye
adj_1[2*N:3*N,2*N:3*N] = adj_spatial_local

for t in all_time:
    np.save('dataset_processed/'+DATASET+'/'+mask_folder_name+'/graph_data/v2/adj_spatial_dist/'+str(t), adj_1)

# %%
for i in tqdm(range(len(all_time))):
    adj_sp,node_feat, node_id = spatial_graph_cluster(all_time[i])
    if i == 0:
        adj1 = adj_sp
        adj2 = adj_sp
        adj3 = adj_sp
    else:
        adj1 = adj2
        adj2 = adj3
        adj3 = adj_sp
    
    # N = len(adj_spatial_local)
    adj = np.zeros([N * 3] * 2)
    adj_eye = np.eye(N)
    # print(adj.shape,adj_eye.shape)
    # print(adj1.shape,adj2.shape,adj3.shape)

    adj_t = adj.copy()
    adj_t[0:N,0:N] = adj1
    adj_t[0:N,N:2*N] = adj_eye
    # adj_2[0:N,2*N:3*N] = adj_eye
    adj_t[N:2*N,0:N] = adj_eye
    adj_t[N:2*N,N:2*N] = adj2
    adj_t[N:2*N,2*N:3*N] = adj_eye
    # adj_2[2*N:3*N,0:N] = adj_eye
    adj_t[2*N:3*N,N:2*N] = adj_eye
    adj_t[2*N:3*N,2*N:3*N] = adj3

    np.save('dataset_processed/'+DATASET+'/'+mask_folder_name+'/graph_data/v2/adj_spatial_cluster/'+str(all_time[i]), adj_t )
    np.save('dataset_processed/'+DATASET+'/'+mask_folder_name+'/graph_data/v2/feat/'+str(all_time[i]), node_feat )

np.save('dataset_processed/'+DATASET+'/'+mask_folder_name+'/graph_data/v2/all_node_id', node_id )
