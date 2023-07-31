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
# %%
# mask_folder_name = 'mask05'
if not os.path.exists('dataset_processed/'+DATASET+'/'+mask_folder_name+'/graph_data/v2/adj_temporal_new'):
    os.makedirs('dataset_processed/'+DATASET+'/'+mask_folder_name+'/graph_data/v2/adj_temporal_new')
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
# %%
with open('dataset_processed/'+DATASET+'/'+mask_folder_name+'/label_id.txt','r') as f:
    lines = f.read().split('\n')[:-1]
labeled_id = [int(x) for x in lines]
print('label',len(labeled_id))
# %%

all_data = pd.read_csv('dataset_processed/'+DATASET+'/all_processed_data_9box_nexty.csv')
all_time = all_data.time.unique()


# %%
def temporal_graph_dtw(timestamp,dist_thresh=25,connect_ratio = 0.05,pre_slot=8):
    # try:
        
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
    if DATASET=='aqi':
        label_feat = label_feat[label_feat['time'].isin(time_list)][['id','PM2.5']].reset_index(drop=True)
    if DATASET=='dengue':
        label_feat = label_feat[label_feat['time'].isin(time_list)][['id','egg_num']].reset_index(drop=True)

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
    return df_adj.values
    # except:

    #     print('error',timestamp)
    #     df_adj = np.zeros((len(labeled_id),len(labeled_id)))
    #     return df_adj
# %%
if DATASET=='aqi':
    nei_file_name = 'label_grid_15neighbor_dist.csv'
if DATASET=='dengue':
    nei_file_name = 'label_grid_100neighbor_dist.csv'
grid_df = pd.read_csv('dataset_processed/'+DATASET+'/'+mask_folder_name+'/'+nei_file_name)
sub_df = grid_df[grid_df['grid_id'].isin(labeled_id)].reset_index()
N = len(sub_df)
# %%
for i in tqdm(range(len(all_time))):
    adj_tmp = temporal_graph_dtw(all_time[i],dist_thresh=25,connect_ratio = 0.1,pre_slot=48)
    if i == 0:
        adj1 = adj_tmp
        adj2 = adj_tmp
        adj3 = adj_tmp
    else:
        adj1 = adj2
        adj2 = adj3
        adj3 = adj_tmp
    
    # N = len(adj_spatial_local)
    adj = np.zeros([N * 3] * 2)
    adj_eye = np.eye(N)

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

    np.save('dataset_processed/'+DATASET+'/'+mask_folder_name+'/graph_data/v2/adj_temporal_new/'+str(all_time[i]), adj_t )

# %%
