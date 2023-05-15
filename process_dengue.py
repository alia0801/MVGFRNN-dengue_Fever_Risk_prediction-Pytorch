# %%
import numpy as np 
import pandas as pd
import random
import os
from config import *
DATASET='dengue'
# %%
grids = pd.read_csv('dataset_processed/'+DATASET+'/grid.csv')
grids

# %%
aq_data = pd.read_csv('dataset_processed/'+DATASET+'/all_processed_data_9box_nexty.csv')
aq_data['id'] = aq_data['id'].astype(int)
del aq_data['Unnamed: 0']
aq_data
# %%
all_labeled_id = list(aq_data.id.unique())
all_labeled_id
# %%
labeled_rate = 0.7
mask_folder = 'mask07'
if not os.path.exists('dataset_processed/'+DATASET+'/'+mask_folder):
    os.makedirs('dataset_processed/'+DATASET+'/'+mask_folder)
# %%
labeled_id = random.sample(all_labeled_id,k=round(len(all_labeled_id)*labeled_rate))
len(labeled_id)
# %%
unlabeled_id = list(set(all_labeled_id)-set(labeled_id))
unlabeled_id
# %%
file = open('dataset_processed/'+DATASET+'/'+mask_folder+'/label_id.txt','w')
for item in labeled_id:
	file.write(str(item)+"\n")
file.close()

# %%
train_id = random.sample(unlabeled_id,k=round(len(unlabeled_id)*0.8))
remain_id = list(set(unlabeled_id)-set(train_id))
valid_id = random.sample(remain_id,k=round(len(remain_id)*0.5))
test_id = list(set(remain_id)-set(valid_id))
print(len(train_id),len(valid_id),len(test_id))
# %%
file = open('dataset_processed/'+DATASET+'/'+mask_folder+'/train_id.txt','w')
for item in train_id:
	file.write(str(item)+"\n")
file.close()

file = open('dataset_processed/'+DATASET+'/'+mask_folder+'/valid_id.txt','w')
for item in valid_id:
	file.write(str(item)+"\n")
file.close()

file = open('dataset_processed/'+DATASET+'/'+mask_folder+'/test_id.txt','w')
for item in test_id:
	file.write(str(item)+"\n")
file.close()

# with open('dataset_processed/'+DATASET+'/'+mask_folder+'/label_id.txt','r') as f:
#     lines = f.read().split('\n')[:-1]
# labeled_id = [int(x) for x in lines]
# with open('dataset_processed/'+DATASET+'/'+mask_folder+'/train_id.txt','r') as f:
#     lines = f.read().split('\n')[:-1]
# train_id = [int(x) for x in lines]
# with open('dataset_processed/'+DATASET+'/'+mask_folder+'/valid_id.txt','r') as f:
#     lines = f.read().split('\n')[:-1]
# valid_id = [int(x) for x in lines]
# with open('dataset_processed/'+DATASET+'/'+mask_folder+'/test_id.txt','r') as f:
#     lines = f.read().split('\n')[:-1]
# test_id = [int(x) for x in lines]
unlabeled_id = train_id+valid_id+test_id
# %%

for i in range(10):
	remain_id = unlabeled_id.copy()
	select_id = random.sample(remain_id,k=round(len(unlabeled_id)*0.1))
	remain_id = list(set(remain_id)-set(select_id))
	file = open('dataset_processed/'+DATASET+'/'+mask_folder+'/unlabeled_split_'+str(i)+'.txt','w')
	for item in select_id:
		file.write(str(item)+"\n")
	file.close()
# %%

grids

# %%
# grids['x_center'] = (grids['left']+grids['right'])*0.5
# grids['y_center'] = (grids['top']+grids['bottom'])*0.5

# %%
grid_df_new = grids[grids['grid_id'].isin(all_labeled_id)]
grid_df_new
# %%
grid_df_label = grid_df_new[grid_df_new['grid_id'].isin(labeled_id)].reset_index(drop=True)
grid_df_label
# %%
grid_df_unlabel = grid_df_new[~grid_df_new['grid_id'].isin(labeled_id)].reset_index(drop=True)
grid_df_unlabel
# %%
labeled_coor = grid_df_label[['x_center','y_center']].values
unlabeled_coor = grid_df_unlabel[['x_center','y_center']].values
labeled_coor,unlabeled_coor
# %%
from sklearn.neighbors import NearestNeighbors

# grid_df_neighbor = pd.read_csv('dataset_processed/aqi/label_grid_15neighbor_dist.csv')
k_neighbor = 100

def trans_for_id_with_dist_2(kneighbors,dist):
    ids = grid_df_label.grid_id.values
    neighbors_id = {}
    for i in range(len(kneighbors)):
        now_id = kneighbors[i]
        neighbors_id['nearest_dist_'+str(i+1)] = dist[i]
    return neighbors_id 

def trans_for_id_with_dist_3(kneighbors,dist):
    ids = grid_df_label.grid_id.values
    neighbors_id = {}
    for i in range(len(kneighbors)):
        now_id = kneighbors[i]
        neighbors_id['nearest_'+str(i+1)] = ids[now_id-1]
    return neighbors_id 

# %%
labeled_coor_tmp = labeled_coor.copy()
knn = NearestNeighbors(n_neighbors=k_neighbor)
knn.fit(labeled_coor_tmp)
d, nei = knn.kneighbors(labeled_coor_tmp)

label_all_neighbor_with_dist = []
label_all_neighbor_id=[]
for i in range(len(nei)):
    dist = d[i][1:]
    kneighbors = nei[i][1:]
    neighbors_dist = trans_for_id_with_dist_2(kneighbors,dist)
    neighbors_id = trans_for_id_with_dist_3(kneighbors,dist)
    label_all_neighbor_with_dist.append(neighbors_dist)
    label_all_neighbor_id.append(neighbors_id)

# %%
label_grid_df_neighbor = grid_df_label.copy(deep=True)
label_grid_df_neighbor = pd.DataFrame(label_all_neighbor_id)
label_grid_df_neighbor['grid_id'] = grid_df_label.grid_id.values
label_grid_df_neighbor
# %%
label_all_neighbor_dist_df = pd.DataFrame(label_all_neighbor_with_dist)
label_all_neighbor_dist_df['grid_id'] = grid_df_label.grid_id.values
label_all_neighbor_dist_df
# %%
grid_df_label.rename(columns={'id': 'grid_id'}, inplace=True)
label_grid_df_neighbor = pd.merge(grid_df_label,label_grid_df_neighbor, on='grid_id')
label_grid_df_neighbor = pd.merge(label_grid_df_neighbor,label_all_neighbor_dist_df, on='grid_id')
label_grid_df_neighbor
# %%
label_grid_df_neighbor.to_csv('dataset_processed/'+DATASET+'/'+mask_folder+'/label_grid_100neighbor_dist.csv')

# %%
# find neighbors for unlabeled grid
all_neighbor_with_dist = []
all_neighbor_id=[]
for i in range(len(unlabeled_coor)):
    labeled_coor_tmp = labeled_coor.copy()
    labeled_coor_tmp = np.insert(labeled_coor_tmp, 0,unlabeled_coor[i], axis=0)
    knn = NearestNeighbors(n_neighbors=k_neighbor+1)
    knn.fit(labeled_coor_tmp)
    d, nei = knn.kneighbors([unlabeled_coor[i]])
    dist = d[0][1:]
    kneighbors = nei[0][1:]
    neighbors_dist = trans_for_id_with_dist_2(kneighbors,dist)
    neighbors_id = trans_for_id_with_dist_3(kneighbors,dist)
    all_neighbor_with_dist.append(neighbors_dist)
    all_neighbor_id.append(neighbors_id)

# %%
all_neighbor_with_dist

# %%
all_neighbor_id

# %%
grid_df_neighbor = grid_df_unlabel.copy(deep=True)
all_neighbor_df = pd.DataFrame(all_neighbor_id)
all_neighbor_df['grid_id'] = grid_df_neighbor.grid_id.values
all_neighbor_df
# %%
all_neighbor_dist_df = pd.DataFrame(all_neighbor_with_dist)
all_neighbor_dist_df['grid_id'] = grid_df_neighbor.grid_id.values
all_neighbor_dist_df
# %%
grid_df_neighbor = pd.merge(grid_df_neighbor,all_neighbor_df, on='grid_id')
grid_df_neighbor = pd.merge(grid_df_neighbor,all_neighbor_dist_df, on='grid_id')
grid_df_neighbor
# %%
print(grid_df_neighbor.columns)
# %%
grid_df_neighbor.to_csv('dataset_processed/'+DATASET+'/'+mask_folder+'/unlabel_grid_100neighbor_dist.csv')
# %%
