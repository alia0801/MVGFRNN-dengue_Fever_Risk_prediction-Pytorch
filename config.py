import torch
VIEW_NUM = 4
DATASET = 'dengue'
# DATASET = 'aqi'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(DEVICE)
wandb_pjname = 'dengue-diff-mask'
# wandb_pjname = 'aqi-test-predict-org'

mask_folder_name = 'mask07'
# mask_folder_name = ''

fuse_adj_method = 'add' 
alpha_multiview_fusion = 0.8
graph_path = 'dataset_processed/'+DATASET+'/'+mask_folder_name+'/graph_data/v2/'
add_labeled_embed = False

k_neighbor = 10
random_seed = 99
batch_size = 8#1024
shuffle=True
model_name = 'org' 
#MODELS = {"org":MVGFRNN ,"no_f": MVGFRNN_no_fusion, "no_g": MVGFRNN_no_graph, "no_idw": MVGFRNN_no_idw, "no_res": MVGFRNN_no_residual}

if DATASET == 'aqi':
    PREV_SLOT = 12
    model_output_size=12
if DATASET == 'dengue':
    PREV_SLOT = 4
    model_output_size=1

lr=0.0001#0.002#0.0003
max_epoch = 20
patient = 3

nearest_col = []
near_dist_col = []
for k in range(k_neighbor):
    nearest_col.append('nearest_'+str(k+1))
    near_dist_col.append('nearest_dist_'+str(k+1))
