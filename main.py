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
from util import *
from config import *
from models import *
import argparse
torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--cv_k', type=int, default=0, help='k of cross validation')
parser.add_argument('--model_name', type=str, default='org', help='model name')
parser.add_argument('--lr', type=float, default=lr, help='learning rate')
args = parser.parse_args()
print('cv_k=',args.cv_k)
print('model_name=',args.model_name)
print('lr=',args.lr)
cv_k=args.cv_k
model_name=args.model_name
lr = args.lr

MODELS = {"org":MVGFRNN ,"no_f": MVGFRNN_no_fusion, "no_g": MVGFRNN_no_graph, "no_idw": MVGFRNN_no_idw, "no_res": MVGFRNN_no_residual, 'ztyao': ZTYao, '1view': OneView}

wandb.init(
    # set the wandb project where this run will be logged
    project=wandb_pjname,#"aqi-test-predict-org",
    
    # track hyperparameters and run metadata
    config={
    "dataset":DATASET,
    "VIEW_NUM": VIEW_NUM,
    "k_neighbor": k_neighbor,
    "batch_size": batch_size,
    "PREV_SLOT":PREV_SLOT,
    "max_epoch": max_epoch,
    "patient":patient,
    "learning_rate":lr,
    "cv_k":cv_k,
    "alpha_multiview_fusion":alpha_multiview_fusion,
    "model_name":model_name,
    "mask_folder_name":mask_folder_name,
    }
)

def print_evaluation_results(huber, rmse, mae, mape):
    if DATASET=='dengue':
        print("Loss : %.4f RMSE : %.4f MAE : %.4f MAPE: %.4f" %( huber, rmse, mae, mape))
    if DATASET=='aqi':
        print('Loss :',huber)
        print('RMSE :',rmse)
        print('MAE :',mae)
        print('MAPE :',mape)

def log_evaluation_results(huber, rmse, mae, mape ,stat='Training'):
    if DATASET=='dengue':
        logging.info("%s... Loss : %.4f  RMSE : %.4f  MAE : %.4f MAPE : %.4f" %(stat, huber, rmse, mae, mape))
    if DATASET=='aqi':
        logging.info(stat)
        # loss_str = ''
        rmse_str = ''
        mae_str = ''
        mape_str = ''
        for i in range(len(rmse)):
            # loss_str += (str(round(huber[i],5))+'\t')
            rmse_str += (str(round(rmse[i],5))+'\t')
            mae_str += (str(round(mae[i],5))+'\t')
            mape_str += (str(round(mape[i],5))+'\t')

        logging.info("Loss"+str(huber))
        logging.info('RMSE'+rmse_str)
        logging.info('MAE'+mae_str)
        logging.info('MAPE'+mape_str)

if __name__ == '__main__':

    log_file_timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_filename = log_file_timestr+'.log' #datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S.log")
    logging.basicConfig(level=logging.INFO, filename='./log/'+log_filename, filemode='w',
	format='[%(asctime)s %(levelname)-8s] %(message)s',
	datefmt='%Y%m%d %H:%M:%S',
	)

    # print("[Epoch %2d] Training RMSE : %.4f Training MAE : %.4f" %( epoch, avg_rmse, avg_mae))
    # logging.info('Fusion+Multi-view')
    logging.info('model_name: '+model_name)
    logging.info('mask_folder_name: '+mask_folder_name)
    # logging.info('GRU Decoder')
    if add_labeled_embed:
        logging.info('with label embedding by mlp&lstm')
    else:
        logging.info('no label embedding by mlp&lstm')
    logging.info('hyper-parm')
    logging.info("VIEW_NUM = %3d" %( VIEW_NUM))
    if VIEW_NUM==4:
        logging.info("fuse_adj_method = "+fuse_adj_method)
    if VIEW_NUM==1:
        logging.info("graph_type = "+graph_type)
    logging.info("k_neighbor = %3d" %( k_neighbor))
    logging.info("batch_size = %4d"%(batch_size))
    logging.info("historical_T = %2d"%(PREV_SLOT))
    logging.info("model_output_size = %2d"%(model_output_size))
    logging.info("learning_rate = %.5f"%(lr))
    logging.info("max_epoch = %3d"%(max_epoch))
    logging.info("patient = %2d"%(patient))
    logging.info("cv_k = %2d"%(cv_k))
    logging.info("alpha_multiview_fusion = %.2f"%(alpha_multiview_fusion))

    nearest_col = []
    near_dist_col = []
    for k in range(k_neighbor):
        nearest_col.append('nearest_'+str(k+1))
        near_dist_col.append('nearest_dist_'+str(k+1))

    if DATASET=='dengue':
        train_dataset = station_data_dengue(mode='train',cv_k=cv_k)
        valid_dataset = station_data_dengue(mode='valid',cv_k=cv_k)
        test_dataset = station_data_dengue(mode='test',cv_k=cv_k)
    if DATASET=='aqi':
        train_dataset = station_data_AQI(mode='train',cv_k=cv_k)
        valid_dataset = station_data_AQI(mode='valid',cv_k=cv_k)
        test_dataset = station_data_AQI(mode='test',cv_k=cv_k)

    torch.manual_seed(random_seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=False)#, pin_memory=True
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=False)

    net = MODELS[model_name](num_station=k_neighbor, output_size=model_output_size).to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.01)

    if not os.path.exists('./model/'+log_file_timestr+'/'):
        os.makedirs('./model/'+log_file_timestr+'/')
    # if load_weight:
    #     # net.load_state_dict(torch.load('model/'+log_file_timestr+'/tmp_save_model.pt'))
    #     net.load_state_dict(torch.load(load_weight_path))

    loss_func_rmse = RMSELoss()
    loss_func_mae = MAELoss()
    loss_func_huber = HuberLoss()

    print("validation set performance before training...")
    time.sleep(1)
    test_huber_before, test_rmse_before, test_mae_before,test_mape_before = testing(net, valid_loader)
    # print("Huber : %.4f RMSE : %.4f MAE : %.4f" %( test_huber_before, test_rmse_before, test_mae_before))#RMSE : 40.9928 MAE : 19.3708
    # logging.info("Before training... Huber : %.4f  RMSE : %.4f  MAE : %.4f" %( test_huber_before, test_rmse_before, test_mae_before))
    print_evaluation_results(test_huber_before, test_rmse_before, test_mae_before, test_mape_before)
    log_evaluation_results(test_huber_before, test_rmse_before, test_mae_before , test_mape_before,'before training')
    torch.cuda.empty_cache()

    patient_count=0
    best_epoch = -1
    best_valid_loss = test_huber_before#100000000000000
    for epoch in range(max_epoch):
        # if epoch<2:
        #     lr=0.0005
        try:
            # torch.cuda.empty_cache()
            # if epoch!=0:
            #     net.load_state_dict(torch.load('model/tmp_save_model.pt'))
            #     # torch.save(net.state_dict(), 'model/tmp_save_model.pt')
            net.train()
            running_loss_rmse = 0.0
            running_loss_mae = 0.0
            running_loss_mape = 0.0
            running_loss_huber = 0.0
            for ovi_target, meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label_out, inv_dis_label,label_id_list, timestamp in tqdm(train_loader):
                optimizer.zero_grad()
                ovi_target = torch.nan_to_num(ovi_target)
                meo_unlabel = torch.nan_to_num(meo_unlabel)
                feature_unlabel = torch.nan_to_num(feature_unlabel)
                ovi_label = torch.nan_to_num(ovi_label)
                meo_label = torch.nan_to_num(meo_label)
                feature_label_out = torch.nan_to_num(feature_label_out)

                h_t = torch.zeros(ovi_target.shape[0],32,device=DEVICE)#.to(device=device)
                output = net(meo_unlabel, feature_unlabel, ovi_label, meo_label, feature_label_out, inv_dis_label, h_t, timestamp, label_id_list)
                # print('output',output)
                # print('ovi_target',ovi_target)
                
                output = torch.nan_to_num(output)
                # print(output.shape,ovi_target.shape)#torch.Size([128, 1])
                # print('ovi_target is nan?',torch.isnan(ovi_target).any())
                # print('output is nan?',torch.isnan(output).any())
                # loss = loss_func_huber(output, ovi_target)
                rmse = loss_func_rmse(output, ovi_target)
                mae = loss_func_mae(output, ovi_target)
                mape = loss_func_mape(output, ovi_target)
                if model_name=='ztyao':
                    loss = rmse*0.5+mae*0.5
                else:
                    loss = rmse*0.2+mae*0.8
                loss.backward()
                optimizer.step()
                running_loss_huber += loss#.item()
                running_loss_rmse += rmse#.item()
                running_loss_mae += mae#.item()
                running_loss_mape += mape#.item()
                # print('output',output)
                # print('rmse, mae, avg', rmse, mae, loss)
            # print('output',output)
            # print('ovi_target',ovi_target)
            avg_huber = running_loss_huber/len(train_loader)
            avg_rmse = running_loss_rmse/len(train_loader)
            avg_mae = running_loss_mae/len(train_loader)
            avg_mape = running_loss_mape/len(train_loader)
            torch.save(net.state_dict(), './model/'+log_file_timestr+'/tmp_save_model.pt')
            print("[Epoch %2d] Training Huber : %.4f Training RMSE : %.4f Training MAE : %.4f Training MAPE : %.4f" %( epoch, avg_huber, avg_rmse, avg_mae, avg_mape))
            # print("[Epoch %2d]"%epoch)
            # print_evaluation_results(avg_huber, avg_rmse, avg_mae)
            logging.info("[Epoch %2d] Training Huber : %.4f Training RMSE : %.4f Training MAE : %.4f Training MAPE : %.4f" %( epoch, avg_huber, avg_rmse, avg_mae, avg_mape))
            wandb.log({"Training Huber": avg_huber,"Training RMSE": avg_rmse, "Training MAE": avg_mae, "Training MAPE": avg_mape})

            print("Start to validation...")
            time.sleep(1)
            valid_huber, valid_rmse, valid_mae,valid_mape = testing(net, valid_loader)
            # print("[Epoch %2d] Validation Huber : %.4f Validation RMSE : %.4f Validation MAE : %.4f" %(epoch, valid_huber, valid_rmse, valid_mae))
            print("[Epoch %2d]"%epoch)
            print_evaluation_results(valid_huber, valid_rmse, valid_mae, valid_mape)
            logging.info("[Epoch %2d]"%epoch)
            log_evaluation_results(valid_huber, valid_rmse, valid_mae, valid_mape,'Validation')
            wandb.log({"Validation Huber": valid_huber, "Validation RMSE": valid_rmse, "Validation MAE": valid_mae,  "Validation MAPE": valid_mape})

            if valid_huber<best_valid_loss:
                best_valid_loss = valid_huber
                patient_count = 0
                torch.save(net.state_dict(), './model/'+log_file_timestr+'/best_wts_my_model.pt')
                print('Save model at epoch ',epoch)
                best_epoch = epoch
                logging.info('Save model at epoch '+str(epoch))
                
            else:
                patient_count+=1
                if patient_count == patient:
                    if best_epoch==-1:
                        print("Start to testing...")
                        time.sleep(1)
                        test_huber, test_rmse, test_mae, test_mape = testing(net, test_loader)
                        print("[Epoch %2d]"%epoch)
                        print_evaluation_results(test_huber, test_rmse, test_mae, test_mape)
                        logging.info("[Epoch %2d]"%epoch)
                        log_evaluation_results(test_huber, test_rmse, test_mae , test_mape,'Testing')
                        wandb.log({"Testing Huber": test_huber, "Testing RMSE": test_rmse, "Testing MAE": test_mae, "Testing MAPE": test_mape})
                    break
            time.sleep(1)

            print("Start to testing...")
            time.sleep(1)
            test_huber, test_rmse, test_mae, test_mape = testing(net, test_loader)
            # print("[Epoch %2d] Testing Huber : %.4f Testing RMSE : %.4f Testing MAE : %.4f" %(epoch, test_huber, test_rmse, test_mae))
            print("[Epoch %2d]"%epoch)
            print_evaluation_results(test_huber, test_rmse, test_mae, test_mape)
            logging.info("[Epoch %2d]"%epoch)
            log_evaluation_results(test_huber, test_rmse, test_mae, test_mape ,'Testing')
            wandb.log({"Testing Huber": test_huber, "Testing RMSE": test_rmse, "Testing MAE": test_mae, "Testing MAPE": test_mape})
            time.sleep(1)
            
        except Exception as e:
            torch.save(net.state_dict(), './model/'+log_file_timestr+'/tmp_save_model.pt')
            # logging.error("Catch an exception.", exc_info=True)
            # logging.exception('Catch an exception.')
            logging.exception('error in epoch'+str(epoch)) # print('error in epoch',epoch)
            # logging.error("error message:", exc_info=True) # print('error message:', str(e))
            break
    
    logging.info('best epoch: '+str(best_epoch))

    wandb.save(model_name+".h5")
