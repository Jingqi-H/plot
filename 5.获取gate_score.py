import os
import sys
# 添加项目根目录到Python路径，为了import自己的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
import yaml
import argparse
import torch.utils.data as data
import torch.optim as optim
from timeit import default_timer as timer
import pandas as pd
import h5py
import glob
from pathlib import Path
import numpy as np

from models.model_moe import CustomMoEMMoE_select_expert
from datasets.ShanxiData import ShanxiData_case
from utils.core_utils import _init_model, val_worker_init_fn, _init_loaders
from utils.process_args import _process_args
from main import seed_torch, dict_to_namespace




'moe模型里面每个专家的权重'


def init_data_model(args, return_a=False, return_gate=False):
    print(f'init dataset...')
    # val_dataset = ShanxiData_case(args.Data, state='val', seed=args.seed, 
    #                               set_multi_task=True, is_dfs=False)
    # val_loader = data.DataLoader(val_dataset, pin_memory=True,
    #                             batch_size=1, 
    #                             num_workers=8, 
    #                             shuffle=False,
    #                             worker_init_fn=val_worker_init_fn)  
    train_loader, val_loader = _init_loaders(args)

    print(f'init moe_wsi model...')
    model = CustomMoEMMoE_select_expert(feat_input=args.encoding_dim, experts_out=512, towers_out=[args.Data.n_bins] + args.Data.multi_task.out_dim,
                                        towers_hidden=32, tasks=args.Data.multi_task.task_num, top_k=4, num_expert=8,set_multi_task=True, 
                                        return_a=return_a, return_gate=return_gate)
    return train_loader, val_loader, model

def save_gate(args, device, save_path):
    train_loader, val_loader, model = init_data_model(args, return_a=False, return_gate=True)
    print(f'val_loader长度：{len(val_loader)}')

    ss_file = f"{args.results_dir}/s_0_checkpoint_bestCindex.pt"
    model.load_state_dict(torch.load(ss_file,weights_only=False))
    model = model.to(torch.device('cuda'))
    print('Successfully init data and model!')  
    model.eval()

    data_dict = {}
    ids = []
    for batch_idx, (data_WSI, label, event_time, c, slide_id_wax, coords, _) in enumerate(train_loader):
        ids.append(slide_id_wax[0])

        with torch.no_grad():
            out = model(data_WSI = data_WSI.squeeze().to(device)) # 结果是list，5个元素，每个元素是8个专家的权重
        gate_weights = []
        for i, tensor in enumerate(out):
            gate_weights.append(tensor.cpu().numpy())
        data_dict[slide_id_wax[0]] = np.array(gate_weights)
    for batch_idx, (data_WSI, label, event_time, c, slide_id_wax, coords, _) in enumerate(val_loader):
        ids.append(slide_id_wax[0])

        with torch.no_grad():
            out = model(data_WSI = data_WSI.squeeze().to(device)) # 结果是list，5个元素，每个元素是8个专家的权重
        gate_weights = []
        for i, tensor in enumerate(out):
            gate_weights.append(tensor.cpu().numpy())
        data_dict[slide_id_wax[0]] = np.array(gate_weights)

    #     if batch_idx  == 11:
    #         break
    # print(ids)
    # fsfdsf
    np.savez_compressed(save_path, **data_dict)


def save_patch_scores(args, device, save_coords_scores_dir, target_idx):
    train_loader, val_loader, model = init_data_model(args, return_a=True, return_gate=False)

    ss_file = f"{args.results_dir}/s_0_checkpoint_bestCindex.pt"
    model.load_state_dict(torch.load(ss_file,weights_only=False))
    model = model.to(torch.device('cuda'))
    print('Successfully init data and model!')  
    model.eval()

    # for batch_idx, (data_WSI, label, event_time, c, slide_id_wax, coords, _) in enumerate(train_loader):
        
    #     save_path = f'{save_coords_scores_dir}/{slide_id_wax[0]}.h5'
    #     if os.path.exists(save_path):
    #         continue
    #     if not slide_id_wax[0] in target_idx:
    #         continue

    #     print(f'当前处理{slide_id_wax[0]}')
    #     with torch.no_grad():
    #         A_raw = model(data_WSI = data_WSI.squeeze().to(device)) # 结果是list，5个元素，每个元素是8个专家的权重

    #     A_raw = A_raw.squeeze()
    #     scores = A_raw.cpu().numpy()
        
    #     with h5py.File(save_path, 'w') as h5f:
    #         h5f.create_dataset("coords", data=coords.squeeze().numpy())
    #         h5f.create_dataset("scores", data=scores)

        # if batch_idx  == 10:
        #     break
    for batch_idx, (data_WSI, label, event_time, c, slide_id_wax, coords, _) in enumerate(val_loader):
        
        save_path = f'{save_coords_scores_dir}/{slide_id_wax[0]}.h5'
        if os.path.exists(save_path):
            continue
        if not slide_id_wax[0] in target_idx:
            continue

        print(f'当前处理{slide_id_wax[0]}')
        with torch.no_grad():
            A_raw = model(data_WSI = data_WSI.squeeze().to(device)) # 结果是list，5个元素，每个元素是8个专家的权重

        A_raw = A_raw.squeeze()
        scores = A_raw.cpu().numpy()
        
        with h5py.File(save_path, 'w') as h5f:
            h5f.create_dataset("coords", data=coords.squeeze().numpy())
            h5f.create_dataset("scores", data=scores)


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


# exp_id = '0.0001loadloss'
exp_id = 'final'
print(exp_id)

result_dir = 'results/moe_wsi/' + exp_id

args = _process_args()
args.patient_level = False

seed_torch(device, args.seed)
with open('config/SXCH.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
args.Data = dict_to_namespace(cfg['Data'])
args.results_dir = result_dir



'1 保存gate权重'
save_dir = f'PLOTS/5.热图/{exp_id}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = f'{save_dir}/all_gates.npz'
save_gate(args, device, save_path)
print('predict gates end.')

'2 保存所有patch的attention scores'

# # 获取要画热图的样本 target_idx
# svs_dir = '/media/ubuntu/DATA/SXCH_svs'
# sheet_dict = pd.read_excel(
#     f'PLOTS/5.热图/{exp_id}/SXCH-Val-用于筛选样本.xlsx',
#     sheet_name=['L', 'H']  # 指定要读取的sheet名称列表
# )
# df_L = sheet_dict['L']  # "L" sheet的数据
# df_H = sheet_dict['H']  # "H" sheet的数据
# final_df = pd.concat([df_L, df_H], axis=0)
# target_idx = []
# for i in os.listdir(svs_dir):
#     slide_id = i.split('.')[0]
#     if slide_id in final_df['slide_id'].values:
#         wax_id = final_df[final_df['slide_id'] == slide_id]['slide_id_wax'].values[0]
#         target_idx.append(wax_id)
target_idx = ['201501316-1,2-HE#1', '201712444-1,2-HE#0']
print(f'要处理的样本数：{len(target_idx)}')

save_coords_scores_dir = f'PLOTS/5.热图/{exp_id}/coords_scores'
if not os.path.exists(save_coords_scores_dir):
    os.makedirs(save_coords_scores_dir)
save_patch_scores(args, device, save_coords_scores_dir, target_idx)