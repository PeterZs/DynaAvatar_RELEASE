import argparse
import os
import os.path as osp
from glob import glob
import csv
import json
from tqdm import tqdm
import numpy as np
import shutil

import torch
from pytorch3d.transforms import axis_angle_to_matrix

def arg_parser(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root_dir', type=str, required=True, dest='dataset_root_dir', help='root directory containing original dataset.')
    parser.add_argument('--target_root_dir', type=str, required=True, dest='target_root_dir', help='root directory containing reannotated dataset.')
    args = parser.parse_args()
    return args


args = arg_parser()
dataset_root_dir = args.dataset_root_dir
target_root_dir = args.target_root_dir

subject_id_list = os.listdir(dataset_root_dir)
subject_id_list.sort(key=lambda x: int(x[-2:]))

for subject_id in subject_id_list: 
    for seq in ['Sequence1', 'Sequence2']: 
        if not osp.exists(osp.join(dataset_root_dir, subject_id, seq)): 
            continue
        
        print(f"Sequence name: {subject_id}_{seq[-1]}")
        target_subject_dir = osp.join(target_root_dir, f"{subject_id}_{seq[-1]}")
        os.makedirs(target_subject_dir, exist_ok=True)
        
        cam_idx_list = [159, 160, 152, 151, 144, 143, 136, 135, 128, 127, 126, \
            112, 111, 110, 109, 96, 95, 94, 93, 80, 79, 78, 77, 56, 55, 54, 53, \
                40, 39, 38, 37, 24, 23, 22, 21, 8, 7, 6, 5]
        print("Start copying imgs and masks...")
        for cam_idx in tqdm(cam_idx_list): 
            os.makedirs(osp.join(target_subject_dir, 'images', f"{cam_idx:03d}"), exist_ok=True)
            os.makedirs(osp.join(target_subject_dir, 'masks', f"{cam_idx:03d}"), exist_ok=True)
            
            imgs_dir = osp.join(dataset_root_dir, subject_id, seq, '4x', 'rgbs', f"Cam{cam_idx:03d}")
            img_path_list = glob(osp.join(imgs_dir, '*.jpg'))            
            for img_path in img_path_list: 
                frame_idx = img_path.split('/')[-1].split('_rgb')[1][:-4]  # 06d
                shutil.copy2(img_path, osp.join(target_subject_dir, 'images', f'{cam_idx:03d}', f'{int(frame_idx):06d}.jpg'))
                mask_path = osp.join(imgs_dir, '..', '..', 'masks', f"Cam{cam_idx:03d}", f"Cam{cam_idx:03d}_mask{frame_idx}.png")
                shutil.copy2(mask_path, osp.join(target_subject_dir, 'masks', f'{cam_idx:03d}', f'{int(frame_idx):06d}.png'))
                
        # cam idx
        cam_param_path = osp.join(dataset_root_dir, subject_id, seq, '4x', 'calibration.csv')
        cam_param_save = {}
        cam_idx_set = set(cam_idx_list)
        with open(cam_param_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['name']
                if int(name[-3:]) in cam_idx_set: 
                    w = int(row['w']); h = int(row['h'])
                    rx = float(row['rx']); ry = float(row['ry']); rz = float(row['rz'])
                    tx = float(row['tx']); ty = float(row['ty']); tz = float(row['tz'])
                    fx = float(row['fx']); fy = float(row['fy']); px = float(row['px']); py = float(row['py'])

                    R_cw = axis_angle_to_matrix(torch.tensor([rx, ry, rz], dtype=torch.float32)).numpy()
                    t_cw = np.array([tx, ty, tz], dtype=float).reshape(3)
                    R_wc = R_cw.T
                    t_wc = - R_cw.T @ t_cw
                    focal = np.array([fx*w, fy*h], dtype=np.float32)
                    princpt = np.array([px*w, py*h], dtype=np.float32)
                    
                    cam_param_save[f"{name[-3:]}"] = {
                        "R": R_wc.tolist(),
                        "t": t_wc.tolist(), 
                        "focal": focal.tolist(),
                        "princpt": princpt.tolist()
                    }
        with open(osp.join(target_subject_dir, 'cam_params.json'), 'w') as f: 
            json.dump(cam_param_save, f, indent=4)
        print("Saved cam params at", osp.join(target_subject_dir, 'cam_params.json'))
        
        print(f"[FIN] {subject_id}_{seq}")
        