# built-in
import os
import cv2
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import json
from collections import Counter
import random
import re
from tqdm import tqdm
import math

# ml
import torch
from torchvision import transforms
from torch.utils.data import Dataset, random_split, WeightedRandomSampler
import torch.nn.functional as F

# own
from config import GlobalConfig

img_save_path = './check/'
plt.ion()
fig, axs = plt.subplots(3, 4, figsize=(25, 25))
def visualize(data, idx):
    for ax in axs.flat:
        if ax.figure is not None:
            ax.clear()

    bev_w_obj = data['bev']
    tl = data['tl_label']
    
    # RGB
    rgb = data['rgb']
    image = rgb.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # (H, W, C)
    image = np.clip(image, 0, 1)
    axs[0,0].imshow(image)
    axs[0,0].axis("off")
    axs[0,0].set_title("RGB Image")

    # p = Path(img_path)
    # img_path = f"{p.parts[3]}/{'_'.join(p.parts[4].split('_')[-2:])}/{'_'.join(p.parts[5].split('_')[:2])}/{p.stem.split('_')[0]}"
    # axs[1,2].text(
    #     0.5, 0.5, 
    #     f"{img_path}", 
    #     fontsize=25, ha='center', va='center'
    # )
    # axs[1,2].axis("off")
    # axs[1,2].set_title("img path")
    
    #* GT waypoint, target point in local coordinate
    waypoints_GT_local = data['local_checkpoint']
    target_point = data['local_targetpoint']
    x_coords = waypoints_GT_local[:, 0]
    y_coords = waypoints_GT_local[:, 1]
    targetpoint_np = target_point.numpy()
    target_x = targetpoint_np[0][0]
    target_y = targetpoint_np[0][1]
    axs[2, 1].scatter(x_coords, y_coords, color='blue', label='Waypoints') # checkpoint
    axs[2, 1].plot(x_coords, y_coords, color='gray', linestyle='--', label='Path') # checkpoint
    axs[2, 1].scatter(target_x, target_y, color='red', label='Target Point', s=100, edgecolor='black') # targetpoint
    axs[2, 1].axhline(0, color='black', linewidth=0.5)  # x축 (차량 기준선)
    axs[2, 1].axvline(0, color='black', linewidth=0.5)  # y축 (차량 기준선)
    axs[2, 1].set_title("Local Waypoints")
    axs[2, 1].set_xlabel("X (px)")
    axs[2, 1].set_ylabel("Y (px)")
    axs[2, 1].legend()
    axs[2, 1].grid()
    axs[2, 1].axis('equal')

    
    bev_label = bev_w_obj.cpu().numpy()
    axs[2,0].imshow(bev_label)
    axs[2,0].axis("off")
    axs[2,0].set_title("BEV Label")
    
    speed = data['speed']
    target_ctrl = data['target_ctrl']
    
    axs[0,3].text(
        0.5, 0.5, 
        f"Current speed: {speed.item():.2f} km/h\n"
        f"TL: {tl}\n"
        f"GT steer: {target_ctrl[0].clone().detach().cpu().numpy()}\n"
        f"GT accel: {target_ctrl[1].clone().detach().cpu().numpy()}", 
        fontsize=25, ha='center', va='center'
    )
    axs[0,3].axis("off")
    axs[0,3].set_title("ego")

    
    crop_rgb = data['crop_rgb']
    crop_rgb = crop_rgb.numpy()
    crop_rgb = crop_rgb.transpose(1, 2, 0)
    axs[1,0].imshow(crop_rgb)
    axs[1,0].axis("off")
    axs[1,0].set_title("Crop")
    
    plt.tight_layout()

    os.makedirs(img_save_path, exist_ok=True)
    name = f'{idx}.png'
    save_path = os.path.join(img_save_path, name)
    plt.savefig(save_path)


def transform_world_to_local(x_world, y_world, ego_x, ego_y, ego_yaw):
    dx = x_world - ego_x
    dy = y_world - ego_y
    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)

    # x_local, y_local = Ego 기준 로컬좌표
    x_local = dx * cos_yaw + dy * sin_yaw       # 앞(ego yaw=0시 x방향)이 +x
    y_local = -dx * sin_yaw + dy * cos_yaw      # 왼쪽이 +y
    return x_local, y_local

def transform_to_bev_coordinates_vertical(xy_local_list, pixels_per_meter=6.67, image_center=(150,0)): #!!
  # 이미지 위에서 아래 방향으로 waypoint 나오도록 계산
  bev_coords = []
  for x, y in xy_local_list:
    pixel_x = float(y * pixels_per_meter + image_center[0]) #-y
    pixel_y = float(x * pixels_per_meter + image_center[1]) #-x
    bev_coords.append((pixel_x, pixel_y)) 
    
  return bev_coords

def transform_from_bev_coordinates(bev_coords, pixels_per_meter=6.67, image_center=(150, 0)):
    local_coords = []
    for pixel_x, pixel_y in bev_coords:
        y = (pixel_x - image_center[0]) / pixels_per_meter
        x = (pixel_y - image_center[1]) / pixels_per_meter
        local_coords.append((x, y))
        
    return local_coords

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])
    
def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


class MORAIDataset(Dataset):
    def __init__(self, root_dir, train=True, ratio=0.8, cache_file="", config=""):
        super().__init__()

        self.root_dir = root_dir
        self.train = train
        self.ratio = ratio
        self.config = config
        self.target_idx = 50
        self.len_wp = 100
        
        self.data = []
        
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform_croprgb = transforms.Compose([
            transforms.Resize((180, 640)),
            transforms.ToTensor(),
        ])
        
        # load already made data
        if cache_file and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                self.data = pickle.load(f)
            print(f"--- Load dta from cache file: {cache_file} ---")

        # newly updated data
        else:
            # (1) 모든 폴더 리스트 가져오기
            scenario_list = [f for f in sorted(os.listdir(self.root_dir))
                         if f.startswith("R_KR_PG_KATRI")]
            # 각 시나리오 에서...
            for scenario in tqdm(scenario_list, desc="Processing scenarios"):
                scenario_path = os.path.join(self.root_dir, scenario)
                if not os.path.isdir(scenario_path):
                    continue
                # m = re.search(r"Scenario_(\d+)", scenario)
                # if m:
                #     num = int(m.group(1))
                #     if num in [123, 124]:
                #         pass
                #     else:
                #         continue
                
            # # (2) 폴더 섞기
            # random.seed(42)  # 동일한 split을 위해 고정
            # random.shuffle(scenario_list)
                            
            # # (3) 80% Train / 20% Validation Split
            # train_size = int(len(scenario_list) * self.ratio)
            # train_scenarios = scenario_list[:train_size]
            # val_scenarios = scenario_list[train_size:]

            # # (4) Train/Validation 선택
            # if self.train:
            #     selected_scenarios = train_scenarios
            #     print(f"Train dataset: {len(selected_scenarios)} folders")
            # else:
            #     selected_scenarios = val_scenarios
            #     print(f"Validation dataset: {len(selected_scenarios)} folders")

            
                #--- traffic_light_set.json 파일을 로드
                if scenario.startswith("R_KR_PG_KATRI"):
                    traffic_light_json_path = os.path.join(self.root_dir, "Map_Data", "R_KR_PG_KATRI", "traffic_light_set.json")
                elif scenario.startswith("R_KR_PR_Pangyo"):
                    traffic_light_json_path = os.path.join(self.root_dir, "Map_Data", "R_KR_PR_Pangyo_UTM", "traffic_light_set.json")
                elif scenario.startswith("R_KR_PR_Sangam"):
                    traffic_light_json_path = os.path.join(self.root_dir, "Map_Data", "R_KR_PR_Sangam_DP", "traffic_light_set.json")
                elif scenario.startswith("R_KR_PR_Sejong"):
                    traffic_light_json_path = os.path.join(self.root_dir, "Map_Data", "R_KR_PR_Sejong_South_DP", "traffic_light_set.json")

                with open(traffic_light_json_path, "r") as f:
                    self.traffic_lights = json.load(f)

                # 여기까지 통과하면 "R_KR..._XX" 형태 + XX in [0,50]
                if not os.path.isdir(scenario_path):
                    continue
                traffic_info_path = os.path.join(scenario_path, "TRAFFIC_INFO", f"{scenario}.json")
                if not os.path.exists(traffic_info_path):
                    continue
                with open(traffic_info_path, "r") as f:
                    traffic_info = json.load(f)
                
                globalpath_file = os.path.join(scenario_path, "global_path_cmd.csv")
                if os.path.exists(globalpath_file):
                    globalpath = pd.read_csv(globalpath_file)
                    globalpath_pos = []
                    for idx, row in globalpath.iterrows():
                        x, y, z = row['PositionX (m)'], row['PositionY (m)'], row['PositionZ (m)']
                        globalpath_pos.append((x, y))
                    kdtree = KDTree(globalpath_pos)
                else:
                    print(f"--- [WARNING] No global path file in {scenario_path} ---")

                # camera images and ego info directories
                camera_dirs = [os.path.join(scenario_path, f"CAMERA_{i}_640_360") for i in range(1, 4)]
                ego_dir = os.path.join(scenario_path, "EGO_INFO")
                map_dir = os.path.join(scenario_path, "BEV_GT_5CAMS")
                
                # CAMERA_1 폴더 기준으로 파일 리스트 얻기
                # 기준이 될 CAMERA_1 폴더
                camera_1_dir = camera_dirs[0]
                if not os.path.isdir(camera_1_dir):
                    continue
                img_files = sorted(
                            [f for f in os.listdir(camera_1_dir) if f.endswith(".jpeg")],
                            key=lambda x: int(x.split("_")[0])
                        )
                
                # max speed 찾기
                # max_velocity = 0
                # if len(img_files) == 0:
                #     continue
                # last_img_num = int(img_files[-1].split("_")[0])
                # img_nums = list(range(10, last_img_num + 1, 10))

                # for num in img_nums:
                #     ego_file = os.path.join(ego_dir, f"{num}.txt")
                #     if not os.path.exists(ego_file):
                #         continue
                #     with open(ego_file, 'r') as f:
                #         for line in f:
                #             key, *values = line.split(":")
                #             if key.strip() == "velocity":
                #                 vel_x, vel_y, _ = map(float, values[0].split())
                #                 speed = np.linalg.norm([vel_x, vel_y])  # norm(vx, vy)
                #                 max_velocity = max(max_velocity, speed)  # 가장 큰 속도 갱신
                                
                # 이미지 파일 각각에 대해
                # for img_file in img_files:
                for img_file in tqdm(img_files, desc=f"Images in {scenario}", leave=False):
                    if img_file.endswith(".jpeg"):
                        num = img_file.split("_")[0]
                        
                        # 5개 카메라 이미지를 한 묶음으로 만들기
                        img_paths = []
                        for cdir in camera_dirs:
                            path_candidate = os.path.join(cdir, img_file)
                            if not os.path.exists(path_candidate):
                                # 5개 중 하나라도 없으면 스킵
                                img_paths = []
                                break
                            img_paths.append(path_candidate)
                        
                        ego_file = os.path.join(ego_dir, f"{num}.txt")
                        future_ego_file = [os.path.join(ego_dir, f"{int(num)+i*2}.txt") for i in range(1, self.len_wp+1)] 
                        map_file = os.path.join(map_dir, f"{num}_bevgt.png")
                        object_info_file = os.path.join(scenario_path,"OBJECT_INFO", f"object_info_{num}.txt")

                        if os.path.exists(ego_file) and  os.path.exists(map_file) and all(os.path.exists(future_ego) for future_ego in future_ego_file):
                            #* bev gt paths
                            bevgt_paths = [os.path.join(map_dir, map_file)]

                            ego = []
                            with open(ego_file, 'r') as f:
                                for line in f:
                                    key, *values = line.split(":")
                                    if values:
                                        values = values[0].split()
                                        try:
                                            if key.strip() == "position":
                                                ego.extend(map(float, values))  # pos_x, pos_y, pos_z
                                                pos_xy = values[:2].copy()
                                            elif key.strip() == "orientation":
                                                heading = float(values[2]) # rot_z #* EGO_INFO에서 heading 불러오기
                                                heading = np.deg2rad(heading)
                                                ego.extend(map(float, values))  # orient_x, orient_y, orient_z
                                            elif key.strip() == "enu_velocity":
                                                ego.extend(map(float, values))  # enu_vel_x, enu_vel_y, enu_vel_z
                                            elif key.strip() == "velocity":
                                                # ego.extend(map(float, values))  # vel_x, vel_y, vel_z
                                                vel_x, vel_y, _ = map(float, values)
                                                speed = np.linalg.norm([vel_x, vel_y])
                                                # norm_speed = speed / max_velocity if max_velocity > 0 else 0  # 정규화
                                                # ego.extend([vel_x, vel_y, norm_speed])   # vel_x, vel_y, norm_speed
                                                ego.extend([vel_x, vel_y, speed])   # vel_x, vel_y, norm_speed
                                            elif key.strip() == "angularVelocity":
                                                ego.extend(map(float, values))  # ang_vel_x, ang_vel_y, ang_vel_z
                                            elif key.strip() == "acceleration":
                                                ego.extend(map(float, values))  # accel_x, accel_y, accel_z
                                            elif key.strip() == "accel":
                                                # clipped_acc = torch.clamp(torch.tensor(float(values[0]), dtype=torch.float32), max=0.4)
                                                ego.append(float(values[0]))  # accel
                                            elif key.strip() == "brake":
                                                # clipped_brake = torch.clamp(torch.tensor(float(values[0]), dtype=torch.float32), max=0.2)
                                                ego.append(float(values[0]))  # Append the clipped value
                                            elif key.strip() == "steer":
                                                ego.append(float(values[0]))  # steer
                                            elif key.strip() == "trafficlightid": # traffic light id
                                                traffic_light_id = str(values[0])
                                                ego.append(traffic_light_id)
                                            
                                        except ValueError:
                                            continue
                                      
                            #* obstacle  
                            obstacles = []
                            with open(object_info_file, "r") as f_obj:
                                for line in f_obj:
                                    try:
                                        parts = line.strip().split()
                                        if len(parts) < 14:
                                            continue
                                        # parts[0]: class name, parts[1:4]: cuboid center (global)
                                        center_global = (float(parts[1]), float(parts[2]))
                                        # width: parts[8]
                                        length = float(parts[7])
                                        width = float(parts[8])
                                        yaw = float(parts[6]) # degree
                                        dim = (length, width, yaw)
                                        
                                        vx_world = float(parts[10])
                                        vy_world = float(parts[11])
                                        speed = (vx_world, vy_world)
                                        obstacles.append([center_global, dim, speed])
                                    except ValueError:
                                        continue
                        
                            #* target point
                            checkpoint = [] #! waypoints
                            # velocities = [] #! velocities

                            for idx, future_ego in enumerate(future_ego_file):
                                # print(idx, future_ego)
                                with open(future_ego, 'r') as f:
                                    for line in f:
                                        key, *values = line.split(":")
                                        if values:
                                            values = values[0].split()
                                            try:
                                                if key.strip() == "position":
                                                    pos_x, pos_y = map(float, values[:2])
                                                    checkpoint.append([pos_x, pos_y])
                                                # elif key.strip() == "velocity" and idx == len(future_ego_file) - 1:
                                                #     #! target speed
                                                #     vel_x, vel_y, vel_z = map(float, values)
                                                #     speed = np.linalg.norm(np.array([vel_x, vel_y, vel_z]))
                                                #     # norm_speed = np.linalg.norm([vel_x, vel_y]) / max_velocity if max_velocity > 0 else 0  # 정규화 #! 위에 지우고
                                                #     velocities.append(speed)

                                                    
                                            except ValueError:
                                                continue
                            
                            _, idx = kdtree.query(pos_xy) #! 'N'm of future global path
                            if idx + self.target_idx >= len(globalpath_pos): # if idx + 50 exceeds the given global path length
                                continue
                            
                            local_targetpoint = self.transform_global_to_local(globalpath_pos[idx+self.target_idx][:2], ego[0:2], heading) #! target point
                            local_checkpoint = self.transform_global_to_local(checkpoint, ego[0:2], heading) #! waypoint
                            bev_targetpoint = transform_to_bev_coordinates_vertical([local_targetpoint])
                            bev_checkpoint = transform_to_bev_coordinates_vertical(local_checkpoint)

                            # local_checkpoint = np.array(local_checkpoint)
                            # velocities = np.array(velocities)
                            # vel_column = velocities.reshape(-1, 1)
                            # local_checkpoint = np.concatenate([local_checkpoint, vel_column], axis=1)

                            # find the closest waypoint
                            # high_level_command = globalpath_cmd[idx]
                            
                            if traffic_light_id == "null":
                                label = 0
                            else: 
                                traffic_code_map = {
                                    0: 0,   # None
                                    1: 1,   # Red
                                    4: 2,   # Yellow
                                    5: 3,   # Red with Yellow
                                    16: 4,  # Green
                                    20: 5,  # Yellow with Green
                                    32: 6,  # GreenLeft
                                    33: 7,  # Red with GreenLeft
                                    36: 8,  # Yellow with GreenLeft
                                    48: 9,  # Green with GreenLeft
                                }
                                traffic_status = traffic_info[str(num)][traffic_light_id]

                                tl_obj = next((obj for obj in self.traffic_lights if obj["idx"] == traffic_light_id), None)
                                if tl_obj is None:
                                    label = 0
                                else:
                                    tl_point = np.array(tl_obj["point"][:2], dtype=float)
                                    ego_pos_arr = np.array(pos_xy, dtype=float)

                                    tl_heading_rad = np.deg2rad(tl_obj["heading"])
                                    tl_direction = np.array([np.cos(tl_heading_rad), np.sin(tl_heading_rad)])
                                    offset = 10.0
                                    new_tl_point = tl_point + offset * tl_direction
                                    
                                    # ego의 위치와 조절된 신호등 기준 위치 간의 상대 벡터 및 투영 계산
                                    ego_direction = np.array([np.cos(heading), np.sin(heading)])
                                    relative = ego_pos_arr - new_tl_point
                                    proj = np.dot(relative, ego_direction)
                                    
                                    if proj >= 0:
                                        label = 0
                                    else:
                                        # 아직 신호등 전방에 있다면, 거리 기준을 적용
                                        threshold = 30.0
                                        dist = np.linalg.norm(relative)
                                        if dist <= threshold:
                                            traffic_index = traffic_code_map.get(traffic_status, 0)  # 알 수 없는 값이면 기본 0(None) 처리
                                            label = traffic_index
                                        else:
                                            label = 0
                                            
                            self.data.append((img_paths, bevgt_paths, bev_targetpoint, bev_checkpoint, ego, label, obstacles))
                            
            # save cache file
            if cache_file:
                with open(cache_file, "wb") as f:
                    pickle.dump(self.data, f)
                print(f"--- Save data to {cache_file} ---")
            else:
                with open("./dataset/cache.pkl", "wb") as f:
                    pickle.dump(self.data, f)
                print(f"--- Save data to ./dataset/cache.pkl ---")              
        
        
        
        # Train / Validation 데이터 셋 분리
        # self.data = self.data[:len(self.data)//18]
        
        random.seed(42)
        random.shuffle(self.data)
        train_size = int(len(self.data) * self.ratio)
        val_size = len(self.data) - train_size

        if self.train:
            self.data = self.data[:train_size]
            print("Train size: ", train_size)
        else:
            self.data = self.data[train_size:]
            print("Val size: ", val_size)
                   
    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        # img_paths, bev_paths, bev_targetpoint, bev_checkpoint, ego, label, obstacles = self.data[idx]
        img_paths, bev_paths, bev_targetpoint, bev_checkpoint, ego, label, obstacles, bev_globalpaths = self.data[idx]
        
        target_prefix = '/nas_data'
        new_prefix = '/home/jelee/nas_data' #todo
        img_paths = [img_path.replace(target_prefix, new_prefix, 1) for img_path in img_paths]
        bev_paths = [bev_path.replace(target_prefix, new_prefix, 1) for bev_path in bev_paths]
        
        # target_prefix = 'BEV_GT_1CAM'
        # new_prefix = 'BEV_GT_3CAMS'
        # bev_paths = [bev_path.replace(target_prefix, new_prefix, 1) for bev_path in bev_paths]
        
        local_tp = transform_from_bev_coordinates(bev_targetpoint)
        local_tp = torch.tensor(local_tp, dtype=torch.float32).squeeze(0)
        local_cp = transform_from_bev_coordinates(bev_checkpoint)
        local_cp = torch.tensor(local_cp, dtype=torch.float32)
        local_path = transform_from_bev_coordinates(bev_globalpaths)
        local_path = torch.tensor(local_path, dtype=torch.float32)

        ego = torch.tensor(ego[:-1], dtype=torch.float32)
        target_ctrl = torch.tensor([ego[20], ego[18]-ego[19]], dtype=torch.float32)
        speed = ego[11]

        # bevgt = [Image.open(bev_path) for bev_path in bev_paths]
        # bevgt_img = bevgt[0]
        # bevgt_img_np = np.array(bevgt_img)
        # bevgt_img_rotated = cv2.rotate(bevgt_img_np, cv2.ROTATE_180)
        
        #* bev_w_obj (for drawing)
        # bev_w_obj = torch.tensor(bevgt_img_rotated) #!!
        
        rgb_img = Image.open(img_paths[0])
        if self.transform_rgb:
            rgb_tensor = self.transform_rgb(rgb_img)
        else:
            rgb_tensor = transforms.ToTensor()(rgb_img)

        return {
            "rgb": rgb_tensor,
            # "bev": bev_w_obj,
            
            "local_targetpoint": local_tp,
            "local_checkpoint": local_cp,
            
            "ego": ego,
            "speed": speed,
            
            "target_ctrl": target_ctrl,
            
            "tl_label": label,
            # "img_path": img_paths[0]
            "local_path": local_path
        }
    
    def convert_color_to_class(self, bev):
        bev = np.array(bev)

        h, w, _ = bev.shape
        class_map = np.zeros((h, w), dtype=np.uint8)

        for color, class_id in self.config.bev_class_mapping.items():
            mask = (bev == color).all(axis=-1)
            class_map[mask] = class_id
        
        return torch.tensor(class_map, dtype=torch.uint8)
    
    def transform_global_to_local(self, waypoints, ego_position, ego_heading):
        waypoints = np.array(waypoints)
        ego_position = np.array(ego_position)
        
        # Waypoints와 ego_position 간의 차이를 계산
        delta = waypoints - ego_position  # delta_x, delta_y 계산
        
        # 헤딩 각도 변환
        cos_theta = np.cos(-ego_heading)
        sin_theta = np.sin(-ego_heading)
        
        # 회전 행렬을 적용하여 좌표 변환
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        local_coords = delta @ rotation_matrix.T

        return local_coords



if __name__ == "__main__":
    root_dir = "/home/justin/NAS_DATA/hyundai_challenge_2025/Official_data_all"
    config = GlobalConfig()
    
    dataset = MORAIDataset(
        root_dir,
        train=True,
        cache_file="official.pkl",
        config=config
    )
    
    for i in range(len(dataset)):
        print(i)
        a = dataset[i]
        visualize(a, i)
        