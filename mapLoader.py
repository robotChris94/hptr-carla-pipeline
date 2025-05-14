# map_loader.py
# 載入路徑設定檔，設定地圖與 ego 起始點
'''
用於:
1. 讀 JSON
2. 將主車送到指定起點位置（spawn）
 '''

import carla
import json

def load_route_config(config_path):
    with open(config_path, 'r') as f:
        route = json.load(f)
    return route

def apply_spawn(world, ego, route):
    loc = route['start']
    spawn = carla.Transform(
        carla.Location(x=loc[0], y=loc[1], z=loc[2]),
        carla.Rotation(yaw=loc[3])
    )
    ego.set_transform(spawn)
    return spawn