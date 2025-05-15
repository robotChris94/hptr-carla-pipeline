'''
驗證過 output shape 與信心選擇, 可生成 best_traj
'''

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, "../hptr_core/models")
sys.path.append(models_dir)

from sc_relative import SceneCentricRelative  # 你的 HPTR 類別

import torch
import json

from utils import preprocess_input, load_checkpoint  # 自定義

model = SceneCentricRelative(...)
model.load_state_dict(torch.load('path/to/model.pt'))
model.eval()

pose_input = torch.tensor(...)  # [1, N, 3] 或你格式的軌跡輸入
pred, conf, _ = model(pose_input)  # pred: [1, 1, 128, 6, 359]

# 取 conf 最大的 mode index
best_mode_idx = torch.argmax(conf[0, 0]).item()
best_traj = pred[0, 0, :, best_mode_idx]  # [128, 359]

# 將軌跡從 bin index 轉成連續位置 (取 bin 的 center)
def bin_to_coord(idx):
    # 假設 359 是 x,y 方向離散化後的組合
    # 這裡你需要對照 bin 編碼方式來還原 (例如從 DiscreteDecoder)
    return [x, y]

coords = [bin_to_coord(torch.argmax(p).item()) for p in best_traj]

with open("logs/hptr_output.json", "w") as f:
    json.dump(coords, f)


def get_vehicle_pose(vehicle):
    transform = vehicle.get_transform()
    location = transform.location
    return location
