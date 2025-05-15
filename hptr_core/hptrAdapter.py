# hptr_adapter.py
# 功能：將 CARLA 世界轉換為 HPTR 所需的 polyline tensor

import torch
import numpy as np
import carla

# 假設我們採用 (N, L, D) 格式，其中 N 條 polyline，每條有 L 個點，每點 D 維特徵

MAX_POLYLINE = 128
MAX_POINTS = 20
FEATURE_DIM = 8  # 可依照模型調整
num_agent_type = 3

def carla2hptr(world, ego, history, num_agent_type=3):
    # 假設有 128 個 entity（agent, map, tl），每個都給一個 placeholder
    B = 1
    N = 128
    L = 20  # past steps
    D_attr = 68
    D_pose = 3
    
    

    batch = {
        "agent_valid": torch.ones(B, N, 1, dtype=torch.bool),             # 全部為有效 agent
        "agent_type": torch.zeros(B, N, num_agent_type, dtype=torch.long),           # 全部 type = 0
        "agent_attr": torch.randn(B, N, 1, 68),                      # 隨機 attr
        "agent_pose": torch.randn(B, N, D_pose),                   # 隨機 pose

        "map_valid": torch.ones(B, N, 1, dtype=torch.bool),
        "map_attr": torch.randn(B, N, 1, 38),
        "map_pose": torch.randn(B, N, D_pose),

        "tl_valid": torch.zeros(B, N, 1, dtype=torch.bool),
        "tl_attr": torch.zeros(B, N, 1, 38),
        "tl_pose": torch.zeros(B, N, D_pose),        
    }
    
    print("🧪 agent_pose.shape =", batch["agent_pose"].shape)
    print("🧪 map_pose.shape   =", batch["map_pose"].shape)
    print("🧪 tl_pose.shape    =", batch["tl_pose"].shape)
    
    return batch
