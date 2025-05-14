import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from hptr_core.hptrWrapper import HPTRWrapper

# 建立 mock batch（注意：這只是示意，請根據你真實應用調整數值與 shape）
n_scene = 1
n_agent = 128
n_map = 32
n_pl_node = 5 
n_tl = 10
n_pred = 6
hidden_dim = 256
n_tl_hist = 20

batch = {
    "agent_valid": torch.ones((n_scene, n_agent, 20), dtype=torch.bool),
    "agent_type": torch.zeros((n_scene, n_agent, 3), dtype=torch.float32),          # one-hot
    "agent_attr": torch.zeros((n_scene, n_agent, 20, 68), dtype=torch.float32),         # 和 model config 對齊
    "agent_pose": torch.zeros((n_scene, n_agent, 3), dtype=torch.float32),       # [x, y, yaw, bbox, vel]
    "map_attr": torch.zeros((n_scene, n_map, 5, 38), dtype=torch.float32),
    "map_valid": torch.ones((n_scene, n_map, 5), dtype=torch.bool),
    "map_pose": torch.zeros((n_scene, n_map, 3), dtype=torch.float32),
    "tl_valid": torch.ones((n_scene, n_tl), dtype=torch.bool),               # 改成 2 維
    "tl_attr": torch.zeros((n_scene, n_tl, 5), dtype=torch.float32),
    "tl_pose": torch.zeros((n_scene, n_tl, 3), dtype=torch.float32),
}

total_len_pose = batch["map_pose"].shape[1] + batch["tl_pose"].shape[1] + batch["agent_pose"].shape[1]
total_len_valid = batch["map_valid"].shape[1] + batch["tl_valid"].shape[1] + batch["agent_valid"].shape[1]
print(f"🔎 pose_input total entities: {total_len_pose}, emb_invalid total entities: {total_len_valid}")

# 載入模型
model_path = os.path.join("hptr_core", "models", "model.ckpt")
model = HPTRWrapper(model_path, device='cpu')

# 推論
conf, pred = model.predict(batch)

conf, pred = model.predict(batch)
print("✅ 推論成功！")
print("conf shape:", conf.shape)
print("pred shape:", pred.shape)
