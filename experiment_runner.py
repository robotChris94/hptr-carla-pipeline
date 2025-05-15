'''
CLI input:
python experiment_runner.py \
  --model_path models/model.ckpt \
  --config_path configs/route_Town03.json \
  --log logs/town03_run1.json \
  --repeat 10
'''
# experiment_runner.py
# 自動化實驗主控腳本：記錄真實 ground truth 與計算 minADE

import sys
import os
sys.path.append(os.path.abspath("./hptr_core"))
import time
import json
import random
import argparse
import torch
import carla
import numpy as np
from collections import deque

sys.path.append("carlaRunner/carla_interface") 
from carla_interface import get_vehicle_pose
from hptrWrapper import HPTRWrapper
from hptrAdapter import carla2hptr
from PIDController import PIDController
from mapLoader import load_route_config, apply_spawn

def run_single_episode(world, ego, model, controller, route, log_path, seed, collision_events):
    # ========== Step 1: 建立 batch ==========
    history = []
    batch = carla2hptr(world, ego, history, num_agent_type=len(model.model.dist_limit_agent))

    # ========== Step 2: 模型推論 ==========
    conf, pred = model.predict(batch)

    # ========== Step 3: 擷取 ego agent 的最佳預測軌跡 ==========
    conf = conf[0, 0]             # (128, 6)
    pred = pred[0, 0]             # (128, 6, 359) ← flat format
    conf_ego = conf[0]            # (6,)
    pred_ego = pred[0]            # (6, 359)

    best_mode = torch.argmax(conf_ego).item()
    flat = pred_ego[best_mode]    # (359,)

    # 修補 reshape 問題：確保長度能整除 2
    if flat.shape[0] % 2 != 0:
        print(f"⚠️ Warning: pred length {flat.shape[0]} is not divisible by 2, trimming last element")
        flat = flat[:flat.shape[0] // 2 * 2]

    best_traj = flat.reshape(-1, 2)  # (T, 2)

    # ========== Step 4: 儲存 best_traj ==========
    os.makedirs("logs", exist_ok=True)
    with open("logs/hptr_output.json", "w") as f:
        json.dump(best_traj.cpu().tolist(), f)
    print(f"✅ Saved best trajectory to logs/hptr_output.json (T={len(best_traj)})")

    # ========== Step 5: 計算真實軌跡與 minADE ==========
    gt_traj = []
    for _ in range(best_traj.shape[0]):
        loc = get_vehicle_pose(ego)
        gt_traj.append([loc.x, loc.y])
    gt_traj_np = np.array(gt_traj[:len(best_traj)])
    best_pred_np = best_traj[:len(gt_traj_np)].cpu().numpy()
    result = {
        "minADE": float(np.mean(np.linalg.norm(best_pred_np - gt_traj_np, axis=1))),
        "collision": len(collision_events) > 0,
        "completion": float(controller.compute_completion(gt_traj_np, route)),
        "timestamp": time.time(),
        "seed": seed,
        "route": route['name'],
    }

    # ========== Step 6: 寫入 log ==========
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--log', type=str, default="logs/experiment.json")
    parser.add_argument('--repeat', type=int, default=10)
    args = parser.parse_args()

    route = load_route_config(args.config_path)

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    client.load_world(route['name'].split('_')[0])
    world = client.get_world()
    
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]  # 可改為你喜歡的車型
    spawn_point = world.get_map().get_spawn_points()[0]
    ego = world.spawn_actor(vehicle_bp, spawn_point)
    
    apply_spawn(world, ego, route)

    collision_events = []
    collision_bp = world.get_blueprint_library().find("sensor.other.collision")
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
    collision_sensor.listen(lambda e: collision_events.append(e))
    
    try:
        model = HPTRWrapper(args.model_path)
        controller = PIDController()

        for i in range(args.repeat):
            seed = 1000 + i
            collision_events.clear()
            run_single_episode(world, ego, model, controller, route, args.log, seed, collision_events)

    finally:
        if 'collision_sensor' in locals():
            print("Cleaning up sensor...")
            collision_sensor.stop()
            collision_sensor.destroy()
        if 'ego' in locals():
            ego.destroy()

    

if __name__ == '__main__':
    main()
        
    
