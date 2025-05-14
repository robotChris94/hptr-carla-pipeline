# experiment_runner.py
# 自動化實驗主控腳本：跑多模型、多場景、重複次數，儲存 log

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

import os
import time
import json
import random
import argparse
import carla
import numpy as np
from collections import deque

from hptr_wrapper import HPTRWrapper
from hptr_adapter import carla2hptr
from PIDController import PIDController
from map_loader import load_route_config, apply_spawn

def run_single_episode(world, ego, model, controller, route, log_path, seed, collision_events):
    history = deque(maxlen=20)
    random.seed(seed)
    world.tick()

    result = {
        'minADE': None,
        'collision': False,
        'completion': 0.0,
        'timestamp': time.time(),
        'seed': seed,
        'route': route['name']
    }

    end_loc = carla.Location(x=route['end'][0], y=route['end'][1], z=route['end'][2])
    gt_traj = []  # ground truth 歷史路徑記錄

    try:
        for step in range(100):
            world.tick()
            location = ego.get_location()
            gt_traj.append([location.x, location.y])  # 記錄真實位置

            history.append(location)
            polyline_tensor = carla2hptr(world, ego, history)
            pred = model.predict(polyline_tensor)
            best_traj = pred[0]  # shape = (80, 2)

            control = controller.run_trajectory(ego, best_traj[:20])
            ego.apply_control(control)
            time.sleep(0.05)

            dist_to_goal = location.distance(end_loc)
            if dist_to_goal < 3.0:
                result['completion'] = 1.0
                break

            if collision_events:
                result['collision'] = True
                break

        if 'completion' not in result:
            result['completion'] = 0.0

        # 計算真實 minADE（與 ground truth 對齊）
        gt_traj_np = np.array(gt_traj[:len(best_traj)])
        best_pred_np = best_traj[:len(gt_traj)].cpu().numpy()
        dists = np.linalg.norm(best_pred_np - gt_traj_np, axis=1)
        preds_np = pred.cpu().numpy()  # (6, 80, 2)
        gt_traj_np = np.array(gt_traj[:len(preds_np[0])])
        minADE = float(min([
        np.mean(np.linalg.norm(preds_np[i][:len(gt_traj_np)] - gt_traj_np, axis=1))
        for i in range(preds_np.shape[0])
        ]))
        result['minADE'] = minADE

    except Exception as e:
        result['error'] = str(e)

    finally:
        with open(log_path, 'a') as f:
            f.write(json.dumps(result) + '\n')


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

    ego = world.get_actors().filter('vehicle.*')[0]
    apply_spawn(world, ego, route)

    collision_events = []
    collision_bp = world.get_blueprint_library().find("sensor.other.collision")
    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=ego)
    collision_sensor.listen(lambda e: collision_events.append(e))

    model = HPTRWrapper(args.model_path)
    controller = PIDController()

    for i in range(args.repeat):
        seed = 1000 + i
        collision_events.clear()
        run_single_episode(world, ego, model, controller, route, args.log, seed, collision_events)

if __name__ == '__main__':
    main()
