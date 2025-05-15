import json
import carla
import time
from pid_controller import PIDController
from carla_interface import get_vehicle_pose, apply_vehicle_control

# 讀取 best_traj
with open("logs/hptr_output.json") as f:
    traj = json.load(f)

client = carla.Client('localhost', 2000)
world = client.get_world()
vehicle = ...  # Spawn vehicle 或取得已存在車輛

steering_pid = PIDController(1.0, 0.0, 0.1)
target_idx = 0

while target_idx < len(traj):
    current_pose = get_vehicle_pose(vehicle)  # [x, y, yaw]
    target = traj[target_idx]

    # 計算方向誤差（Pure Pursuit 可進一步加速）
    dx = target[0] - current_pose[0]
    dy = target[1] - current_pose[1]
    error_heading = ...  # 將 (dx, dy) 與車輛朝向比較

    steer_cmd = steering_pid.control(error_heading, dt=0.05)
    throttle_cmd = 0.4

    apply_vehicle_control(vehicle, steer=steer_cmd, throttle=throttle_cmd)
    time.sleep(0.05)

    # 若靠近 target 則前往下一點
    if (dx**2 + dy**2)**0.5 < 1.0:
        target_idx += 1
