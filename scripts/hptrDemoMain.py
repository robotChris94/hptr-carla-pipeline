# hptr_demo_main.py
# 手動調試用
# 主控程式：整合 HPTR + Adapter + PID 控制器

import carla
import time
import collections

from hptr_wrapper import HPTRWrapper
from hptr_adapter import carla2hptr
from pid_controller import PIDController

history = collections.deque(maxlen=20)

def main():
    # 連線到 CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    # 取得主車輛
    ego = world.get_actors().filter('vehicle.*')[0]  # 預設只取第一台
    controller = PIDController()

    # 載入 HPTR 模型
    hptr = HPTRWrapper("./models/hptr_model.pt")

    while True:
        world.tick()
        transform = ego.get_transform()
        location = transform.location
        history.append(location)

        polyline_tensor = carla2hptr(world, ego, history)
        pred = hptr.predict(polyline_tensor)

        # 選擇最可能的軌跡
        best_traj = pred[0]  # shape: (80, 2)
        target_point = best_traj[10].tolist()  # 預測 0.5s 後的位置為目標

        control = controller.run_step(ego, target_point)
        ego.apply_control(control)
        time.sleep(0.05)

if __name__ == '__main__':
    main()
