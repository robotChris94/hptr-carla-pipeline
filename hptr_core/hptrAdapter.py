# hptr_adapter.py
# 功能：將 CARLA 世界轉換為 HPTR 所需的 polyline tensor

import torch
import numpy as np
import carla

# 假設我們採用 (N, L, D) 格式，其中 N 條 polyline，每條有 L 個點，每點 D 維特徵

MAX_POLYLINE = 128
MAX_POINTS = 20
FEATURE_DIM = 8  # 可依照模型調整

def carla2hptr(world, ego_vehicle, history_buffer):
    """
    將 CARLA 場景中的資訊轉換為 polyline tensor，供 HPTR 使用。
    history_buffer: collections.deque 存過去 N 次 tick 的 ego 座標。
    """
    polylines = []

    # 1. ego 歷史軌跡 polyline
    ego_points = []
    for loc in list(history_buffer)[-MAX_POINTS:]:
        ego_points.append([loc.x, loc.y, 0, 0, 0, 0, 0, 1])  # 最後一維 one-hot = ego
    while len(ego_points) < MAX_POINTS:
        ego_points.insert(0, ego_points[0])  # padding
    polylines.append(ego_points)

    # 2. 動態車輛（附近其他車）
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego_vehicle.id:
            continue
        loc = actor.get_location()
        poly = [[loc.x, loc.y, 0, 0, 0, 0, 1, 0]] * MAX_POINTS
        polylines.append(poly)
        if len(polylines) >= MAX_POLYLINE:
            break

    # 3. 道路線（可略或簡化為直線）
    # 在正式版本可用 map.get_waypoint() 拿 lane centerline

    polyline_tensor = torch.tensor(polylines[:MAX_POLYLINE], dtype=torch.float32)
    return polyline_tensor

# 注意：你要在主程式裡建立 history_buffer = deque(maxlen=MAX_POINTS)
# 並每 tick 時記錄 ego 的位置進去
