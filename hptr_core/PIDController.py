# pid_controller.py
# 功能：簡單雙迴路 PID 控制器（速度 + 轉向）

'''
未來，if 實際導航/複雜場景追蹤, 可改用:
    Pure Pursuit Controller（簡單幾何控制）
    MPC（需考慮動態模型與優化成本）
'''

import carla
import math

class PID:
    def __init__(self, kp, ki, kd, dt=0.05):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0

    def step(self, error):
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

class PIDController:
    def __init__(self):
        self.steer_pid = PID(1.0, 0.0, 0.1)
        self.throttle_pid = PID(1.5, 0.1, 0.05)

    def run_step(self, vehicle, target_point, target_speed=10.0):
        """
        vehicle: carla.Vehicle
        target_point: [x, y] (m) in world coordinates
        target_speed: desired speed in km/h
        """
        transform = vehicle.get_transform()
        location = transform.location
        yaw = math.radians(transform.rotation.yaw)

        # 計算前輪與目標點的角度差
        dx = target_point[0] - location.x
        dy = target_point[1] - location.y
        angle_to_target = math.atan2(dy, dx)
        angle_diff = angle_to_target - yaw
        steer = self.steer_pid.step(angle_diff)
        steer = max(-1.0, min(1.0, steer))

        # 速度控制
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6
        throttle = self.throttle_pid.step(target_speed - speed)
        throttle = max(0.0, min(1.0, throttle))

        return carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0)
