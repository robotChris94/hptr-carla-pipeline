def get_vehicle_pose(vehicle):
    transform = vehicle.get_transform()
    location = transform.location
    yaw = transform.rotation.yaw
    return [location.x, location.y, yaw]

def apply_vehicle_control(vehicle, steer, throttle):
    control = carla.VehicleControl()
    control.steer = max(min(steer, 1.0), -1.0)
    control.throttle = max(min(throttle, 1.0), 0.0)
    vehicle.apply_control(control)
