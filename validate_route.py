# 用 map.get_waypoint() 驗證起點與終點是否在車道上

map = world.get_map()
start_loc = carla.Location(x=route["start"][0], y=route["start"][1], z=route["start"][2])
end_loc = carla.Location(x=route["end"][0], y=route["end"][1], z=route["end"][2])

start_wp = map.get_waypoint(start_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
end_wp = map.get_waypoint(end_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

if start_wp and end_wp:
    print("✅ Start & end are valid drivable points.")
else:
    print("❌ Invalid start or end point.")
