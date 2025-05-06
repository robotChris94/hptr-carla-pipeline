FROM nvidia/cuda:12.3.1-cudnn8-runtime-ubuntu22.04

# 1. ROS2 Foxy
RUN apt-get update && \
    apt-get install -y curl gnupg lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list
RUN apt-get update && apt-get install -y ros-foxy-desktop python3-rosdep

# 2. Python env
RUN apt-get install -y python3-pip
RUN pip install torch==2.2.0+cu121 torchvision --index-url https://download.pytorch.org/whl/cu121
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3. CARLA SDK (only need PythonAPI)
COPY Carla_PythonAPI/ /opt/carla
ENV PYTHONPATH=$PYTHONPATH:/opt/carla
