#!/usr/bin/env bash
set -e

cd /root/humble_ws
sudo apt-get update
rosdep install --from-paths src --ignore-src --rosdistro=humble -y
source /opt/ros/humble/setup.sh
colcon build
source install/local_setup.bash