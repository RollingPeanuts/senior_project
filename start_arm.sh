#!/bin/bash
if [ "$#" -gt 1 ]; then
    echo "USAGE: $0 [script]"
    echo "  script: path to python3 script for robot arm to run on startup"
    exit 1
fi

# Change arm type to match appropriate model
arm_type="px100"

# Setup commands to be run
start_cmd="roslaunch interbotix_xsarm_control xsarm_control.launch robot_model:=${arm_type}"
torque_enable_cmd="rosservice call /${arm_type}/torque_enable \"{cmd_type: 'group', name: 'all', enable: true}\""

# Start robot arm and rviz interface
gnome-terminal -e "${start_cmd}"

# Wait for robot arm initialization to complete, then enable torque on robot arm
sleep 7
eval "${torque_enable_cmd}"

if [ "$#" -eq 1 ]; then
    python3 $1
fi

echo "Run scripts or commands for the robot arm in this terminal"

