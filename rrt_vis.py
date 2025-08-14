#!/usr/bin/env python3
import rospy
import time
import numpy as np
from std_msgs.msg import Float64MultiArray
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
from meshcat.visualizer import Visualizer
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
current_path = os.path.abspath(__file__)
parent_path_1 = os.path.dirname(current_path)        # 当前文件夹
parent_path_2 = os.path.dirname(parent_path_1)        # 父目录
parent_path_3 = os.path.dirname(parent_path_2)        # 爷爷目录
parent_path_4 = os.path.dirname(parent_path_3)   
sys.path.append(parent_path_1)
sys.path.append(parent_path_2)
sys.path.append(parent_path_3)
sys.path.append(parent_path_4)
from planning.RRTConnect import *

# === Robot Model Selection ===
robot_name = "g1"  # or "galaxea"

g1_dir = "./g1_description"
g1_urdf = g1_dir + "/g1_29dof.urdf"

galaxea_dir = "./Galaxea_R1_URDF/r1_v2_1_0"
galaxea_urdf = galaxea_dir + "/r1_v2_1_0.urdf"

if robot_name == "galaxea":
    urdf_path = galaxea_urdf
    package_path = galaxea_dir
    left_arm_joints = [
        "left_arm_joint1",
        "left_arm_joint2",
        "left_arm_joint3",
        "left_arm_joint4",
        "left_arm_joint5",
        "left_arm_joint6",
    ]
    ee_name = "left_gripper_joint"
elif robot_name == "g1":
    urdf_path = g1_urdf
    package_path = g1_dir
    left_arm_joints = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    ee_name = "L_ee"
else:
    raise ValueError("Unknown robot name")

# === Load Robot ===
if(robot_name == "galaxea"):
    robot = Robot(galaxea_urdf, galaxea_dir)
elif(robot_name == "g1"):
    robot = Robot(g1_urdf, g1_dir)
viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model, meshcat.Visualizer())
viz.initViewer(open=True)
viz.loadViewerModel()

# === ROS Callback ===
def callback(msg):

    q = msg.data[3:]
    full_q = pin.randomConfiguration(robot.model) * 0
    cnt = 0
    for joint_name in left_arm_joints:
        joint_id = robot.model.getJointId(joint_name)
        idx_q = robot.model.joints[joint_id].idx_q
        full_q[idx_q] = q[cnt]
        cnt += 1
    print("full_q", full_q)
    viz.display(full_q)

# === ROS Node ===
def listener():
    rospy.init_node('planning_viz_node', anonymous=True)
    rospy.Subscriber('planning_topic', Float64MultiArray, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
