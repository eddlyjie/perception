#!/usr/bin/env python
import rospy
import numpy as np
import os
from std_msgs.msg import Float64MultiArray, Bool
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
current_path = os.path.abspath(__file__)
parent_path_1 = os.path.dirname(current_path)        
parent_path_2 = os.path.dirname(parent_path_1)        
parent_path_3 = os.path.dirname(parent_path_2)        
parent_path_4 = os.path.dirname(parent_path_3)   
sys.path.append(parent_path_1)
sys.path.append(parent_path_2)
sys.path.append(parent_path_3)
sys.path.append(parent_path_4)
from ultralytics import YOLO
from utils import *
import math
from planning.utils_planning import *
from planning.RRTConnect import *

class ReactiveTrajectoryReplayer:
    def __init__(self,realsense,perception,box_perception,check_grasp=False, step_size=0.005):

        self.check = check_grasp
        self.step_size = step_size

        self.robot_goal_pub = rospy.Publisher('/g1_29dof/robot_teleop', Float64MultiArray, queue_size=10)
        self.planning_pub = rospy.Publisher('planning_topic', Float64MultiArray, queue_size=10)
        self.gripper_pub = rospy.Publisher('/gripper_topic', Bool, queue_size=10)
        self.realsense = realsense
        self.perception = perception
        self.index = 1
        self.prev_joint_goal = None
        self.prev_gripper_state = None
        self.trajectory = []
        self.prev_robot_goal = None
        self.prev_gripper_state = None
        self.use_left_hand = False
        self.single_arm = False
        self.plan = True
        # if self.plan:
        self.reset_planner()
    
    def reset_planner(self):
        robot_name = "g1"
        g1_dir = "./g1_description"
        g1_urdf = g1_dir + "/g1_29dof.urdf"
        galaxea_dir = "./Galaxea_R1_URDF/r1_v2_1_0"
        galaxea_urdf = galaxea_dir + "/r1_v2_1_0.urdf"
        if(robot_name == "galaxea"):
            self.robot = Robot(galaxea_urdf, galaxea_dir)
        elif(robot_name == "g1"):
            self.robot = Robot(g1_urdf, g1_dir)
        self.planner = Planner(self.robot)
        self.cur_config = np.zeros(self.robot.model.nq)
        galaxea_left_arm_joints = ["left_arm_joint1",
                       "left_arm_joint2",
                       "left_arm_joint3",
                       "left_arm_joint4",
                       "left_arm_joint5",
                       "left_arm_joint6"]
        galaxea_left_ee_name = "left_gripper_joint"

        g1_left_arm_joints = ["left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_joint",
                        "left_wrist_roll_joint",
                        "left_wrist_pitch_joint",
                        "left_wrist_yaw_joint"]
        g1_left_ee_name = "L_ee"
        galaxea_right_arm_joints = [
            "right_arm_joint1",
            "right_arm_joint2",
            "right_arm_joint3",
            "right_arm_joint4",
            "right_arm_joint5",
            "right_arm_joint6",
        ]
        galaxea_right_ee_name = "right_gripper_joint"
        g1_right_arm_joints = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        g1_right_ee_name = "R_ee"
        if(robot_name == "galaxea"):
            self.right_arm_joints = galaxea_right_arm_joints
            self.left_arm_joints = galaxea_left_arm_joints
            self.right_ee_name = galaxea_right_ee_name
            self.left_ee_name = galaxea_left_ee_name
        elif(robot_name == "g1"):
            self.left_arm_joints = g1_left_arm_joints
            self.right_arm_joints = g1_right_arm_joints
            self.left_ee_name = g1_left_ee_name
            self.right_ee_name = g1_right_ee_name


    def count_data_files(self):
        count = 0
        while True:
            idx_str = f"{count+1:03d}"
            if os.path.exists(os.path.join(self.data_dir, f"robot_goal_{idx_str}.npy")):
                count += 1
            else:
                break
        return count

    def interpolate_translation_between_goals(self, start_vec, end_vec):
        left_start = start_vec[[3, 7, 11]]
        left_end = end_vec[[3, 7, 11]]
        right_start = start_vec[[19, 23, 27]]
        right_end = end_vec[[19, 23, 27]]

        left_distance = np.linalg.norm(left_end - left_start)
        right_distance = np.linalg.norm(right_end - right_start)
        max_distance = max(left_distance, right_distance)
        n_seg = max(1, int(np.ceil(max_distance / self.step_size)))

        left_interp = np.linspace(left_start, left_end, n_seg + 1, endpoint=True)
        right_interp = np.linspace(right_start, right_end, n_seg + 1, endpoint=True)

        interp_points = []
        for i in range(1, n_seg + 1):
            new_point = start_vec.copy()
            new_point[3], new_point[7], new_point[11] = left_interp[i]
            new_point[19], new_point[23], new_point[27] = right_interp[i]
            interp_points.append(new_point)
        return interp_points

    def perform_grasp_check(self):
        # if self.index == 8:
        K = np.load("perception/camera_intrinsics.npy")
        robot_goal = self.prev_robot_goal.copy()
        return check_object_in_hand(self.perception, self.realsense,self.object_id, robot_goal[[19, 23, 27]])
        # return True

    def save_place_goal(self):
        """
        Save a processed robot goal and gripper state to the given directory.
        First: save current robot_goal with z=0.
        Then: save a zeroed robot_goal with index +1.
        """
        if self.single_arm:
            save_dir = "perception/planned_goals/place/"
            base_idx = 1

            # 初始夹爪状态为 close
            robot_goal = self.prev_robot_goal.copy()
            gripper_state = True  # close
            goal_matrix = robot_goal.reshape(2, 4, 4)

            # Step 1: z=0.04, gripper = close
            goal_matrix[self.hand_idx, 2, 3] -= 0.15
            goal_path = os.path.join(save_dir, f"robot_goal_{base_idx:03d}.npy")
            np.save(goal_path, goal_matrix.flatten())
            gripper_path = os.path.join(save_dir, f"gripper_state_{base_idx:03d}.txt")
            with open(gripper_path, "w") as f:
                f.write("close")

            # Step 2: z=0.04, gripper = open
            next_idx = base_idx + 1
            goal_path = os.path.join(save_dir, f"robot_goal_{next_idx:03d}.npy")
            np.save(goal_path, goal_matrix.flatten())
            gripper_path = os.path.join(save_dir, f"gripper_state_{next_idx:03d}.txt")
            with open(gripper_path, "w") as f:
                f.write("open")

            # Step 3: z=0.2, gripper = open
            next_idx += 1
            goal_matrix[self.hand_idx, 2, 3] += 0.15
            goal_path = os.path.join(save_dir, f"robot_goal_{next_idx:03d}.npy")
            np.save(goal_path, goal_matrix.flatten())
            gripper_path = os.path.join(save_dir, f"gripper_state_{next_idx:03d}.txt")
            with open(gripper_path, "w") as f:
                f.write("open")

            # Step 4: home_goal, gripper = open
            next_idx += 1
            zero_goal = self.home_goal.copy()
            zero_goal_path = os.path.join(save_dir, f"robot_goal_{next_idx:03d}.npy")
            np.save(zero_goal_path, zero_goal.flatten())
            gripper_path = os.path.join(save_dir, f"gripper_state_{next_idx:03d}.txt")
            with open(gripper_path, "w") as f:
                f.write("open")
        else:
            save_dir = "perception/planned_goals/place/"
            base_idx = 1

            # 初始夹爪状态为 close
            robot_goal = self.prev_robot_goal.copy()
            gripper_state = True  # close
            goal_matrix = robot_goal.reshape(2, 4, 4)

            # Step 1: z=0.04, gripper = close
            goal_matrix[0, 2, 3] -= 0.15
            goal_matrix[1, 2, 3] -= 0.15

            goal_path = os.path.join(save_dir, f"robot_goal_{base_idx:03d}.npy")
            np.save(goal_path, goal_matrix.flatten())
            gripper_path = os.path.join(save_dir, f"gripper_state_{base_idx:03d}.txt")
            with open(gripper_path, "w") as f:
                f.write("close")

            # Step 2: z=0.04, gripper = open
            next_idx = base_idx + 1
            goal_path = os.path.join(save_dir, f"robot_goal_{next_idx:03d}.npy")
            np.save(goal_path, goal_matrix.flatten())
            gripper_path = os.path.join(save_dir, f"gripper_state_{next_idx:03d}.txt")
            with open(gripper_path, "w") as f:
                f.write("open")

            # Step 3: z=0.2, gripper = open
            next_idx += 1
            goal_matrix[0, 2, 3] += 0.15
            goal_matrix[1, 2, 3] += 0.15
            goal_path = os.path.join(save_dir, f"robot_goal_{next_idx:03d}.npy")
            np.save(goal_path, goal_matrix.flatten())
            gripper_path = os.path.join(save_dir, f"gripper_state_{next_idx:03d}.txt")
            with open(gripper_path, "w") as f:
                f.write("open")

            # Step 4: home_goal, gripper = open
            next_idx += 1
            zero_goal = self.home_goal.copy()
            zero_goal_path = os.path.join(save_dir, f"robot_goal_{next_idx:03d}.npy")
            np.save(zero_goal_path, zero_goal.flatten())
            gripper_path = os.path.join(save_dir, f"gripper_state_{next_idx:03d}.txt")
            with open(gripper_path, "w") as f:
                f.write("open")

    def set_hand_config(self, use_left_hand: bool):
        self.hand_idx = 0 if use_left_hand else 1
        self.ee_name = self.left_ee_name if use_left_hand else self.right_ee_name
        self.arm_joints = self.left_arm_joints if use_left_hand else self.right_arm_joints
        self.joint_indices = [self.robot.model.getJointId(joint)-1 for joint in self.arm_joints]

    def plan_trajectory(self, cart_goals, cur_config):
        if self.single_arm:
            joint_goals = []
            if getattr(self, "prev_joint_goal", None) is None:
                cur_config = np.zeros(self.robot.model.nq)
                self.prev_joint_goal = cur_config.copy()
            else:
                cur_config = self.prev_joint_goal.copy()
            robot_config = np.zeros(self.robot.model.nq)
            for idx, cart in enumerate(cart_goals):
                cart = cart.reshape(2,4,4)
                cart = cart[self.hand_idx]
                print(cart)
                # print(self.ee_name)
                q, status = self.robot.IK(
                    cart,
                    robot_config,
                    self.ee_name,
                    self.arm_joints
                )
                print(status)
                cart_tol = np.array([0.03, 0.03, 0.03, 0.0, 0.0, 0.02])
                while((status == False) or not self.robot.collision_free(q)):
                    cart_new = self.robot.sample_cart(cart, cart_tol)
                    q, status = self.robot.IK(cart_new, robot_config, self.ee_name, self.arm_joints)
                    print(status)
                    print(cart_new)
                if status == 0:
                    rospy.logwarn(f"[Single Arm] IK failed at idx {idx}, status={status}, skipping")
                    continue
                joint_wp = q[self.joint_indices]
                joint_goals.append(joint_wp)
                self.prev_joint_goal[self.joint_indices] = joint_wp
            # import ipdb; ipdb.set_trace()
            
            print("joint_goals", joint_goals)
            planned_traj = self.planner.plan(joint_goals, robot_config, self.arm_joints)
            self.full_traj = np.tile(cur_config, (len(planned_traj), 1))
            for i, q in enumerate(planned_traj):
                self.full_traj[i][self.joint_indices] = q

        else:
            left_joint_goals = []
            right_joint_goals = []
            self.keyjoints = []
            if getattr(self, "prev_joint_goal", None) is None:
                cur_config = np.zeros(self.robot.model.nq)
                self.prev_joint_goal = cur_config.copy()
            else:
                cur_config = self.prev_joint_goal.copy()
            # left_config = cur_config[15:22].copy()
            # right_config = cur_config[22:].copy()
            robot_config = np.zeros(self.robot.model.nq)
            for idx, cart in enumerate(cart_goals):
                # Left arm
                # import ipdb; ipdb.set_trace()
                cart = cart.reshape(2,4,4)
                cart_left = cart[0]
                self.set_hand_config(True)
                q_left, status_left = self.robot.IK(
                    cart_left,
                    robot_config,
                    self.left_ee_name,
                    self.left_arm_joints
                )
                cart_tol = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
                while(not status_left or not self.robot.collision_free(q_left)):
                    new_cart = self.robot.sample_cart(cart_left, cart_tol)
                    q_left, status_left = self.robot.IK(new_cart, robot_config, self.left_ee_name, self.left_arm_joints)
                if status_left == 0:
                    rospy.logwarn(f"[Dual Arm - Left] IK failed at idx {idx}, status={status_left}")
                    continue
                joint_wp_left = q_left[[self.robot.model.getJointId(j)-1 for j in self.left_arm_joints]]
                self.prev_joint_goal[15:22] = joint_wp_left
                left_joint_goals.append(joint_wp_left)

                # Right arm
                cart_right = cart[1]
                self.set_hand_config(False)
                q_right, status_right = self.robot.IK(
                    cart_right,
                    robot_config,
                    self.right_ee_name,
                    self.right_arm_joints
                )
                cart_tol = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])
                while(not status_right or not self.robot.collision_free(q_right)):
                    new_cart = self.robot.sample_cart(cart_right, cart_tol)
                    q_right, status_right = self.robot.IK(new_cart, robot_config, self.right_ee_name, self.right_arm_joints)
                # import ipdb; ipdb.set_trace()
                if status_right == 0:
                    rospy.logwarn(f"[Dual Arm - Right] IK failed at idx {idx}, status={status_right}")
                    continue
                joint_wp_right = q_right[[self.robot.model.getJointId(j)-1 for j in self.right_arm_joints]]
                self.prev_joint_goal[22:] = joint_wp_right
                right_joint_goals.append(joint_wp_right)
                self.keyjoints.append(self.prev_joint_goal)
            # 轨迹规划
            # import ipdb; ipdb.set_trace()
            # import ipdb; ipdb.set_trace()
            planned_traj_left = self.planner.plan(left_joint_goals, robot_config, self.left_arm_joints)
            planned_traj_right = self.planner.plan(right_joint_goals, robot_config, self.right_arm_joints)

            # 合并轨迹
            traj_len = min(len(planned_traj_left), len(planned_traj_right))
            self.full_traj = np.zeros((traj_len, self.robot.model.nq))
            left_indices = [self.robot.model.getJointId(j)-1 for j in self.left_arm_joints]
            right_indices = [self.robot.model.getJointId(j)-1 for j in self.right_arm_joints]

            for i in range(traj_len):
                self.full_traj[i][left_indices] = planned_traj_left[i]
                self.full_traj[i][right_indices] = planned_traj_right[i]


    def acute_angle_deg(self, theta_rad: float) -> float:
        """
        Calculate the acute angle (in degrees) between a line at angle theta_rad
        and the x-axis.

        Parameters:
            theta_rad (float): Angle of the line in radians (can be any real number).

        Returns:
            float: The acute angle between that line and the x-axis, in degrees (0 to 90).
        """
        # Normalize to [0, π)
        rad = abs(theta_rad) % math.pi
        # If larger than 90°, take the supplementary to get the acute angle
        if rad > math.pi / 2:
            rad = math.pi - rad
        return math.degrees(rad)
    
    def rot_x(self,angle_rad: float) -> np.ndarray:
        """
        Return a 4×4 homogeneous transform representing
        a rotation of `angle_rad` radians about the local x-axis.
        """
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        return np.array([
            [1, 0,  0, 0],
            [0, c, -s, 0],
            [0, s,  c, 0],
            [0, 0,  0, 1]
        ])
    def rot_y(self, angle_rad: float) -> np.ndarray:
        """
        Return a 4×4 homogeneous transform representing
        a rotation of `angle_rad` radians about the local y-axis.
        """
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        return np.array([
            [ c, 0,  s, 0],
            [ 0, 1,  0, 0],
            [-s, 0,  c, 0],
            [ 0, 0,  0, 1]
        ])
    def rot_z(self, angle_rad: float) -> np.ndarray:
        """
        Return a 4×4 homogeneous transform representing
        a rotation of `angle_rad` radians about the local z-axis.
        """
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        return np.array([
            [ c, -s, 0, 0],
            [ s,  c, 0, 0],
            [ 0,  0, 1, 0],
            [ 0,  0, 0, 1]
        ])
    def generate_pick_goal(self, object_id, object_location,grasp_pose = None, box_location= None):
        """
        Generate and save a pick sequence: prepick, pick, gripper close, pickup.
        Each step is saved as a 2x4x4 robot_goal and a gripper_state (open/close).
        """
        # theta = 0.2 
        # R = np.array([
        #     [np.cos(theta), np.sin(theta), 0],
        #     [-np.sin(theta),             np.cos(theta), 0            ],
        #     [0,0, 1]
        # ])
        # T = np.eye(4)
        # T[:3, :3] = R
        # Determine which hand to use

        if object_id in ["bowl", "tennis", "can", "wooden-block", "cone", "lollipop", "cloth","candy"]:
            # object_location[1] += 0.055
            use_left_hand = object_location[1] > 0  # y > 0 -> left hand, else right hand
            # print(use_left_hand)
            self.set_hand_config(use_left_hand)
            # self.check = False
            self.data_dir = f"perception/planned_goals/pick/{'bowl' if object_id == 'bowl' else 'object'}/"

            # Base transformation (keep the other hand unchanged)
            base_pose = self.home_goal.copy().reshape(2, 4, 4)
            base_pose[self.hand_idx, :3, 3] = object_location[:3]
            # x_offset = -0.02
            # base_pose[self.hand_idx, 0, 3]+= x_offset
            for i in [0, 1]:
                angle_offset = 0.2 if i == 0 else -0.2
                R_offset    = self.rot_z(angle_offset)
                base_pose[i] = base_pose[i] @ R_offset
            # for i in [0, 1]:
            #     base_pose[i] = base_pose[i] @ x_270
            angle =  np.pi/2 if self.hand_idx == 0 else -np.pi/2
            R     =  self.rot_x(angle)
            base_pose[self.hand_idx] = base_pose[self.hand_idx] @ self.rot_x(object_location[5])
            # Generate 4 steps
            poses = []

            # Step 1: prepick (above object)
            prepick = base_pose.copy()
            prepick[self.hand_idx, 2, 3] += 0.15
            poses.append(prepick)

            # Step 2: pick (at object)
            pick = base_pose.copy()
            poses.append(pick)

            # Step 3: gripper close (same as pick)
            close = pick.copy()
            poses.append(close)

            # Step 4: pickup (lift up)
            pickup = base_pose.copy()
            pickup[self.hand_idx, 2, 3] += 0.15
            poses.append(pickup)

            # Save each step
            for i, pose in enumerate(poses):
                idx = i + 1
                goal_path = os.path.join(self.data_dir, f"robot_goal_{idx:03d}.npy")
                np.save(goal_path, pose.flatten())

                # Gripper state: open for first two, closed after
                gripper_state = "open" if i < 2 else "close"
                gripper_path = os.path.join(self.data_dir, f"gripper_state_{idx:03d}.txt")
                with open(gripper_path, "w") as f:
                    f.write(gripper_state)

        if object_id == "cloth":
                        # object_location[1] += 0.055
            use_left_hand = object_location[1] > 0  # y > 0 -> left hand, else right hand
            # print(use_left_hand)
            self.set_hand_config(use_left_hand)
            # self.check = False
            self.data_dir = f"perception/planned_goals/pick/{'bowl' if object_id == 'bowl' else 'object'}/"

            # Base transformation (keep the other hand unchanged)
            base_pose = self.home_goal.copy().reshape(2, 4, 4)
            base_pose[self.hand_idx, :3, 3] = object_location[:3]
            # x_offset = -0.02
            # base_pose[self.hand_idx, 0, 3]+= x_offset
            for i in [0, 1]:
                angle_offset = 0.2 if i == 0 else -0.2
                R_offset    = self.rot_z(angle_offset)
                base_pose[i] = base_pose[i] @ R_offset
            # for i in [0, 1]:
            #     base_pose[i] = base_pose[i] @ x_270
            angle =  np.pi/2 if self.hand_idx == 0 else -np.pi/2
            R     =  self.rot_x(angle)
            base_pose[self.hand_idx] = base_pose[self.hand_idx] @ self.rot_x(object_location[5])
            # Generate 4 steps
            poses = []

            # Step 1: prepick (above object)
            prepick1 = base_pose.copy()
            prepick1[self.hand_idx, 2, 3] += 0.15
            poses.append(prepick1)

            prepick2 = base_pose.copy()
            angle_offset = 0.5 
            R_offset    = self.rot_y(angle_offset)
            prepick2[self.hand_idx] = prepick2[self.hand_idx] @ R_offset
            poses.append(prepick2)

            
            prepick3 = prepick2.copy()
            prepick3[self.hand_idx, 0, 3] += 0.1
            poses.append(prepick3)


            # Step 2: pick (at object)
            pick = prepick3.copy()
            poses.append(pick)

            # Step 3: gripper close (same as pick)
            close = pick.copy()
            poses.append(close)

            # Step 4: pickup (lift up)
            pickup = base_pose.copy()
            pickup[self.hand_idx, 2, 3] += 0.15
            poses.append(pickup)

            # Save each step
            for i, pose in enumerate(poses):
                idx = i + 1
                goal_path = os.path.join(self.data_dir, f"robot_goal_{idx:03d}.npy")
                np.save(goal_path, pose.flatten())

                # Gripper state: open for first two, closed after
                gripper_state = "open" if i < 2 else "close"
                gripper_path = os.path.join(self.data_dir, f"gripper_state_{idx:03d}.txt")
                with open(gripper_path, "w") as f:
                    f.write(gripper_state)

        if object_id == "box":
            self.data_dir = "perception/planned_goals/pick/box/"
            base_pose = self.home_goal.copy().reshape(2, 4, 4)
            pose1 = grasp_pose[0]
            pose1[0] += 0.02
            use_left_hand = pose1[1] > 0  # y > 0 -> left hand, else right hand
            # print(use_left_hand)
            self.set_hand_config(use_left_hand)
            # self.check = False
            # Base transformation (keep the other hand unchanged)
            base_pose[self.hand_idx, :3, 3] = pose1[:3]
            base_pose[self.hand_idx] = base_pose[self.hand_idx]@ self.rot_x(pose1[5])
            pose2 = grasp_pose[1]
            pose2[0] += 0.02
            use_left_hand = pose2[1] > 0  # y > 0 -> left hand, else right hand
            # print(use_left_hand)
            self.set_hand_config(use_left_hand)
            # self.check = False
            # Base transformation (keep the other hand unchanged)
            base_pose[self.hand_idx, :3, 3] = pose2[:3]
            base_pose[self.hand_idx] = base_pose[self.hand_idx]@ self.rot_x(pose2[5])
            # x_offset = -0.02
            # base_pose[self.hand_idx, 0, 3]+= x_offset
            for i in [0, 1]:
                angle_offset = 0.2 if i == 0 else -0.2
                R_offset    = self.rot_z(angle_offset)
                base_pose[i] = base_pose[i] @ R_offset  
            # for i in [0, 1]:
            #     base_pose[i] = base_pose[i] @ x_270
            # angle =  np.pi/2 if self.hand_idx == 0 else -np.pi/2
            # R     =  self.rot_x(angle)
            # base_pose[self.hand_idx] = base_pose[self.hand_idx] @ self.rot_x(object_location[5])
            # Generate 4 steps
            poses = []
            # Step 1: prepick (above object)
            prepick = base_pose.copy()
            prepick[0, 2, 3] += 0.15
            prepick[1, 2, 3] += 0.15
            poses.append(prepick)
            # Step 2: pick (at object)
            pick = base_pose.copy()
            poses.append(pick)

            # Step 3: gripper close (same as pick)
            close = pick.copy()
            poses.append(close)

            # Step 4: pickup (lift up)
            pickup = base_pose.copy()
            pickup[0, 2, 3] += 0.15
            pickup[1, 2, 3] += 0.15
            poses.append(pickup)

            # Save each step
            for i, pose in enumerate(poses):
                idx = i + 1
                goal_path = os.path.join(self.data_dir, f"robot_goal_{idx:03d}.npy")
                np.save(goal_path, pose.flatten())

                # Gripper state: open for first two, closed after
                gripper_state = "open" if i < 2 else "close"
                gripper_path = os.path.join(self.data_dir, f"gripper_state_{idx:03d}.txt")
                with open(gripper_path, "w") as f:
                    f.write(gripper_state)

    def precompute_trajectory(self):
        self.trajectory = []
        self.total_loaded = self.count_data_files()
        if self.plan:
            joint_goals = []
            gripper_states = []
            for i in range(1, self.total_loaded + 1):
                idx_str = f"{i:03d}"
                joint_path   = os.path.join(self.data_dir, f"robot_goal_{idx_str}.npy")
                gripper_path = os.path.join(self.data_dir, f"gripper_state_{idx_str}.txt")
                if not os.path.exists(joint_path) or not os.path.exists(gripper_path):
                    rospy.logwarn(f"Skipping missing index: {idx_str}")
                    continue
                joint_goals.append(np.load(joint_path))
                with open(gripper_path, "r") as f:
                    gripper_str = f.read().strip().lower()
                    gripper_states.append(gripper_str == "close")
                current_gripper_state = (gripper_str == "close")
                robot_goal = np.load(joint_path)
            if not joint_goals:
                rospy.logwarn("No joint goals found – empty trajectory.")
                return
            cart_goals = []
            if getattr(self, "prev_robot_goal", None) is not None:
                cart_goals.append(self.prev_robot_goal.copy())
                gripper_states.insert(0, self.prev_gripper_state)
            cart_goals.extend(joint_goals)
            self.plan_trajectory(cart_goals, self.prev_joint_goal)
            self.trajectory = []
            wp_ptr = 0
            for q in self.full_traj:
                is_keypoint = wp_ptr < len(joint_goals) and np.allclose(q, self.keyjoints[wp_ptr])
                if is_keypoint:
                    pause = True
                    current_gripper = gripper_states[wp_ptr]
                    wp_ptr += 1
                else:
                    pause = False
                    current_gripper = gripper_states[wp_ptr - 1]
                self.trajectory.append({
                    "joints": q[12:],
                    "gripper": current_gripper,
                    "pause": pause
                })

            rospy.loginfo(f"Precomputed {len(self.trajectory)} joint-space steps.")
            self.prev_robot_goal = robot_goal
            self.prev_gripper_state = current_gripper_state

        else:
            for i in range(1, self.total_loaded + 1):
                idx_str = f"{i:03d}"

                robot_goal_path = os.path.join(self.data_dir, f"robot_goal_{idx_str}.npy")
                gripper_path = os.path.join(self.data_dir, f"gripper_state_{idx_str}.txt")

                if not os.path.exists(robot_goal_path) or not os.path.exists(gripper_path):
                    rospy.logwarn("Skipping missing index: %s", idx_str)
                    continue

                robot_goal = np.load(robot_goal_path)
                # print(robot_goal)
                with open(gripper_path, "r") as f:
                    gripper_str = f.read().strip().lower()
                current_gripper_state = (gripper_str == "close")

                pause = (self.prev_gripper_state is not None and
                        current_gripper_state != self.prev_gripper_state)
                # import ipdb; ipdb.set_trace()
                _interp_dims = [3, 7, 11, 19, 23, 27]
                if self.prev_robot_goal is not None:
                    for i in range(robot_goal.shape[0]):
                        if i not in _interp_dims:
                            self.prev_robot_goal[i] = robot_goal[i]
                    interpolated = self.interpolate_translation_between_goals(
                        self.prev_robot_goal, robot_goal)
                    for point in interpolated:
                        self.trajectory.append({
                            "point": point,
                            "gripper": current_gripper_state,
                            "pause": False
                        })

                self.trajectory.append({
                    "point": robot_goal,
                    "gripper": current_gripper_state,
                    "pause": pause
                })

                self.prev_robot_goal = robot_goal
                self.prev_gripper_state = current_gripper_state

            rospy.loginfo("Precomputed %d trajectory steps.", len(self.trajectory))

    def packing_goals(self,packing_box_location):
        poses = []
        self.data_dir = "perception/planned_goals/pack/"
        base_pose = self.home_goal.copy().reshape(2, 4, 4)
        print(packing_box_location)
        packing_box_location[2] += 0.1
        #Step 1: side1 (front box)
        side1 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.2 if i == 0 else -0.2
            side1[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side1[i] = side1[i] @ R_offset  
            side1[i, 0, 3] = packing_box_location[0]-0.1
        poses.append(side1)
        #Step 2: side2 (middle box)
        side2 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.2 if i == 0 else -0.2
            side2[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side2[i] = side2[i] @ R_offset  
            side2[i, 0, 3] = packing_box_location[0]
        poses.append(side2)
        #Step 3: side3 (above box)
        side3 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.2 if i == 0 else -0.2
            side3[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side3[i] = side3[i] @ R_offset  
            side3[i, 2, 3] = packing_box_location[2]+0.1
            side3[i, 0, 3] = packing_box_location[0]
        poses.append(side3)
        #Step 4: side4 (in box)
        side4 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.16 if i == 0 else -0.16
            side4[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side4[i] = side4[i] @ R_offset  
            side4[i, 2, 3] = packing_box_location[2] + 0.1
            side4[i, 0, 3] = packing_box_location[0]
        poses.append(side4)
        print(poses)
        # Save each step
        for i, pose in enumerate(poses):
            idx = i + 1
            goal_path = os.path.join(self.data_dir, f"robot_goal_{idx:03d}.npy")
            np.save(goal_path, pose.flatten())
            # Gripper state: open for first two, closed after
            gripper_state = "close" if i == 3 else "close"
            gripper_path = os.path.join(self.data_dir, f"gripper_state_{idx:03d}.txt")
            with open(gripper_path, "w") as f:
                f.write(gripper_state)



    def pack(self):
        self.check = False
        self.data_dir = "perception/planned_goals/pack/"
        self.single_arm = False
        self.precompute_trajectory()
        self.publish_trajectory()
        return
    def reset_upper_body(self):
        self.check = False
        self.data_dir = "perception/planned_goals/reset/"
        self.single_arm = False
        self.precompute_trajectory()
        self.home_goal = self.prev_robot_goal.copy()
        self.publish_trajectory()
        return
    
   
    def pick(self,object_id):
        if object_id == "bowl":
            self.single_arm = True
            self.object_id = object_id
            self.check = False
            self.data_dir = "perception/planned_goals/pick/bowl/"
            self.precompute_trajectory()
            self.publish_trajectory()
            self.save_place_goal()
        elif object_id in ["tennis", "can", "wooden-block", "cone", "lollipop","cloth","candy"]:
            self.single_arm = True
            self.object_id = object_id
            self.check = False
            self.data_dir = "perception/planned_goals/pick/object/"
            self.precompute_trajectory()
            self.publish_trajectory()
            self.save_place_goal()
        elif object_id == "box":
            self.object_id = object_id
            self.check = False
            self.single_arm = False
            self.data_dir = "perception/planned_goals/pick/box/"
            self.precompute_trajectory()
            self.publish_trajectory()
            self.save_place_goal()
        elif object_id == "large":
            self.object_id = object_id
            self.check = True
            self.data_dir = "perception/planned_goals/pick/large/"
            self.precompute_trajectory()
            self.publish_trajectory()
            self.save_place_goal()
        elif object_id == "connected":
            self.object_id = object_id
            self.check = True
            self.data_dir = "perception/planned_goals/pick/candy/"
            self.precompute_trajectory()
            self.publish_trajectory()
            self.save_place_goal()
        return
    
    def place(self):
        self.check = False
        self.data_dir = "perception/planned_goals/place/"
        self.precompute_trajectory()
        self.publish_trajectory()
        return

    def home(self):
        self.single_arm = False
        self.check = False
        self.data_dir = "perception/planned_goals/home/"
        self.precompute_trajectory()
        self.publish_trajectory()
        return
    def publish_trajectory(self):
        if self.plan:
            for i, step in enumerate(self.trajectory):
                self.gripper_pub.publish(Bool(data=step["gripper"]))
                if step["pause"]:
                    rospy.loginfo("Gripper state changed at step %d, pausing...", i)
                    rospy.sleep(3)

                self.planning_pub.publish(Float64MultiArray(data=step["joints"].tolist()))
                rospy.sleep(0.1)
            if self.check and not self.perform_grasp_check():
                rospy.logwarn("Grasp check failed, stopping.")
            else:
                rospy.loginfo("Grasp check passed, continuing.")
            rospy.loginfo("Publishing trajectory...")
        else:
            for i, step in enumerate(self.trajectory):
                self.gripper_pub.publish(Bool(data=step["gripper"]))
                if step["pause"]:
                    rospy.loginfo("Gripper state changed at step %d, pausing...", i)
                    rospy.sleep(3)

                self.robot_goal_pub.publish(Float64MultiArray(data=step["point"].tolist()))
                rospy.sleep(0.1)

            if self.check and not self.perform_grasp_check():
                rospy.logwarn("Grasp check failed, stopping.")
            else:
                rospy.loginfo("Grasp check passed, continuing.")
            rospy.loginfo("Finished publishing trajectory.")

    def run(self):
        self.precompute_trajectory()
        self.publish_trajectory()

if __name__ == '__main__':
    realsense = Realsense()
    for i in range(50):
        realsense.get_frames()
    K = np.load("perception/camera_intrinsics.npy")
    box_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_picking_box2.pt"))
    object_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_wbcd10.pt"))
    object_id = "bowl"
    check_object_in_hand(object_perception, realsense,object_id, [0.28,-0.07,0.08])
