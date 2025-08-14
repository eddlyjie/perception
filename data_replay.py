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
from perception.utils import *
import math
from planning.utils_planning import *
from planning.RRTConnect import *

    
class ReactiveTrajectoryReplayer:
    def __init__(self,realsense,perception,box_perception,check_grasp=False, step_size=0.005):

        self.check = check_grasp
        self.step_size = step_size

        self.robot_goal_pub = rospy.Publisher('/g1_29dof/robot_teleop', Float64MultiArray, queue_size=10)
        self.planning_pub = rospy.Publisher('planning_topic', Float64MultiArray, queue_size=10)
        self.left_gripper_pub = rospy.Publisher('/left_gripper_topic', Bool, queue_size=10, latch=True)
        self.right_gripper_pub = rospy.Publisher('/right_gripper_topic', Bool, queue_size=10, latch=True)
        self.realsense = realsense
        self.perception = perception
        self.index = 1
        self.prev_joint_goal = None
        self.prev_left_gripper_state = False
        self.prev_right_gripper_state = False
        self.trajectory = []
        self.prev_robot_goal = None
        self.use_left_hand = False
        self.single_arm = False
        self.plan = True
        self.goal_types = None
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
        self.dual_arm_joints = ["left_shoulder_pitch_joint",
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
                        "right_wrist_yaw_joint",]
        self.dual_joint_indices = [self.robot.model.getJointId(joint)-1 for joint in self.dual_arm_joints]

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
        rospy.sleep(1)
        K = np.load("perception/camera_intrinsics.npy")
        robot_goal = self.prev_robot_goal.copy()
        self.set_hand_config(self.use_left_hand)
        if self.hand_idx == 0:  # left hand
            hand_pos = robot_goal[[3, 7, 11]]
        else:  # right hand
            hand_pos = robot_goal[[19, 23, 27]]

        return check_object_in_hand(self.perception, self.realsense, self.object_id, hand_pos)
        # return True

    def save_place_goal(self):
        """
        Save a processed robot goal and gripper state to the given directory.
        First: save current robot_goal with z=0.
        Then: save a zeroed robot_goal with index +1.
        """
        if self.single_arm:
            self.set_hand_config(self.use_left_hand)
            save_dir = "perception/planned_goals/place/"
            base_idx = 1

            robot_goal = self.prev_robot_goal.copy()
            goal_matrix = robot_goal.reshape(2, 4, 4)

            # Step 1: z=0.04, gripper = close
            goal_matrix[self.hand_idx, 2, 3] -= 0.2
            goal_path = os.path.join(save_dir, f"robot_goal_{base_idx:03d}.npy")
            np.save(goal_path, goal_matrix.flatten())

            # Step 2: z=0.04, gripper = open
            next_idx = base_idx + 1
            goal_path = os.path.join(save_dir, f"robot_goal_{next_idx:03d}.npy")
            np.save(goal_path, goal_matrix.flatten())

            # Step 3: z=0.2, gripper = open
            next_idx += 1
            goal_matrix[self.hand_idx, 2, 3] += 0.2
            goal_path = os.path.join(save_dir, f"robot_goal_{next_idx:03d}.npy")
            np.save(goal_path, goal_matrix.flatten())

            # Step 4: home_goal, gripper = open
            next_idx += 1
            zero_goal = self.home_goal.copy()
            zero_goal_path = os.path.join(save_dir, f"robot_goal_{next_idx:03d}.npy")
            np.save(zero_goal_path, zero_goal.flatten())

        else:
            save_dir = "perception/planned_goals/place/"
            base_idx = 1

            robot_goal = self.prev_robot_goal.copy()
            goal_matrix = robot_goal.reshape(2, 4, 4)

            # Step 1: z=0.04, gripper = close
            goal_matrix[0, 2, 3] -= 0.2
            goal_matrix[1, 2, 3] -= 0.2
            goal_path = os.path.join(save_dir, f"robot_goal_{base_idx:03d}.npy")
            np.save(goal_path, goal_matrix.flatten())

            # Step 2: z=0.04, gripper = open
            next_idx = base_idx + 1
            goal_path = os.path.join(save_dir, f"robot_goal_{next_idx:03d}.npy")
            np.save(goal_path, goal_matrix.flatten())

            # Step 3: z=0.2, gripper = open
            next_idx += 1
            goal_matrix[0, 2, 3] += 0.2
            goal_matrix[1, 2, 3] += 0.2
            goal_path = os.path.join(save_dir, f"robot_goal_{next_idx:03d}.npy")
            np.save(goal_path, goal_matrix.flatten())

            # Step 4: home_goal, gripper = open
            next_idx += 1
            zero_goal = self.home_goal.copy()
            zero_goal_path = os.path.join(save_dir, f"robot_goal_{next_idx:03d}.npy")
            np.save(zero_goal_path, zero_goal.flatten())

        # Write gripper states
        for i in range(4):
            idx = i+1
            gripper_state = "open" if i in [1, 2, 3] else "close"
            with open(os.path.join(self.data_dir, f"gripper_left_state_{idx:03d}.txt"), "w") as f:
                f.write(gripper_state if self.hand_idx == 0 else "open")
            with open(os.path.join(self.data_dir, f"gripper_right_state_{idx:03d}.txt"), "w") as f:
                f.write(gripper_state if self.hand_idx == 1 else "open")
    def stir_goal(self, object_id,object_location):
        self.use_left_hand = object_location[1] > 0  # y > 0 -> left hand, else right hand
        # print(use_left_hand)
        self.set_hand_config(self.use_left_hand)
        # self.check = False
        self.data_dir = "perception/planned_goals/stir/"

        # Base transformation (keep the other hand unchanged)
        base_pose = self.home_goal.copy().reshape(2, 4, 4)
        base_pose[self.hand_idx, :3, 3] = object_location[:3]
        # for i in [0, 1]:
        #     angle_offset = 0.2 if i == 0 else -0.2
        #     R_offset    = self.rot_z(angle_offset)
        #     base_pose[i] = base_pose[i] @ R_offset
        # for i in [0, 1]:
        #     base_pose[i] = base_pose[i] @ x_270
        angle =  np.pi/2 if self.hand_idx == 0 else -np.pi/2
        R     =  self.rot_x(angle)
        base_pose[self.hand_idx] = base_pose[self.hand_idx] @ self.rot_x(object_location[5])
        # Generate 4 steps
        poses = []

        # Step 1: prepick (above object)
        prepick = base_pose.copy()
        prepick[self.hand_idx, 2, 3] += 0.2
        poses.append(prepick)

        # Step 2: pick (at object)
        pick = base_pose.copy()
        poses.append(pick)

        # Step 3: move (same as pick)
        move = pick.copy()
        move[self.hand_idx, 0, 3] += 0.1
        poses.append(move)

        # Step 4: pickup (lift up)
        pickup = base_pose.copy()
        pickup[self.hand_idx, 2, 3] += 0.2
        poses.append(pickup)

        # Save each step
        self.goal_types = []
        for i, pose in enumerate(poses):
            idx = i + 1
            goal_path = os.path.join(self.data_dir, f"robot_goal_{idx:03d}.npy")
            np.save(goal_path, pose.flatten())

            gripper_state = "close"
            goal_type = "soft" if i != 1 else "hard"  
            self.goal_types.append(goal_type)

            with open(os.path.join(self.data_dir, f"gripper_left_state_{idx:03d}.txt"), "w") as f:
                f.write(gripper_state if self.hand_idx == 0 else "open")
            with open(os.path.join(self.data_dir, f"gripper_right_state_{idx:03d}.txt"), "w") as f:
                f.write(gripper_state if self.hand_idx == 1 else "open")

    def seperate_goal(self, object_id,object_location):
        self.use_left_hand = object_location[1] > 0  # y > 0 -> left hand, else right hand
        # print(use_left_hand)
        self.set_hand_config(self.use_left_hand)
        # self.check = False
        self.data_dir = "perception/planned_goals/stir/"

        # Base transformation (keep the other hand unchanged)
        base_pose = self.home_goal.copy().reshape(2, 4, 4)
        base_pose[self.hand_idx, :3, 3] = object_location[:3]
        # for i in [0, 1]:
        #     angle_offset = 0.2 if i == 0 else -0.2
        #     R_offset    = self.rot_z(angle_offset)
        #     base_pose[i] = base_pose[i] @ R_offset
        # for i in [0, 1]:
        #     base_pose[i] = base_pose[i] @ x_270
        angle =  np.pi/2 if self.hand_idx == 0 else -np.pi/2
        R     =  self.rot_x(angle)
        base_pose[self.hand_idx] = base_pose[self.hand_idx] @ self.rot_x(object_location[5])
        # Generate 4 steps
        poses = []

        # Step 1: prepick (above object)
        prepick = base_pose.copy()
        prepick[self.hand_idx, 2, 3] += 0.2
        poses.append(prepick)

        # Step 2: pick (at object)
        pick = base_pose.copy()
        poses.append(pick)

        # Step 3: move (same as pick)
        move = pick.copy()
        move[self.hand_idx, 0, 3] += 0.1
        poses.append(move)

        # Step 4: pickup (lift up)
        pickup = base_pose.copy()
        pickup[self.hand_idx, 2, 3] += 0.2
        poses.append(pickup)

        # Save each step
        self.goal_types = []
        for i, pose in enumerate(poses):
            idx = i + 1
            goal_path = os.path.join(self.data_dir, f"robot_goal_{idx:03d}.npy")
            np.save(goal_path, pose.flatten())

            gripper_state = "close"
            goal_type = "soft" if i != 1 else "hard"  
            self.goal_types.append(goal_type)

            with open(os.path.join(self.data_dir, f"gripper_left_state_{idx:03d}.txt"), "w") as f:
                f.write(gripper_state if self.hand_idx == 0 else "open")
            with open(os.path.join(self.data_dir, f"gripper_right_state_{idx:03d}.txt"), "w") as f:
                f.write(gripper_state if self.hand_idx == 1 else "open")
    def set_hand_config(self, use_left_hand: bool):
        self.hand_idx = 0 if use_left_hand else 1
        self.ee_name = self.left_ee_name if use_left_hand else self.right_ee_name
        self.arm_joints = self.left_arm_joints if use_left_hand else self.right_arm_joints
        self.joint_indices = [self.robot.model.getJointId(joint)-1 for joint in self.arm_joints]

    def plan_trajectory(self, cart_goals, cur_config, goal_types=None):
        # if self.single_arm:
        #     joint_goals = []
        #     self.keyjoints = []
        #     if getattr(self, "prev_joint_goal", None) is None:
        #         cur_config = np.zeros(self.robot.model.nq)
        #         self.prev_joint_goal = cur_config.copy()
        #         # self.keyjoints = [self.prev_joint_goal.copy()]
        #     else:
        #         cur_config = self.prev_joint_goal.copy()
        #         joint_goals = [self.prev_joint_goal.copy()[self.joint_indices]]
        #         self.keyjoints = [self.prev_joint_goal.copy()]
        #     robot_config = np.zeros(self.robot.model.nq)
        #     for idx, cart in enumerate(cart_goals):
        #         cart = cart.reshape(2,4,4)
        #         cart = cart[self.hand_idx]
        #         q, status = self.robot.IK(
        #             cart,
        #             robot_config,
        #             self.ee_name,
        #             self.arm_joints
        #         )
        #         q_test = self.prev_joint_goal.copy()
        #         q_test[self.joint_indices] = q[self.joint_indices]
        #         if goal_types and goal_types[idx] == "soft":
        #             cart_tol = np.array([0.03, 0.03, 0.03, 0.0, 0.0, 0.02])
        #             max_attempts = 10
        #             attempt = 0
        #             while (status == False or not self.robot.collision_free(q_test)) and attempt < max_attempts:
        #                 print(f"IK failed (attempt {attempt+1}), sampling new cartesian goal")
        #                 cart_new = self.robot.sample_cart(cart, cart_tol)
        #                 q, status = self.robot.IK(cart_new, robot_config, self.ee_name, self.arm_joints)
        #                 q_test[self.joint_indices] = q[self.joint_indices]
        #                 attempt += 1

        #             if status == 0 or not self.robot.collision_free(q_test):
        #                 raise RuntimeError(f"[Single Arm] IK failed at idx {idx} after {max_attempts} attempts.")

        #         else:
        #             if status == 0 or not self.robot.collision_free(q_test):
        #                 raise RuntimeError(f"[Single Arm] Hard goal IK failed at idx {idx}, status={status}")
        #         joint_wp = q[self.joint_indices]
        #         joint_goals.append(joint_wp)
        #         self.prev_joint_goal[self.joint_indices] = joint_wp
        #         self.keyjoints.append(self.prev_joint_goal.copy())  

        #     planned_traj = self.planner.plan(joint_goals, cur_config, self.arm_joints)
        #     self.full_traj = np.tile(cur_config, (len(planned_traj), 1))
        #     for i, q in enumerate(planned_traj):
        #         self.full_traj[i][self.joint_indices] = q

        # else:           
            dual_joint_goals = []
            self.keyjoints = []
            if getattr(self, "prev_joint_goal", None) is None:
                cur_config = np.zeros(self.robot.model.nq)
                self.prev_joint_goal = cur_config.copy()
                # self.keyjoints = [self.prev_joint_goal.copy()]
            else:
                cur_config = self.prev_joint_goal.copy()
                dual_joint_goals = [self.prev_joint_goal.copy()[self.dual_joint_indices]]        
                self.keyjoints = [self.prev_joint_goal.copy()]

            robot_config = np.zeros(self.robot.model.nq)
            for idx, cart in enumerate(cart_goals):
                cart = cart.reshape(2,4,4)
                left_cart = cart[0, :, :]
                right_cart = cart[1, :, :]
                cart_tol = np.array([0, 0, 0, 0, 0, 0])
                left_new_cart = np.copy(left_cart)
                right_new_cart = np.copy(right_cart)

                if goal_types and goal_types[idx] == "soft":
                    cart_tol = np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1])

                self.set_hand_config(True)
                left_full_q, left_status = self.robot.IK(left_cart, robot_config, self.left_ee_name, self.left_arm_joints)

                self.set_hand_config(False)
                right_full_q, right_status = self.robot.IK(right_cart, robot_config, self.right_ee_name, self.right_arm_joints)

                full_q = np.copy(robot_config)
                full_q[15:22] = left_full_q[15:22]
                full_q[22:] = right_full_q[22:]
                if goal_types and goal_types[idx] == "soft":
                    while (not left_status or not right_status or not self.robot.collision_free(full_q)):
                        left_new_cart = self.robot.sample_cart(left_cart, cart_tol)
                        right_new_cart = self.robot.sample_cart(right_cart, cart_tol)

                        self.set_hand_config(True)
                        left_full_q, left_status = self.robot.IK_opt(left_new_cart, robot_config, self.left_ee_name, self.left_arm_joints)
                        self.set_hand_config(False)
                        right_full_q, right_status = self.robot.IK_opt(right_new_cart, robot_config, self.right_ee_name, self.right_arm_joints)

                        full_q[15:22] = left_full_q[15:22]
                        full_q[22:] = right_full_q[22:]

                    if not left_status or not right_status:
                        rospy.logwarn(f"[Dual Arm] IK failed at idx {idx}, skipping")
                        continue

                else:
                    if not left_status:
                        print(left_cart)
                        raise RuntimeError(f"[Dual Arm - Left] Hard goal IK failed at idx {idx}, status={left_status}")
                    if not right_status:
                        print(right_cart)
                        raise RuntimeError(f"[Dual Arm - Right] Hard goal IK failed at idx {idx}, status={right_status}")
                    if not self.robot.collision_free(full_q):
                        raise RuntimeError(f"[Dual Arm] Hard goal collision detected at idx {idx}")

                dual_joint_wp = full_q[self.dual_joint_indices]
                self.prev_joint_goal[self.dual_joint_indices] = dual_joint_wp
                dual_joint_goals.append(dual_joint_wp)
                self.keyjoints.append(self.prev_joint_goal.copy())
            print(dual_joint_goals)
            path = self.planner.plan(dual_joint_goals, robot_config, self.dual_arm_joints)
            traj_len = len(path)
            self.full_traj = np.zeros((traj_len, self.robot.model.nq))
            dual_indices = self.dual_joint_indices 
            for i in range(traj_len):
                dual_i = min(i, len(path) - 1)
                self.full_traj[i][dual_indices] = path[dual_i]



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

        if object_id in ["bowl", "tennis", "can", "wooden-block", "cone", "lolipop", "candy","bread","cloth"]:
            if object_id == "bowl":
                x_offset = -0.03
            else:
                x_offset = 0.00
            # object_location[1] += 0.055
            self.use_left_hand = object_location[1] > 0  # y > 0 -> left hand, else right hand
            # print(use_left_hand)
            self.set_hand_config(self.use_left_hand)
            # self.check = False
            self.data_dir = f"perception/planned_goals/pick/{'bowl' if object_id == 'bowl' else 'object'}/"

            # Base transformation (keep the other hand unchanged)
            base_pose = self.home_goal.copy().reshape(2, 4, 4)
            base_pose[self.hand_idx, :3, 3] = object_location[:3]
            base_pose[self.hand_idx, 0, 3]+= x_offset
            # for i in [0, 1]:
            #     angle_offset = 0.2 if i == 0 else -0.2
            #     R_offset    = self.rot_z(angle_offset)
            #     base_pose[i] = base_pose[i] @ R_offset
            # for i in [0, 1]:
            #     base_pose[i] = base_pose[i] @ x_270
            angle =  np.pi/2 if self.hand_idx == 0 else -np.pi/2
            R     =  self.rot_x(angle)
            base_pose[self.hand_idx] = base_pose[self.hand_idx] @ self.rot_x(object_location[5])
            # Generate 4 steps
            poses = []

            # Step 1: prepick (above object)
            prepick = base_pose.copy()
            prepick[self.hand_idx, 2, 3] += 0.2
            poses.append(prepick)

            # Step 2: pick (at object)
            pick = base_pose.copy()
            poses.append(pick)

            # Step 3: gripper close (same as pick)
            close = pick.copy()
            poses.append(close)

            # Step 4: pickup (lift up)
            pickup = base_pose.copy()
            pickup[self.hand_idx, 2, 3] += 0.2
            poses.append(pickup)

            # Save each step
            self.goal_types = []
            for i, pose in enumerate(poses):
                idx = i + 1
                goal_path = os.path.join(self.data_dir, f"robot_goal_{idx:03d}.npy")
                np.save(goal_path, pose.flatten())

                gripper_state = "open" if i < 2 else "close"
                goal_type = "soft" if i != 1 else "hard"  
                self.goal_types.append(goal_type)

                with open(os.path.join(self.data_dir, f"gripper_left_state_{idx:03d}.txt"), "w") as f:
                    f.write(gripper_state if self.hand_idx == 0 else "open")
                with open(os.path.join(self.data_dir, f"gripper_right_state_{idx:03d}.txt"), "w") as f:
                    f.write(gripper_state if self.hand_idx == 1 else "open")


        # if object_id == "cloth":
        #                 # object_location[1] += 0.055
        #     self.use_left_hand = object_location[1] > 0  # y > 0 -> left hand, else right hand
        #     # print(use_left_hand)
        #     self.set_hand_config(self.use_left_hand)
        #     # self.check = False
        #     self.data_dir = f"perception/planned_goals/pick/cloth/"

        #     # Base transformation (keep the other hand unchanged)
        #     base_pose = self.home_goal.copy().reshape(2, 4, 4)
        #     base_pose[self.hand_idx, :3, 3] = object_location[:3]
        #     # x_offset = -0.02
        #     # base_pose[self.hand_idx, 0, 3]+= x_offset
        #     for i in [0, 1]:
        #         angle_offset = 0.2 if i == 0 else -0.2
        #         R_offset    = self.rot_z(angle_offset)
        #         base_pose[i] = base_pose[i] @ R_offset
        #     # for i in [0, 1]:
        #     #     base_pose[i] = base_pose[i] @ x_270
        #     angle =  np.pi/2 if self.hand_idx == 0 else -np.pi/2
        #     R     =  self.rot_x(angle)
        #     base_pose[self.hand_idx] = base_pose[self.hand_idx] @ self.rot_x(object_location[5])
        #     # Generate 4 steps
        #     poses = []

        #     # Step 1: prepick (above object)
        #     prepick1 = base_pose.copy()
        #     prepick1[self.hand_idx, 2, 3] += 0.2
        #     poses.append(prepick1)

        #     prepick2 = base_pose.copy()
        #     angle_offset = 0.5 
        #     R_offset    = self.rot_y(angle_offset)
        #     prepick2[self.hand_idx] = prepick2[self.hand_idx] @ R_offset
        #     poses.append(prepick2)

            
        #     prepick3 = prepick2.copy()
        #     prepick3[self.hand_idx, 0, 3] += 0.1
        #     poses.append(prepick3)


        #     # Step 2: pick (at object)
        #     pick = prepick3.copy()
        #     poses.append(pick)

        #     # Step 3: gripper close (same as pick)
        #     close = pick.copy()
        #     poses.append(close)

        #     # Step 4: pickup (lift up)
        #     pickup = base_pose.copy()
        #     pickup[self.hand_idx, 2, 3] += 0.2
        #     poses.append(pickup)

        #     # Save each step
        #     self.goal_types = []
        #     for i, pose in enumerate(poses):
        #         idx = i + 1
        #         goal_path = os.path.join(self.data_dir, f"robot_goal_{idx:03d}.npy")
        #         np.save(goal_path, pose.flatten())

        #         gripper_state = "open" if i < 2 else "close"
        #         goal_type = "soft" if i != 1 else "hard"  
        #         self.goal_types.append(goal_type)

        #         with open(os.path.join(self.data_dir, f"gripper_left_state_{idx:03d}.txt"), "w") as f:
        #             f.write(gripper_state if self.hand_idx == 0 or not self.single_arm else "open")
        #         with open(os.path.join(self.data_dir, f"gripper_right_state_{idx:03d}.txt"), "w") as f:
        #             f.write(gripper_state if self.hand_idx == 1 or not self.single_arm else "open")


        if object_id == "box":
            self.data_dir = "perception/planned_goals/pick/box/"
            base_pose = self.home_goal.copy().reshape(2, 4, 4)
            pose1 = grasp_pose[0]
            pose1[0] += 0.02
            pose1[1] -= 0.02

            self.use_left_hand = pose1[1] > 0  # y > 0 -> left hand, else right hand
            # print(use_left_hand)
            self.set_hand_config(self.use_left_hand)
            # self.check = False
            # Base transformation (keep the other hand unchanged)
            base_pose[self.hand_idx, :3, 3] = pose1[:3]
            base_pose[self.hand_idx] = base_pose[self.hand_idx]@ self.rot_x(pose1[5])
            pose2 = grasp_pose[1]
            pose2[0] += 0.02
            pose2[1] -= 0.02

            self.use_left_hand = pose2[1] > 0  # y > 0 -> left hand, else right hand
            # print(use_left_hand)
            self.set_hand_config(self.use_left_hand)
            # self.check = False
            # Base transformation (keep the other hand unchanged)
            base_pose[self.hand_idx, :3, 3] = pose2[:3]
            base_pose[self.hand_idx] = base_pose[self.hand_idx]@ self.rot_x(pose2[5])
            # x_offset = -0.02
            # base_pose[self.hand_idx, 0, 3]+= x_offset
            # for i in [0, 1]:
            #     angle_offset = 0.2 if i == 0 else -0.2
            #     R_offset    = self.rot_z(angle_offset)
            #     base_pose[i] = base_pose[i] @ R_offset  
            # for i in [0, 1]:
            #     base_pose[i] = base_pose[i] @ x_270
            # angle =  np.pi/2 if self.hand_idx == 0 else -np.pi/2
            # R     =  self.rot_x(angle)
            # base_pose[self.hand_idx] = base_pose[self.hand_idx] @ self.rot_x(object_location[5])
            # Generate 4 steps
            poses = []
            # Step 1: prepick (above object)
            prepick = base_pose.copy()
            prepick[0, 2, 3] += 0.2
            prepick[1, 2, 3] += 0.2
            poses.append(prepick)
            # Step 2: pick (at object)
            pick = base_pose.copy()
            poses.append(pick)

            # Step 3: gripper close (same as pick)
            close = pick.copy()
            poses.append(close)

            # Step 4: pickup (lift up)
            pickup = base_pose.copy()
            pickup[0, 2, 3] += 0.2
            pickup[1, 2, 3] += 0.2
            poses.append(pickup)

            # Save each step
            self.goal_types = []
            for i, pose in enumerate(poses):
                idx = i + 1
                goal_path = os.path.join(self.data_dir, f"robot_goal_{idx:03d}.npy")
                np.save(goal_path, pose.flatten())

                gripper_state = "open" if i < 2 else "close"
                goal_type = "soft" if i != 1 else "hard"  
                self.goal_types.append(goal_type)

                with open(os.path.join(self.data_dir, f"gripper_left_state_{idx:03d}.txt"), "w") as f:
                    f.write(gripper_state)
                with open(os.path.join(self.data_dir, f"gripper_right_state_{idx:03d}.txt"), "w") as f:
                    f.write(gripper_state)


    def precompute_trajectory(self):
        self.trajectory = []
        self.total_loaded = self.count_data_files()
        if self.plan:
            joint_goals = []
            gripper_left_states = []
            gripper_right_states = []
            for i in range(1, self.total_loaded + 1):
                idx_str = f"{i:03d}"
                joint_path   = os.path.join(self.data_dir, f"robot_goal_{idx_str}.npy")
                gripper_left_path = os.path.join(self.data_dir, f"gripper_left_state_{idx_str}.txt")
                gripper_right_path = os.path.join(self.data_dir, f"gripper_right_state_{idx_str}.txt")

                if not os.path.exists(gripper_left_path) or not os.path.exists(gripper_right_path):
                    rospy.logwarn("Missing gripper state files at %s", idx_str)
                    continue

                with open(gripper_left_path, "r") as f:
                    gripper_left_str = f.read().strip().lower()
                with open(gripper_right_path, "r") as f:
                    gripper_right_str = f.read().strip().lower()
                gripper_left_state = (gripper_left_str == "close")
                gripper_right_state = (gripper_right_str == "close")
                robot_goal = np.load(joint_path)
                gripper_left_states.append(gripper_left_state)
                gripper_right_states.append(gripper_right_state)
                joint_goals.append(robot_goal)

            if not joint_goals:
                rospy.logwarn("No joint goals found – empty trajectory.")
                return

            cart_goals = []
            if getattr(self, "prev_robot_goal", None) is not None:
                cart_goals.append(self.prev_robot_goal.copy())
                gripper_left_states.insert(0, self.prev_left_gripper_state)
                gripper_right_states.insert(0, self.prev_right_gripper_state)
            cart_goals.extend(joint_goals)
            self.plan_trajectory(joint_goals, self.prev_joint_goal, self.goal_types)

            self.trajectory = []
            wp_ptr = 0
            for q in self.full_traj:
                is_keypoint = wp_ptr < len(joint_goals) and np.allclose(q, self.keyjoints[wp_ptr])
                if is_keypoint:
                    current_left_gripper = gripper_left_states[wp_ptr]
                    current_right_gripper = gripper_right_states[wp_ptr]
                    previous_left_gripper = gripper_left_states[wp_ptr - 1] if wp_ptr > 0 else self.prev_left_gripper_state
                    previous_right_gripper = gripper_right_states[wp_ptr - 1] if wp_ptr > 0 else self.prev_right_gripper_state
                    pause = (current_left_gripper != previous_left_gripper or 
                            current_right_gripper != previous_right_gripper)
                    wp_ptr += 1
                else:
                    pause = False
                    current_left_gripper = gripper_left_states[wp_ptr - 1]
                    current_right_gripper = gripper_right_states[wp_ptr - 1]
                self.trajectory.append({
                    "joints": q[12:],
                    "gripper_left": current_left_gripper,
                    "gripper_right": current_right_gripper,
                    "pause": pause
                })
            rospy.loginfo(f"Precomputed {len(self.trajectory)} joint-space steps.")
            self.prev_robot_goal = robot_goal
            self.prev_left_gripper_state = current_left_gripper
            self.prev_right_gripper_state = current_right_gripper

        else:
            for i in range(1, self.total_loaded + 1):
                idx_str = f"{i:03d}"
                robot_goal_path = os.path.join(self.data_dir, f"robot_goal_{idx_str}.npy")
                gripper_left_path = os.path.join(self.data_dir, f"gripper_left_state_{idx_str}.txt")
                gripper_right_path = os.path.join(self.data_dir, f"gripper_right_state_{idx_str}.txt")

                if not os.path.exists(robot_goal_path) or not os.path.exists(gripper_left_path) or not os.path.exists(gripper_right_path):
                    rospy.logwarn("Skipping missing index: %s", idx_str)
                    continue

                robot_goal = np.load(robot_goal_path)
                with open(gripper_left_path, "r") as f:
                    gripper_left_str = f.read().strip().lower()
                with open(gripper_right_path, "r") as f:
                    gripper_right_str = f.read().strip().lower()
                current_left_gripper_state = (gripper_left_str == "close")
                current_right_gripper_state = (gripper_right_str == "close")

                pause = False
                if (self.prev_left_gripper_state is not None and
                    current_left_gripper_state != self.prev_left_gripper_state) or \
                   (self.prev_right_gripper_state is not None and
                    current_right_gripper_state != self.prev_right_gripper_state):
                    pause = True

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
                            "gripper_left": current_left_gripper_state,
                            "gripper_right": current_right_gripper_state,
                            "pause": False
                        })

                self.trajectory.append({
                    "point": robot_goal,
                    "gripper_left": current_left_gripper_state,
                    "gripper_right": current_right_gripper_state,
                    "pause": pause
                })

                self.prev_robot_goal = robot_goal
                self.prev_left_gripper_state = current_left_gripper_state
                self.prev_right_gripper_state = current_right_gripper_state

            rospy.loginfo("Precomputed %d trajectory steps.", len(self.trajectory))

    def packing_goals(self,packing_box_location):
        poses = []
        self.data_dir = "perception/planned_goals/pack/"
        base_pose = self.home_goal.copy().reshape(2, 4, 4)
        packing_box_location[2] += 0.1
        #Step 1: side1 (front box)
        side1 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.3 if i == 0 else -0.3
            side1[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side1[i] = side1[i] @ R_offset  
            side1[i, 0, 3] = packing_box_location[0]-0.15
            side1[i, 2, 3] = packing_box_location[2]
        poses.append(side1)
        #Step 2: side2 (middle box)
        side2 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.3 if i == 0 else -0.3
            side2[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side2[i] = side2[i] @ R_offset  
            side2[i, 0, 3] = packing_box_location[0]
            side2[i, 2, 3] = packing_box_location[2]
        poses.append(side2)
        #Step 3: side3 (above box)
        side3 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.3 if i == 0 else -0.3
            side3[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side3[i] = side3[i] @ R_offset  
            side3[i, 2, 3] = packing_box_location[2]+0.12
            side3[i, 0, 3] = packing_box_location[0]
        poses.append(side3)
        #Step 4: side4 (in box)
        side4 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.2 if i == 0 else -0.2
            side4[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side4[i] = side4[i] @ R_offset  
            side4[i, 2, 3] = packing_box_location[2] + 0.12
            side4[i, 0, 3] = packing_box_location[0]
        poses.append(side4)
        side4 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/4 if i == 0 else - np.pi/4
            y_offset = 0.2 if i == 0 else -0.2
            side4[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side4[i] = side4[i] @ R_offset  
            side4[i, 2, 3] = packing_box_location[2] + 0.12
            side4[i, 0, 3] = packing_box_location[0]
        poses.append(side4)
        side5 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = 0
            y_offset = 0.2 if i == 0 else -0.2
            side5[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side5[i] = side5[i] @ R_offset  
            side5[i, 2, 3] = packing_box_location[2] + 0.12
            side5[i, 0, 3] = packing_box_location[0]
        poses.append(side5)
        side6 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = 0
            y_offset = 0.2 if i == 0 else -0.2
            side6[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side6[i] = side6[i] @ R_offset  
            side6[i, 2, 3] = packing_box_location[2] + 0.08
            side6[i, 0, 3] = packing_box_location[0]
        poses.append(side6)
        side7 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = 0
            y_offset = 0.2 if i == 0 else -0.2
            side7[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side7[i] = side7[i] @ R_offset  
            side7[i, 2, 3] = packing_box_location[2] + 0.12
            side7[i, 0, 3] = packing_box_location[0]
        poses.append(side7)
        side7 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = 0
            y_offset = 0.2 if i == 0 else -0.2
            side7[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side7[i] = side7[i] @ R_offset  
            side7[i, 2, 3] = packing_box_location[2] + 0.12
            side7[i, 0, 3] = packing_box_location[0]
        poses.append(side7)
        side7 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = 0
            y_offset = 0.3 if i == 0 else -0.3
            side7[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side7[i] = side7[i] @ R_offset  
            side7[i, 2, 3] = packing_box_location[2] + 0.12
            side7[i, 0, 3] = packing_box_location[0]
        poses.append(side7)
        side7 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.3 if i == 0 else -0.3
            side7[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side7[i] = side7[i] @ R_offset  
            side7[i, 2, 3] = packing_box_location[2] +0.12
            side7[i, 0, 3] = packing_box_location[0] 
        poses.append(side7)
        side7 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.3 if i == 0 else -0.3
            side7[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side7[i] = side7[i] @ R_offset  
            side7[i, 2, 3] = packing_box_location[2]
            side7[i, 0, 3] = packing_box_location[0] 
        poses.append(side7)
        side7 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.25 if i == 0 else -0.25
            side7[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side7[i] = side7[i] @ R_offset  
            side7[i, 2, 3] = packing_box_location[2]
            side7[i, 0, 3] = packing_box_location[0] 
        poses.append(side7)
        side7 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.25 if i == 0 else -0.25
            x_offset = 0.1 if i == 0 else -0.1
            side7[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side7[i] = side7[i] @ R_offset  
            side7[i, 2, 3] = packing_box_location[2]
            side7[i, 0, 3] = packing_box_location[0] + x_offset
        poses.append(side7)
        side7 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.25 if i == 0 else -0.25
            x_offset = 0.2 if i == 0 else -0.2
            side7[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            side7[i] = side7[i] @ R_offset  
            side7[i, 2, 3] = packing_box_location[2]
            side7[i, 0, 3] = packing_box_location[0] + x_offset
        poses.append(side7)
        front1 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.3 if i == 0 else -0.3
            front1[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            front1[i] = front1[i] @ R_offset  
            front1[i, 0, 3] = packing_box_location[0]
            front1[i, 2, 3] = packing_box_location[2]
        poses.append(front1)
        front2 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.2 if i == 0 else -0.2
            front2[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            front2[i] = front2[i] @ R_offset  
            front2[i, 0, 3] = packing_box_location[0]
            front2[i, 2, 3] = packing_box_location[2]
        poses.append(front2)
        front3 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.2 if i == 0 else -0.2
            front3[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            front3[i] = front3[i] @ R_offset  
            front3[i, 0, 3] = packing_box_location[0]
            front3[i, 2, 3] = packing_box_location[2]+ 0.1
        poses.append(front3)
        front3 = base_pose.copy()
        for i in [0, 1]:
            angle_offset = np.pi/2 if i == 0 else - np.pi/2
            y_offset = 0.1 if i == 0 else -0.1
            front3[i, 1, 3] = packing_box_location[1] + y_offset
            R_offset    = self.rot_z(angle_offset)
            front3[i] = front3[i] @ R_offset  
            front3[i, 0, 3] = packing_box_location[0]
            front3[i, 2, 3] = packing_box_location[2]+ 0.1
        poses.append(front3)
        print(poses)
        self.goal_types = []
        # Save each step
        for i, pose in enumerate(poses):
            idx = i + 1
            goal_path = os.path.join(self.data_dir, f"robot_goal_{idx:03d}.npy")
            np.save(goal_path, pose.flatten())
            # Gripper state: open for first two, closed after
            goal_type = "soft"
            self.goal_types.append(goal_type)
            gripper_state = "close" if i == 3 else "close"
            with open(os.path.join(self.data_dir, f"gripper_left_state_{idx:03d}.txt"), "w") as f:
                f.write(gripper_state if self.hand_idx == 0 or not self.single_arm else "open")

            with open(os.path.join(self.data_dir, f"gripper_right_state_{idx:03d}.txt"), "w") as f:
                f.write(gripper_state if self.hand_idx == 1 or not self.single_arm else "open")



    def pack(self):
        self.check = False
        self.data_dir = "perception/planned_goals/pack/"
        self.single_arm = False
        self.precompute_trajectory()
        self.publish_trajectory()
        return
    def start_upper_body(self):
        self.check = False
        self.data_dir = "perception/planned_goals/reset/"
        self.single_arm = False
        self.precompute_trajectory()
        self.home_goal = self.prev_robot_goal.copy()
        if self.prev_joint_goal is not None:
            self.joint_home_goal = self.prev_joint_goal.copy()
        self.publish_trajectory()
        return
    
    def reset(self):
        if self.plan:
            self.left_gripper_pub.publish(False)
            self.right_gripper_pub.publish(False)
            self.planning_pub.publish(Float64MultiArray(data=(self.joint_home_goal[12:].tolist())))
            self.prev_joint_goal = self.joint_home_goal.copy()
            self.prev_robot_goal = self.home_goal.copy()
            self.prev_left_gripper_state = False
            self.prev_right_gripper_state = False
    
    
    def stir(self):
        self.single_arm = False
        # self.object_id = object_id
        self.check =    False
        self.data_dir = "perception/planned_goals/stir/"
        prepare_empty_folder(self.data_dir)
        self.precompute_trajectory()
        self.publish_trajectory()
    def pick(self,object_id):
        print(self.prev_joint_goal)
        if object_id == "bowl":
            self.single_arm = True
            self.object_id = object_id
            self.check =    True
            self.data_dir = "perception/planned_goals/pick/bowl/"
            prepare_empty_folder(self.data_dir)
            self.precompute_trajectory()
            self.publish_trajectory()
            self.save_place_goal()
        elif object_id in ["tennis", "can", "wooden-block", "cone", "lolipop","candy","cloth","bread"]:
            self.single_arm = True
            self.object_id = object_id
            self.check = True
            self.data_dir = "perception/planned_goals/pick/object/"
            prepare_empty_folder(self.data_dir)
            self.precompute_trajectory()
            self.publish_trajectory()
            self.save_place_goal()
        elif object_id == "box":
            self.object_id = object_id
            self.check = False
            self.single_arm = False
            self.data_dir = "perception/planned_goals/pick/box/"
            prepare_empty_folder(self.data_dir)
            self.precompute_trajectory()
            self.publish_trajectory()
            self.save_place_goal()
        # elif object_id == "cloth":
        #     self.object_id = object_id
        #     self.check = True
        #     self.data_dir = "perception/planned_goals/pick/cloth/"
        #     self.precompute_trajectory()
        #     self.publish_trajectory()
        #     self.save_place_goal()
        elif object_id == "connected":
            self.object_id = object_id
            self.check = True
            self.data_dir = "perception/planned_goals/pick/candy/"
            prepare_empty_folder(self.data_dir)
            self.precompute_trajectory()
            self.publish_trajectory()
            self.save_place_goal()
        print(self.prev_joint_goal)
        return
    
    def place(self):
        print(self.prev_joint_goal)
        self.check = False
        self.data_dir = "perception/planned_goals/place/"
        prepare_empty_folder(self.data_dir)
        self.precompute_trajectory()
        self.publish_trajectory()
        print(self.prev_joint_goal)
        return

    def home(self):
        self.single_arm = False
        self.check = False
        self.data_dir = "perception/planned_goals/home/"
        prepare_empty_folder(self.data_dir)
        self.precompute_trajectory()
        self.publish_trajectory()
        return
    def publish_trajectory(self):
        if self.plan:
            for i, step in enumerate(self.trajectory):
                self.left_gripper_pub.publish(Bool(data=step["gripper_left"]))
                self.right_gripper_pub.publish(Bool(data=step["gripper_right"]))
                if step["pause"]:
                    rospy.loginfo("Gripper state changed at step %d, pausing...", i)
                    rospy.sleep(3)
                # print(Bool(data=step["gripper_left"]))
                # print(Float64MultiArray(data=step["joints"].tolist()))
                self.planning_pub.publish(Float64MultiArray(data=step["joints"].tolist()))
                rospy.sleep(0.1)
            if self.check and not self.perform_grasp_check():
                rospy.logwarn("Grasp check failed, stopping.")
                raise RuntimeError("Grasp check failed") 
            else:
                rospy.loginfo("Grasp check passed, continuing.")
            rospy.loginfo("Publishing trajectory...")
        else:
            for i, step in enumerate(self.trajectory):
                if self.single_arm:
                    if self.hand_idx == 0:
                        self.left_gripper_pub.publish(Bool(data=step["gripper_left"]))
                    else:
                        self.right_gripper_pub.publish(Bool(data=step["gripper_right"]))
                else:
                    self.left_gripper_pub.publish(Bool(data=step["gripper_left"]))
                    self.right_gripper_pub.publish(Bool(data=step["gripper_right"]))
                if step["pause"]:
                    rospy.loginfo("Gripper state changed at step %d, pausing...", i)
                    rospy.sleep(3)
                self.robot_goal_pub.publish(Float64MultiArray(data=step["point"].tolist()))
                rospy.sleep(0.1)

            if self.check and not self.perform_grasp_check():
                rospy.logwarn("Grasp check failed, stopping.")
                raise RuntimeError("Grasp check failed") 
            else:
                rospy.loginfo("Grasp check passed, continuing.")
            rospy.loginfo("Publishing trajectory...")

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
