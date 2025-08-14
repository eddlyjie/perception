from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tkinter as tk
import pyrealsense2 as rs
import open3d as o3d
import copy
from scipy.spatial import cKDTree
import icp
from yolo_6d_pose import Yolo6DPose
from typing import Tuple

import math

def select_first_valid_x(grasp_list, x_threshold=0.23, offset=0.03):
    """
    Select the first grasp point where x > x_threshold and x < 0.32,
    then offset it 0.03 meters in the opposite direction of yaw.

    Args:
        grasp_list: List of elements like [[x, y, z], yaw].
        x_threshold: Minimum allowed x value.
        offset: Distance to shift opposite to yaw direction.

    Returns:
        A tuple (adjusted_pos, yaw):
            - adjusted_pos: [x, y, z] after applying the offset.
            - yaw: float
        Or (None, None) if no valid point is found.
    """
    for pos, yaw in grasp_list:
        x, y, z = pos
        if x > x_threshold and x < 0.35:
            # Shift position 0.03 meters opposite to yaw direction
            dx = -offset * math.cos(yaw)
            dy = -offset * math.sin(yaw)
            adjusted_pos = [x + dx, y + dy, z]
            return adjusted_pos, yaw
    return None, None


def cam_frame_to_base_frame(object_pose_in_camera):
    # input: 4x4 matrix in camera frame or 1x3 vector pos
    pos_only = False
    if object_pose_in_camera.shape == (3,):
        pos_only = True
        translation = object_pose_in_camera
        object_pose_in_camera = np.eye(4)
        object_pose_in_camera[:3, 3] = translation
    # hard coded transformation
    T = np.array([[0, -0.743, 0.669, 0.047],
                  [-1, 0, 0, 0.055],
                  [0, -0.669, -0.743, 0.46],
                  [0, 0, 0, 1]])
    object_pos_in_base_hom = T @ object_pose_in_camera
    if pos_only:
        object_pos_in_base_hom[:3, :3] = np.eye(3)
    return object_pos_in_base_hom


class Realsense():
    def __init__(self, depth_scale = 0.001):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # Configure streams: RGB and depth
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.device = self.profile.get_device()
        self.depth_sensor = self.device.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        self.depth_stream = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_intrinsics = self.depth_stream.get_intrinsics()
        color_intrinsics = self.color_stream.get_intrinsics()
        downscale = 1
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.K = np.array([[color_intrinsics.fx * downscale, 0, color_intrinsics.ppx * downscale],
                           [0, color_intrinsics.fy * downscale, color_intrinsics.ppy * downscale],
                           [0, 0, 1]])

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image = (depth_image.astype(np.float32) * self.depth_scale / 0.001).astype(int) #convert to mm if it is in cm
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

    def stop(self):
        self.pipeline.stop()


def draw_all_objects(mesh_transform_list):
    """
    Draw all meshes with their transformations and the base coordinate frame.
    """
    geometries = []
    colors = plt.cm.get_cmap("tab10", len(mesh_transform_list))
    i = 0
    for transformation, mesh in mesh_transform_list:
        if mesh is None:
            mesh_copy = o3d.geometry.TriangleMesh.create_box(width=100, height=100, depth=10)
        else:
            mesh_copy = copy.deepcopy(mesh)
        bbox = mesh_copy.get_axis_aligned_bounding_box()
        x_extent = bbox.get_extent()[0]
        if x_extent < 1.0:
            mesh_copy.scale(1000, center = [0,0,0])
            print("[INFO] Mesh scaled from m to mm.")
        transformation_mm = transformation.copy()
        transformation_mm[:3, 3] = transformation[:3, 3] * 1000  # Convert from m to mm
        mesh_copy.transform(transformation_mm)
        
        rgb = colors(i)[:3]
        mesh_copy.paint_uniform_color(rgb)
        mesh_copy.compute_vertex_normals()
        geometries.append(mesh_copy)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150)
        frame.transform(transformation_mm)
        geometries.append(frame)
        i += 1
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    
    geometries.append(base_frame)
    o3d.visualization.draw_geometries(geometries,
                                      window_name="All Objects",
                                      width=1024,
                                      height=768,
                                      zoom=0.8,
                                      front=[-0.5, -0.5, -0.5],
                                      lookat=[0, 0, 0],
                                      up=[0, 0, 1])


def filter_box_pointcloud(box_pointcloud, object_pointcloud_combined, threshold=0.1):
    """
    Remove points from box_pointcloud that are within `threshold` distance 
    to any point in object_pointcloud_combined.
    """
    box_tree = cKDTree(box_pointcloud)
    obj_tree = cKDTree(object_pointcloud_combined)
    pairs = obj_tree.query_ball_tree(box_tree, r=threshold)
    box_indices_to_remove = set(i for sublist in pairs for i in sublist)
    mask = np.ones(len(box_pointcloud), dtype=bool)
    mask[list(box_indices_to_remove)] = False
    return box_pointcloud[mask]


def get_object_and_box_position(object_perception, box_perception, realsense,
                                object_item_yolo_name, box_item_yolo_name,
                                object_mesh, box_mesh):
    color_image, depth_image = realsense.get_frames()
    pointcloud = object_perception.get_pointcloud(np.ones_like(depth_image), depth_image)
    object_poses, object_pointcloud = object_perception.get_6d_pose(
        color_image, depth_image, object_mesh, object_item_yolo_name, visualize=True)
    # print(object_poses)
    if object_pointcloud is None or len(object_pointcloud) == 0:

        print("[WARN] No matching object found.")
        return None
    contours, confs = box_perception.yolo_seg(color_image, box_item_yolo_name, visualize=False)

    if not contours:
        raise ValueError("No contours detected.")
    areas = [cv2.contourArea(c) for c in contours]
    max_idx = np.argmax(areas)
    contours = [contours[max_idx]]
    confs = confs[max_idx]
    if contours is None or len(contours) == 0:
        print("[WARN] No matching box found.")
        return None
    poses = []
    grouped_pointclouds = []
    mask = box_perception.contour_to_mask(contours[0])
    box_pointcloud = box_perception.get_pointcloud(mask, depth_image)
    if box_pointcloud is None or len(box_pointcloud) == 0:
        print("[WARN] No valid depth points in mask.")
        return None
    object_pointcloud_combined = np.vstack(object_pointcloud)
    filtered_box_pointcloud = filter_box_pointcloud(box_pointcloud, object_pointcloud_combined, threshold=0.1)
    if len(filtered_box_pointcloud) == 0:
        print("[WARN] No valid depth points in mask.")
        return None
    box_pose = box_perception.get_6d_pose_from_pointcloud(filtered_box_pointcloud, box_mesh, visualize=False)
    pose_mesh_pairs = [[cam_frame_to_base_frame(pose), object_mesh] for pose in object_poses]
    # Use the provided box_mesh instead of a stored mesh.
    box_pose_mesh_pairs = [[cam_frame_to_base_frame(box_pose), box_mesh]]
    pose_mesh_pairs.extend(box_pose_mesh_pairs)
    draw_all_objects(pose_mesh_pairs)
    return box_pose, object_poses

def demo_soft_object(object_perception, object_yolo_name, realsense, merge_contours = True, vis = True):
    color_image, depth_image = realsense.get_frames()

    contours = object_perception.yolo_seg(color_image, object_yolo_name, visualize=vis)
    if len(contours) == 0:
        return None
    edges = object_perception.find_edge(color_image, contours[0], merge_contours=merge_contours)
    grasp_points = object_perception.find_edge_grasp_point(color_image, contours[0], merge_contours=merge_contours)
    grasp_points_3d = object_perception.find_edge_grasp_point_3d(color_image, depth_image,  contours[0], merge_contours = merge_contours,transformation=object_perception.T)
    if vis == True:
        for _, pt1, pt2 in edges:
            cv2.line(color_image, pt1, pt2, (0, 0, 255), 2)

        for pt, direction in grasp_points:
            pt = np.array(pt, dtype=int)
            arrow_end = (pt + 50 * np.array([np.cos(direction), np.sin(direction)])).astype(int)
            cv2.arrowedLine(color_image, tuple(pt), tuple(arrow_end), (0, 255, 0), 2, tipLength=0.3)
        for pt3d, direction in grasp_points_3d:
            x, y, z = pt3d
            text = f"({x:.2f}, {y:.2f}, {z:.2f}) m"
            idx = grasp_points_3d.index([pt3d, direction])
            pt2d = np.array(grasp_points[idx][0], dtype=int)
            
            cv2.putText(color_image, text, tuple(pt2d + np.array([5, -10])), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        cv2.imshow("Top 10 Longest Clean Edges", color_image)
        cv2.waitKey(0)
        return grasp_points_3d

    else: 
        return grasp_points_3d


if __name__ == "__main__":
    K = np.load("perception/camera_intrinsics.npy")
    # Minimal changes in the main block:
    Demos = ["box_bowl", "candy", "box_box", "box_block", "lolipop", "box_ocean_ball", "cloth"]
    Demo = "box_cone"
    depth_scale = 0.01 #0.01 for realsense d405, 0.001 for d435
    realsense = Realsense(depth_scale= depth_scale)
    box_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_picking_box4.pt"), init_transformation=["x_180"])
    if Demo == "box_cone":
        # Remove the extra parameters from constructor.
        bowl_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_wbcd18.pt"))
        for i in range(50):
            realsense.get_frames()
        while True:
            import time
            start = time.time()
            print(get_object_and_box_position(bowl_perception, box_perception, realsense,
                                        object_item_yolo_name="can",   # passed in at call time
                                        box_item_yolo_name="box",
                                        object_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/can.STL"),
                                        box_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/picking_box.STL")))
            
            end = time.time()
            print("Time taken: ", end - start)

    if Demo == "box_box":
        # Remove the extra parameters from constructor.
        
        object_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_wbcd16.pt"))
        for i in range(50):
            realsense.get_frames()
        while True:
            result = get_object_and_box_position(object_perception, box_perception, realsense,
                                        object_item_yolo_name="box",   # passed in at call time
                                        box_item_yolo_name="box",
                                        object_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/box.STL"),
                                        box_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/picking_box.STL"))
            if result is None:
                continue
            box_pose, object_poses = result
            print(object_poses[0])
    if Demo == "box_block":
        # Remove the extra parameters from constructor.
        object_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_cube.pt"))
        for i in range(50):
            realsense.get_frames()
        while True:
            result = get_object_and_box_position(object_perception, box_perception, realsense,
                                                 object_item_yolo_name="cube",   # passed in at call time
                                                 box_item_yolo_name="box",
                                        object_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/wooden-block.STL"),
                                        box_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/picking_box.STL"))
            if result is None:
                continue
            box_pose, object_poses = result
            print(object_poses[0])
    if Demo == "box_ocean_ball":
        # Remove the extra parameters from constructor.
        object_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_wbcd9.pt"))
        while True:
            result = get_object_and_box_position(object_perception, box_perception, realsense,
                                        object_item_yolo_name="ocean-ball",   # passed in at call time
                                        box_item_yolo_name="box",
                                        object_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/ocean_ball.STL"),
                                        box_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/picking_box.STL"))
            if result is None:
                continue
            box_pose, object_poses = result
            print(object_poses[0])
    if Demo == "candy":
        candy_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_candy2.pt"))
        while True:
            for i in range(50):
                realsense.get_frames()
            while( True):
                demo_soft_object(candy_perception, "candy", realsense, merge_contours= True)
            color_image, depth_image = realsense.get_frames()
            object_poses_base_frame = []
            contours = candy_perception.yolo_seg(color_image, "candy", visualize=True)
            if len(contours) == 0:
                print("[WARN] No matching object found.")
                continue
            edges = candy_perception.find_edge(color_image, contours[0])
            grasp_points = candy_perception.find_edge_grasp_point(color_image, contours[0])
            grasp_points_3d = candy_perception.find_edge_grasp_point_3d(color_image, depth_image, contours[0], transformation=candy_perception.T)
            box_pose = box_perception.get_6d_pose(color_image, depth_image,
                o3d.io.read_triangle_mesh("perception/wbcd_meshes/picking_box.STL"),
                "box", visualize=True)
            for _, pt1, pt2 in edges:
                cv2.line(color_image, pt1, pt2, (0, 0, 255), 2)

            for pt, direction in grasp_points:
                pt = np.array(pt, dtype=int)
                arrow_end = (pt + 50 * np.array([np.cos(direction), np.sin(direction)])).astype(int)
                cv2.arrowedLine(color_image, tuple(pt), tuple(arrow_end), (0, 255, 0), 2, tipLength=0.3)
            for pt3d, direction in grasp_points_3d:
                x, y, z = pt3d
                text = f"({x:.2f}, {y:.2f}, {z:.2f}) m"
                idx = grasp_points_3d.index([pt3d, direction])
                pt2d = np.array(grasp_points[idx][0], dtype=int)

                cv2.putText(color_image, text, tuple(pt2d + np.array([5, -10])), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

            cv2.imshow("Top 10 Longest Clean Edges", color_image)
            cv2.waitKey(0)
            pass
    if Demo == "cloth":
        cloth_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_cloth.pt"))
        for i in range(50):
            realsense.get_frames()
        while( True):
            grasp_points_3d = demo_soft_object(cloth_perception, "cloth", realsense, merge_contours= True)
            print(select_first_valid_x(grasp_points_3d, x_threshold=0.23))
    if Demo == "lolipop":
        for i in range(50):
            color_image,depth_image = realsense.get_frames()
        object_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_lolipop7.pt"))
        while True:
            try:
               while True:
                    box_pose,objpose = get_object_and_box_position(object_perception, box_perception, realsense,
                                            object_item_yolo_name="lollipop",   # passed in at call time
                                            box_item_yolo_name="box",
                                            object_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/lollipop.STL"),
                                            box_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/picking_box.STL"))
                    print(cam_frame_to_base_frame(np.array(box_pose)),cam_frame_to_base_frame(np.array(objpose)))
            except Exception as e:
                print(f"Error: {e}")
    if Demo == "packing_box":
        box_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_picking_box4.pt"))
        for i in range(50):
            color_image,depth_image = realsense.get_frames()
        while( True):
            color_image,depth_image = realsense.get_frames()
            contours, confs = box_perception.get_6d_pose(color_image, depth_image, o3d.io.read_triangle_mesh("perception/wbcd_meshes/picking_box.STL"),"box", visualize=True)
