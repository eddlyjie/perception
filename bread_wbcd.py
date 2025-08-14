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
import rospy
from geometry_msgs.msg import Point,PoseStamped
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import String

#for wbcd task

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
        color_image, depth_image, object_mesh, object_item_yolo_name, visualize=False, select=False)
    
    if object_poses is None or object_pointcloud is None:
        print("[WARN] No matching object found.")
        return None
    
    # Handle single pose case (when select=True)
    if not isinstance(object_poses, list):
        object_poses = [object_poses]
        object_pointcloud = [object_pointcloud]
    
    contours, confs = box_perception.yolo_seg(color_image, box_item_yolo_name, visualize=False)
    if contours is None or len(contours) == 0:
        print("[WARN] No matching object found.")
        return None
    
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

    # draw camera view
    # pose_mesh_pairs_ = [[pose, object_mesh] for pose in object_poses]
    # box_pose_mesh_pairs_ = [[box_pose, box_mesh]]
    # pose_mesh_pairs_.extend(box_pose_mesh_pairs_)
    # draw_all_objects(pose_mesh_pairs_)

    # Prepare pose-mesh pairs
    pose_mesh_pairs = [[cam_frame_to_base_frame(pose), object_mesh] for pose in object_poses]
    box_pose_mesh_pairs = [[cam_frame_to_base_frame(box_pose), box_mesh]]
    pose_mesh_pairs.extend(box_pose_mesh_pairs)
    
    # draw_all_objects(pose_mesh_pairs)
    
    # print("box pose: ", box_pose)
    # print("object poses: ", object_poses)
    # return box_pose, object_poses

    # box_base_pose = cam_frame_to_base_frame(box_pose)
    # print("base box pose: ", box_base_pose)
    object_base_poses = [cam_frame_to_base_frame(pose) for pose in object_poses]
    print("num of objects: ", len(object_base_poses))
    print("base object poses: ", object_base_poses)
    return object_base_poses

def demo_soft_object(object_perception, object_yolo_name, realsense, merge_contours = True, vis =  True):
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

    else: 
        return grasp_points_3d
    
def demo_cloth_corner(object_perception, object_yolo_name, realsense, merge_contours=True, vis=True):
    color_image, depth_image = realsense.get_frames()

    contours = object_perception.yolo_seg(color_image, object_yolo_name, visualize=vis)
    if len(contours) == 0:
        return None

    # Find the convex hull of the contour to get the garment shape
    merged_contour = contours[0]
    if len(merged_contour) == 0:
        return None
    
    # Approximate the contour to a polygon to find corners
    epsilon = 0.02 * cv2.arcLength(merged_contour[0], True)
    approx = cv2.approxPolyDP(merged_contour[0], epsilon, True)
    approx = approx.squeeze()
    
    # If we have less than 4 points, use the bounding rectangle corners
    if len(approx) < 4:
        rect = cv2.minAreaRect(merged_contour[0])
        box = cv2.boxPoints(rect)
        approx = box.astype(np.int32)
    
    # Take the 4 farthest points as corners (for cases where we have more than 4 points)
    if len(approx) > 4:
        # Convert to numpy array
        approx = np.array([point for point in approx])
        
        # Calculate centroid
        centroid = np.mean(approx, axis=0)
        
        # Find the 4 points farthest from the centroid
        distances = [np.linalg.norm(point - centroid) for point in approx]
        farthest_indices = np.argsort(distances)[-4:]
        approx = approx[farthest_indices]
    
    # Now we should have exactly 4 points
    corners_2d = approx
    
    centroid = np.mean(merged_contour[0].squeeze(), axis=0)  # 轮廓中心

    yaw_angles = []
    for pt in corners_2d:
        direction = centroid - pt
        direction = direction / np.linalg.norm(direction)
        yaw = np.arctan2(direction[1], direction[0])
        yaw_angles.append(yaw)
    
    # Calculate 3D positions for each corner
    corners_3d = []
    mask = np.zeros(depth_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, merged_contour, -1, 255, cv2.FILLED)
    
    for corner, yaw in zip(corners_2d, yaw_angles):
        x, y = corner
        window_size = 5  # Pixels around the corner
        x_min = max(x - window_size, 0)
        x_max = min(x + window_size, depth_image.shape[1])
        y_min = max(y - window_size, 0)
        y_max = min(y + window_size, depth_image.shape[0])
        
        # Collect valid depth points in the neighborhood
        points = []
        for j in range(y_min, y_max):
            for i in range(x_min, x_max):
                if mask[j, i] == 0:
                    continue
                z = depth_image[j, i]
                if z == 0 or np.isnan(z):
                    continue
                # Backproject using intrinsics
                fx = object_perception.K[0, 0]
                fy = object_perception.K[1, 1]
                cx = object_perception.K[0, 2]
                cy = object_perception.K[1, 2]
                x3d = (i - cx) * z / fx
                y3d = (j - cy) * z / fy
                points.append([x3d, y3d, z])
        
        if not points:
            continue  # skip if no depth data
        
        avg_point = np.mean(points, axis=0) / 1000  # In camera frame
        avg_point_h = np.append(avg_point, 1.0)  # Homogeneous
        point_base = object_perception.T @ avg_point_h
        corners_3d.append([point_base[:3].tolist(), yaw])
    
    if vis:
        # Draw the corners and normals on the image
        for i, (corner, (pt_3d, yaw)) in enumerate(zip(corners_2d, corners_3d)):
            cv2.circle(color_image, tuple(corner), 5, (0, 0, 255), -1)
            arrow_end = (corner + 30 * np.array([np.cos(yaw), np.sin(yaw)])).astype(int)
            cv2.arrowedLine(color_image, tuple(corner), tuple(arrow_end), (0, 255, 0), 2, tipLength=0.3)
            cv2.putText(color_image, f"Corner {i+1}", (corner[0]+10, corner[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(color_image, f"yaw: {np.degrees(yaw):.1f}°", (corner[0]+10, corner[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw the convex hull
        cv2.drawContours(color_image, merged_contour, -1, (0, 255, 0), 2)
        
        cv2.imshow("Garment Corners Detection", color_image)
        cv2.waitKey(0)
        return corners_3d
    else:
        return corners_3d

    
def demo_cloth_edge(object_perception, object_yolo_name, realsense, merge_contours=True, vis=True):
    color_image, depth_image = realsense.get_frames()

    contours = object_perception.yolo_seg(color_image, object_yolo_name, visualize=vis)
    if len(contours) == 0:
        return None
    
    # Find the convex hull of the contour to get the garment shape
    merged_contour = contours[0]
    if len(merged_contour) == 0:
        return None
    
    # Approximate the contour to a polygon to find edges
    epsilon = 0.02 * cv2.arcLength(merged_contour[0], True)
    approx = cv2.approxPolyDP(merged_contour[0], epsilon, True)
    approx = approx.squeeze()
    
    # If we have less than 4 points, use the bounding rectangle
    if len(approx) < 4:
        rect = cv2.minAreaRect(merged_contour[0])
        box = cv2.boxPoints(rect)
        approx = box.astype(np.int32)
    
    # Ensure we have exactly 4 points (a quadrilateral)
    if len(approx) > 4:
        # Find the 4 points that form the largest quadrilateral
        from itertools import combinations
        max_area = 0
        best_quad = None
        for quad in combinations(approx, 4):
            area = cv2.contourArea(np.array(quad))
            if area > max_area:
                max_area = area
                best_quad = quad
        approx = np.array(best_quad)
    
    # Now we have exactly 4 points representing the garment corners
    # Sort points in clockwise order
    centroid = np.mean(approx, axis=0)
    angles = [np.arctan2(p[1]-centroid[1], p[0]-centroid[0]) for p in approx]
    approx = approx[np.argsort(angles)]
    
    # Calculate midpoints and edge directions for each side
    edge_midpoints = []
    edge_directions = []
    n = len(approx)
    for i in range(n):
        pt1 = approx[i]
        pt2 = approx[(i+1)%n]
        
        # Calculate midpoint
        midpoint = ((pt1[0] + pt2[0])/2, (pt1[1] + pt2[1])/2)
        
        # Calculate edge direction (normalized)
        edge_vec = np.array([pt2[0]-pt1[0], pt2[1]-pt1[1]])
        edge_vec = edge_vec / np.linalg.norm(edge_vec)
        
        # Calculate outward normal (rotate edge vector by 90 degrees)
        normal = np.array([-edge_vec[1], edge_vec[0]])
        
        # Verify normal direction (should point outward)
        test_point = midpoint + 10 * normal
        if cv2.pointPolygonTest(merged_contour[0], tuple(test_point), False) < 0:
            normal = -normal
        
        yaw = np.arctan2(normal[1], normal[0])
        
        edge_midpoints.append(midpoint)
        edge_directions.append(yaw)
    
    # Calculate 3D positions for each edge midpoint
    edge_points_3d = []
    mask = np.zeros(depth_image.shape, dtype=np.uint8)
    cv2.drawContours(mask, merged_contour, -1, 255, cv2.FILLED)
    
    for midpoint, yaw in zip(edge_midpoints, edge_directions):
        x, y = int(midpoint[0]), int(midpoint[1])
        window_size = 5  # Pixels around the midpoint
        
        # Get depth values in a small neighborhood
        points = []
        for j in range(max(y-window_size, 0), min(y+window_size+1, depth_image.shape[0])):
            for i in range(max(x-window_size, 0), min(x+window_size+1, depth_image.shape[1])):
                if mask[j, i] == 0:
                    continue
                z = depth_image[j, i]
                if z == 0 or np.isnan(z):
                    continue
                # Backproject using intrinsics
                fx = object_perception.K[0, 0]
                fy = object_perception.K[1, 1]
                cx = object_perception.K[0, 2]
                cy = object_perception.K[1, 2]
                x3d = (i - cx) * z / fx
                y3d = (j - cy) * z / fy
                points.append([x3d, y3d, z])
        
        if not points:
            continue  # skip if no depth data
        
        avg_point = np.mean(points, axis=0) / 1000  # In camera frame (meters)
        avg_point_h = np.append(avg_point, 1.0)  # Homogeneous
        point_base = object_perception.T @ avg_point_h
        edge_points_3d.append([point_base[:3].tolist(), yaw])
    
    if vis:
        # Draw the edges and midpoints on the image
        for i in range(len(approx)):
            pt1 = tuple(approx[i])
            pt2 = tuple(approx[(i+1)%n])
            midpoint = edge_midpoints[i]
            yaw = edge_directions[i]
            
            # Draw edge
            cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)
            
            # Draw midpoint and normal direction
            cv2.circle(color_image, (int(midpoint[0]), int(midpoint[1])), 5, (0, 0, 255), -1)
            arrow_end = (int(midpoint[0] + 30 * np.cos(yaw)), 
                         int(midpoint[1] + 30 * np.sin(yaw)))
            cv2.arrowedLine(color_image, 
                           (int(midpoint[0]), int(midpoint[1])),
                           arrow_end,
                           (255, 0, 0), 2, tipLength=0.3)
            
            # Add text labels
            cv2.putText(color_image, f"Edge {i+1}", 
                        (int(midpoint[0])+10, int(midpoint[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(color_image, f"{np.degrees(yaw):.1f}°", 
                        (int(midpoint[0])+10, int(midpoint[1])+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        cv2.imshow("Garment Edge Midpoints", color_image)
        cv2.waitKey(0)
        return None
    else:
        return edge_points_3d

    
# ellipse-fitting
def demo_bread_ef(object_perception, object_yolo_name, realsense, transformation = np.eye(4), merge_contours=True, vis=True):
    """
    Returns:
      List of [ [x, y, z], yaw ] in base frame for each bread ellipse midpoint.
    """
    # 1. Capture
    color_image, depth_image = realsense.get_frames()
    K = realsense.K  # 3×3 intrinsics

    # 2. YOLO + Mask segmentation
    seg_output = object_perception.yolo_seg(color_image,
                                            object_yolo_name,
                                            visualize=vis)
    try:
        contours, mask = seg_output
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim == 1:
            H, W = depth_image.shape[:2]
            if mask.size == H*W:
                mask = mask.reshape(H, W)
            else:
                # unexpected shape, disable mask-filtering
                mask = None
    except Exception:
        contours = seg_output
        mask = None

    if not contours:
        return []

    grasp_2d = []
    # 3. Fit ellipses & extract 2D center + yaw
    for cnt in contours:
        # ensure format Nx2
        cnt_np = np.asarray(cnt, dtype=np.int32)
        if cnt_np.ndim == 1 and cnt_np.size % 2 == 0:
            cnt_np = cnt_np.reshape(-1, 2)
        elif cnt_np.ndim == 3 and cnt_np.shape[1] == 1:
            cnt_np = cnt_np.reshape(-1, 2)
        if cnt_np.shape[0] < 5:
            continue

        ellipse = cv2.fitEllipse(cnt_np)
        center_px = np.array(ellipse[0], dtype=int)  # (u, v)
        ang = ellipse[2]
        # rotate by ±90° so major axis points along “bread length”
        ang = ang + 90 if ang < 90 else ang - 90
        yaw = np.deg2rad(ang)

        grasp_2d.append((center_px, yaw))

        if vis:
            cv2.ellipse(color_image, ellipse, (128,255,255), 1)
            cv2.circle(color_image, tuple(center_px), 3, (0,0,255), -1)
            # draw major axis
            rad = np.deg2rad(ang)
            half = max(ellipse[1]) / 2
            d = np.array([np.cos(rad), np.sin(rad)]) * half
            p0 = (center_px - d).astype(int)
            p1 = (center_px + d).astype(int)
            cv2.line(color_image, tuple(p0), tuple(p1), (0,255,0), 2)

    # 4. Backproject & transform to base
    grasp_3d = []
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    for (mid_px, yaw) in grasp_2d:
        u, v = int(mid_px[0]), int(mid_px[1])
        # sample window
        W = 5
        u_min, u_max = max(u-W,0), min(u+W, depth_image.shape[1]-1)
        v_min, v_max = max(v-W,0), min(v+W, depth_image.shape[0]-1)

        pts = []
        for vv in range(v_min, v_max+1):
            for uu in range(u_min, u_max+1):
                if mask is not None and mask[vv,uu] == 0:
                    continue
                z = depth_image[vv,uu]
                if z == 0 or np.isnan(z):
                    continue
                # backproject (z in mm)
                x3 = (uu - cx) * z / fx
                y3 = (vv - cy) * z / fy
                pts.append([x3, y3, z])

        if not pts:
            continue

        avg = np.mean(pts, axis=0) / 1000.0          # → meters in cam frame
        hom = np.append(avg, 1.0)                    # homogeneous
        base_pt = (transformation @ hom)[:3].tolist()
        grasp_3d.append([base_pt, yaw])
    print(grasp_3d)
    # 5. Visualize & return
    if vis:
        cv2.imshow("Bread Grasps", color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return grasp_3d

def demo_bread(object_perception, object_yolo_name, realsense, transformation=np.eye(4), merge_contours=True, vis=True):
    """
    Returns:
    List of [ [x, y, z], yaw ] in base frame for each bread object.
    Uses PCA to determine the longest axis of the object.
    Visualizes all bread objects in a single window.
    """
    # 1. Capture frames
    color_image, depth_image = realsense.get_frames()
    K = realsense.K  # 3×3 intrinsics
    
    # 2. YOLO + Mask segmentation
    seg_output = object_perception.yolo_seg(color_image, object_yolo_name, visualize=False)
    
    if seg_output is None or len(seg_output) == 0:
        print("No objects detected")
        return []
    
    contours, confs = seg_output
    
    if not contours:
        return []
    
    grasp_3d = []
    
    # Create a single visualization image
    vis_img = color_image.copy()
    
    # Process each contour
    for contour_idx, contour in enumerate(contours):
        # Convert contour to proper format
        cnt_np = np.asarray(contour, dtype=np.int32)
        if cnt_np.ndim == 1 and cnt_np.size % 2 == 0:
            cnt_np = cnt_np.reshape(-1, 2)
        elif cnt_np.ndim == 3 and cnt_np.shape[1] == 1:
            cnt_np = cnt_np.reshape(-1, 2)
            
        if cnt_np.shape[0] < 5:
            continue
            
        # Create mask from contour
        mask = np.zeros(depth_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt_np], 0, 255, -1)
        
        # Generate 3D point cloud from mask and depth
        pointcloud = object_perception.get_pointcloud(mask, depth_image)
        
        if pointcloud is None or len(pointcloud) < 10:
            continue
            
        # Create Open3D point cloud for visualization
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pointcloud)
        
        # Compute PCA to find principal axes
        try:
            # Compute covariance matrix
            points_mean = np.mean(pointcloud, axis=0)
            points_centered = pointcloud - points_mean
            cov = np.cov(points_centered.T)
            
            # Calculate eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            # Sort by eigenvalue in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # The first eigenvector is the direction of the longest axis
            major_axis = eigenvectors[:, 0]
            
            # Convert to meters and get center point
            center_point = points_mean / 1000.0  # mm to meters
            
            # Calculate yaw from the major axis direction
            # Project to XY plane for yaw calculation
            axis_xy = np.array([major_axis[0], major_axis[1], 0])
            if np.linalg.norm(axis_xy) > 1e-6:  # Ensure it's not entirely vertical
                axis_xy = axis_xy / np.linalg.norm(axis_xy)
                yaw = np.arctan2(axis_xy[1], axis_xy[0])
            else:
                # Default orientation if the axis is vertical
                yaw = 0.0
                
            # Convert center point to base frame
            center_hom = np.append(center_point, 1.0)  # homogeneous coordinates
            T = np.array([[0, -0.743, 0.669, 0.047],
                          [-1, 0, 0, 0.055],
                          [0, -0.669, -0.743, 0.46],
                          [0, 0, 0, 1]])
            base_center = (T @ center_hom)[:3].tolist()
            
            grasp_3d.append([base_center, yaw])
            
            # Visualization (all on one image)
            if vis:
                # Get color for this bread instance (different for each instance)
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                          (0, 255, 255), (255, 0, 255)]
                color = colors[contour_idx % len(colors)]
                
                # Draw the contour
                cv2.drawContours(vis_img, [cnt_np], 0, color, 2)
                
                # Draw center point
                center_px = object_perception.project_point_to_image(points_mean)
                cv2.circle(vis_img, tuple(center_px.astype(int)), 5, color, -1)
                
                # Draw major axis
                axis_length = np.sqrt(eigenvalues[0]) * 0.5  # Scale based on eigenvalue
                axis_endpoint1 = points_mean + major_axis * axis_length
                axis_endpoint2 = points_mean - major_axis * axis_length
                
                px1 = object_perception.project_point_to_image(axis_endpoint1)
                px2 = object_perception.project_point_to_image(axis_endpoint2)
                
                cv2.line(vis_img, tuple(px1.astype(int)), tuple(px2.astype(int)), color, 2)
                
                # Calculate text position
                text_pos_x = int(center_px[0]) + 10
                text_pos_y = int(center_px[1]) - 10
                
                # Add bread ID
                cv2.putText(vis_img, f"Bread {contour_idx+1}", (text_pos_x, text_pos_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                grasp_pos, grasp_yaw = grasp_3d[contour_idx]  # Extract grasp pose for current bread
                cv2.putText(vis_img, f"Pos: [{grasp_pos[0]:.2f}, {grasp_pos[1]:.2f}, {grasp_pos[2]:.2f}] m", 
                            (text_pos_x, text_pos_y + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(vis_img, f"Yaw: {np.degrees(grasp_yaw):.1f} deg", 
                            (text_pos_x, text_pos_y + 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw arrow indicating grasp direction
                arrow_length = 30
                arrow_start = center_px.astype(int)
                arrow_end = (center_px + np.array([np.cos(yaw), np.sin(yaw)]) * arrow_length).astype(int)
                cv2.arrowedLine(vis_img, tuple(arrow_start), tuple(arrow_end), color, 2)
        
        except Exception as e:
            print(f"Error in PCA calculation for bread {contour_idx+1}: {e}")
            continue

    # Show the visualization with all bread objects
    # if vis and grasp_3d:
    #     cv2.imshow("Bread Objects", vis_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    positions = [item[0] for item in grasp_3d]
    print(positions)
    return positions

def demo_objects(bread_perception, box_perception, 
                 bread_yolo_names, box_yolo_names, 
                 realsense, transformation=np.eye(4), 
                 merge_contours=False, vis=False):
    """
    Detect multiple object types from two YOLO models (e.g., bread + box),
    estimate grasp poses via PCA, and return positions + yaw in base frame.

    Args:
        bread_perception: Yolo6DPose instance for bread/other items
        box_perception: Yolo6DPose instance for picking box
        bread_yolo_names (list[str]): YOLO classes for bread_perception
        box_yolo_names (list[str]): YOLO classes for box_perception
        realsense: RGB-D camera wrapper
        transformation (4x4): camera-to-base extrinsic
        vis (bool): enable visualization
    """
    # 1. Get RGB-D frame
    color_image, depth_image = realsense.get_frames()

    all_contours = []
    all_labels = []

    # 2. Detect using bread_perception
    for obj_name in bread_yolo_names:
        seg_output = bread_perception.yolo_seg(color_image, obj_name, visualize=False)
        if seg_output is not None and len(seg_output) > 0:
            contours, _ = seg_output
            all_contours.extend(contours)
            all_labels.extend([obj_name] * len(contours))

    # 3. Detect using box_perception
    for obj_name in box_yolo_names:
        seg_output = box_perception.yolo_seg(color_image, obj_name, visualize=False)
        if seg_output is not None and len(seg_output) > 0:
            contours, _ = seg_output
            all_contours.extend(contours)
            all_labels.extend([obj_name] * len(contours))

    if not all_contours:
        print("No objects detected")
        return []

    grasp_3d = []
    vis_img = color_image.copy()

    # 4. Process each contour
    for idx, (contour, obj_label) in enumerate(zip(all_contours, all_labels)):
        cnt_np = np.asarray(contour, dtype=np.int32)
        if cnt_np.ndim == 1 and cnt_np.size % 2 == 0:
            cnt_np = cnt_np.reshape(-1, 2)
        elif cnt_np.ndim == 3 and cnt_np.shape[1] == 1:
            cnt_np = cnt_np.reshape(-1, 2)
        if cnt_np.shape[0] < 5:
            continue

        # Create mask
        mask = np.zeros(depth_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt_np], 0, 255, -1)

        # Pick correct perception for pointcloud
        if obj_label in bread_yolo_names:
            pointcloud = bread_perception.get_pointcloud(mask, depth_image)
            proj_func = bread_perception.project_point_to_image
        else:
            pointcloud = box_perception.get_pointcloud(mask, depth_image)
            proj_func = box_perception.project_point_to_image

        if pointcloud is None or len(pointcloud) < 10:
            continue

        try:
            # PCA for yaw
            points_mean = np.mean(pointcloud, axis=0)
            points_centered = pointcloud - points_mean
            cov = np.cov(points_centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            idx_sort = eigenvalues.argsort()[::-1]
            eigenvectors = eigenvectors[:, idx_sort]
            major_axis = eigenvectors[:, 0]

            axis_xy = np.array([major_axis[0], major_axis[1], 0])
            yaw = np.arctan2(axis_xy[1], axis_xy[0]) if np.linalg.norm(axis_xy) > 1e-6 else 0.0

            # Transform to base
            center_point = points_mean / 1000.0
            center_hom = np.append(center_point, 1.0)
            base_center = (transformation @ center_hom)[:3].tolist()

            grasp_3d.append([base_center, yaw, obj_label])

            # Visualization
            if vis:
                label_colors = {
                    "bread": (0, 255, 0),
                    "bowl": (255, 0, 0),
                    "plate": (0, 0, 255),
                    "box": (0, 255, 255)
                }
                color = label_colors.get(obj_label, (200, 200, 200))

                cv2.drawContours(vis_img, [cnt_np], 0, color, 2)
                center_px = proj_func(points_mean)
                cv2.circle(vis_img, tuple(center_px.astype(int)), 5, color, -1)

                axis_length = np.sqrt(eigenvalues[0]) * 0.5
                axis_endpoint1 = points_mean + major_axis * axis_length
                axis_endpoint2 = points_mean - major_axis * axis_length
                px1 = proj_func(axis_endpoint1)
                px2 = proj_func(axis_endpoint2)
                cv2.line(vis_img, tuple(px1.astype(int)), tuple(px2.astype(int)), color, 2)

                cv2.putText(vis_img, f"{obj_label.capitalize()} {idx+1}", 
                            (int(center_px[0])+10, int(center_px[1])-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(vis_img, f"Yaw: {np.degrees(yaw):.1f} deg", 
                            (int(center_px[0])+10, int(center_px[1])+10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(vis_img, f"Pos: [{base_center[0]:.2f}, {base_center[1]:.2f}, {base_center[2]:.2f}]", 
                            (int(center_px[0])+10, int(center_px[1])+30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        except Exception as e:
            print(f"Error processing object {idx+1} ({obj_label}): {e}")
            continue

    if vis and grasp_3d:
        cv2.imshow("Detected Objects", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(grasp_3d)
    return grasp_3d

def demo_lolipop_axe(object_perception, object_yolo_name, realsense, transformation = np.eye(4), merge_contours=True, vis=True):
    # 1) grab images + intrinsics
    color_image, depth_image = realsense.get_frames()
    K = realsense.K
    h, w = depth_image.shape

    # 2) YOLO + mask
    seg_output = object_perception.yolo_seg(color_image,
                                            object_yolo_name,
                                            visualize=vis)
    try:
        contours, mask = seg_output
        # ensure mask is H×W uint8 array
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim == 1:
            if mask.size == h*w:
                mask = mask.reshape(h, w)
            else:
                mask = None
    except Exception:
        contours, mask = seg_output, None

    if not contours:
        return []

    # helper: back‐project a pixel (u,v) + depth_mm → (x,y,z)_m
    def pixel_to_point(u, v, depth):
        z = depth * 0.001
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        return np.array([x, y, z])

    # helper: segment intersection test
    def segments_intersect(p1, p2, p3, p4):
        def orient(a,b,c):
            v = (b[1]-a[1])*(c[0]-b[0]) - (b[0]-a[0])*(c[1]-b[1])
            return 0 if v==0 else (1 if v>0 else 2)
        def on_seg(a,b,c):
            return min(a[0],c[0])<=b[0]<=max(a[0],c[0]) and \
                   min(a[1],c[1])<=b[1]<=max(a[1],c[1])
        o1 = orient(p1,p2,p3); o2 = orient(p1,p2,p4)
        o3 = orient(p3,p4,p1); o4 = orient(p3,p4,p2)
        if o1!=o2 and o3!=o4: return True
        if o1==0 and on_seg(p1,p3,p2): return True
        if o2==0 and on_seg(p1,p4,p2): return True
        if o3==0 and on_seg(p3,p1,p4): return True
        if o4==0 and on_seg(p3,p2,p4): return True
        return False

    # 3) build 2D candidates with midpoint & yaw
    candidates = []
    for cnt in contours:
        cnt = np.asarray(cnt, dtype=np.int32)
        if cnt.ndim==1 and cnt.size%2==0:
            cnt = cnt.reshape(-1,2)
        elif cnt.ndim==3 and cnt.shape[1]==1:
            cnt = cnt.reshape(-1,2)
        if cnt.shape[0] < 5:
            continue

        # min‐area box → major axis
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        center = np.array(rect[0], dtype=int)
        w_rect, h_rect = rect[1]
        ang = rect[2] + 90
        if w_rect < h_rect:
            major_len, angle_adj = h_rect, ang
        else:
            major_len, angle_adj = w_rect, ang + 90

        # yaw in image plane
        yaw = np.deg2rad(angle_adj)

        # endpoints in pixels
        rad = np.deg2rad(angle_adj)
        d = np.array([np.cos(rad), np.sin(rad)]) * (major_len/2)
        start = (center - d).astype(int)
        end   = (center + d).astype(int)

        # drop if outside
        u0,v0 = start; u1,v1 = end
        if not (0<=u0<w and 0<=v0<h and 0<=u1<w and 0<=v1<h):
            continue

        # center depth (for sorting)
        vc, uc = center[1], center[0]
        if 0<=uc<w and 0<=vc<h:
            depth_center = depth_image[vc,uc] * 0.001
        else:
            continue

        candidates.append({
            'start': tuple(start),
            'end':   tuple(end),
            'mid':   tuple(center),
            'yaw':   yaw,
            'depth': depth_center
        })

    if not candidates:
        return []

    # 4) pick nearest non‐intersecting
    candidates.sort(key=lambda c: c['depth'])
    selected = None
    for i, c in enumerate(candidates):
        p1, p2 = c['start'], c['end']
        if not any(segments_intersect(p1,p2, o['start'],o['end'])
                   for j,o in enumerate(candidates) if i!=j):
            selected = c
            break
    if selected is None:
        selected = candidates[0]

    # 5) sample around selected midpoint → 3D + transform
    mid_u, mid_v = selected['mid']
    W = 5
    pts3d = []
    for vv in range(max(0, mid_v-W), min(h, mid_v+W+1)):
        for uu in range(max(0, mid_u-W), min(w, mid_u+W+1)):
            if mask is not None and mask[vv,uu]==0:
                continue
            z = depth_image[vv,uu]
            if z==0 or np.isnan(z):
                continue
            pts3d.append(pixel_to_point(uu, vv, z))

    if not pts3d:
        return []

    avg_cam = np.mean(pts3d, axis=0)        # meters in camera frame
    avg_h   = np.hstack((avg_cam, 1.0))
    base_pt = (transformation @ avg_h)[:3]  # in base frame
    print([[base_pt.tolist(), selected['yaw']]])
    if vis:
        cv2.line(color_image,
                 selected['start'],
                 selected['end'],
                 (0, 255, 0), 2)
        midpt = ((selected['start'][0] + selected['end'][0])//2,
                 (selected['start'][1] + selected['end'][1])//2)
        cv2.circle(color_image, midpt, 5, (0, 0, 255), -1)
        cv2.imshow("Selected Lollipop Axis Only", color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # return a list of [ [x,y,z], yaw ]
    return [[base_pt.tolist(), selected['yaw']]]


if __name__ == "__main__":

    try:
        rospy.get_rostime()  # 测试ROS是否已经初始化
        print("ROS already initialized, skipping init_node")
    except:
        rospy.init_node('mcp_execution', anonymous=True)







    K = np.load("perception/camera_intrinsics.npy")
    # Minimal changes in the main block:
    Demos = ["box_bowl", "candy", "box_box", "box_block", "lolipop", "box_ocean_ball", "cloth", "bread", "lolipop_axe", "bowl", "all"]
    Demo = "all"
    depth_scale = 0.01 #0.01 for realsense d405, 0.001 for d435
    realsense = Realsense(depth_scale= depth_scale)
    box_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_picking_box4.pt"), init_transformation=["x_180"])
    if Demo == "box_bowl":
        # Remove the extra parameters from constructor.
        bowl_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_wbcd8.pt"))
        
        while True:
            get_object_and_box_position(bowl_perception, box_perception, realsense,
                                        object_item_yolo_name="bowl",   # passed in at call time
                                        box_item_yolo_name="box",
                                        object_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/bowl.STL"),
                                        box_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/picking_box.STL"))
    if Demo == "box_box":
        # Remove the extra parameters from constructor.
        
        object_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_wbcd8.pt"))
        while True:
            result = get_object_and_box_position(object_perception, box_perception, realsense,
                                        object_item_yolo_name="box",   # passed in at call time
                                        box_item_yolo_name="box",
                                        object_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/paper-box.STL"),
                                        box_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/picking_box.STL"))
            if result is None:
                continue
            box_pose, object_poses = result
            print(object_poses[0])
    if Demo == "box_block":
        # Remove the extra parameters from constructor.
        object_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_wbcd18.pt"))
        while True:
            result = get_object_and_box_position(object_perception, box_perception, realsense,
                                                 object_item_yolo_name="wooden-block",   # passed in at call time
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
            demo_cloth_corner(cloth_perception, "cloth", realsense, merge_contours= False)
    if Demo == "lolipop":
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
    if Demo == "bread":
        bread_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_bread.pt"))
        for _ in range(50):
            realsense.get_frames()
        while True:
            demo_bread(bread_perception, "bread", realsense, merge_contours= False)
    if Demo == "lolipop_axe":
        lolipop_axe_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/best.pt"))
        for _ in range(50):
            realsense.get_frames()
        while True:
            demo_lolipop_axe(lolipop_axe_perception, "lolipop", realsense, merge_contours= False)
    if Demo == "bowl":
        bread_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_wbcd18.pt"))
        for _ in range(50):
            realsense.get_frames()
        # return each bowl's [meters in front of base, meters to the left of base, meters above base]
        for i in range(10):
            demo_bread(bread_perception, "bowl", realsense, merge_contours= False)
    if Demo == "all":
        bread_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_wbcd18.pt"))
        box_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_picking_box4.pt"))

        for _ in range(50):
            realsense.get_frames()
        for i in range(10):
            positions = demo_objects(
                bread_perception, box_perception,
                ["bowl","wooden-block","tennis","bread","can","cone","lollipop","ocean-ball","paper-box"],
                ["box"],
                realsense,
                transformation=np.array([[0, -0.743, 0.669, 0.047],
                                        [-1, 0, 0, 0.055],
                                        [0, -0.669, -0.743, 0.46],
                                        [0, 0, 0, 1]]),
                vis=False
            )
            if i==9:
                print("Reached 9th iteration, starting continuous publishing...")
                pub = rospy.Publisher('/object_poses', String, queue_size=1)
                # pub = rospy.Publisher('/object_data', String, queue_size=1)
                rospy.sleep(0.5)  # 给publisher时间初始化

                rate = rospy.Rate(10)  # 10Hz发布频率，可以调整
                
                while not rospy.is_shutdown():
                    # 如果你需要实时检测，取消下面这行的注释
                    # positions = demo_bread(bread_perception, "bowl", realsense, merge_contours=False)
                        data_list = []
                        for i, pos in enumerate(positions):
                            try:
                                xyz_list = pos[0]  # [x, y, z]
                                yaw_angle = float(pos[1])  # yaw
                                object_name = pos[2]  # object
                                
                                x, y, z = xyz_list[0], xyz_list[1], xyz_list[2]
                                
                                # 格式：x,y,z,yaw,object_name
                                data_str = f"{x},{y},{z},{yaw_angle},{object_name}"
                                data_list.append(data_str)
                                
                                rospy.loginfo(f"Object: {object_name}, x:{x:.3f}, y:{y:.3f}, z:{z:.3f}, yaw:{yaw_angle:.3f}")
                                
                            except (IndexError, TypeError, ValueError) as e:
                                rospy.logwarn(f"Error processing position {i}: {pos}, error: {e}")
                                continue
                        
                        # 用分号分隔多个物体
                        final_data = ";".join(data_list)
                        
                        # 发布数据
                        msg = String()
                        msg.data = final_data
                        pub.publish(msg)
                        
                        rospy.loginfo(f"Published: {final_data}")
                        rate.sleep()
                break
                






            