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
from scipy.spatial.transform import Rotation as R
from typing import Tuple
from yolo_6d_pose import Yolo6DPose
import math
import os
import shutil

def prepare_empty_folder(folder_path):
    """
    Create the folder if it doesn't exist.
    If it exists, remove all contents (files and subfolders).
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    else:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

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
        return edge_points_3d
    else:
        return edge_points_3d


def bread_grasp_pose(object_perception, object_yolo_name, realsense, transformation = np.eye(4), vis=True):
    """
    Returns:
      List of [ [x, y, z], yaw ] in base frame for each bread ellipse midpoint.
    """
    # 1. Capture
    color_image, depth_image = realsense.get_frames()
    center_poses = object_perception.get_object_center(color_image, depth_image, object_yolo_name, visualize=vis)
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
    for cnt in contours[0]:
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
    for i, (mid_px, yaw) in enumerate(grasp_2d):
        u, v = int(mid_px[0]), int(mid_px[1])
        # sample window
        W = 3
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
        print(base_pt)
        hom = np.append(center_poses[i], 1.0)   
        base_pt = (transformation@hom)[:3].tolist()  # in base frame
        print(base_pt)
        grasp_3d.append([base_pt, yaw])
    # print(grasp_3d)
    # 5. Visualize & return
    if vis:
        cv2.imshow("Bread Grasps", color_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return grasp_3d

def demo_lolipop_axe(object_perception, object_yolo_name, realsense, transformation = np.eye(4), vis=True):
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
        print("[WARN] No contours found.")
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
    for cnt in contours[0]:
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
        print("[WARN] No candidates found.")
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

def select_pose(poses_in_base, object_type='cone'):
    """
    Select and return the best pose based on object type:
    - For 'cone': use y-axis up/forward/backward classification.
    - For 'can': only check if y-axis is up or not.

    Returns:
        (pose_xyzrpy, type_id)
        where type_id = 1 (up), 2 (forward), 3 (backward)
    """
    up_poses = []
    forward_poses = []
    backward_poses = []

    # 分类
    for pose in poses_in_base:
        y_axis = pose[:3, 1]
        y_z = y_axis[2]
        y_x = y_axis[0]

        if y_z > 0.7:
            up_poses.append(pose)
        elif y_x > 0:
            forward_poses.append(pose)
        else:
            backward_poses.append(pose)

    def get_pose_x(p):
        return p[0, 3]  

    if object_type == 'cone':
        if len(up_poses) > 0:
            best_up = min(up_poses, key=get_pose_x)
            print("best up pose: ", best_up)
            return matrix_to_xyzrpy(best_up), 1

        best_forward = min(forward_poses, key=get_pose_x) if len(forward_poses) > 0 else None
        best_backward = min(backward_poses, key=get_pose_x) if len(backward_poses) > 0 else None

        if best_forward is not None and best_backward is not None:
            if get_pose_x(best_forward) < get_pose_x(best_backward):
                return matrix_to_xyzrpy(best_forward), 2
            else:
                return project_y_axis_offset(best_backward, offset=-0.12), 3
        elif best_forward is not None:
            return matrix_to_xyzrpy(best_forward), 2
        elif best_backward is not None:
            return project_y_axis_offset(best_backward, offset=-0.12), 3

    elif object_type == 'can':
        if len(up_poses) > 0:
            best_up = min(up_poses, key=get_pose_x)
            return matrix_to_xyzrpy(best_up), 1
        else:
            other_poses = forward_poses + backward_poses
            if len(other_poses) > 0:
                best_other = min(other_poses, key=get_pose_x)
                return project_y_axis_offset(best_other, offset=0), 2

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


def check_object_in_hand(perception, realsense, object,expected_object_pos):
    """
    Check if the object is in the gripper.
    Args:
        perception (Yolo6DPose): An instance of the Yolo6DPose class for object detection and pose estimation.
        realsense (Realsense): An instance of the Realsense class for depth image acquisition.
    Returns:
        bool: True if the object is in the gripper, False otherwise.
    """
    import time
    time.sleep(0.5)
    if object == "bowl":
        z_offset = 0.07
    elif object == "box":
        z_offset = 0.02
    elif object == "can":
        z_offset = 0.08
    elif object == "cone":
        z_offset = 0.18
    else:
        z_offset = 0.02
    import time
    start = time.time()
    color_image, depth_image = realsense.get_frames()
    object_pos_cam_frame = perception.get_object_center(color_image, depth_image,object, visualize=True)
    if object_pos_cam_frame is None:
        print("No object detected, object not in gripper")
        return False
    object_pos_base_frame = [cam_frame_to_base_frame(pose) for pose in object_pos_cam_frame]
    z_values = [T[2, 3] for T in object_pos_base_frame]
    max_idx = np.argmax(z_values)
    max_z = z_values[max_idx]
    print("max z: ", max_z)
    print("expected z: ", expected_object_pos[2])
    dist = np.linalg.norm(max_z+z_offset - expected_object_pos[2])
    print("object pos in base frame: ", object_pos_base_frame)
    print("expected object pos in base frame: ", expected_object_pos)
    end = time.time()
    print("time taken: ", end - start)
    if dist > 0.08:
        print("object dist to expected: ", dist)
        print("Object is not in the gripper")
        return False
    else:
        print("object dist to expected: ", dist)
        print("Object is in the gripper")
        return True
    
def select_first_valid_x(grasp_list, x_threshold=0.23, offset=0):
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
        if y>0:
            offset = offset
        else: 
            offset = -offset
        if x > x_threshold and x < 0.32:
            # Shift position 0.03 meters opposite to yaw direction
            dx = -offset * math.cos(yaw)
            dy = -offset * math.sin(yaw)
            adjusted_pos = [x + dx, y + dy, z]
            return adjusted_pos, yaw
    return None, None



def project_y_axis_offset(T: np.ndarray, offset: float = 0.03, visualize: bool = False):
    """
    Projects the local Y axis of a 4×4 transform onto the world XY plane,
    moves a point by 'offset' along the opposite direction of that projection,
    and returns the resulting (x, y) position and normal angle (in radians).

    If visualize=True, displays a plot showing the original frame, projected Y axis,
    offset movement, and resulting new position with normal.
    """
    # Extract rotation matrix and position
    R, P = T[:3, :3], T[:3, 3]

    # Local Y axis in world coordinates
    y_world = R @ np.array([0, 1, 0])

    # Project Y axis onto XY plane
    proj = y_world[:2]
    if np.linalg.norm(proj) < 1e-8:
        raise ValueError("Projected Y axis is too small to determine a direction")

    # Compute opposite direction
    dir_unit = -proj / np.linalg.norm(proj)

    # Offset position
    new_pos = P[:2] + dir_unit * offset

    # Calculate angle of normal
    normal_angle = np.arctan2(dir_unit[0], -dir_unit[1])

    if visualize:
        plt.figure(figsize=(6, 6))
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')

        # Plot original frame position
        plt.plot(P[0], P[1], 'ko', label='Original Position')

        # Plot projected Y axis
        plt.arrow(P[0], P[1], proj[0]*0.1, proj[1]*0.1, 
                  head_width=0.01, color='green', label='Projected Y axis')

        # Plot offset direction (opposite)
        plt.arrow(P[0], P[1], dir_unit[0]*offset, dir_unit[1]*offset, 
                  head_width=0.01, color='red', label='Offset Direction')

        # Plot new position
        plt.plot(new_pos[0], new_pos[1], 'bx', markersize=10, label='New Position')

        # Plot normal at new position
        plt.arrow(new_pos[0], new_pos[1],
                  np.cos(normal_angle)*0.05, np.sin(normal_angle)*0.05,
                  head_width=0.01, color='blue', label='Normal Direction')

        plt.legend()
        plt.title('Projection and Offset Visualization')
        plt.gca().set_aspect('equal')
        plt.grid(True)
        plt.show()

    return new_pos, normal_angle

def base_relative_goals(object_relative_goals, object_pos_in_base):
    """
    Map a trajectory of goals defined relative to object position, to be relative to base frame.
    Only concerning about goal1 (left hand) for now
    Args:
        object_relative_goals (list of numpy.ndarray): A list of 4x4 transformation matrices representing
            the trajectory of goals relative to a object position.
        object_pos_in_base (numpy.ndarray): A 1D array of shape (3,) representing the position of the object
            in the base frame.
    Returns:
        list of numpy.ndarray: A list of 4x4 transformation matrices representing the trajectory of goals
            relative to the base frame.
    """
    base_relative_trajectory = object_relative_goals.copy()
    for i in range(len(object_relative_goals)):
        base_relative_trajectory[i][0, :3, 3] += object_pos_in_base
    return base_relative_trajectory

def matrix_to_xyzrpy(T):
    """
    Convert a 4x4 transformation matrix into [x, y, z, roll, pitch, yaw].
    """
    translation = T[:3, 3]
    rotation_matrix = T[:3, :3]
    rpy = R.from_matrix(rotation_matrix).as_euler('xyz', degrees=False)
    return np.concatenate((translation, rpy))

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
        self.spatial = rs.spatial_filter()   
        self.temporal = rs.temporal_filter() 
        self.hole_filling = rs.hole_filling_filter()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = self.spatial.process(aligned_depth_frame)
        aligned_depth_frame = self.temporal.process(aligned_depth_frame)
        # aligned_depth_frame = self.hole_filling.process(aligned_depth_frame)
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
        color_image, depth_image, object_mesh, object_item_yolo_name, visualize=False)
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
    box_pose = box_perception.get_6d_pose_from_pointcloud(filtered_box_pointcloud, box_mesh, visualize= False)
    pose_mesh_pairs = [[cam_frame_to_base_frame(pose), object_mesh] for pose in object_poses]
    # Use the provided box_mesh instead of a stored mesh.
    box_pose_mesh_pairs = [[cam_frame_to_base_frame(box_pose), box_mesh]]
    pose_mesh_pairs.extend(box_pose_mesh_pairs)
    # draw_all_objects(pose_mesh_pairs)
    return box_pose, object_poses

def demo_soft_object(object_perception, object_yolo_name, realsense, merge_contours = True, vis = False):
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
    Demo = "lolipop_axe"
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
        
        object_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_wbcd10.pt"))
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
        object_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_wbcd10.pt"))
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
            grasp_points_3d = demo_soft_object(cloth_perception, "cloth", realsense, merge_contours= False)
            print(select_first_valid_x(grasp_points_3d, x_threshold=0.23))
    if Demo == "lolipop":
        object_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segment_lolipop2.pt"))
        while True:
            try:
               while True:
                    box_pose,objpose = get_object_and_box_position(object_perception, box_perception, realsense,
                                            object_item_yolo_name="lollipop",   # passed in at call time
                                            box_item_yolo_name="box",
                                            object_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/lollipop.STL"),
                                            box_mesh=o3d.io.read_triangle_mesh("perception/wbcd_meshes/picking_box.STL"))
                    print(cam_frame_to_base_frame(np.array(box_pose)),cam_frame_to_base_frame(np.array(objpose)))
                    print(project_y_axis_offset(np.array(objpose[0]), offset=0.03))
            except Exception as e:
                print(f"Error: {e}")

    if Demo == "lolipop_axe":
        lolipop_axe_perception = Yolo6DPose(K, YOLO("perception/yolo_segmentation/segemet_lollipop_clu.pt"))
        for _ in range(50):
            realsense.get_frames()
        while True:
            demo_lolipop_axe(lolipop_axe_perception, "lolipop", realsense)
