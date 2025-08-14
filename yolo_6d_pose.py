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

def angle_between(v1, v2):
    """Returns angle in degrees between two vectors."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


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
                  [-1, 0, 0,  0.055],
                  [0, -0.669, -0.743, 0.46],
                  [0, 0, 0, 1]])
    object_pos_in_base_hom = T @ object_pose_in_camera
    if pos_only:
        object_pos_in_base_hom[:3, :3] = np.eye(3)
    return object_pos_in_base_hom


class Yolo6DPose():
    # Modified constructor: remove storing of item, item_yolo_name, and mesh.
    def __init__(self, K, yolo_seg_model, init_transformation=None):
        self.yolo_seg_model = yolo_seg_model
        self.H = 480
        self.W = 640
        self.K = K
        self.max_instances = 6
        self.T = np.array([[0, -0.743, 0.669, 0.047],
                  [-1, 0, 0,  0.055],
                  [0, -0.669, -0.743, 0.46],
                  [0, 0, 0, 1]])
        self.init_transformation = init_transformation

    # The segmentation method now takes the object’s YOLO label name as an argument.
    def yolo_seg(self, frame, item_yolo_name, visualize=False):
        results = self.yolo_seg_model.predict(frame, show=visualize, conf=0.7)
        if results is None:
            return None
        img = np.copy(results[0].orig_img)
        H, W = img.shape[:2]
        num_objects = len(results[0].boxes)
        contours = []
        confs = []
        for i in range(num_objects):
            class_id = int(results[0].boxes.cls[i])
            class_name = self.yolo_seg_model.names[class_id]
            # Replace self.item_yolo_name with the method argument.
            if class_name.lower() == item_yolo_name:
                contour = results[0].masks.xy[i].astype(int)
                contours.append(contour)
                confs.append(results[0].boxes.conf[i])
                if len(contours) > self.max_instances:
                    break
        return contours, confs

    # Similarly, update keypoint detection to take the label name as an argument.
    def yolo_keypoint(self, frame, item_yolo_name, yolo_keypoint_model, visualize=True):
        results = yolo_keypoint_model.predict(frame, show=visualize, conf=0.7)
        if results is None:
            return None
        img = np.copy(results[0].orig_img)
        H, W = img.shape[:2]
        num_objects = len(results[0].boxes)
        interest_points = []
        for i in range(num_objects):
            class_id = int(results[0].boxes.cls[i])
            class_name = yolo_keypoint_model.names[class_id]
            if class_name.lower() == item_yolo_name:
                point = results[0].keypoints[i].xy[0][0]
                interest_points.append(point)
        return interest_points

    def contour_to_mask(self, contour):
        mask = np.zeros((self.H, self.W), np.uint8)
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)
        return mask

    def get_pointcloud(self, mask, depth_image, trim=0.05):
        """
        Generate a 3D point cloud from a binary mask and a depth image using camera intrinsics.
        Returns: Nx3 array of points (in millimeters)
        """
        ys, xs = np.where(mask > 0)
        zs = depth_image[ys, xs]
        valid = zs > 0
        xs = xs[valid]
        ys = ys[valid]
        zs = zs[valid]
        if len(xs) == 0:
            return None
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        xs_3d = (xs - cx) * zs / fx
        ys_3d = (ys - cy) * zs / fy
        pointcloud = np.stack((xs_3d, ys_3d, zs), axis=-1)
        median = np.median(pointcloud, axis=0)
        distances = np.linalg.norm(pointcloud - median, axis=1)
        threshold = np.percentile(distances, (1 - trim) * 100)
        pointcloud = pointcloud[distances <= threshold]
        return pointcloud

    def get_center_pos_raw(self, pointcloud):
        if pointcloud.size == 0:
            return None
        center_pos = np.mean(pointcloud, axis=0)
        return center_pos

    def get_bounding_box_3d(self, pointcloud):
        if pointcloud.size == 0:
            return None
        x_vals = pointcloud[:, 0]
        y_vals = pointcloud[:, 1]
        z_vals = pointcloud[:, 2]
        xmin, xmax = np.percentile(x_vals, [0.5, 99.5])
        ymin, ymax = np.percentile(y_vals, [0.5, 99.5])
        zmin, zmax = np.percentile(z_vals, [5, 95])
        return np.array([xmax, xmin, ymax, ymin, zmax, zmin])
    
    def project_point_to_image(self, point_3d):
        """Project a 3D point to 2D image coordinates using camera intrinsics"""
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        
        # Handle zero depth case
        z = point_3d[2]
        if abs(z) < 1e-6:
            z = 1e-6
            
        x = (point_3d[0] * fx) / z + cx
        y = (point_3d[1] * fy) / z + cy
        
        return np.array([x, y])

    def draw_3d_box(self, image, bounding_box):
        xmax, xmin, ymax, ymin, zmax, zmin = bounding_box
        corners_3d = np.array([
            [xmin, ymin, zmin],
            [xmax, ymin, zmin],
            [xmax, ymax, zmin],
            [xmin, ymax, zmin],
            [xmin, ymin, zmax],
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmin, ymax, zmax]
        ])
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        zs = corners_3d[:, 2]
        zs[zs == 0] = 1e-6
        xs = (corners_3d[:, 0] * fx) / zs + cx
        ys = (corners_3d[:, 1] * fy) / zs + cy
        points_2d = np.stack((xs, ys), axis=-1).astype(int)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
        ]
        for i, j in edges:
            pt1 = tuple(points_2d[i])
            pt2 = tuple(points_2d[j])
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        return image

    # In methods that needed the stored YOLO name, we now add a parameter (item_yolo_name).
    def show_3d_box(self, rgb_image, depth_image, item_yolo_name):
        seg = self.yolo_seg(rgb_image, item_yolo_name, visualize=False)
        if seg is None:
            print("[WARN] No matching object found.")
            return None
        # Use the first found contour.
        contour = seg[0][0]
        self.H, self.W = rgb_image.shape[:2]
        mask = self.contour_to_mask(contour)
        pointcloud = self.get_pointcloud(mask, depth_image)
        if pointcloud is None or pointcloud.size == 0:
            print("[WARN] No valid depth points in mask.")
            return None
        center_pos = self.get_center_pos_raw(pointcloud)
        bounding_box = self.get_bounding_box_3d(pointcloud)
        image_with_box = np.copy(rgb_image)
        highlight = cv2.addWeighted(image_with_box, 1.0,
                                    np.full_like(image_with_box, 200), 0.5, 0)
        mask_3c = cv2.merge([mask, mask, mask])
        image_with_box = np.where(mask_3c > 0, highlight, image_with_box)
        image_with_box = self.draw_3d_box(image_with_box, bounding_box)
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        x, y, z = center_pos
        if z != 0:
            u = int((x * fx) / z + cx)
            v = int((y * fy) / z + cy)
            cv2.circle(image_with_box, (u, v), 6, (0, 0, 255), -1)
            text = f"({x:.2f}, {y:.2f}, {z:.2f})"
            cv2.putText(image_with_box, text, (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("bounding box", image_with_box)
        cv2.waitKey(1)
        return center_pos

    def get_object_center(self, rgb_image, depth_image, item_yolo_name, visualize=False):
        contours, confs = self.yolo_seg(rgb_image, item_yolo_name, visualize=visualize)
        if contours is None or len(contours) == 0:
            print("[WARN] No matching object found.")
            return None
        self.H, self.W = rgb_image.shape[:2]
        center_poss = []
        for contour, conf in zip(contours, confs):
            if conf > 0.6:
                mask = self.contour_to_mask(contour)
                pointcloud = self.get_pointcloud(mask, depth_image)
                if pointcloud is None or pointcloud.size == 0:
                    print("[WARN] No valid depth points in mask.")
                    return None
                center_pos = self.get_center_pos_raw(pointcloud)
                # Convert from mm to meters.
                x, y, z = center_pos
                center_pos = np.array([x / 1e3, y / 1e3, z / 1e3])
                center_poss.append(center_pos)
                if visualize and False:
                    import matplotlib.pyplot as plt
                    import matplotlib.patches as patches
                    depth_vis = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
                    depth_vis = (depth_vis * 255).astype(np.uint8)
                    fig, ax = plt.subplots()
                    ax.imshow(depth_vis, cmap="jet")
                    x_min, y_min = np.min(contour[:, 0]), np.min(contour[:, 1])
                    w = np.max(contour[:, 0]) - x_min
                    h = np.max(contour[:, 1]) - y_min
                    rect = patches.Rectangle((x_min, y_min), w, h, linewidth=2,
                                             edgecolor='green', facecolor='none')
                    ax.add_patch(rect)
                    plt.title("Depth Image with Contour")
                    plt.axis("off")
                    plt.show()
        return center_poss

    def get_interest_point_pos(self, rgb_image, depth_image, item_yolo_name, yolo_keypoint_model, visualize=False):
        interest_points = self.yolo_keypoint(rgb_image, item_yolo_name, yolo_keypoint_model, visualize)
        seg = self.yolo_seg(rgb_image, item_yolo_name, visualize=False)
        if seg is None:
            print("[WARN] No matching object found.")
            return None
        contour = seg[0][0]
        mask = self.contour_to_mask(contour)
        if interest_points is None:
            print("[WARN] No keypoints found.")
            return None
        for point in interest_points:
            x, y = int(point[0]), int(point[1])
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                mask_region = mask[max(0, y - 2):min(self.H, y + 3),
                                   max(0, x - 2):min(self.W, x + 3)]
                depth_region = depth_image[max(0, y - 2):min(self.H, y + 3),
                                           max(0, x - 2):min(self.W, x + 3)]
                valid_depths = depth_region[mask_region > 0]
                if len(valid_depths) > 0:
                    avg_depth = np.median(valid_depths)
                    final_positions = (x, y, avg_depth)
                    if visualize:
                        vis_img = rgb_image.copy()
                        cv2.circle(vis_img, (x, y), 5, (0, 255, 0), -1)
                        cv2.putText(vis_img, f"({x}, {y}, {avg_depth:.2f})", (x + 10, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.imshow("Final Position Visualization", vis_img)
                        cv2.waitKey(1)
                    return final_positions
        distances = []
        for point in interest_points:
            x, y = int(point[0]), int(point[1])
            distances.append(cv2.pointPolygonTest(contour, (x, y), True))
        best_interest_point = interest_points[np.argmin(distances)]
        x, y = int(best_interest_point[0]), int(best_interest_point[1])
        closest_point = contour[np.argmin(np.linalg.norm(contour - np.array([x, y]), axis=1))]
        closest_point = np.array(closest_point).astype(int)
        cx, cy = closest_point[0], closest_point[1]
        mask_region = mask[max(0, cy - 4):min(self.H, cy + 5),
                           max(0, cx - 4):min(self.W, cx + 5)]
        depth_region = depth_image[max(0, cy - 4):min(self.H, cy + 5),
                                   max(0, cx - 4):min(self.W, cx + 5)]
        valid_depths = depth_region[mask_region > 0]
        avg_depth = np.mean(valid_depths)
        final_positions = (cx, cy, avg_depth)
        if visualize:
            vis_img = rgb_image.copy()
            cv2.circle(vis_img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(vis_img, f"({cx}, {cy}, {avg_depth:.2f})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Final Position Visualization", vis_img)
            cv2.waitKey(1)
        return final_positions

    # Here, add a 'mesh' argument so that ICP uses the provided mesh instead of a stored one.
    def get_6d_pose(self, rgb_image, depth_image, mesh, item_yolo_name, visualize=False, seg_threshold=0.6, select = False):
        if not select:
            contours, confs = self.yolo_seg(rgb_image, item_yolo_name, visualize=visualize)
            if contours is None:
                print("[WARN] No matching object found.")
                return rgb_image, None
            self.H, self.W = rgb_image.shape[:2]
            poses = []
            grouped_pointclouds = []
            for contour, conf in zip(contours, confs):
                if conf > seg_threshold:
                    mask = self.contour_to_mask(contour)
                    pointcloud = self.get_pointcloud(mask, depth_image)
                    if pointcloud is None or pointcloud.size == 0:
                        print("[WARN] No valid depth points in mask.")
                        return None
                    ref_pc = mesh.sample_points_uniformly(number_of_points=2000)
                    ref_pc_array = np.asarray(ref_pc.points)
                    x_diff = np.max(ref_pc_array[:, 0]) - np.min(ref_pc_array[:, 0])
                    if x_diff > 1:
                        print("\n WARNING\n mesh seems to be in mm. Converting units.")
                        ref_pc.points = o3d.utility.Vector3dVector(np.asarray(ref_pc.points))
                    else:
                        ref_pc.points = o3d.utility.Vector3dVector(np.asarray(ref_pc.points) * 1000)
                    pc = o3d.geometry.PointCloud()
                    pc.points = o3d.utility.Vector3dVector(pointcloud)
                    import time
                    start = time.time()
                    rot = np.array([
                        [1.0,      0.0,       0.0],
                        [0.0, -0.66913,  0.74314],
                        [0.0,  -0.74314,  -0.66913]
                    ])
                    transformation = np.eye(4)
                    transformation[:3, :3] = rot
                    pose, rmse = icp.get_6d_pose(ref_pc, pc, threshold=50, init_transformations=[transformation])
                    end = time.time()
                    print(f"ICP time: {end - start:.4f} seconds")
                    pose_m = pose.copy()
                    pose_m[:3, 3] = pose[:3, 3] / 1e3  # convert from mm to m
                    if visualize:
                        icp.draw_registration_result(ref_pc, pc, pose)
                    poses.append(pose_m)
                    grouped_pointclouds.append(pointcloud)
            return poses, grouped_pointclouds
        else:
            contours, confs = self.yolo_seg(rgb_image, item_yolo_name, visualize=visualize)
            if contours is None:
                print("[WARN] No matching object found.")
                return None, None
            self.H, self.W = rgb_image.shape[:2]
            poses = []
            grouped_pointclouds = []
            for contour, conf in zip(contours, confs):
                if conf > seg_threshold:
                    mask = self.contour_to_mask(contour)
                    pointcloud = self.get_pointcloud(mask, depth_image)
                    if pointcloud is None or pointcloud.size == 0:
                        print("[WARN] No valid depth points in mask.")
                        continue
                    ref_pc = mesh.sample_points_uniformly(number_of_points=2000)
                    ref_arr = np.asarray(ref_pc.points)
                    if np.max(ref_arr[:,0]) - np.min(ref_arr[:,0]) <= 1:
                        ref_pc.points = o3d.utility.Vector3dVector(ref_arr * 1000)
                    pc = o3d.geometry.PointCloud()
                    pc.points = o3d.utility.Vector3dVector(pointcloud)
                    rot = np.array([[1.0, 0.0, 0.0], [0.0, -0.66913, 0.74314], [0.0, -0.74314, -0.66913]])
                    init_tf = np.eye(4)
                    init_tf[:3,:3] = rot
                    pose, rmse = icp.get_6d_pose(ref_pc, pc, threshold=50, init_transformations=[init_tf])
                    pose_m = pose.copy()
                    pose_m[:3,3] = pose[:3,3]/1000.0
                    poses.append(pose_m)
                    grouped_pointclouds.append(pointcloud)
            # Select nearest non-intersecting
            if poses:
                # build bboxes (xmin,xmax,ymin,ymax,zmin,zmax) in meters
                bboxes = []
                for pts in grouped_pointclouds:
                    bb = self.get_bounding_box_3d(pts)
                    xmin, xmax = bb[1]/1000.0, bb[0]/1000.0
                    ymin, ymax = bb[3]/1000.0, bb[2]/1000.0
                    zmin, zmax = bb[5]/1000.0, bb[4]/1000.0
                    bboxes.append((xmin, xmax, ymin, ymax, zmin, zmax))
                def boxes_intersect(b1, b2):
                    return not (b1[1] < b2[0] or b2[1] < b1[0] or b1[3] < b2[2] or b2[3] < b1[2] or b1[5] < b2[4] or b2[5] < b1[4])
                valid = [i for i in range(len(poses)) if not any(boxes_intersect(bboxes[i], bboxes[j]) for j in range(len(poses)) if j!=i)]
                if not valid:
                    valid = list(range(len(poses)))
                dists = [np.linalg.norm(poses[i][:3,3]) for i in valid]
                sel = valid[int(np.argmin(dists))]
                return poses[sel], grouped_pointclouds[sel]
            return None, None

    def get_6d_pose_from_pointcloud(self, pointcloud, mesh, visualize=False):
        ref_pc = mesh.sample_points_uniformly(number_of_points=2000)
        ref_pc_array = np.asarray(ref_pc.points)
        x_diff = np.max(ref_pc_array[:, 0]) - np.min(ref_pc_array[:, 0])
        if x_diff > 1:
            print("\n WARNING\n mesh seems to be in mm. Converting units.")
            ref_pc.points = o3d.utility.Vector3dVector(np.asarray(ref_pc.points))
        else:
            ref_pc.points = o3d.utility.Vector3dVector(np.asarray(ref_pc.points) * 1000)
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pointcloud)
        import time
        start = time.time()
        rot = np.array([
            [1.0,      0.0,       0.0],
            [0.0, -0.66913,  0.74314],
            [0.0,  -0.74314,  -0.66913]
        ])
        transformation = np.eye(4)
        transformation[:3, :3] = rot
        pose, rmse = icp.get_6d_pose(ref_pc, pc, threshold=50,init_transformations=[transformation])
        end = time.time()
        print(f"ICP time: {end - start:.4f} seconds")
        pose_m = pose.copy()
        pose_m[:3, 3] = pose[:3, 3] / 1e3
        if visualize:
            icp.draw_registration_result(ref_pc, pc, pose)
        return pose_m

    def merge_contours(self, frame, contours):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for contour in contours:
            cv2.fillPoly(mask, [np.array(contour)], 255)

        merged_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return merged_contours

    def find_edge(self, frame, contours, merge_contours = True, visualize=False):
        if merge_contours:
            merged_contours = self.merge_contours(frame, contours)
        else:
            merged_contours = contours
        frame_copy = frame.copy()
        approx_edges = []

        for cnt in merged_contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            approx = approx.squeeze()
            if len(approx.shape) == 1:
                approx = approx[np.newaxis, :]
            if approx.shape[0] < 2:
                continue
            

            # Determine which edges to skip based on sharp inward turns
            skip_edge_indices = set()
            n = len(approx)
            for i in range(n):
                pt_prev = approx[i - 1]
                pt_curr = approx[i]
                pt_next = approx[(i + 1) % n]

                v1 = pt_curr - pt_prev
                v2 = pt_next - pt_curr
                angle = angle_between(v1, v2)

                if angle > 120:
                    # Mark both edges adjacent to the corner
                    skip_edge_indices.add((i - 1) % n)
                    skip_edge_indices.add(i)

            # Add valid edges
            for i in range(n):
                if i in skip_edge_indices:
                    continue
                pt1 = tuple(approx[i])
                pt2 = tuple(approx[(i + 1) % n])
                length = np.linalg.norm(np.array(pt1) - np.array(pt2))
                if length < 10:
                    continue
                approx_edges.append((length, pt1, pt2))

        # Sort and draw
        approx_edges.sort(key=lambda x: -x[0])
        top_edges = approx_edges[:10]
        if visualize:
            for _, pt1, pt2 in top_edges:
                cv2.line(frame_copy, pt1, pt2, (0, 0, 255), 2)

            cv2.imshow("Top 10 Longest Clean Edges", frame_copy)
            cv2.waitKey(0)
        return top_edges

    def find_edge_grasp_point(self, frame, contours, merge_contours=True):
        """
        Calls self.find_edge to get valid straight edges.
        For each edge, computes:
        - midpoint
        - outward-facing normal angle (theta in radians)
        Returns:
        List of [midpoint(x, y), theta]
        """
        # Create mask from the contour
        # 
        # for contour in contours:
        #     cv2.fillPoly(mask, [np.array(contour)], 255)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, contours, 255)

        # Use find_edge to get top N clean edges: list of (length, pt1, pt2)
        top_edges = self.find_edge(frame, contours, merge_contours=merge_contours)

        grasp_points = []
        for _, pt1, pt2 in top_edges:
            pt1 = np.array(pt1, dtype=np.float32)
            pt2 = np.array(pt2, dtype=np.float32)
            midpoint = (pt1 + pt2) / 2.0

            # Edge vector
            edge_vec = pt2 - pt1
            edge_vec /= np.linalg.norm(edge_vec)

            # Outward normal (rotate +90°)
            normal = np.array([-edge_vec[1], edge_vec[0]])
            test_points_positive = [(midpoint + i * normal).astype(int) for i in range(2, 11)]
            test_points_negative = [(midpoint - i * normal).astype(int) for i in range(2, 11)]

            # Count points inside the mask for both sides
            positive_count = sum(
                1 for pt in test_points_positive
                if 0 <= pt[1] < mask.shape[0] and 0 <= pt[0] < mask.shape[1] and mask[pt[1], pt[0]] > 0
            )
            negative_count = sum(
                1 for pt in test_points_negative
                if 0 <= pt[1] < mask.shape[0] and 0 <= pt[0] < mask.shape[1] and mask[pt[1], pt[0]] > 0
            )

            # Choose the side with fewer points in the mask as the outward normal
            if positive_count > negative_count:
                normal = -normal

            theta = float(np.arctan2(normal[1], normal[0]))
            grasp_points.append([midpoint.tolist(), theta])

        return grasp_points
    
    def find_edge_grasp_point_3d(self, color_frame, depth_frame, contour, merge_contours = True, transformation=np.eye(4)):
        """
        Compute 3D grasp points from 2D contour and edge grasp locations.

        Args:
            color_frame: RGB image (used for shape/size only)
            depth_frame: depth map in meters
            contour: the merged contour (as list of points)
            transformation: 4x4 camera-to-base transformation

        Returns:
            List of [3D position in base frame (x, y, z), yaw in radians]
        """

        grasp_2d = self.find_edge_grasp_point(color_frame, contour, merge_contours=merge_contours)

        # Prepare contour mask
        mask = np.zeros(depth_frame.shape, dtype=np.uint8)
        cv2.fillPoly(mask, contour, 255)

        grasp_3d = []

        for midpoint, yaw in grasp_2d:
            x, y = int(midpoint[0]), int(midpoint[1])
            window_size = 5  # Pixels around the midpoint
            x_min = max(x - window_size, 0)
            x_max = min(x + window_size, depth_frame.shape[1])
            y_min = max(y - window_size, 0)
            y_max = min(y + window_size, depth_frame.shape[0])

            # Collect 3D points in a small neighborhood that are inside the mask
            points = []
            for j in range(y_min, y_max):
                for i in range(x_min, x_max):
                    if mask[j, i] == 0:
                        continue
                    z = depth_frame[j, i]
                    if z == 0 or np.isnan(z):
                        continue
                    # Backproject using intrinsics
                    fx = self.K[0, 0]
                    fy = self.K[1, 1]
                    cx = self.K[0, 2]
                    cy = self.K[1, 2]
                    x3d = (i - cx) * z / fx
                    y3d = (j - cy) * z / fy
                    points.append([x3d, y3d, z])

            if not points:
                continue  # skip if no depth data
            
            avg_point = np.mean(points, axis=0) / 1000  # In camera frame
            avg_point_h = np.append(avg_point, 1.0)  # Homogeneous

            point_base = transformation @ avg_point_h
            grasp_3d.append([point_base[:3].tolist(), yaw])

        return grasp_3d
    

