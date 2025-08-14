import open3d as o3d
import sys
import os
import numpy as np
import copy
import time

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0, 0, 0])
    object_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0, origin=[0, 0, 0])
    object_frame.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp, coordinate_frame, object_frame],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])

def get_6d_pose(cad_pc, cam_pc, threshold=50, init_transformations=None):
    # Define common rotation matrices
    def rot_x(theta):
        return np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta),  np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

    def rot_y(theta):
        return np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

    def rot_z(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta),  np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    rot = np.array([
            [1.0,      0.0,       0.0],
            [0.0, -0.66913,  0.74314],
            [0.0,  -0.74314,  -0.66913]
        ])
    transformation = np.eye(4)
    transformation[:3, :3] = rot
    if init_transformations is None:
        # Use default orientation candidates
        init_transformations = [
            transformation,transformation@rot_y(np.pi / 2),
            transformation@rot_z(np.pi / 2),transformation@rot_z(-np.pi / 2),
            transformation@rot_x(np.pi / 2),transformation@rot_x(-np.pi / 2)
        ]

    pc_center = np.mean(cam_pc.points, axis=0).reshape((3, 1))
    min_rmse = float("inf")
    best_trans = None
    
    for trans_init in init_transformations:
        ref_pc = copy.deepcopy(cad_pc)
        ref_pc_center = np.mean(ref_pc.points, axis=0).reshape((3, 1))

        # Initial transformation: align centroids + candidate rotation
        t_init = np.identity(4)
        t_init[:3, 3] = -(ref_pc_center - pc_center)[:, 0]
        t_init = t_init @ trans_init
        # Estimate normals
        # ref_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=30))
        # cam_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.001, max_nn=30))

        # Run ICP
        reg_p2p = o3d.pipelines.registration.registration_icp(
            ref_pc, cam_pc, threshold, t_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
        )

        # Track best result
        if reg_p2p.inlier_rmse < min_rmse:
            min_rmse = reg_p2p.inlier_rmse
            best_trans = reg_p2p.transformation

    return best_trans, min_rmse


if __name__ == "__main__":
    print("Using: ", sys.version)
    print("Working Dir: ", os.getcwd())

    mesh = o3d.io.read_triangle_mesh("./perception/ref_pc.STL")
    if not mesh.is_empty() and mesh.has_triangles():
        print("Mesh loaded successfully with triangles")
    else:
        quit()

    ref_pc = mesh.sample_points_uniformly(number_of_points=2000)
    points = np.load("./perception/pc.npy") * 1000 # m to mm
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    threshold = 100
    t1 = time.time()
    trans, rmse = get_6d_pose(ref_pc, pc, threshold)
    print(trans)
    print(rmse, time.time() - t1)

    draw_registration_result(ref_pc, pc, trans)