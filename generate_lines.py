import numpy as np
import matplotlib.pyplot as plt

def generate_segments_with_center_output(rect_center, rotation_angle_rad, rect_length=18.5, rect_width=13.5, visualize=False):
    """
    Given the center and rotation (in radians) of a rectangle,
    generate two 8.5-length segments extending from rectangle corners
    at 45 degrees to the rectangle edges, adjust them if needed, 
    and return their center points and angles.

    Args:
        rect_center: (x, y) tuple
        rotation_angle_rad: rotation in radians
        rect_length: rectangle length
        rect_width: rectangle width
        visualize: whether to draw the result

    Returns:
        (center1, angle1), (center2, angle2)
    """

    cx, cy = rect_center
    half_l = rect_length / 2
    half_w = rect_width / 2

    # Rectangle local corners
    local_corners = np.array([
        [ half_l,  half_w],
        [-half_l,  half_w],
        [-half_l, -half_w],
        [ half_l, -half_w],
    ])

    # Rotation matrix (rotation angle is already radian)
    theta = rotation_angle_rad
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ])

    # World corners after rotation and translation
    world_corners = local_corners @ R.T + np.array([cx, cy])

    diagonals = [
        (world_corners[0], world_corners[2]),
        (world_corners[1], world_corners[3]),
    ]
    diag_indices = [(0, 2), (1, 3)]

    # Choose the diagonal most aligned with Y axis
    min_angle = float('inf')
    best_diag = None
    best_diag_idx = None
    for idx, (pt1, pt2) in enumerate(diagonals):
        vec = pt2 - pt1
        vec_norm = vec / np.linalg.norm(vec)
        angle_to_y = np.arccos(np.clip(np.abs(np.dot(vec_norm, np.array([0,1]))), -1.0, 1.0))
        if angle_to_y < min_angle:
            min_angle = angle_to_y
            best_diag = (pt1, pt2)
            best_diag_idx = idx

    ptA, ptB = best_diag
    idxA, idxB = diag_indices[best_diag_idx]

    # Create two 8.5m long segments at 45°
    segment_length = 8.5

    local_x_dir = R @ np.array([1.0, 0.0])
    local_y_dir = R @ np.array([0.0, 1.0])

    def rotate_vec(vec, angle_deg):
        angle_rad = np.deg2rad(angle_deg)
        rot = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        return rot @ vec

    dir1 = rotate_vec(local_x_dir, 45)
    dir2 = rotate_vec(local_y_dir, 45)

    def build_segment(center, direction, length):
        offset = (length / 2) * (direction / np.linalg.norm(direction))
        start = center - offset
        end = center + offset
        return (start, end)

    segment1 = build_segment(ptA, dir1, segment_length)
    segment2 = build_segment(ptB, dir2, segment_length)

    # Check if endpoints are inside rectangle, if yes, rotate 90°
    def point_in_rectangle(pt, rect_corners):
        def is_left(p0, p1, p2):
            return np.cross(p1 - p0, p2 - p0) >= 0
        return (is_left(rect_corners[0], rect_corners[1], pt) and
                is_left(rect_corners[1], rect_corners[2], pt) and
                is_left(rect_corners[2], rect_corners[3], pt) and
                is_left(rect_corners[3], rect_corners[0], pt))

    def rotate_segment_90(segment):
        center = (segment[0] + segment[1]) / 2
        rel_start = segment[0] - center
        rel_end = segment[1] - center
        R90 = np.array([
            [0, -1],
            [1,  0]
        ])
        new_start = center + R90 @ rel_start
        new_end = center + R90 @ rel_end
        return (new_start, new_end)

    def check_and_rotate(segment, rect_corners):
        if point_in_rectangle(segment[0], rect_corners) or point_in_rectangle(segment[1], rect_corners):
            return rotate_segment_90(segment)
        else:
            return segment

    segment1 = check_and_rotate(segment1, world_corners)
    segment2 = check_and_rotate(segment2, world_corners)

    # Move segments along corner bisector
    def move_along_bisector(segment, corner_idx, rect_corners, distance):
        corner = rect_corners[corner_idx]
        if corner_idx == 0:
            edge1 = rect_corners[1] - corner
            edge2 = rect_corners[3] - corner
        elif corner_idx == 1:
            edge1 = rect_corners[0] - corner
            edge2 = rect_corners[2] - corner
        elif corner_idx == 2:
            edge1 = rect_corners[3] - corner
            edge2 = rect_corners[1] - corner
        else:
            edge1 = rect_corners[2] - corner
            edge2 = rect_corners[0] - corner

        edge1 = edge1 / np.linalg.norm(edge1)
        edge2 = edge2 / np.linalg.norm(edge2)

        bisector = (edge1 + edge2)
        bisector = bisector / np.linalg.norm(bisector)

        move_vec = bisector * distance
        return (segment[0] + move_vec, segment[1] + move_vec)

    segment1 = move_along_bisector(segment1, idxA, world_corners, distance=2.0)
    segment2 = move_along_bisector(segment2, idxB, world_corners, distance=2.0)

    # Calculate center and orientation
    center1 = (segment1[0] + segment1[1]) / 2
    center2 = (segment2[0] + segment2[1]) / 2

    vec1 = segment1[1] - segment1[0]
    vec2 = segment2[1] - segment2[0]
    angle1 = np.arctan2(vec1[1], vec1[0])
    angle2 = np.arctan2(vec2[1], vec2[0])

    if visualize:
        plt.figure(figsize=(8,8))
        world_corners_closed = np.vstack([world_corners, world_corners[0]])
        plt.plot(world_corners_closed[:,0], world_corners_closed[:,1], 'k-', label="Rectangle")

        for i, seg in enumerate([segment1, segment2]):
            plt.plot([seg[0][0], seg[1][0]], [seg[0][1], seg[1][1]], label=f"Segment {i+1}")
            plt.scatter(*seg[0], color='r')
            plt.scatter(*seg[1], color='r')

        plt.scatter(*center1, color='b', marker='x', label='Center 1')
        plt.scatter(*center2, color='g', marker='x', label='Center 2')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('scaled')
        plt.grid(True)
        plt.legend()
        plt.title(f"Center=({cx:.2f},{cy:.2f}), rotation={rotation_angle_rad:.1f}°")
        plt.show()

    return (center1, angle1), (center2, angle2)
def angle_between(v1, v2):
    """Return angle between v1 and v2 in radians, range [0, π]."""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def acute_angle_deg(v1, v2):
    """Return acute angle between v1 and v2 in degrees."""
    ang = np.degrees(angle_between(v1, v2))
    return ang if ang <= 90 else 180 - ang

def point_to_segment_distance(C, A, B):
    """
    Shortest distance from point C to segment AB (clamped to endpoints).
    C, A, B: np.array([x, y])
    """
    AB = B - A
    t = np.dot(C - A, AB) / np.dot(AB, AB)
    t = np.clip(t, 0.0, 1.0)
    closest = A + t * AB
    return np.linalg.norm(C - closest)

def segment_to_segment_distance(A1, B1, A2, B2):
    """
    Shortest distance between segment A1B1 and segment A2B2.
    """
    d1 = point_to_segment_distance(A1, A2, B2)
    d2 = point_to_segment_distance(B1, A2, B2)
    d3 = point_to_segment_distance(A2, A1, B1)
    d4 = point_to_segment_distance(B2, A1, B1)
    return min(d1, d2, d3, d4)

def segment_avoids_square_diagonals(dir_vec, square_yaw_rad, min_deg=10):
    """
    Case3 helper: ensure dir_vec's acute angle to both rotated diagonals > min_deg.
    """
    diags = [np.array([1.0, 1.0]), np.array([1.0, -1.0])]
    R = np.array([[np.cos(square_yaw_rad), -np.sin(square_yaw_rad)],
                  [np.sin(square_yaw_rad),  np.cos(square_yaw_rad)]])
    for d in diags:
        if acute_angle_deg(dir_vec, R @ d) <= min_deg:
            return False
    return True

# ---------- Case 1: Midpoint on Circle Edge ----------

def find_segment_case1(
    circles, radius,
    rect_center, rect_w, rect_h, rect_yaw_deg,
    segment_length=8.5, num_offset=100, margin=2.0, min_edge_dist=5.0
):
    half = segment_length / 2
    theta = np.radians(rect_yaw_deg)
    R_inv = np.array([[np.cos(-theta), -np.sin(-theta)],
                      [np.sin(-theta),  np.cos(-theta)]])
    angles1 = np.linspace(np.pi/2, 0, num_offset//2, endpoint=False)
    angles2 = np.linspace(3*np.pi/2, np.pi, num_offset//2, endpoint=False)
    angles = np.concatenate([angles1, angles2])
    dirs = [np.array([np.cos(a), np.sin(a)]) for a in angles]

    for cx, cy in sorted(circles, key=lambda c: c[0]):
        center = np.array([cx, cy])
        for d in dirs:
            midpoint = center + radius * d
            if midpoint[1] < 0: 
                d = np.array([-d[0], d[1]]) 
                midpoint = center + radius * d
            if midpoint[0] >= 32:
                continue
            A = midpoint - d * half
            B = midpoint + d * half

            # 1) region check
            ok = True
            for P in (A, B):
                Lp = R_inv @ (P - rect_center)
                if not (-rect_w/2 + margin <= Lp[0] <= rect_w/2 - margin and
                        -rect_h/2 + margin <= Lp[1] <= rect_h/2 - margin):
                    ok = False
                    break
            if not ok:
                continue

            # 2) gap to other circles
            for ox, oy in circles:
                if (ox, oy) == (cx, cy):
                    continue
                if point_to_segment_distance(np.array([ox, oy]), A, B) <= (radius + min_edge_dist):
                    ok = False
                    break
            if not ok:
                continue

            return A, B, center
    return None

# ---------- Case 2: Midpoint at Circle Center ----------

def find_segment_case2(
    circles, radius,
    rect_center, rect_w, rect_h, rect_yaw_deg,
    segment_length=8.5, num_offset=100, margin=2.0, min_edge_dist=2.0
):
    half = segment_length / 2
    theta = np.radians(rect_yaw_deg)
    R_inv = np.array([[np.cos(-theta), -np.sin(-theta)],
                      [np.sin(-theta),  np.cos(-theta)]])
    angles1 = np.linspace(np.pi/2, 0, num_offset//2, endpoint=False)
    angles2 = np.linspace(3*np.pi/2, np.pi, num_offset//2, endpoint=False)
    angles = np.concatenate([angles1, angles2])
    dirs = [np.array([np.cos(a), np.sin(a)]) for a in angles]

    for cx, cy in sorted(circles, key=lambda c: c[0]):
        center = np.array([cx, cy])
 
        for d in dirs:
            if center[1] < 0: 
                d = np.array([-d[0], d[1]]) 
            A = center - d * half
            B = center + d * half

            # 1) region check
            ok = True
            for P in (A, B):
                Lp = R_inv @ (P - rect_center)
                if not (-rect_w/2 + margin <= Lp[0] <= rect_w/2 - margin and
                        -rect_h/2 + margin <= Lp[1] <= rect_h/2 - margin):
                    ok = False
                    break
            if not ok:
                continue

            # 2) gap to other circles
            for ox, oy in circles:
                if (ox, oy) == (cx, cy):
                    continue
                if point_to_segment_distance(np.array([ox, oy]), A, B) <= (radius + min_edge_dist):
                    ok = False
                    break
            if not ok:
                continue

            return A, B, center
    return None

# ---------- Case 3: Square-centered & avoid diagonals + gap to other squares ----------

def find_segment_case3(
    square_center, square_yaw_deg,
    rect_center, rect_w, rect_h, rect_yaw_deg,
    square_size=5.0, segment_length=8.5,
    min_diag_angle_deg=10, num_offset=100, margin=2.0, min_edge_dist=2,
    all_squares=None
):
    half = segment_length / 2
    theta_box = np.radians(rect_yaw_deg)
    Rb_inv = np.array([[np.cos(-theta_box), -np.sin(-theta_box)],
                       [np.sin(-theta_box),  np.cos(-theta_box)]])
    angles1 = np.linspace(np.pi/2, 0, num_offset//2, endpoint=False)
    angles2 = np.linspace(3*np.pi/2, np.pi, num_offset//2, endpoint=False)
    angles = np.concatenate([angles1, angles2])
    dirs = [np.array([np.cos(a), np.sin(a)]) for a in angles]

    sq_yaw_rad = np.radians(square_yaw_deg)

    # Precompute other square edges
    other_edges = []
    half_s = square_size / 2
    corners = np.array([[-half_s, -half_s],
                        [ half_s, -half_s],
                        [ half_s,  half_s],
                        [-half_s,  half_s],
                        [-half_s, -half_s]])
    for c, y in (all_squares or []):
        if np.allclose(c, square_center):
            continue
        th = np.radians(y)
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]])
        poly = (R @ corners.T).T + c
        for i in range(len(poly)-1):
            other_edges.append((poly[i], poly[i+1]))

    for d in dirs:
        if square_center[1] < 0: 
            d = np.array([-d[0], d[1]]) 
        if not segment_avoids_square_diagonals(d, sq_yaw_rad, min_diag_angle_deg):
            continue
        A = np.array(square_center) - d * half
        B = np.array(square_center) + d * half

        # 1) region check
        ok = True
        for P in (A, B):
            Lp = Rb_inv @ (P - rect_center)
            if not (-rect_w/2 + margin <= Lp[0] <= rect_w/2 - margin and
                    -rect_h/2 + margin <= Lp[1] <= rect_h/2 - margin):
                ok = False
                break
        if not ok:
            continue

        # 2) gap to other squares
        for E1, E2 in other_edges:
            if segment_to_segment_distance(A, B, E1, E2) <= min_edge_dist:
                ok = False
                break
        if not ok:
            continue

        return A, B, square_center
    return None

# ---------- Unified Interface & Visualization ----------

def compute_segment_from_object_box(
    object_poses, box_pose,
    mode='case1',
    rect_w=45.5, rect_h=30,
    object_radius=11.5/2, box_radius=6.5/2,
    square_size=5.0,
    segment_length=8.5, num_offset=30,
    margin=2.0, min_edge_dist=2.0,
    visualize=False
):
    """
    object_poses: list of [x, y, 0, 0, 0, yaw_rad]
    box_pose:     [x, y, 0, 0, 0, yaw_rad]
    mode: 'case1', 'case2', or 'case3'
    """
    box_xy = np.array(box_pose[:2])
    # box_yaw_deg = np.degrees(box_pose[4])
    box_yaw_deg = np.degrees(0)

    # print(object_poses)
    # print(box_pose)
    if mode in ('case1', 'case2'):
        circles = [tuple(obj[:2]) for obj in object_poses]
        if mode == 'case1':
            seg = find_segment_case1(
                circles, object_radius, box_xy, rect_w, rect_h, box_yaw_deg,
                segment_length, num_offset, margin, 3
            )
        else:
            seg = find_segment_case2(
                circles, box_radius, box_xy, rect_w, rect_h, box_yaw_deg,
                segment_length, num_offset, margin, min_edge_dist
            )
    else:  # case3
        squares = [(obj[:2], np.degrees(obj[5])) for obj in object_poses]
        seg = None
        for center, yaw in sorted(squares, key=lambda s: s[0][0]):
            cand = find_segment_case3(
                center, yaw, box_xy, rect_w, rect_h, box_yaw_deg,
                square_size, segment_length, 10, num_offset, margin,
                min_edge_dist, all_squares=squares
            )
            if cand is not None:
                seg = cand
                break

    if seg is None:
        return None, None

    A, B, center = seg
    midpoint = (A + B) / 2
    angle = np.arctan2((B - A)[1], (B - A)[0])

    if visualize:
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        # Region
        corners = np.array([[-rect_w/2, -rect_h/2],
                            [ rect_w/2, -rect_h/2],
                            [ rect_w/2,  rect_h/2],
                            [-rect_w/2,  rect_h/2],
                            [-rect_w/2, -rect_h/2]])
        thb = np.radians(box_yaw_deg)
        Rb = np.array([[np.cos(thb), -np.sin(thb)],
                       [np.sin(thb),  np.cos(thb)]])
        box_poly = (Rb @ corners.T).T + box_xy
        ax.plot(box_poly[:,0], box_poly[:,1], 'k--', label='Region')

        # Draw objects
        if mode in ('case1', 'case2'):
            radius = object_radius if mode=='case1' else box_radius
            for obj in object_poses:
                circ = plt.Circle(obj[:2], radius, edgecolor='black',
                                  facecolor='lightgray', alpha=0.5)
                ax.add_patch(circ)
        else:
            for obj in object_poses:
                half_s = square_size / 2
                corners_sq = np.array([[-half_s,-half_s],[ half_s,-half_s],
                                       [ half_s, half_s],[-half_s, half_s],
                                       [-half_s,-half_s]])
                ths = obj[5]
                Rs = np.array([[np.cos(ths), -np.sin(ths)],
                               [np.sin(ths),  np.cos(ths)]])
                poly = (Rs @ corners_sq.T).T + obj[:2]
                ax.plot(poly[:,0], poly[:,1], color='orange')

        # Draw segment
        ax.plot([A[0], B[0]], [A[1], B[1]], 'r-', lw=2, label=mode)
        ax.scatter(center[0], center[1], c='g', marker='x')
        ax.legend(); plt.grid(True); plt.show()

    return midpoint, angle

# ---------- Example Usage ----------

if __name__ == "__main__":
    obj_list = [
        [31.0, 6, 0, 0, 0, np.radians(0)],
        [32.0, 0.0, 0, 0, 0, np.radians(30)],
        [48.0, -2.0, 0, 0, 0, np.radians(60)]
    ]
    region_pose = [39.0, 10, 0, 0, 0, np.radians(10)]

    for mode in ['case1', 'case2', 'case3']:
        mid, ang = compute_segment_from_object_box(
            obj_list, region_pose, mode=mode, visualize=True
        )
        print(f"--- {mode.upper()} ---")
        if mid is not None:
            print("Midpoint:", mid, "Angle(deg):", np.degrees(ang))
        else:
            print("No valid segment found")

    center = (40.0, 0.0)

    rotation_angle_deg = 3.14

    seg1_info, seg2_info = generate_segments_with_center_output(
        center,
        rotation_angle_deg,
        visualize=True
    )

    print("Segment 1 Center and Angle:", seg1_info)
    print("Segment 2 Center and Angle:", seg2_info)
