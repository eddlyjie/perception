import pyrealsense2 as rs
import numpy as np
import cv2

def is_similar(img1, img2, thresh=1.0):
    """比较两张图是否相似（像素平均差小于阈值）"""
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    return np.mean(diff) < thresh

def wait_for_stable_frames(threshold=15, sim_thresh=1.5):
    # 初始化 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)

    try:
        stable_count = 0
        prev_gray = None

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                if is_similar(gray, prev_gray, sim_thresh):
                    stable_count += 1
                    print(f"Stable frame count: {stable_count}")
                else:
                    stable_count = 0
                    print("Frame changed, resetting count.")
            else:
                print("Initializing first frame...")

            if stable_count >= threshold:
                print("✅ Frames stabilized. Proceeding.")
                return color_image, depth_image

            prev_gray = gray.copy()
            key = cv2.waitKey(1)
            if key == 27:  # ESC to break
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# 示例调用
if __name__ == "__main__":
    color_img, depth_img = wait_for_stable_frames()
    cv2.imshow("Stable Frame", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
