import pyrealsense2 as rs
import numpy as np
import cv2

# Create a pipeline and configure the streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create an align object to align the depth frame to the color frame
align_to = rs.stream.color
align = rs.align(align_to)

# Scale factor to enlarge the display image
scale_factor = 1.5

try:
    while True:
        # Wait for a new set of frames
        frames = pipeline.wait_for_frames()
        # Align the depth frame to the color frame
        aligned_frames = align.process(frames)
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            continue

        # Convert frame data to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply a colormap on the depth image for better visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Stack the color and depth images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Option 1: Resize the final image to be larger using a scale factor
        enlarged_image = cv2.resize(
            images,
            (
                int(images.shape[1] * scale_factor),
                int(images.shape[0] * scale_factor)
            )
        )
        
        # Option 2: Alternatively, you can set a larger window (uncomment the following two lines if preferred)
        # cv2.namedWindow('Aligned RGB and Depth', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('Aligned RGB and Depth', int(images.shape[1] * scale_factor), int(images.shape[0] * scale_factor))

        # Display the enlarged image
        cv2.imshow('Aligned RGB and Depth', enlarged_image)
        
        # Exit on 'q' key press
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

finally:
    # Stop the pipeline and destroy all windows
    pipeline.stop()
    cv2.destroyAllWindows()
