import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2

from utilities import getDepthScale, removeMaskNoise, getIntrinsic, deprojectPixelToPoints, postprocessDepth


THRESHOLD = 1   # maximum distance, in meter


def capture_one_frame(pipeline, metadata, name="realsense"):
    align = metadata["align"]
    scale = metadata["scale"]
    intrinsic  = metadata["intrinsic"]

    frame = pipeline.wait_for_frames()

    # depth to color alignment
    
    aligned_frames = align.process(frame)

    depth = aligned_frames.get_depth_frame()
    color = aligned_frames.get_color_frame()

    if not depth or not color: print("Fail to get depth/color frame")

    depth = postprocessDepth(depth)

    depth_arr = np.asanyarray(depth.get_data()) # np.array [720 x 1280    ] dtype = uint16
    color_arr = np.asanyarray(color.get_data()) # np.array [720 x 1280 x 3] dtype = uint8

    valid_mask = np.where((depth_arr >  (THRESHOLD / scale)) | (depth_arr <= 0), 0, 1).astype(np.uint8)
    depth_arr  = np.where(valid_mask == 0, 0, depth_arr)
    color_arr  = np.where(valid_mask[:, :, np.newaxis] == 0, 0, color_arr)

    pts_3d     = deprojectPixelToPoints(intrinsic, depth_arr, step=1)
    pts_bgr    = (color_arr / 255).flatten().reshape(720 * 1280, 3)
    pts_rgb    = np.hstack((pts_bgr[:, 2:3], pts_bgr[:, 1:2], pts_bgr[:, 0:1]))

    pts_cloud  = o3d.geometry.PointCloud()
    pts_cloud.points = o3d.utility.Vector3dVector(pts_3d)
    pts_cloud.colors = o3d.utility.Vector3dVector(pts_rgb)

    o3d.visualization.draw_geometries([pts_cloud])
    np.savez(name, points = pts_3d, image = color_arr)



def main_loop_fn(pipeline, metadata):
    global THRESHOLD

    align = metadata["align"]
    scale = metadata["scale"]

    frame = pipeline.wait_for_frames()

    # depth to color alignment
    aligned_frames = align.process(frame)

    depth = aligned_frames.get_depth_frame()
    color = aligned_frames.get_color_frame()

    if not depth or not color: return

    depth_arr = np.asanyarray(depth.get_data()) # np.array [720 x 1280    ] dtype = uint16
    color_arr = np.asanyarray(color.get_data()) # np.array [720 x 1280 x 3] dtype = uint8

    image_mask = np.where((depth_arr >  (THRESHOLD / scale)) | (depth_arr <= 0), 0, 1).astype(np.uint8)
    image_mask = removeMaskNoise(image_mask, kernel_size=3)
    depth_masked = image_mask * depth_arr

    image_mask   = np.expand_dims(image_mask, axis=(2,))  # np.array [720 x 1280 x 1]
    color_masked = image_mask * color_arr                 # boardcasting by np

    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', color_masked)
    cv2.waitKey(1)



if __name__ == "__main__":
    # Create a pipeline
    pipeline = rs.pipeline()

    # Configuration
    config   = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16 , 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Initialize RealSense pipeline
    profile  = pipeline.start(config)
    
    # Camera configuration from pipeline
    metadata = {
        "scale": getDepthScale(profile),
        "align": rs.align(rs.stream.color),
        "intrinsic": getIntrinsic(profile)
    }

    try:
        # while True: main_loop_fn(pipeline, metadata)
        capture_one_frame(pipeline, metadata, "data/cloud3")
    finally:
        print("Program Exit. stopping RealSense pipeline ...")
        pipeline.stop()

