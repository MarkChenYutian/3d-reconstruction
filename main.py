import pyrealsense2 as rs
import numpy as np
import cv2

from utilities import getDepthScale, removeMaskNoise, getIntrinsic, deprojectPixelToPoints


THRESHOLD = 1


def capture_one_frame(pipeline, metadata, name="realsense"):
    align = metadata["align"]
    scale = metadata["scale"]
    rs_K  = metadata["intrinsic"]

    frame = pipeline.wait_for_frames()

    # depth to color alignment
    aligned_frames = align.process(frame)

    depth = aligned_frames.get_depth_frame()
    color = aligned_frames.get_color_frame()

    if not depth or not color: print("Fail to get depth/color frame")

    depth_arr = np.asanyarray(depth.get_data()) # np.array [720 x 1280    ] dtype = uint16
    color_arr = np.asanyarray(color.get_data()) # np.array [720 x 1280 x 3] dtype = uint8

    intrinsic = {
        "coeffs": rs_K[0].coeffs,
        "fx"    : rs_K[0].fx,
        "fy"    : rs_K[0].fy,
        "height": rs_K[0].height,
        "width" : rs_K[0].width,
        "ppx"   : rs_K[0].ppx,
        "ppy"   : rs_K[0].ppy,
        "model" : rs_K[0].model
    }

    print(intrinsic)

    np.savez(name, depth=depth_arr, color=color_arr)


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

    deprojectPixelToPoints(metadata["intrinsic"], depth_masked)

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
        while True: main_loop_fn(pipeline, metadata)
        # capture_one_frame(pipeline, metadata, "data/rs3")
    finally:
        print("Program Exit. stopping RealSense pipeline ...")
        pipeline.stop()

