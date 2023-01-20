import pyrealsense2 as rs
import numpy as np
import open3d as o3d

from utilities import getDepthScale, getIntrinsic, deprojectPixelToPoints, postprocessDepth, removeOutlier, samplingColor


THRESHOLD = 1   # maximum distance, in meter
AVG_FRAME = 15  # number of consecutive frames used to perform averaging
PRE_FRAME = 15  # number of frames used for preheat (adjust exposure, etc)
PTS_SPACE = 5   # number of pixel skipped between two spatial point


def capture_one_frame(pipeline, metadata, name="realsense"):
    align = metadata["align"]
    scale = metadata["scale"]
    intrinsic  = metadata["intrinsic"]
    temporal_avg = rs.temporal_filter()

    depth_frames = []
    aligned_frame = None
    for i in range(AVG_FRAME):
        frame = pipeline.wait_for_frames()
        # depth to color alignment
        aligned_frame = align.process(frame)
        depth_frames.append(aligned_frame.get_depth_frame())

    depth = None
    for i in range(AVG_FRAME):
        depth = temporal_avg.process(depth_frames[i])
    
    color = aligned_frame.get_color_frame()

    if not depth or not color: print("Fail to get depth/color frame")

    depth     = postprocessDepth(depth)
    depth_arr = np.asanyarray(depth.get_data()) # np.array [720 x 1280    ] dtype = uint16
    color_arr = np.asanyarray(color.get_data()) # np.array [720 x 1280 x 3] dtype = uint8

    valid_mask = np.where((depth_arr >  (THRESHOLD / scale)) | (depth_arr <= 0), 0, 1).astype(np.uint8)
    depth_arr  = np.where(valid_mask == 0, 0, depth_arr)
    color_arr  = np.where(valid_mask[:, :, np.newaxis] == 0, 0, color_arr)

    pts_3d     = deprojectPixelToPoints(intrinsic, depth_arr, step=PTS_SPACE)
    color_arr  = samplingColor(color_arr, step=PTS_SPACE)
    pts_bgr    = (color_arr / 255).flatten().reshape(-1, 3)
    pts_rgb    = np.hstack((pts_bgr[:, 2:3], pts_bgr[:, 1:2], pts_bgr[:, 0:1]))

    pts_cloud  = o3d.geometry.PointCloud()
    pts_cloud.points = o3d.utility.Vector3dVector(pts_3d)
    pts_cloud.colors = o3d.utility.Vector3dVector(pts_rgb)

    pts_cloud  = removeOutlier(pts_cloud)

    # add normal estimation, for surface reconstruction
    pts_cloud.estimate_normals()
    o3d.visualization.draw_geometries([pts_cloud])

    # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pts_cloud)
    # o3d.visualization.draw_geometries([tetra_mesh])


    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pts_cloud, 0.3)
    # o3d.visualization.draw_geometries([mesh])
    
    np.savez(name, points = pts_cloud.points, image = pts_cloud.colors, step=PTS_SPACE)


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
        # Preheat - allow auto-exposure to adjust
        # Skip first several frames to give the Auto-Exposure time to adjust
        for x in range(PRE_FRAME):
            pipeline.wait_for_frames()
        
        # Create a capture
        capture_one_frame(pipeline, metadata, "data/rotate_1")
    finally:
        print("Program Exit. stopping RealSense pipeline ...")
        pipeline.stop()

