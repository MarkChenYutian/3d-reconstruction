import pyrealsense2 as rs
import numpy as np
import cv2


# Get the depth scale of [color sensor, depth sensor]
def getDepthScale(profile):
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    return depth_scale

def getIntrinsic(profile):
    streams = profile.get_streams()
    depth_intrinsic = None
    color_intrinsic = None
    
    for stream in streams:
        if stream.stream_type() == rs.stream.depth:
            depth_intrinsic = stream.as_video_stream_profile().intrinsics
        elif stream.stream_type() == rs.stream.color:
            color_intrinsic = stream.as_video_stream_profile().intrinsics
    
    return color_intrinsic, depth_intrinsic


# remove noise points in depth mask by running nms
# depth_mask: np.array [720 x 1280] dtype = uint8
def removeMaskNoise(depth_mask, kernel_size=3):
    dilated_mask = cv2.dilate(depth_mask, np.ones(kernel_size))
    eroded_mask  = cv2.erode(dilated_mask, np.ones(kernel_size))
    return eroded_mask


def deprojectPixelToPoints(intrinsic, depth_arr, x_range=(0, 1280), y_range=(0, 720), step=20):
    mesh_x, mesh_y = np.meshgrid(
        np.arange(x_range[0], x_range[1], step),
        np.arange(y_range[0], y_range[1], step)
    )

    mesh_x = np.expand_dims(mesh_x.flatten(), axis=(1,))
    mesh_y = np.expand_dims(mesh_y.flatten(), axis=(1,))
    pts    = np.hstack((mesh_x, mesh_y)).tolist()
    deproject = lambda pt: rs.rs2_deproject_pixel_to_point(intrinsic[0], pt, depth_arr[640, 360])
    return list(map(deproject, pts))
