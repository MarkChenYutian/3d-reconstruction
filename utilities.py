import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d


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


def postprocessDepth(depth_frame):
    # Depth - Disparsity conversion
    depth_to_disparity = rs.disparity_transform(True )
    disparity_to_depth = rs.disparity_transform(False)
    
    # Domain-transform Edge-preserving Smoothing
    spatial = rs.spatial_filter()

    depth_frame = depth_to_disparity.process(depth_frame)
    depth_frame = spatial.process(depth_frame)
    depth_frame = disparity_to_depth.process(depth_frame)


    return depth_frame


def deprojectPixelToPoints(intrinsic, depth_arr, x_range=(0, 1280), y_range=(0, 720), step=20):
    mesh_x, mesh_y = np.meshgrid(
        np.arange(x_range[0], x_range[1], step),
        np.arange(y_range[0], y_range[1], step)
    )

    mesh_x = np.expand_dims(mesh_x.flatten(), axis=(1,))
    mesh_y = np.expand_dims(mesh_y.flatten(), axis=(1,))
    pts    = np.hstack((mesh_x, mesh_y)).tolist()

    def deproject(pt):
        return rs.rs2_deproject_pixel_to_point(intrinsic[1], pt, depth_arr[pt[1], pt[0]])

    return np.array(list(map(deproject, pts)))


def samplingColor(color_arr, x_range=(0, 1280), y_range=(0, 720),  step=1):
    mesh_x, mesh_y = np.meshgrid(
        np.arange(x_range[0], x_range[1], step),
        np.arange(y_range[0], y_range[1], step)
    )
    mesh_x = np.expand_dims(mesh_x.flatten(), axis=(1,))
    mesh_y = np.expand_dims(mesh_y.flatten(), axis=(1,))
    pts    = np.hstack((mesh_y, mesh_x))
    print(pts.shape)
    return color_arr[pts[:, 0], pts[:, 1]]


def visualize_npz_file(fileName: str) -> None:
    npz = np.load(fileName)
    pts_3d, pts_rgb = npz["points"], npz["image"]

    pts_cloud  = o3d.geometry.PointCloud()
    pts_cloud.points = o3d.utility.Vector3dVector(pts_3d)
    pts_cloud.colors = o3d.utility.Vector3dVector(pts_rgb)

    o3d.visualization.draw_geometries([pts_cloud])

def removeOutlier(pointcloud):
    pointcloud_down = pointcloud.voxel_down_sample(voxel_size = 0.2)
    # cl, ind = pointcloud_down.remove_radius_outlier(nb_points=32, radius=5)
    cl, ind = pointcloud_down.remove_statistical_outlier(nb_neighbors=20,
                                                         std_ratio=2.0)
    return cl

if __name__ == "__main__":
    visualize_npz_file("./data/sparse_1.npz")
