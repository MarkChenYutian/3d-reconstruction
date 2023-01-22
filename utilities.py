import pyrealsense2 as rs
import numpy as np
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


def deprojectPixelToPoints(intrinsic, depth_arr, x_range=(0, 1280), y_range=(0, 720), step=1):
    mesh_x, mesh_y = np.meshgrid(
        np.arange(x_range[0], x_range[1], step),
        np.arange(y_range[0], y_range[1], step)
    )

    mesh_x = np.expand_dims(mesh_x.flatten(), axis=(1,))
    mesh_y = np.expand_dims(mesh_y.flatten(), axis=(1,))
    pts    = np.hstack((mesh_x, mesh_y)).tolist()

    def deproject(pt):
        return rs.rs2_deproject_pixel_to_point(intrinsic[1], pt, depth_arr[pt[1], pt[0]])

    T = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    pts_3d = np.array(list(map(deproject, pts)))
    pts_3d = pts_3d @ T
    return pts_3d


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


def sample_cloud(pts_cloud, step=5):
    pts_vec = np.asarray(pts_cloud.points)
    col_vec = np.asarray(pts_cloud.colors)
    print("Original:", pts_vec.shape)
    pts_idx = np.arange(0, pts_vec.shape[0], step)
    pts_sampled = pts_vec[pts_idx]
    col_sampled = col_vec[pts_idx]

    print("Sampled:", pts_sampled.shape)

    pts_cloud_sampled = o3d.geometry.PointCloud()
    pts_cloud_sampled.points = o3d.utility.Vector3dVector(pts_sampled)
    pts_cloud_sampled.colors = o3d.utility.Vector3dVector(col_sampled)
    return pts_cloud_sampled


def load_cloud(name):
    npz = np.load("data/{}.npz".format(name))
    points, image = npz["points"], npz["image"]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(image)
    
    return cloud

def visualize_npz_file(fileName: str) -> None:
    npz = np.load(fileName)
    pts_3d, pts_rgb = npz["points"], npz["image"]

    pts_cloud  = o3d.geometry.PointCloud()
    pts_cloud.points = o3d.utility.Vector3dVector(pts_3d)
    pts_cloud.colors = o3d.utility.Vector3dVector(pts_rgb)

    o3d.visualization.draw_geometries([pts_cloud])

def removeOutlier(pointcloud):
    pointcloud_down = pointcloud.voxel_down_sample(voxel_size = 0.2)
    cl, ind = pointcloud_down.remove_statistical_outlier(nb_neighbors=20,
                                                         std_ratio=2.0)
    return cl


def removeSmallCluster(pointcloud):
    clusters = np.array(pointcloud.cluster_dbscan(eps=20, min_points=800, print_progress=True))
    bad_pts  = (clusters == -1).nonzero()[0].tolist()
    pointcloud = pointcloud.select_by_index(bad_pts, invert=True)
    return pointcloud


def removePlane(pointcloud):
    plane_model, inliers = pointcloud.segment_plane(distance_threshold=8, ransac_n=3, num_iterations=3000)
    clean_points = pointcloud.select_by_index(inliers, invert=True)
    return removeOutlier(removeSmallCluster(clean_points))


def mergeClouds(clouds):
    pts = o3d.geometry.PointCloud()
    for cloud in clouds:
        new_pts = cloud.points
        new_col = cloud.colors
        all_pts = pts.points
        all_col = pts.colors

        merge_pts  = np.concatenate([all_pts, new_pts], axis=0)
        merge_col  = np.concatenate([all_col, new_col], axis=0)
        pts.points = o3d.utility.Vector3dVector(merge_pts)
        pts.colors = o3d.utility.Vector3dVector(merge_col)

    return pts

if __name__ == "__main__":
    # visualize_npz_file("./data/rotate_3.npz")
    for i in range(24):
        print("Cleaning ... " + str(i))
        cloud = load_cloud("demo/real_" + str(i))
        cloud = removePlane(cloud)
        o3d.visualization.draw_geometries([cloud])

        np.savez("./data/demo_noground/real_" + str(i), points = cloud.points, image = cloud.colors)
