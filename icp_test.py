import numpy as np
import open3d as o3d
import time

from utilities import load_cloud, sample_cloud


def to_tensor(vec):
    return o3d.core.Tensor(np.asarray(vec))


def to_tcloud(cloud):
    tcloud = o3d.t.geometry.PointCloud()
    tcloud.point.positions = to_tensor(cloud.points)
    tcloud.point.colors = to_tensor(cloud.colors)
    return tcloud


def icp(source_cloud, target_cloud):
    source, target = to_tcloud(source_cloud), to_tcloud(target_cloud)
    target.estimate_normals()
    max_correspondence_distance = 10000
    init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)

    estimation_method = (
        o3d.t.pipelines.registration.TransformationEstimationForColoredICP()
    )
    # sigma = 10
    # estimation_method = (
    #     o3d.t.pipelines.registration.TransformationEstimationForColoredICP(
    #         o3d.t.pipelines.registration.robust_kernel.RobustKernel(
    #             o3d.t.pipelines.registration.robust_kernel.RobustKernelMethod.TukeyLoss, sigma
    #         )
    #     )
    # )
    # criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=3)

    voxel_size = 1

    res = o3d.t.pipelines.registration.icp(
        source,
        target,
        max_correspondence_distance,
        init_source_to_target=init_source_to_target,
        estimation_method=estimation_method,
        voxel_size=voxel_size,
        # criteria=criteria
    )

    return res.transformation


def merge(source, target):
    source_down = sample_cloud(source, step=50)
    target_down = sample_cloud(target, step=50)

    

    T = icp(source_down, target_down)

    source   = to_tcloud(source)
    source_t = source.transform(T)
    source_t = source_t.to_legacy()

    source_pts = np.asarray(source_t.points)
    source_col = np.asarray(source_t.colors)

    target_pts = np.asarray(target.points)
    target_col = np.asarray(target.colors)

    merge_pts  = np.concatenate([source_pts, target_pts], axis=0)
    merge_col  = np.concatenate([source_col, target_col], axis=0)

    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(merge_pts)
    new_cloud.colors = o3d.utility.Vector3dVector(merge_col)

    new_cloud = new_cloud.voxel_down_sample(2)

    o3d.visualization.draw_geometries([new_cloud])

    return new_cloud
    




if __name__ == "__main__":
    # merged = merge(
    #             merge(load_cloud("real_1"), load_cloud("real_0")),
    #             load_cloud("real_2")
    #          )
    merged = load_cloud("real_0")
    for i in range(1, 11):
        merged = merge(merged, load_cloud("real_" + str(i)))
    # merged = merge(merge(load_cloud("rotate_1"), load_cloud("rotate_2")), load_cloud("rotate_3"))
    
    merged.estimate_normals()
    o3d.visualization.draw_geometries([merged])

    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            # merged, o3d.utility.DoubleVector([25, 50]))
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
