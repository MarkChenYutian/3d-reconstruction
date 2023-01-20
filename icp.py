import numpy as np
import open3d as o3d
import time


def to_tensor(vec):
    return o3d.core.Tensor(np.asarray(vec))


def load_cloud(name):
    npz = np.load("data/{}.npz".format(name))
    points, image = npz["points"], npz["image"]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(image)

    # o3d.visualization.draw_geometries([cloud])
    return cloud


def to_tcloud(cloud):
    tcloud = o3d.t.geometry.PointCloud()
    tcloud.point.positions = to_tensor(cloud.points)
    tcloud.point.colors = to_tensor(cloud.colors)
    return tcloud


def icp(source, target):
    source, target = to_tcloud(load_cloud(source)), to_tcloud(load_cloud(target))
    target.estimate_normals()
    max_correspondence_distance = 1000
    init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)
    # init_source_to_target = np.asarray(
    #     [[1, 0, 0, -3.5], [0, 1, 0, -24], [0, 0, 1, 65], [0, 0, 0, 1]], dtype=float
    # )
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
    voxel_size = 0.5

    start = time.time()
    res = o3d.t.pipelines.registration.icp(
        source,
        target,
        max_correspondence_distance,
        init_source_to_target=init_source_to_target,
        estimation_method=estimation_method,
        # criteria=criteria,
        voxel_size=voxel_size
    )
    print(time.time() - start)
    print(res.transformation)
    print(res.fitness)
    print(res.inlier_rmse)

    source_new = source.clone()
    source_new.transform(res.transformation)

    o3d.visualization.draw_geometries([source.to_legacy(), target.to_legacy()])
    o3d.visualization.draw_geometries([source_new.to_legacy(), target.to_legacy()])


def main():
    source = "rotate_2"
    target = "rotate_3"
    icp(source, target)


if __name__ == "__main__":
    main()
