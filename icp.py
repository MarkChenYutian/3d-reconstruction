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
    max_correspondence_distance = 10000
    estimation_method = o3d.t.pipelines.registration.TransformationEstimationForColoredICP()

    start = time.time()
    res = o3d.t.pipelines.registration.icp(source, target, max_correspondence_distance, estimation_method=estimation_method)
    print(time.time() - start)

    source_new = source.clone()
    source_new.transform(res.transformation)

    o3d.visualization.draw_geometries([source.to_legacy(), target.to_legacy()])
    o3d.visualization.draw_geometries([source_new.to_legacy(), target.to_legacy()])


def main():
    source = "sparse_1"
    target = "sparse_2"
    icp(source, target)


if __name__ == "__main__":
    main()
