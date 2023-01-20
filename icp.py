import numpy as np
import open3d as o3d
import time


def to_tensor(vec):
    return o3d.core.Tensor(np.asarray(vec))


def load_cloud(name):
    npz = np.load("data/{}.npz".format(name))
    points, image = npz["points"], npz["image"]
    bgr = (image / 256).flatten().reshape(720 * 1280, 3)
    rgb = np.hstack((bgr[:, 2:3], bgr[:, 1:2], bgr[:, 0:1]))

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(rgb)

    # o3d.visualization.draw_geometries([cloud])
    return cloud


def to_tcloud(cloud):
    tcloud = o3d.t.geometry.PointCloud()
    tcloud.point.positions = to_tensor(cloud.points)
    tcloud.point.colors = to_tensor(cloud.colors)
    return tcloud


def icp(source, target):
    print("Load point-clouds: ", time.time())
    source, target = to_tcloud(load_cloud(source)), to_tcloud(load_cloud(target))
    max_correspondence_distance = 10000
    # callback_after_iteration = lambda updated_result_dict: print(
    #     "Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
    #         updated_result_dict["iteration_index"].item(),
    #         updated_result_dict["fitness"].item(),
    #         updated_result_dict["inlier_rmse"].item(),
    #     )
    # )

    print("ICP: ", time.time())
    res = o3d.t.pipelines.registration.icp(source, target, max_correspondence_distance)

    print("Print stats: ", time.time())
    print(" - Fitness: ", res.fitness)
    print(" - RMSE: ", res.inlier_rmse)
    print(" - Transformation: ", res.transformation)

    print("Transform: ", time.time())
    source_new = source.clone()
    source_new.transform(res.transformation)

    print("Visualize: ", time.time())
    o3d.visualization.draw_geometries([source.to_legacy(), source_new.to_legacy()])


def main():
    source = "sparse_1"
    target = "sparse_2"
    icp(source, target)


if __name__ == "__main__":
    main()
