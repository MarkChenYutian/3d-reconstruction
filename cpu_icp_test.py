import numpy as np
import open3d as o3d


from utilities import load_cloud
from cpu_icp import cpu_icp


def clean(cloud, preview=False, bogus=False):
    if not bogus:
        cloud, _ = cloud.remove_radius_outlier(nb_points=100, radius=10)
    if preview:
        o3d.visualization.draw_geometries([cloud])
    return cloud


def merge(target, source, preview=False, dist=35):
    # source = source.uniform_down_sample(1)
    # target = target.uniform_down_sample(1)
    T = cpu_icp(source, target, dist)
    # T = np.eye(4, 4)
    # print(T)

    # source_t = source.transform(T)

    # source_pts = np.asarray(source_t.points)
    # source_col = np.asarray(source_t.colors)
    # target_pts = np.asarray(target.points)
    # target_col = np.asarray(target.colors)
    # merge_pts = np.concatenate([source_pts, target_pts], axis=0)
    # merge_col = np.concatenate([source_col, target_col], axis=0)
    #
    # new_cloud = o3d.geometry.PointCloud()
    # new_cloud.points = o3d.utility.Vector3dVector(merge_pts)
    # new_cloud.colors = o3d.utility.Vector3dVector(merge_col)
    #
    # # new_cloud = new_cloud.uniform_down_sample(2)
    #
    # if preview:
    #     o3d.visualization.draw_geometries([new_cloud])

    return T
    # return new_cloud


def merge_tier():
    zero = [
        clean(load_cloud("full_lessground/real_{}".format(i)), False, bogus)
        for i, bogus in zip(range(4), [False, False, True, True])
    ]
    T01 = merge(zero[0], zero[1], False, 35)
    T12 = merge(zero[1], zero[2], False, 100)
    T23 = merge(zero[2], zero[3], False, 100)

    o3d.visualization.draw_geometries(
        [
            zero[0],
            zero[1].transform(T01),
            zero[2].transform(T12).transform(T01),
            zero[3].transform(T23).transform(T12).transform(T01),
        ]
    )

    # one = [
    #     merge(zero[i], zero[(i + 1) % 24], True, dist)
    #     for i, dist in zip(range(3), [35, 100, 200])
    # ]
    # 0 1 2 3


if __name__ == "__main__":
    merge_tier()
    # o3d.visualization.draw_geometries(B)
