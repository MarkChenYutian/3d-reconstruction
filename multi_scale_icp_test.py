import numpy as np
import open3d as o3d


from utilities import load_cloud, sample_cloud
from regular_icp import regular_icp


def merge(source, target):
    source_down = sample_cloud(source, step=10)
    target_down = sample_cloud(target, step=10)

    T = regular_icp(source_down, target_down)
    print(T)

    source_t = source.transform(T)

    source_pts = np.asarray(source_t.points)
    source_col = np.asarray(source_t.colors)
    target_pts = np.asarray(target.points)
    target_col = np.asarray(target.colors)
    merge_pts = np.concatenate([source_pts, target_pts], axis=0)
    merge_col = np.concatenate([source_col, target_col], axis=0)

    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(merge_pts)
    new_cloud.colors = o3d.utility.Vector3dVector(merge_col)

    new_cloud = new_cloud.voxel_down_sample(2)

    o3d.visualization.draw_geometries([new_cloud])

    return new_cloud


if __name__ == "__main__":
    merged = load_cloud("complex/real_1")
    for i in range(2, 11):
        merged = merge(merged, load_cloud("complex/real_" + str(i)))
