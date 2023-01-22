import numpy as np
import open3d as o3d

def visualize_npz_file(fileName: str) -> None:
    npz = np.load(fileName)
    pts_3d, pts_rgb = npz["points"], npz["image"]

    pts_cloud  = o3d.geometry.PointCloud()
    pts_cloud.points = o3d.utility.Vector3dVector(pts_3d)
    pts_cloud.colors = o3d.utility.Vector3dVector(pts_rgb)

    o3d.visualization.draw_geometries([pts_cloud])

visualize_npz_file("./data/full_export.npz")

