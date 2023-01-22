import numpy as np
import open3d as o3d


from utilities import load_cloud, mergeClouds
from cpu_icp import cpu_icp

icp = cpu_icp


def merge():
    clouds = [load_cloud("full_noground/real_{}".format(i)) for i in range(24)]
    dists = {3: 5, 10: 8, 11: 13, 12: 25, 13: 10}  # 13: bad
    transitions = [icp(clouds[i], clouds[(i + 1) % 24], dists.get(i, 35)) for i in range(24)]

    # spot correction for 13
    t = np.eye(4, 4)
    start = 14
    for i in range(23):
        t = t @ transitions[start % 24]
        start += 1
    t = np.linalg.inv(t)
    transitions[13] = icp(clouds[13], clouds[14], 5, init=t)

    transitions_updated = []
    for i in range(24):
        t = np.eye(4, 4)
        for j in range(23):
            t = t @ transitions[(i + j + 1) % 24]
        t = np.linalg.inv(t)
        transitions_updated.append(icp(clouds[i], clouds[(i + 1) % 24], 5, init=t))

    start = 10
    t = np.eye(4, 4)
    prefixes = []
    geometries = []
    for i in range(24):
        prefixes.append(t)
        t = t @ transitions_updated[(start + i) % 24]
    for i in range(0, 24, 6):
        geometries.append(clouds[(start + i) % 24].transform(prefixes[i]))
    
    merged = mergeClouds(geometries)

    np.savez("./data/full_export", points = merged.points, image = merged.colors)

    o3d.io.write_point_cloud("./data/full.pts", merged)
    
    o3d.visualization.draw_geometries([merged])


if __name__ == "__main__":
    merge()
