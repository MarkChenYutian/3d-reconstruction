import numpy as np
import open3d as o3d


from utilities import load_cloud, mergeClouds, removePlane
from cpu_icp import cpu_icp

icp = cpu_icp


def merge():
    clouds = [load_cloud("demo_noground/real_{}".format(i)) for i in range(22)]
    # dists = {3: 5, 10: 8, 11: 13, 12: 25, 13: 10}  # 13: bad
    dists = {20: 10000, 21: 10000}
    # dists = dict()
    transitions = [icp(clouds[i], clouds[(i + 1) % 22], dists.get(i, 35)) for i in range(22)]

    print("Optimization")

    # spot correction for 13
    t = np.eye(4, 4)
    start = 1
    for i in range(21):
        t = t @ transitions[start % 22]
        start += 1
    t = np.linalg.inv(t)
    transitions[0] = icp(clouds[0], clouds[1], 5, init=t)

    transitions_updated = []
    for i in range(22):
        t = np.eye(4, 4)
        for j in range(21):
            t = t @ transitions[(i + j + 1) % 22]
        t = np.linalg.inv(t)
        transitions_updated.append(icp(clouds[i], clouds[(i + 1) % 22], 5, init=t))

    start = 1
    t = np.eye(4, 4)
    prefixes = []
    geometries = []
    for i in range(22):
        prefixes.append(t)
        t = t @ transitions_updated[(start + i) % 22]
    for i in range(0, 22, 6):
        geometries.append(clouds[(start + i) % 22].transform(prefixes[i]))
    
    merged = mergeClouds(geometries)

    np.savez("./data/demo_export", points = merged.points, image = merged.colors)
    
    o3d.visualization.draw_geometries([merged])


if __name__ == "__main__":
    merge()
