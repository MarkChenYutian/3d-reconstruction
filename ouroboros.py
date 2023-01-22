import open3d as o3d


from utilities import load_cloud
from cpu_icp import cpu_icp


def merge():
    zero = [load_cloud("full_noground/real_{}".format(i)) for i in range(24)]
    temp = [load_cloud("full_noground/real_{}".format(i)) for i in range(24)]
    D = {3: 5, 10: 8, 11: 13, 12: 25, 13: 1}  # 13: bad
    for i in range(24):
        t = cpu_icp(zero[i], zero[(i + 1) % 24], D.get(i, 35))
        o3d.visualization.draw_geometries([zero[i], temp[(i + 1) % 24].transform(t)])


if __name__ == "__main__":
    merge()
