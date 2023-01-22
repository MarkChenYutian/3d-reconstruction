import time
import numpy as np
import open3d as o3d


def to_tcloud(cloud):
    tcloud = o3d.t.geometry.PointCloud()
    tcloud.point.positions = o3d.core.Tensor(np.asarray(cloud.points))
    tcloud.point.colors = o3d.core.Tensor(np.asarray(cloud.colors))
    return tcloud


def gpu_icp(target, source, dist, init=np.eye(4, 4)):
    source, target = to_tcloud(source), to_tcloud(target)
    target.estimate_normals()
    init = o3d.core.Tensor.from_numpy(init)
    estimation_method = (
        o3d.t.pipelines.registration.TransformationEstimationForColoredICP()
    )
    start = time.time()
    res = o3d.t.pipelines.registration.icp(
        source,
        target,
        dist,
        init,
        estimation_method,
    )
    print("ICP time: {}".format(time.time() - start))

    return res.transformation.numpy()
