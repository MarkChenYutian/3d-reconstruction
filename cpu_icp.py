import time
import numpy as np
import open3d as o3d


def cpu_icp(target, source, dist, init=np.eye(4, 4)):
    target.estimate_normals()
    start = time.time()
    res = o3d.pipelines.registration.registration_colored_icp(
        source,
        target,
        dist,
        init,
    )
    print("ICP time: {}".format(time.time() - start))

    return res.transformation
