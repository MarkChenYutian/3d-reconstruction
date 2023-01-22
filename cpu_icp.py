import time
import numpy as np
import open3d as o3d

from utilities import sample_cloud


def cpu_icp(target, source, dist, init=np.eye(4, 4)):
    # source = sample_cloud(source, step=2)
    # target = sample_cloud(target, step=2)
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
