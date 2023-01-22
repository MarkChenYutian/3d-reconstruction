import time
import numpy as np
import open3d as o3d


def regular_icp(source, target):
    target.estimate_normals()
    # init = np.random.rand(4, 4)
    start = time.time()
    res = o3d.pipelines.registration.registration_colored_icp(
        source,
        target,
        30000,
        # init
    )
    print("ICP time: {}".format(time.time() - start));

    return res.transformation
