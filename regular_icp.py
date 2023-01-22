import time
import open3d as o3d


def regular_icp(source, target):
    target.estimate_normals()
    start = time.time()
    res = o3d.pipelines.registration.registration_colored_icp(
        source,
        target,
        300
    )
    print("ICP time: {}".format(time.time() - start));

    return res.transformation
