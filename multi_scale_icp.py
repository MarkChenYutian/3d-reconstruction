import time
import open3d as o3d


def multi_scale_icp(source, target):
    target.estimate_normals()
    voxel_sizes = o3d.utility.DoubleVector([1, 0.5, 0.25])
    criteria_list = [
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.0001, 0.0001, 20),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.00001, 0.00001, 15),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(0.000001, 0.000001, 10)
    ]
    max_correspondence_distances = o3d.utility.DoubleVector([300, 140, 70])
    init_source_to_target = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)
    estimation = o3d.t.pipelines.registration.TransformationEstimationForColoredICP()

    start = time.time()
    res = o3d.t.pipelines.registration.multi_scale_icp(
        source,
        target,
        voxel_sizes,
        criteria_list,
        max_correspondence_distances,
        init_source_to_target,
        estimation
    )
    print("ICP time: {}".format(time.time() - start));

    return res.transformation
