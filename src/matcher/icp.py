from numpy import ndarray
from open3d import pipelines

from ply import Ply
from utils.setup_logging import setup_logging
from utils.timer import BenchmarkResults, timer

logger = setup_logging(__name__)


def refine_registration(
    src: Ply,
    tgt: Ply,
    init_trans: ndarray,
    voxel_size: float,
    benchmark_results: BenchmarkResults | None = None,
) -> pipelines.registration.RegistrationResult:
    """ICPによる精密位置合わせ.

    Args:
        src: ソース点群
        tgt: ターゲット点群
        init_trans: 初期変換行列
        voxel_size: ボクセルサイズ
        benchmark_results: ベンチマーク結果を格納するオブジェクト

    Returns:
        位置合わせ結果
    """
    dist_thresh = voxel_size * 0.4

    with timer("ICP refinement") as t:
        result = pipelines.registration.registration_icp(
            src.pcd,
            tgt.pcd,
            dist_thresh,
            init_trans,
            pipelines.registration.TransformationEstimationPointToPlane(),
        )

    if benchmark_results:
        benchmark_results.add_timing("ICP", t.elapsed_seconds)
        benchmark_results.add_fitness("ICP", result.fitness)
        benchmark_results.add_rmse("ICP", result.inlier_rmse)

    logger.info("ICP refinement: %s", result)
    return result
