import numpy as np
import open3d as o3d
from open3d import pipelines

from ply import Ply
from utils.setup_logging import setup_logging
from utils.timer import BenchmarkResults, timer

logger = setup_logging(__name__)


def global_registration(
    src: Ply,
    tgt: Ply,
    voxel_size: float,
    iteration: int = 30,
    benchmark_results: BenchmarkResults | None = None,
) -> pipelines.registration.RegistrationResult:
    """FPFHを使用したRANSACベースのグローバル位置合わせ.

    Args:
        src: ソース点群
        tgt: ターゲット点群
        voxel_size: ボクセルサイズ
        iteration: RANSACのイテレーション数
        benchmark_results: ベンチマーク結果を格納するオブジェクト

    Returns:
        位置合わせ結果
    """
    if src.pcd_fpfh is None or tgt.pcd_fpfh is None:
        msg = "FPFH features are required for global_registration. Set use_fpfh=True when creating Ply objects."
        raise ValueError(msg)

    dist_thresh = voxel_size * 1.5

    with timer("RANSAC (with FPFH)") as t:
        result = pipelines.registration.registration_ransac_based_on_feature_matching(
            src.pcd_down,
            tgt.pcd_down,
            src.pcd_fpfh,
            tgt.pcd_fpfh,
            True,  # noqa: FBT003
            dist_thresh,
            pipelines.registration.TransformationEstimationPointToPoint(False),  # noqa: FBT003
            3,
            [
                pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
            ],
            pipelines.registration.RANSACConvergenceCriteria(iteration, 0.999),
        )

    if benchmark_results:
        benchmark_results.add_timing("RANSAC (with FPFH)", t.elapsed_seconds)
        benchmark_results.add_fitness("RANSAC (with FPFH)", result.fitness)
        benchmark_results.add_rmse("RANSAC (with FPFH)", result.inlier_rmse)

    logger.info("Global RANSAC result: %s", result)
    return result


def global_registration_without_fpfh(
    src: Ply,
    tgt: Ply,
    voxel_size: float,
    iteration: int = 30,
    benchmark_results: BenchmarkResults | None = None,
) -> pipelines.registration.RegistrationResult:
    """FPFHを使用しないRANSACベースのグローバル位置合わせ.

    最近傍点探索で対応点を生成し、RANSACで外れ値を除去します。

    Args:
        src: ソース点群
        tgt: ターゲット点群
        voxel_size: ボクセルサイズ
        iteration: RANSACのイテレーション数
        benchmark_results: ベンチマーク結果を格納するオブジェクト

    Returns:
        位置合わせ結果
    """
    dist_thresh = voxel_size * 1.5

    with timer("RANSAC (without FPFH)") as t:
        # KDTreeを使って最近傍点を探索し対応点を生成
        tgt_tree = o3d.geometry.KDTreeFlann(tgt.pcd_down)
        correspondences = []

        src_points = np.asarray(src.pcd_down.points)
        for i, point in enumerate(src_points):
            _, idx, _ = tgt_tree.search_knn_vector_3d(point, 1)
            correspondences.append([i, idx[0]])

        correspondences_vec = o3d.utility.Vector2iVector(correspondences)

        result = pipelines.registration.registration_ransac_based_on_correspondence(
            src.pcd_down,
            tgt.pcd_down,
            correspondences_vec,
            dist_thresh,
            pipelines.registration.TransformationEstimationPointToPoint(False),  # noqa: FBT003
            3,
            [
                pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh),
            ],
            pipelines.registration.RANSACConvergenceCriteria(iteration, 0.999),
        )

    if benchmark_results:
        benchmark_results.add_timing("RANSAC (without FPFH)", t.elapsed_seconds)
        benchmark_results.add_fitness("RANSAC (without FPFH)", result.fitness)
        benchmark_results.add_rmse("RANSAC (without FPFH)", result.inlier_rmse)

    logger.info("Global RANSAC (without FPFH) result: %s", result)
    return result
