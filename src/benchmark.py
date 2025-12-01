"""FPFHベンチマークスクリプト.

FPFHを用いた場合と用いなかった場合の処理時間・精度の比較を行います。
KDTreeパラメータの影響も調査できます。
"""

from pathlib import Path

from matcher.icp import refine_registration
from matcher.ransac import global_registration, global_registration_without_fpfh
from ply import KDTreeParams, Ply
from utils.setup_logging import setup_logging
from utils.timer import BenchmarkResults

logger = setup_logging(__name__)

DATA_DIRECTORY = (Path(__file__).parent / ".." / "3d_data").resolve()


def run_benchmark_with_fpfh(
    src_path: Path,
    tgt_path: Path,
    voxel_size: float,
    kdtree_params: KDTreeParams,
    iteration: int = 30,
) -> BenchmarkResults:
    """FPFHを使用したベンチマーク.

    Args:
        src_path: ソース点群ファイルパス
        tgt_path: ターゲット点群ファイルパス
        voxel_size: ボクセルサイズ
        kdtree_params: KDTreeのパラメータ
        iteration: RANSACのイテレーション数

    Returns:
        ベンチマーク結果
    """
    results = BenchmarkResults()

    # FPFH計算を含む前処理 (時間計測はPly内部で行われる)
    src_ply = Ply(
        src_path,
        voxel_size,
        use_fpfh=True,
        kdtree_params=kdtree_params,
        benchmark_results=results,
    )
    tgt_ply = Ply(
        tgt_path,
        voxel_size,
        use_fpfh=True,
        kdtree_params=kdtree_params,
        benchmark_results=results,
    )

    # RANSAC (FPFHを使用)
    ransac_result = global_registration(
        src_ply,
        tgt_ply,
        voxel_size,
        iteration,
        benchmark_results=results,
    )

    # ICP精密位置合わせ
    refine_registration(
        src_ply,
        tgt_ply,
        ransac_result.transformation,
        voxel_size,
        benchmark_results=results,
    )

    return results


def run_benchmark_without_fpfh(
    src_path: Path,
    tgt_path: Path,
    voxel_size: float,
    iteration: int = 30,
) -> BenchmarkResults:
    """FPFHを使用しないベンチマーク.

    Args:
        src_path: ソース点群ファイルパス
        tgt_path: ターゲット点群ファイルパス
        voxel_size: ボクセルサイズ
        iteration: RANSACのイテレーション数

    Returns:
        ベンチマーク結果
    """
    results = BenchmarkResults()

    # FPFHなしで前処理
    src_ply = Ply(
        src_path,
        voxel_size,
        use_fpfh=False,
        benchmark_results=results,
    )
    tgt_ply = Ply(
        tgt_path,
        voxel_size,
        use_fpfh=False,
        benchmark_results=results,
    )

    # RANSAC (対応点ベース、FPFHなし)
    ransac_result = global_registration_without_fpfh(
        src_ply,
        tgt_ply,
        voxel_size,
        iteration,
        benchmark_results=results,
    )

    # ICP精密位置合わせ
    refine_registration(
        src_ply,
        tgt_ply,
        ransac_result.transformation,
        voxel_size,
        benchmark_results=results,
    )

    return results


def run_kdtree_params_comparison(
    src_path: Path,
    tgt_path: Path,
    voxel_size: float,
) -> dict[str, BenchmarkResults]:
    """異なるKDTreeパラメータでのベンチマーク比較.

    Args:
        src_path: ソース点群ファイルパス
        tgt_path: ターゲット点群ファイルパス
        voxel_size: ボクセルサイズ

    Returns:
        パラメータ名をキーとしたベンチマーク結果の辞書
    """
    param_sets = {
        "default": KDTreeParams(),
        "small_radius": KDTreeParams(
            normal_radius_multiplier=1.5,
            fpfh_radius_multiplier=3.0,
        ),
        "large_radius": KDTreeParams(
            normal_radius_multiplier=3.0,
            fpfh_radius_multiplier=7.0,
        ),
        "small_nn": KDTreeParams(
            normal_max_nn=15,
            fpfh_max_nn=50,
        ),
        "large_nn": KDTreeParams(
            normal_max_nn=60,
            fpfh_max_nn=200,
        ),
    }

    all_results = {}
    for name, params in param_sets.items():
        logger.info("Running benchmark with KDTree params: %s", name)
        all_results[name] = run_benchmark_with_fpfh(src_path, tgt_path, voxel_size, params)

    return all_results


def print_comparison_report(
    with_fpfh: BenchmarkResults,
    without_fpfh: BenchmarkResults,
) -> None:
    """FPFHあり/なしの比較レポートを出力.

    Args:
        with_fpfh: FPFHを使用した場合の結果
        without_fpfh: FPFHを使用しなかった場合の結果
    """
    print("\n" + "=" * 70)
    print("FPFH Comparison Report")
    print("=" * 70)

    print("\n[With FPFH]")
    print(with_fpfh.summary())

    print("\n[Without FPFH]")
    print(without_fpfh.summary())

    # 差分の計算
    print("\n[Comparison]")

    # 処理時間の比較
    fpfh_total = sum(t for k, t in with_fpfh.timings.items() if "FPFH" in k)
    ransac_with = with_fpfh.timings.get("RANSAC (with FPFH)", 0)
    ransac_without = without_fpfh.timings.get("RANSAC (without FPFH)", 0)

    print(f"  FPFH calculation time: {fpfh_total:.4f} seconds")
    print(f"  RANSAC (with FPFH): {ransac_with:.4f} seconds")
    print(f"  RANSAC (without FPFH): {ransac_without:.4f} seconds")
    print(
        f"  RANSAC speedup without FPFH: {ransac_with / ransac_without:.2f}x"
        if ransac_without > 0
        else "  RANSAC speedup: N/A"
    )

    # Fitness比較
    fitness_with = with_fpfh.fitness_scores.get("RANSAC (with FPFH)", 0)
    fitness_without = without_fpfh.fitness_scores.get("RANSAC (without FPFH)", 0)

    print(f"\n  Fitness (with FPFH): {fitness_with:.6f}")
    print(f"  Fitness (without FPFH): {fitness_without:.6f}")
    print(f"  Fitness difference: {fitness_with - fitness_without:.6f}")

    print("=" * 70)


def print_kdtree_comparison_report(results: dict[str, BenchmarkResults]) -> None:
    """KDTreeパラメータ比較レポートを出力.

    Args:
        results: パラメータ名をキーとした結果辞書
    """
    print("\n" + "=" * 70)
    print("KDTree Parameters Comparison Report")
    print("=" * 70)

    # ヘッダー
    print(f"\n{'Params':<15} {'FPFH Time':>12} {'RANSAC Time':>12} {'Fitness':>12} {'RMSE':>12}")
    print("-" * 70)

    for name, result in results.items():
        fpfh_time = sum(t for k, t in result.timings.items() if "FPFH" in k)
        ransac_time = result.timings.get("RANSAC (with FPFH)", 0)
        fitness = result.fitness_scores.get("RANSAC (with FPFH)", 0)
        rmse = result.inlier_rmse.get("RANSAC (with FPFH)", 0)

        print(f"{name:<15} {fpfh_time:>12.4f} {ransac_time:>12.4f} {fitness:>12.6f} {rmse:>12.6f}")

    print("=" * 70)


def main() -> None:
    """ベンチマークのメイン関数."""
    import argparse

    parser = argparse.ArgumentParser(description="FPFH Benchmark")
    parser.add_argument(
        "--mode",
        choices=["compare", "kdtree"],
        default="compare",
        help="Benchmark mode: 'compare' for FPFH vs non-FPFH, 'kdtree' for KDTree params comparison",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=DATA_DIRECTORY / "source.ply",
        help="Source PLY file path",
    )
    parser.add_argument(
        "--tgt",
        type=Path,
        default=DATA_DIRECTORY / "target.ply",
        help="Target PLY file path",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        help="Voxel size for downsampling",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=30,
        help="RANSAC iteration count",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        logger.info("Running FPFH comparison benchmark...")

        # FPFHあり
        logger.info("=== With FPFH ===")
        with_fpfh = run_benchmark_with_fpfh(
            args.src,
            args.tgt,
            args.voxel_size,
            KDTreeParams(),
            args.iteration,
        )

        # FPFHなし
        logger.info("=== Without FPFH ===")
        without_fpfh = run_benchmark_without_fpfh(
            args.src,
            args.tgt,
            args.voxel_size,
            args.iteration,
        )

        print_comparison_report(with_fpfh, without_fpfh)

    elif args.mode == "kdtree":
        logger.info("Running KDTree parameters comparison benchmark...")
        results = run_kdtree_params_comparison(args.src, args.tgt, args.voxel_size)
        print_kdtree_comparison_report(results)


if __name__ == "__main__":
    main()
