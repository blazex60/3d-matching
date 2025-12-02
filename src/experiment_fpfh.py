import time
from pathlib import Path
import open3d as o3d
import numpy as np

# 既存のモジュールをインポート
# ※このスクリプトは src/ ディレクトリ直下に置くことを想定しています
from ply.ply import Ply
from matcher.ransac import global_registration
from matcher.icp import refine_registration
from utils.setup_logging import setup_logging

logger = setup_logging(__name__)
DATA_DIRECTORY = (Path(__file__).parent / ".." / "3d_data").resolve()


class ConfigurablePly(Ply):
    """
    実験用にパラメータを変更可能なPlyクラス
    """

    def __init__(self, path: Path, voxel_size: float, fpfh_radius_mult: float = 5.0, fpfh_max_nn: int = 100) -> None:
        self.fpfh_radius_mult = fpfh_radius_mult
        self.fpfh_max_nn = fpfh_max_nn
        self.fpfh_time = 0.0
        # 親クラスの__init__を呼ぶと_preprocessが走るため、ここでオーバーライドの準備が必要だが、
        # 設計上_preprocessが__init__内で呼ばれるため、パラメータ設定後に手動でロードする形式に変更するか、
        # あるいは_preprocessをオーバーライドしてその中で計測を行う。

        self.path = path
        if not self.path.exists():
            raise FileNotFoundError(f"Ply file not found: {self.path}")

        self.pcd = self._load(self.path)

        # Preprocess with timing
        start_time = time.perf_counter()
        self.pcd_down, self.pcd_fpfh = self._preprocess_custom(self.pcd, voxel_size)
        end_time = time.perf_counter()
        self.fpfh_time = end_time - start_time

        self._add_normals(self.pcd, voxel_size)

    def _preprocess_custom(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float,
    ) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        # ダウンサンプリング
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30),
        )

        # FPFH計算（パラメータ変更可能箇所）
        radius_feature = voxel_size * self.fpfh_radius_mult
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=self.fpfh_max_nn),
        )
        return pcd_down, pcd_fpfh


def evaluate_result(src_down, tgt_down, transformation, threshold):
    """結果のFitnessとRMSEを評価する"""
    evaluation = o3d.pipelines.registration.evaluate_registration(src_down, tgt_down, threshold, transformation)
    return evaluation.fitness, evaluation.inlier_rmse


def run_experiment():
    voxel_size = 0.01
    src_path = DATA_DIRECTORY / "sample.ply"
    tgt_path = DATA_DIRECTORY / "target.ply"

    print(f"{'=' * 80}")
    print(f"{'Experiment: FPFH Parameters & RANSAC Performance':^80}")
    print(f"{'=' * 80}")
    print(f"{'Case':<30} | {'FPFH Time (s)':<12} | {'RANSAC Time (s)':<12} | {'Fitness':<10} | {'RMSE':<10}")
    print(f"{'-' * 80}")

    # 実験設定リスト
    # (ケース名, radius倍率, max_nn)
    settings = [
        ("Standard (5.0x, 100)", 5.0, 100),
        ("Small Radius (2.5x, 100)", 2.5, 100),
        ("Large Radius (10.0x, 100)", 10.0, 100),
        ("Low Neighbors (5.0x, 30)", 5.0, 30),
        ("High Neighbors (5.0x, 200)", 5.0, 200),
    ]

    results = []

    for name, rad_mult, max_nn in settings:
        # 1. データ読み込みとFPFH計算（時間計測含む）
        src_ply = ConfigurablePly(src_path, voxel_size, fpfh_radius_mult=rad_mult, fpfh_max_nn=max_nn)
        tgt_ply = ConfigurablePly(tgt_path, voxel_size, fpfh_radius_mult=rad_mult, fpfh_max_nn=max_nn)

        total_fpfh_time = src_ply.fpfh_time + tgt_ply.fpfh_time

        # 2. RANSAC実行（時間計測）
        start_ransac = time.perf_counter()
        result_trans = global_registration(src_ply, tgt_ply, voxel_size)
        end_ransac = time.perf_counter()
        ransac_time = end_ransac - start_ransac

        # 3. 評価
        fitness, rmse = evaluate_result(src_ply.pcd_down, tgt_ply.pcd_down, result_trans, voxel_size * 1.5)

        print(f"{name:<30} | {total_fpfh_time:<12.4f} | {ransac_time:<12.4f} | {fitness:<10.4f} | {rmse:<10.4f}")
        results.append((name, total_fpfh_time, ransac_time, fitness))

    # 4. 比較対象：FPFH/RANSACを用いない場合（初期位置のまま、あるいはランダム）
    # Open3DのGlobal RANSACは特徴量が必須のため、「FPFHを用いないRANSAC」は定義できません。
    # 代わりに「Global Registrationを行わなかった場合（Identity）」をベースラインとして表示します。
    print(f"{'-' * 80}")
    src_ply_base = ConfigurablePly(src_path, voxel_size)
    tgt_ply_base = ConfigurablePly(tgt_path, voxel_size)
    identity_trans = np.identity(4)
    fit_base, rmse_base = evaluate_result(
        src_ply_base.pcd_down, tgt_ply_base.pcd_down, identity_trans, voxel_size * 1.5
    )

    print(
        f"{'No Global Reg (Baseline)':<30} | {'0.0000':<12} | {'0.0000':<12} | {fit_base:<10.4f} | {rmse_base:<10.4f}"
    )
    print(f"{'=' * 80}")

    # 結果の考察を出力
    print("\n--- Summary & Insights ---")
    best_fit = max(results, key=lambda x: x[3])
    fastest_fpfh = min(results, key=lambda x: x[1])

    print(f"Best Fitness: {best_fit[3]:.4f} (Case: {best_fit[0]})")
    print(f"Fastest FPFH Calculation: {fastest_fpfh[1]:.4f}s (Case: {fastest_fpfh[0]})")
    print("\nNote: 'FPFHを用いないRANSAC'について")
    print("Open3Dの global_registration (RANSAC) は特徴量マッチングに基づくため、FPFH等の特徴量が必須です。")
    print("上記 'No Global Reg' は、FPFH/RANSACを使用せずに初期位置から開始した場合のスコアであり、")
    print("FPFHを用いたGlobal Registrationがいかに初期位置合わせに貢献しているかを示します。")


if __name__ == "__main__":
    run_experiment()
