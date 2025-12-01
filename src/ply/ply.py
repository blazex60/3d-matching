from dataclasses import dataclass
from pathlib import Path

import open3d as o3d

from utils.setup_logging import setup_logging
from utils.timer import BenchmarkResults, timer

logger = setup_logging(__name__)


@dataclass
class KDTreeParams:
    """KDTreeSearchParamHybridのパラメータを格納するデータクラス."""

    # 法線推定用パラメータ
    normal_radius_multiplier: float = 2.0
    normal_max_nn: int = 30

    # FPFH計算用パラメータ
    fpfh_radius_multiplier: float = 5.0
    fpfh_max_nn: int = 100


class Ply:
    def __init__(
        self,
        path: Path,
        voxel_size: float,
        *,
        use_fpfh: bool = True,
        kdtree_params: KDTreeParams | None = None,
        benchmark_results: BenchmarkResults | None = None,
    ) -> None:
        self.path = path
        self.kdtree_params = kdtree_params or KDTreeParams()
        self.benchmark_results = benchmark_results
        self.fpfh_time: float = 0.0

        if not self.path.exists():
            msg = f"Ply file not found: {self.path}"
            raise FileNotFoundError(msg)
        if self.path.suffix.lower() != ".ply":
            msg = f"File is not a ply file: {self.path}"
            raise TypeError(msg)

        self.pcd = self._load(self.path)

        if use_fpfh:
            self.pcd_down, self.pcd_fpfh = self._preprocess(self.pcd, voxel_size)
        else:
            self.pcd_down = self.pcd.voxel_down_sample(voxel_size)
            self.pcd_down.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * self.kdtree_params.normal_radius_multiplier,
                    max_nn=self.kdtree_params.normal_max_nn,
                ),
            )
            self.pcd_fpfh = None

        self._add_normals(self.pcd, voxel_size)
        logger.info("Successfully loaded and preprocessed ply file: %s", self.path)

    def _load(self, path: Path) -> o3d.geometry.PointCloud:
        pcd = o3d.io.read_point_cloud(str(path))
        if not pcd.has_points():
            msg = f"Point cloud is empty: {path}"
            logger.error(msg)
            raise ValueError(msg)
        return pcd

    def _preprocess(
        self,
        pcd: o3d.geometry.PointCloud,
        voxel_size: float,
    ) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        pcd_down = pcd.voxel_down_sample(voxel_size)
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * self.kdtree_params.normal_radius_multiplier,
                max_nn=self.kdtree_params.normal_max_nn,
            ),
        )

        # FPFH計算の時間計測
        with timer(f"FPFH calculation ({self.path.name})") as t:
            pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                pcd_down,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=voxel_size * self.kdtree_params.fpfh_radius_multiplier,
                    max_nn=self.kdtree_params.fpfh_max_nn,
                ),
            )
        self.fpfh_time = t.elapsed_seconds

        if self.benchmark_results:
            self.benchmark_results.add_timing(
                f"FPFH ({self.path.name})", t.elapsed_seconds
            )

        return pcd_down, pcd_fpfh

    def _add_normals(self, pcd: o3d.geometry.PointCloud, voxel_size: float) -> None:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * self.kdtree_params.normal_radius_multiplier,
                max_nn=self.kdtree_params.normal_max_nn,
            ),
        )


if __name__ == "__main__":
    from pathlib import Path

    voxel_size = 0.01
    src_path = Path.cwd() / "3d_data" / "sample.ply"
    tgt_path = Path.cwd() / "3d_data" / "target.ply"

    src_ply = Ply(src_path, voxel_size)
    tgt_ply = Ply(tgt_path, voxel_size)

    logger.info("Source PLY: %s", src_ply.path)
    logger.info("Target PLY: %s", tgt_ply.path)
