from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
        path: Path | str | None,
        voxel_size: float,
        *,
        use_fpfh: bool = True,
        kdtree_params: KDTreeParams | None = None,
        benchmark_results: BenchmarkResults | None = None,
        pcd: o3d.geometry.PointCloud | None = None,
        name: str = "unknown",
    ) -> None:
        self.kdtree_params = kdtree_params or KDTreeParams()
        self.benchmark_results = benchmark_results
        self.fpfh_time: float = 0.0
        self.name = name

        if pcd is not None:
            # 直接PointCloudを渡された場合
            self.path = None
            self.pcd = pcd
            self.name = name
        elif path is not None:
            # ファイルパスから読み込む場合
            self.path = Path(path) if isinstance(path, str) else path
            if not self.path.exists():
                msg = f"Ply file not found: {self.path}"
                raise FileNotFoundError(msg)
            if self.path.suffix.lower() != ".ply":
                msg = f"File is not a ply file: {self.path}"
                raise TypeError(msg)
            self.pcd = self._load(self.path)
            self.name = self.path.name
        else:
            msg = "Either 'path' or 'pcd' must be provided"
            raise ValueError(msg)

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
        logger.info("Successfully loaded and preprocessed: %s", self.name)

    @classmethod
    def from_bunny(
        cls,
        voxel_size: float,
        *,
        use_fpfh: bool = True,
        kdtree_params: KDTreeParams | None = None,
        benchmark_results: BenchmarkResults | None = None,
        transform: np.ndarray | None = None,
        name: str = "bunny",
    ) -> "Ply":
        """Open3DのBunnyメッシュから点群を生成.

        Args:
            voxel_size: ボクセルサイズ
            use_fpfh: FPFH特徴量を計算するか
            kdtree_params: KDTreeのパラメータ
            benchmark_results: ベンチマーク結果を格納するオブジェクト
            transform: 適用する変換行列 (4x4)
            name: 点群の名前

        Returns:
            Plyオブジェクト
        """
        bunny = o3d.data.BunnyMesh()
        mesh = o3d.io.read_triangle_mesh(bunny.path)
        pcd = mesh.sample_points_poisson_disk(number_of_points=50000)

        if transform is not None:
            pcd.transform(transform)

        return cls(
            path=None,
            voxel_size=voxel_size,
            use_fpfh=use_fpfh,
            kdtree_params=kdtree_params,
            benchmark_results=benchmark_results,
            pcd=pcd,
            name=name,
        )

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
        with timer(f"FPFH calculation ({self.name})") as t:
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
                f"FPFH ({self.name})",
                t.elapsed_seconds,
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
    import numpy as np

    voxel_size = 0.002  # bunnyは小さいのでvoxel_sizeも小さく

    # ソース: bunnyそのまま
    src_ply = Ply.from_bunny(voxel_size, name="bunny_source")

    # ターゲット: bunnyを回転・移動させたもの
    transform = np.eye(4)
    transform[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0.1, 0.2, 0.3])
    transform[:3, 3] = [0.02, 0.01, 0.005]
    tgt_ply = Ply.from_bunny(voxel_size, transform=transform, name="bunny_target")

    logger.info("Source: %s", src_ply.name)
    logger.info("Target: %s", tgt_ply.name)
