from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import open3d as o3d
import open3d.visualization.gui as o3dv_gui  # pyright: ignore[reportMissingImports]
import open3d.visualization.rendering as o3dv_rendering  # pyright: ignore[reportMissingImports]

from matcher.icp import refine_registration
from matcher.ransac import global_registration
from utils.setup_logging import setup_logging
from utils.timer import BenchmarkResults

if TYPE_CHECKING:
    from ply import Ply

logger = setup_logging(__name__)


class VisualzerProtocol(Protocol):
    def update_geometry(self, geometry: o3d.geometry.Geometry) -> None: ...

    def poll_events(self) -> None: ...

    def update_renderer(self) -> None: ...


class VisualizeInfoProtocol(Protocol):
    fitness: float
    inlier_rmse: float


SOURCE_NAME = "source"
TARGET_NAME = "target"


class VisualizeMatcher:
    def __init__(self, source: Ply, target: Ply, *, window_name: str = "RANSAC & ICP Render") -> None:
        self.source = source
        self.target = target
        self.window_name = window_name

        self.app = o3dv_gui.Application.instance
        self.app.initialize()

        self.iter_num = 0
        self.max_iter = 0
        self.voxel_size = 0
        self.is_logging = False
        self._result = None
        self._is_executed_icp = False

        # ベンチマーク結果を格納
        self.benchmark_results = BenchmarkResults()
        self._log_messages: list[str] = []

    def _setup_app(self) -> None:
        self.window = self.app.create_window(self.window_name, 1200, 800)
        window = self.window

        # ==== Scene / Material ====
        self.scene = o3dv_rendering.Open3DScene(window.renderer)
        self.material = o3dv_rendering.MaterialRecord()
        self.scene.add_geometry(SOURCE_NAME, self.source.pcd, self.material)
        self.scene.add_geometry(TARGET_NAME, self.target.pcd, self.material)

        # ==== 左側 GUI レイアウト ====
        em = window.theme.font_size
        gui_layout = o3dv_gui.Vert(0, o3dv_gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        # 左側の幅を 350px に拡張
        gui_layout.frame = o3dv_gui.Rect(
            window.content_rect.x,
            window.content_rect.y,
            350,
            window.content_rect.height,
        )

        # ステータスラベル
        self.label = o3dv_gui.Label("RANSAC Fitness: progressing...")
        gui_layout.add_child(self.label)

        # ログ表示用のラベル (複数行)
        gui_layout.add_child(o3dv_gui.Label(""))  # スペーサー
        log_header = o3dv_gui.Label("=== Benchmark Log ===")
        gui_layout.add_child(log_header)

        self.log_label = o3dv_gui.Label("")
        gui_layout.add_child(self.log_label)

        # ==== 右側 SceneWidget ====
        self.scene_widget = o3dv_gui.SceneWidget()
        self.scene_widget.scene = self.scene
        self.scene_widget.setup_camera(
            60.0,
            self.scene.bounding_box,
            self.scene.bounding_box.get_center(),
        )

        self.scene_widget.frame = o3dv_gui.Rect(
            gui_layout.frame.get_right(),
            window.content_rect.y,
            window.content_rect.width - gui_layout.frame.width,
            window.content_rect.height,
        )

        # ==== Window に直接 add ====
        window.add_child(gui_layout)
        window.add_child(self.scene_widget)

        # 初回描画
        window.post_redraw()

    def invoke(self, voxel_size: float, ransac_iteration: int, *, is_logging: bool) -> None:
        self.voxel_size = voxel_size
        self.max_iter = ransac_iteration
        self.is_logging = is_logging

        # FPFH計算時間をログに追加
        fpfh_src_time = getattr(self.source, "fpfh_time", 0.0)
        fpfh_tgt_time = getattr(self.target, "fpfh_time", 0.0)
        if fpfh_src_time > 0:
            self._log_messages.append(f"FPFH (source): time={fpfh_src_time:.3f}s")
        if fpfh_tgt_time > 0:
            self._log_messages.append(f"FPFH (target): time={fpfh_tgt_time:.3f}s")

        self._setup_app()

        # 1. 毎ループmain threadから呼び出される処理
        self.window.set_on_tick_event(lambda: self._on_tick())

        # 2. RANSAC/ICP は別スレッドに逃がす
        self.app.run_in_thread(self._worker_loop)

        self.app.run()

    def _on_tick(self) -> bool:
        # キー入力など GUI 系の処理だけ行う。なければ単に False でもよい
        return False  # ここで True を返すと毎フレーム再描画要求になる

    def _worker_loop(self) -> None:
        while self.iter_num < self.max_iter:
            # ここは別スレッド → ICP/RANSAC 計算だけ
            result = global_registration(
                self.source,
                self.target,
                self.voxel_size,
                iteration=1,
                benchmark_results=self.benchmark_results,
            )
            self.iter_num += 1

            # 処理時間を取得
            elapsed = self.benchmark_results.timings.get("RANSAC (with FPFH)", 0)

            # main thread で geometry を触るために post_to_main_thread
            self.app.post_to_main_thread(
                self.window,
                lambda res=result, t=elapsed: self._apply_result(res, elapsed_time=t),
            )

        # RANSAC 終了後に ICP 一回
        if self._result is not None:
            icp_result = refine_registration(
                self.source,
                self.target,
                self._result.transformation,
                self.voxel_size,
                benchmark_results=self.benchmark_results,
            )
            elapsed = self.benchmark_results.timings.get("ICP", 0)
            self.app.post_to_main_thread(
                self.window,
                lambda res=icp_result, t=elapsed: self._apply_result(res, is_icp=True, elapsed_time=t),
            )

    def _apply_result(
        self,
        result: o3d.pipelines.registration.RegistrationResult,
        *,
        is_icp: bool = False,
        elapsed_time: float = 0.0,
    ) -> None:
        self._result = result

        # ここは main thread 確定なので GUI 触ってよい
        self.source.pcd.transform(result.transformation)

        if self.scene.has_geometry(SOURCE_NAME):
            self.scene.remove_geometry(SOURCE_NAME)
        self.scene.add_geometry(SOURCE_NAME, self.source.pcd, self.material)

        # ステータス更新
        method = "ICP" if is_icp else f"RANSAC [{self.iter_num}/{self.max_iter}]"
        self.label.text = f"{method} Fitness: {result.fitness:.4f}"

        # ログメッセージを追加 (処理時間も含む)
        log_entry = (
            f"{method}: fitness={result.fitness:.4f}, "
            f"rmse={result.inlier_rmse:.4f}, "
            f"time={elapsed_time:.3f}s"
        )
        self._log_messages.append(log_entry)

        # 最新10件のログを表示
        recent_logs = self._log_messages[-10:]
        self.log_label.text = "\n".join(recent_logs)

        self.window.post_redraw()


if __name__ == "__main__":
    import numpy as np

    from ply import Ply

    voxel_size = 0.002  # bunnyは小さいのでvoxel_sizeも小さく

    # ソース: bunnyそのまま
    src_ply = Ply.from_bunny(voxel_size, name="bunny_source")

    # ターゲット: bunnyを回転・移動させたもの
    transform = np.eye(4)
    transform[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0.1, 0.2, 0.3])
    transform[:3, 3] = [0.02, 0.01, 0.005]
    tgt_ply = Ply.from_bunny(voxel_size, transform=transform, name="bunny_target")

    visualizer = VisualizeMatcher(src_ply, tgt_ply)
    visualizer.invoke(voxel_size, ransac_iteration=3, is_logging=True)
