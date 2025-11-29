from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import open3d as o3d
import open3d.visualization.gui as o3dv_gui  # pyright: ignore[reportMissingImports]
import open3d.visualization.rendering as o3dv_rendering  # pyright: ignore[reportMissingImports]

from matcher.icp import refine_registration
from matcher.ransac import global_registration
from utils.setup_logging import setup_logging

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

    def _setup_app(self) -> None:
        self.window = self.app.create_window(self.window_name, 800, 600)
        window = self.window

        # ==== Scene / Material ====
        self.scene = o3dv_rendering.Open3DScene(window.renderer)
        self.material = o3dv_rendering.MaterialRecord()
        self.scene.add_geometry(SOURCE_NAME, self.source.pcd, self.material)
        self.scene.add_geometry(TARGET_NAME, self.target.pcd, self.material)

        # ==== 左側 GUI レイアウト ====
        em = window.theme.font_size
        gui_layout = o3dv_gui.Vert(0, o3dv_gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))
        # 左側の幅を 250px に決め打ち
        gui_layout.frame = o3dv_gui.Rect(
            window.content_rect.x,
            window.content_rect.y,
            250,
            window.content_rect.height,
        )

        self.label = o3dv_gui.Label("RANSAC Fitness: progressing...")
        gui_layout.add_child(self.label)

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
            result = global_registration(self.source, self.target, self.voxel_size, iteration=1)
            self.iter_num += 1

            # main thread で geometry を触るために post_to_main_thread
            self.app.post_to_main_thread(self.window, lambda res=result: self._apply_result(res))

        # RANSAC 終了後に ICP 一回
        if self._result is not None:
            icp_result = refine_registration(self.source, self.target, self._result.transformation, self.voxel_size)
            self.app.post_to_main_thread(self.window, lambda res=icp_result: self._apply_result(res))

    def _apply_result(self, result: o3d.pipelines.registration.RegistrationResult) -> None:
        self._result = result

        # ここは main thread 確定なので GUI 触ってよい
        self.source.pcd.transform(result.transformation)

        if self.scene.has_geometry(SOURCE_NAME):
            self.scene.remove_geometry(SOURCE_NAME)
        self.scene.add_geometry(SOURCE_NAME, self.source.pcd, self.material)

        self.label.text = f"Fitness: {result.fitness:.4f}"
        self.window.post_redraw()


if __name__ == "__main__":
    from pathlib import Path

    from ply import Ply

    voxel_size = 0.01
    base_path = Path(__file__).parent.parent.parent / "3d_data"
    src_path = base_path / "source.ply"
    tgt_path = base_path / "target.ply"

    src_ply = Ply(src_path, voxel_size)
    tgt_ply = Ply(tgt_path, voxel_size)

    visualizer = VisualizeMatcher(src_ply, tgt_ply)
    visualizer.invoke(voxel_size, ransac_iteration=3, is_logging=True)
