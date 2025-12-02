"""Bunnyデータを生成して保存するスクリプト."""

from pathlib import Path

import numpy as np
import open3d as o3d


def main():
    # 保存先ディレクトリ
    data_dir = Path("/app/3d_data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Bunnyメッシュを読み込み
    bunny = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny.path)

    # 点群にサンプリング
    pcd = mesh.sample_points_poisson_disk(number_of_points=50000)

    # ソース点群として保存
    source_path = data_dir / "source.ply"
    o3d.io.write_point_cloud(str(source_path), pcd)
    print(f"Saved source point cloud to {source_path}")

    # ターゲット点群を作成（回転・移動）
    transform = np.eye(4)
    # 回転 (X軸周りに30度、Y軸周りに45度)
    R = o3d.geometry.get_rotation_matrix_from_xyz([np.deg2rad(30), np.deg2rad(45), 0])
    transform[:3, :3] = R
    # 移動
    transform[:3, 3] = [0.05, 0.05, 0.05]

    pcd_target = pcd.transform(transform)

    # ターゲット点群として保存
    target_path = data_dir / "target.ply"
    o3d.io.write_point_cloud(str(target_path), pcd_target)
    print(f"Saved target point cloud to {target_path}")


if __name__ == "__main__":
    main()
