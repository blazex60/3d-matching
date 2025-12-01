import open3d as o3d

if __name__ == "__main__":
    dataset = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(dataset.path)
