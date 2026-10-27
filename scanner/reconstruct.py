"""3D reconstruction through COLMAP dense stereo and Open3D meshing."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d


def _find_colmap() -> str | None:
    return shutil.which("colmap")


def run_colmap_dense(workspace: Path, image_dir: Path) -> Path:
    """Run COLMAP dense stereo using pre-computed poses. Returns fused PLY path."""
    colmap = _find_colmap()
    if colmap is None:
        raise RuntimeError(
            "COLMAP not found. Install from https://colmap.github.io/install.html"
        )

    sparse_dir = workspace / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    dense_dir = workspace / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        colmap, "image_undistorter",
        "--image_path", str(image_dir),
        "--input_path", str(sparse_dir),
        "--output_path", str(dense_dir),
        "--output_type", "COLMAP",
    ], check=True)

    subprocess.run([
        colmap, "patch_match_stereo",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--PatchMatchStereo.geom_consistency", "true",
    ], check=True)

    fused_path = dense_dir / "fused.ply"
    subprocess.run([
        colmap, "stereo_fusion",
        "--workspace_path", str(dense_dir),
        "--workspace_format", "COLMAP",
        "--output_path", str(fused_path),
    ], check=True)

    return fused_path


def pointcloud_to_mesh(ply_path: Path, output_dir: Path,
                       depth: int = 9) -> tuple[Path, Path]:
    """Poisson surface reconstruction. Returns (stl_path, obj_path)."""
    pcd = o3d.io.read_point_cloud(str(ply_path))

    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    densities = np.asarray(densities)
    threshold = np.quantile(densities, 0.01)
    vertices_to_remove = densities < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.compute_vertex_normals()

    output_dir.mkdir(parents=True, exist_ok=True)
    stl_path = output_dir / "model.stl"
    obj_path = output_dir / "model.obj"

    o3d.io.write_triangle_mesh(str(stl_path), mesh)
    o3d.io.write_triangle_mesh(str(obj_path), mesh)

    return stl_path, obj_path
