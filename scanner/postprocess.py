"""Mesh post-processing: clean, decimate, export."""

from pathlib import Path

import numpy as np
import open3d as o3d


def clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    triangle_clusters, cluster_n_tri, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_tri = np.asarray(cluster_n_tri)

    if len(cluster_n_tri) > 1:
        largest = cluster_n_tri.argmax()
        triangles_to_remove = triangle_clusters != largest
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()

    return mesh


def decimate_mesh(mesh: o3d.geometry.TriangleMesh,
                  target_triangles: int = 100_000) -> o3d.geometry.TriangleMesh:
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    return mesh


def process_and_export(input_path: Path, output_dir: Path,
                       target_triangles: int = 100_000) -> tuple[Path, Path]:
    mesh = o3d.io.read_triangle_mesh(str(input_path))
    mesh = clean_mesh(mesh)
    mesh = decimate_mesh(mesh, target_triangles)
    mesh.compute_vertex_normals()

    output_dir.mkdir(parents=True, exist_ok=True)
    stl_path = output_dir / "model_final.stl"
    obj_path = output_dir / "model_final.obj"

    o3d.io.write_triangle_mesh(str(stl_path), mesh)
    o3d.io.write_triangle_mesh(str(obj_path), mesh)

    return stl_path, obj_path
