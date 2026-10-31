"""Test COLMAP pose export conventions."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from scanner.pose import export_colmap_poses


def _camera():
    matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1],
    ], dtype=np.float64)
    return matrix


def _export(poses):
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "sparse" / "0"
        export_colmap_poses(poses, _camera(), (640, 480), out)
        return (out / "images.txt").read_text(), (out / "cameras.txt").read_text()


def _image_row(text):
    for line in text.splitlines():
        if line and not line.startswith("#"):
            return line
    raise AssertionError("no image row written")


def test_identity_pose_writes_uninverted_extrinsics():
    poses = [{"rvec": [0.0, 0.0, 0.0], "tvec": [1.0, 2.0, 3.0],
              "path": "/frames/frame_0001.png"}]
    images, cameras = _export(poses)

    assert "1 PINHOLE 640 480 800.0 800.0 320.0 240.0" in cameras

    fields = _image_row(images).split()
    assert fields[0] == "1"
    assert [float(v) for v in fields[1:5]] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert [float(v) for v in fields[5:8]] == pytest.approx([1.0, 2.0, 3.0])
    assert fields[8] == "1"
    assert fields[9] == "frame_0001.png"


def test_rotated_pose_matches_opencv_extrinsics():
    rvec = [0.0, 0.0, np.pi / 2.0]
    poses = [{"rvec": rvec, "tvec": [1.0, 0.0, 0.0],
              "path": "/frames/frame_0002.png"}]
    images, _ = _export(poses)

    fields = _image_row(images).split()
    half = np.sqrt(2.0) / 2.0
    assert [float(v) for v in fields[1:5]] == pytest.approx([half, 0.0, 0.0, half])
    assert [float(v) for v in fields[5:8]] == pytest.approx([1.0, 0.0, 0.0])

    rmat, _ = cv2.Rodrigues(np.array(rvec))
    assert rmat @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])
