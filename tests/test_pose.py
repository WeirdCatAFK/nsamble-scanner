"""Test pose estimation with synthetic marker data."""

import numpy as np
import cv2

from scanner.config import MARKER_IDS, marker_corners_3d
from scanner.pose import estimate_turntable_pose


def _synthetic_camera():
    camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros(5, dtype=np.float64)
    return camera_matrix, dist_coeffs


def test_solvepnp_round_trip():
    camera_matrix, dist_coeffs = _synthetic_camera()

    rvec_true = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    tvec_true = np.array([0.0, 0.0, 500.0], dtype=np.float64)

    all_corners = []
    all_ids = []

    for marker_id in MARKER_IDS[:6]:
        pts_3d = np.array(marker_corners_3d(marker_id), dtype=np.float64)
        pts_2d, _ = cv2.projectPoints(
            pts_3d, rvec_true, tvec_true, camera_matrix, dist_coeffs
        )
        all_corners.append(pts_2d.reshape(1, 4, 2).astype(np.float32))
        all_ids.append([marker_id])

    ids = np.array(all_ids, dtype=np.int32)
    result = estimate_turntable_pose(all_corners, ids, camera_matrix, dist_coeffs)

    assert result is not None
    rvec_est, tvec_est, _ = result
    assert np.allclose(rvec_est.flatten(), rvec_true, atol=0.01)
    assert np.allclose(tvec_est.flatten(), tvec_true, atol=1.0)
