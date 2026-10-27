"""Object segmentation using the detected disc plane."""

from pathlib import Path

import cv2
import numpy as np

from scanner.config import DISC_RADIUS_MM


def create_disc_mask(
    image_shape: tuple[int, int],
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    num_points: int = 72,
) -> np.ndarray:
    """Project the disc circle into image space and create a binary mask."""
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    circle_3d = np.array([
        [DISC_RADIUS_MM * np.cos(a), DISC_RADIUS_MM * np.sin(a), 0.0]
        for a in angles
    ], dtype=np.float64)

    projected, _ = cv2.projectPoints(circle_3d, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2).astype(np.int32)

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, projected, 255)
    return mask


def mask_frame(
    image: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    mask = create_disc_mask(image.shape, rvec, tvec, camera_matrix, dist_coeffs)
    result = image.copy()
    result[mask == 0] = 0
    return result


def mask_all_frames(
    frame_dir: Path,
    poses: list[dict],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for pose in poses:
        img = cv2.imread(pose["path"])
        if img is None:
            continue

        rvec = np.array(pose["rvec"])
        tvec = np.array(pose["tvec"])
        masked = mask_frame(img, rvec, tvec, camera_matrix, dist_coeffs)

        out_path = output_dir / Path(pose["path"]).name
        cv2.imwrite(str(out_path), masked)
        saved.append(out_path)

    return saved
