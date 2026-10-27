"""Camera calibration using a ChArUco board."""

import json
from pathlib import Path

import cv2
import numpy as np

CHARUCO_COLS = 7
CHARUCO_ROWS = 5
CHARUCO_SQUARE_MM = 30.0
CHARUCO_MARKER_MM = 22.0


def create_charuco_board():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return cv2.aruco.CharucoBoard(
        (CHARUCO_COLS, CHARUCO_ROWS),
        CHARUCO_SQUARE_MM / 1000.0,
        CHARUCO_MARKER_MM / 1000.0,
        dictionary,
    )


def generate_charuco_image(output_path: str | Path, dpi: int = 150):
    board = create_charuco_board()
    px_per_m = dpi / 0.0254
    width = int(CHARUCO_COLS * CHARUCO_SQUARE_MM / 1000.0 * px_per_m)
    height = int(CHARUCO_ROWS * CHARUCO_SQUARE_MM / 1000.0 * px_per_m)
    img = board.generateImage((width, height))
    cv2.imwrite(str(output_path), img)
    return Path(output_path)


def calibrate_from_images(image_paths: list[Path]) -> dict:
    """Calibrate camera from ChArUco board images. Returns intrinsics dict."""
    board = create_charuco_board()
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        detector_params,
    )

    all_charuco_corners = []
    all_charuco_ids = []
    image_size = None

    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) < 4:
            continue

        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        if ret >= 4:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)

    if len(all_charuco_corners) < 3:
        raise ValueError(
            f"Need at least 3 valid frames, got {len(all_charuco_corners)}"
        )

    ret, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, image_size, None, None
    )

    return {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "image_size": list(image_size),
        "reprojection_error": ret,
    }


def save_calibration(cal: dict, path: str | Path):
    Path(path).write_text(json.dumps(cal, indent=2))


def load_calibration(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(Path(path).read_text())
    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["dist_coeffs"])
    return camera_matrix, dist_coeffs
