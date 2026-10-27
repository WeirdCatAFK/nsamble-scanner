"""ArUco marker detection and turntable pose estimation."""

from pathlib import Path

import cv2
import numpy as np

from scanner.config import (
    MARKER_DICTIONARY,
    MARKER_IDS,
    marker_corners_3d,
)


def _build_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICTIONARY)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_markers(image: np.ndarray) -> tuple[list, np.ndarray | None]:
    detector = _build_detector()
    corners, ids, _ = detector.detectMarkers(image)
    return corners, ids


def estimate_turntable_pose(
    corners: list,
    ids: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Estimate disc pose from detected markers. Returns (rvec, tvec, angle_deg) or None."""
    obj_points = []
    img_points = []

    id_set = set(MARKER_IDS)
    for i, marker_id in enumerate(ids.flatten()):
        if int(marker_id) not in id_set:
            continue
        corners_3d = marker_corners_3d(int(marker_id))
        for pt3d, pt2d in zip(corners_3d, corners[i][0]):
            obj_points.append(pt3d)
            img_points.append(pt2d)

    if len(obj_points) < 4:
        return None

    obj_points = np.array(obj_points, dtype=np.float64)
    img_points = np.array(img_points, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    angle_rad = np.arctan2(rmat[1, 0], rmat[0, 0])
    angle_deg = np.degrees(angle_rad)

    return rvec, tvec, float(angle_deg)


def process_frames(
    frame_dir: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[dict]:
    """Detect markers and estimate pose for all frames in a directory."""
    results = []
    frames = sorted(frame_dir.glob("frame_*.png"))

    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids = detect_markers(gray)

        if ids is None or len(ids) < 2:
            print(f"  {frame_path.name}: insufficient markers")
            continue

        pose = estimate_turntable_pose(corners, ids, camera_matrix, dist_coeffs)
        if pose is None:
            print(f"  {frame_path.name}: pose estimation failed")
            continue

        rvec, tvec, angle = pose
        results.append({
            "path": str(frame_path),
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
            "angle_deg": angle,
            "markers_detected": len(ids),
        })
        print(f"  {frame_path.name}: {angle:.1f} deg, {len(ids)} markers")

    return results


def _rotation_matrix_to_quaternion(r: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to (qw, qx, qy, qz)."""
    trace = np.trace(r)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (r[2, 1] - r[1, 2]) * s
        qy = (r[0, 2] - r[2, 0]) * s
        qz = (r[1, 0] - r[0, 1]) * s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    return float(qw), float(qx), float(qy), float(qz)


def export_colmap_poses(
    poses: list[dict],
    camera_matrix: np.ndarray,
    image_size: tuple[int, int],
    output_dir: Path,
):
    """Write COLMAP-format cameras.txt and images.txt."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    w, h = image_size

    with open(output_dir / "cameras.txt", "w") as f:
        f.write("# Camera list: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    with open(output_dir / "images.txt", "w") as f:
        f.write("# Image list: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, "
                "CAMERA_ID, NAME\n")
        for i, pose in enumerate(poses):
            rvec = np.array(pose["rvec"])
            tvec = np.array(pose["tvec"])
            rmat, _ = cv2.Rodrigues(rvec)

            rmat_inv = rmat.T
            tvec_inv = -rmat_inv @ tvec

            qw, qx, qy, qz = _rotation_matrix_to_quaternion(rmat_inv)

            name = Path(pose["path"]).name
            f.write(f"{i+1} {qw} {qx} {qy} {qz} "
                    f"{tvec_inv[0]} {tvec_inv[1]} {tvec_inv[2]} 1 {name}\n")
            f.write("\n")

    (output_dir / "points3D.txt").write_text("# Empty — dense stereo only\n")
