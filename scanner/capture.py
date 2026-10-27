"""Frame capture from the Logitech C920. Press SPACE to capture, Q to finish."""

from datetime import datetime
from pathlib import Path

import cv2


def capture_photos(output_dir: str | Path, camera_index: int = 0,
                   resolution: tuple[int, int] = (1920, 1080)) -> list[Path]:
    """Live preview. Press SPACE to capture, Q to finish."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    saved: list[Path] = []
    print("SPACE=capture  Q=finish")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(
            display, f"Frames: {len(saved)}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )
        cv2.imshow("Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            path = output_dir / f"frame_{len(saved):04d}.png"
            cv2.imwrite(str(path), frame)
            saved.append(path)
            print(f"  Saved {path.name}")

    cap.release()
    cv2.destroyAllWindows()
    return saved


def extract_frames_from_video(video_path: str | Path, output_dir: str | Path,
                              every_n: int = 10) -> list[Path]:
    """Extract every Nth frame from a video file."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    saved: list[Path] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            path = output_dir / f"frame_{len(saved):04d}.png"
            cv2.imwrite(str(path), frame)
            saved.append(path)
        frame_idx += 1

    cap.release()
    return saved


def create_session_dir(base: str | Path = "scans") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = Path(base) / ts / "frames"
    session.mkdir(parents=True, exist_ok=True)
    return session
