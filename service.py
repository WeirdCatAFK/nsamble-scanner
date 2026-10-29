"""Scanner service: accepts video uploads, runs reconstruction, returns STL."""

import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from scanner.calibration import load_calibration
from scanner.pose import process_frames, export_colmap_poses
from scanner.segment import mask_all_frames
from scanner.reconstruct import run_colmap_dense, pointcloud_to_mesh
from scanner.postprocess import process_and_export

app = FastAPI(title="nsamble-scanner")

WORK_DIR = Path("/data/scans")
WORK_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_PATH = Path("/data/calibration.json")

jobs: dict[str, dict] = {}
queue: deque[str] = deque()
lock = threading.Lock()


def extract_frames(video_path: Path, output_dir: Path, target_count: int = 36) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0

    step = max(1, total // target_count)
    saved = 0
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(output_dir / f"frame_{saved:04d}.png"), frame)
        saved += 1

    cap.release()
    return saved


def run_pipeline(job_id: str):
    job = jobs[job_id]
    scan_dir = Path(job["scan_dir"])

    try:
        job["status"] = "extracting_frames"
        frame_dir = scan_dir / "frames"
        count = extract_frames(Path(job["video_path"]), frame_dir)
        if count == 0:
            raise RuntimeError("No frames extracted from video.")
        job["frame_count"] = count

        cal_path = scan_dir / "calibration.json" if (scan_dir / "calibration.json").exists() else CALIBRATION_PATH
        if not cal_path.exists():
            raise RuntimeError("No calibration file found.")
        camera_matrix, dist_coeffs = load_calibration(str(cal_path))

        first_frame = next(frame_dir.glob("frame_*.png"))
        img = cv2.imread(str(first_frame))
        image_size = (img.shape[1], img.shape[0])

        job["status"] = "detecting_markers"
        poses = process_frames(frame_dir, camera_matrix, dist_coeffs)
        if not poses:
            raise RuntimeError("No valid poses found in frames.")
        job["poses_found"] = len(poses)

        colmap_dir = scan_dir / "colmap"
        export_colmap_poses(poses, camera_matrix, image_size, colmap_dir / "sparse" / "0")

        job["status"] = "masking"
        masked_dir = scan_dir / "masked"
        mask_all_frames(frame_dir, poses, camera_matrix, dist_coeffs, masked_dir)

        job["status"] = "dense_reconstruction"
        fused = run_colmap_dense(colmap_dir, masked_dir)

        job["status"] = "meshing"
        output_dir = scan_dir / "output"
        stl, _ = pointcloud_to_mesh(fused, output_dir)

        job["status"] = "postprocessing"
        stl_final, _ = process_and_export(stl, output_dir)

        job["status"] = "done"
        job["result_path"] = str(stl_final)
        job["completed_at"] = time.time()

    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = time.time()


def worker():
    while True:
        job_id = None
        with lock:
            if queue:
                job_id = queue.popleft()

        if job_id is None:
            time.sleep(0.5)
            continue

        run_pipeline(job_id)


threading.Thread(target=worker, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/scans")
async def create_scan(
    video: UploadFile = File(...),
    calibration: UploadFile | None = File(None),
):
    job_id = str(uuid.uuid4())[:8]
    scan_dir = WORK_DIR / job_id
    scan_dir.mkdir(parents=True, exist_ok=True)

    video_path = scan_dir / video.filename
    with open(video_path, "wb") as f:
        while chunk := await video.read(1024 * 1024):
            f.write(chunk)

    if calibration:
        cal_path = scan_dir / "calibration.json"
        with open(cal_path, "wb") as f:
            while chunk := await calibration.read(1024 * 1024):
                f.write(chunk)

    job = {
        "id": job_id,
        "status": "queued",
        "video_path": str(video_path),
        "scan_dir": str(scan_dir),
        "created_at": time.time(),
        "completed_at": None,
        "error": None,
        "result_path": None,
    }

    with lock:
        jobs[job_id] = job
        queue.append(job_id)

    return {"id": job_id, "status": "queued"}


@app.get("/scans")
def list_scans():
    ordered = sorted(jobs.values(), key=lambda j: j["created_at"])
    return [
        {
            "id": j["id"],
            "status": j["status"],
            "created_at": j["created_at"],
            "completed_at": j["completed_at"],
        }
        for j in ordered
    ]


@app.get("/scans/{job_id}")
def get_scan(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Scan not found.")
    job = jobs[job_id]
    return {
        "id": job["id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "completed_at": job["completed_at"],
        "error": job["error"],
        "has_result": job["result_path"] is not None,
    }


@app.get("/scans/{job_id}/model")
def get_model(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Scan not found.")
    job = jobs[job_id]
    if job["status"] != "done" or not job["result_path"]:
        raise HTTPException(400, f"Scan not ready. Status: {job['status']}")
    return FileResponse(
        job["result_path"],
        media_type="model/stl",
        filename="model_final.stl",
    )


@app.post("/calibration")
async def upload_calibration(file: UploadFile = File(...)):
    with open(CALIBRATION_PATH, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)
    return {"status": "ok", "path": str(CALIBRATION_PATH)}


@app.get("/calibration")
def check_calibration():
    return {"exists": CALIBRATION_PATH.exists()}
