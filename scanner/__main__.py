"""CLI entry point: python -m scanner <command>."""

import argparse
import sys
from pathlib import Path


def cmd_disc(args):
    from scanner.disc import generate_disc_svg, generate_lasercut_svg, generate_sheets_svg
    out = generate_disc_svg(args.output)
    print(f"Generated: {out}")
    print(f"Marker positions: {out.with_suffix('.json')}")
    if args.lasercut:
        lc_dir = Path(args.output).parent
        files = generate_lasercut_svg(lc_dir)
        for f in files:
            print(f"Laser-cut SVG: {f}")
    if args.sheets:
        lc_dir = Path(args.output).parent
        sheets = generate_sheets_svg(lc_dir)
        for s in sheets:
            print(f"Sheet: {s}")


def cmd_calibrate(args):
    from scanner.calibration import (
        calibrate_from_images, generate_charuco_image, save_calibration,
    )
    if args.generate_board:
        path = generate_charuco_image(args.generate_board)
        print(f"ChArUco board: {path}")
        return

    images = sorted(Path(args.image_dir).glob("*.png"))
    if not images:
        images = sorted(Path(args.image_dir).glob("*.jpg"))
    cal = calibrate_from_images(images)
    save_calibration(cal, args.output)
    print(f"Reprojection error: {cal['reprojection_error']:.4f}")
    print(f"Saved: {args.output}")


def cmd_capture(args):
    from scanner.capture import capture_photos, create_session_dir
    session = create_session_dir(args.output_base)
    frames = capture_photos(session, camera_index=args.camera)
    print(f"Captured {len(frames)} frames in {session}")


def cmd_process(args):
    from scanner.calibration import load_calibration
    from scanner.pose import process_frames, export_colmap_poses
    from scanner.segment import mask_all_frames
    from scanner.reconstruct import run_colmap_dense, pointcloud_to_mesh
    from scanner.postprocess import process_and_export

    scan_dir = Path(args.scan_dir)
    frame_dir = scan_dir / "frames" if (scan_dir / "frames").exists() else scan_dir

    camera_matrix, dist_coeffs = load_calibration(args.calibration)
    image_size = (1920, 1080)

    print("Detecting markers and estimating poses...")
    poses = process_frames(frame_dir, camera_matrix, dist_coeffs)
    if not poses:
        print("No valid poses found.")
        sys.exit(1)
    print(f"  {len(poses)} frames with valid poses")

    colmap_dir = scan_dir / "colmap"
    export_colmap_poses(poses, camera_matrix, image_size, colmap_dir / "sparse" / "0")

    print("Masking frames...")
    masked_dir = scan_dir / "masked"
    mask_all_frames(frame_dir, poses, camera_matrix, dist_coeffs, masked_dir)

    print("Running COLMAP dense stereo...")
    fused = run_colmap_dense(colmap_dir, masked_dir)

    print("Meshing...")
    output_dir = scan_dir / "output"
    stl, obj = pointcloud_to_mesh(fused, output_dir)

    print("Post-processing...")
    stl_final, obj_final = process_and_export(stl, output_dir)

    print(f"Done. Output in {output_dir}/")
    print(f"  STL: {stl_final}")
    print(f"  OBJ: {obj_final}")


def main():
    parser = argparse.ArgumentParser(prog="nsamble-scan")
    sub = parser.add_subparsers(dest="command", required=True)

    p_disc = sub.add_parser("disc", help="Generate fiducial disc SVG")
    p_disc.add_argument("-o", "--output", default="fiducial_disc.svg")
    p_disc.add_argument("--lasercut", action="store_true",
                        help="Also generate laser-cut gear turntable SVG")
    p_disc.add_argument("--sheets", action="store_true",
                        help="Nest parts onto 300x300 mm sheets for cutting")
    p_disc.set_defaults(func=cmd_disc)

    p_cal = sub.add_parser("calibrate", help="Camera calibration")
    p_cal.add_argument("image_dir", nargs="?", default=".")
    p_cal.add_argument("-o", "--output", default="calibration.json")
    p_cal.add_argument("--generate-board", metavar="PATH")
    p_cal.set_defaults(func=cmd_calibrate)

    p_cap = sub.add_parser("capture", help="Capture scan frames")
    p_cap.add_argument("-c", "--camera", type=int, default=0)
    p_cap.add_argument("-o", "--output-base", default="scans")
    p_cap.set_defaults(func=cmd_capture)

    p_proc = sub.add_parser("process", help="Run full reconstruction pipeline")
    p_proc.add_argument("scan_dir")
    p_proc.add_argument("--calibration", default="calibration.json")
    p_proc.set_defaults(func=cmd_process)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
