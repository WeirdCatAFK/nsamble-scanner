"""Scanner configuration: disc geometry and marker layout."""

import math
import cv2

DISC_DIAMETER_MM = 220.0
DISC_RADIUS_MM = DISC_DIAMETER_MM / 2.0

MARKER_DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_COUNT = 12
MARKER_SIZE_MM = 15.0
MARKER_RING_RADIUS_MM = 95.0
MARKER_IDS = list(range(MARKER_COUNT))

GEAR_MODULE = 2.0
GEAR_PRESSURE_ANGLE_DEG = 20.0
GEAR_RING_TEETH = 120
DRIVE_GEAR_TEETH = 20
PEG_DIAMETER_MM = 6.0
BASE_PLATE_DIAMETER_MM = 240.0
MDF_THICKNESS_MM = 3.0
TAB_COUNT = 8
TAB_WIDTH_MM = 8.0
TAB_DEPTH_MM = 5.0

BOX_INNER_W_MM = 280.0
BOX_INNER_D_MM = 250.0
BOX_WALL_H_MM = 10.0
BOX_DISC_OFFSET_Y_MM = 130.0
BOX_WALL_TAB_W_MM = 10.0
BOX_WALL_TAB_D_MM = MDF_THICKNESS_MM
SHEET_W_MM = 300.0
SHEET_H_MM = 300.0
WASHER_OUTER_R_MM = 10.0
WASHER_HOLE_R_MM = 2.9
CRANK_HOLE_COUNT = 6
CRANK_HOLE_RADIUS_MM = PEG_DIAMETER_MM / 2.0
CRANK_HOLE_RING_RADIUS_MM = 115.0
GEAR_CLEARANCE_MM = 1.0
SPACER_DIAMETER_MM = 30.0
SPACER_COUNT = math.ceil(
    (BOX_WALL_H_MM + MDF_THICKNESS_MM + GEAR_CLEARANCE_MM) / MDF_THICKNESS_MM
)


def marker_positions() -> dict[int, tuple[float, float]]:
    """Return marker ID to (x, y) center in mm, origin at disc center."""
    positions = {}
    for i, marker_id in enumerate(MARKER_IDS):
        angle_rad = math.radians(i * (360.0 / MARKER_COUNT))
        x = MARKER_RING_RADIUS_MM * math.cos(angle_rad)
        y = MARKER_RING_RADIUS_MM * math.sin(angle_rad)
        positions[marker_id] = (round(x, 3), round(y, 3))
    return positions


def marker_corners_3d(marker_id: int) -> list[tuple[float, float, float]]:
    """Return 4 corners of a marker in 3D (z=0), order: TL, TR, BR, BL."""
    cx, cy = marker_positions()[marker_id]
    half = MARKER_SIZE_MM / 2.0
    return [
        (cx - half, cy - half, 0.0),
        (cx + half, cy - half, 0.0),
        (cx + half, cy + half, 0.0),
        (cx - half, cy + half, 0.0),
    ]
