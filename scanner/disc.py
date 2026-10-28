"""Generate printable SVG fiducial disc and laser-cut gear turntable."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import svgwrite

from scanner.config import (
    BASE_PLATE_DIAMETER_MM,
    BOX_DISC_OFFSET_Y_MM,
    BOX_INNER_D_MM,
    BOX_INNER_W_MM,
    BOX_WALL_H_MM,
    BOX_WALL_TAB_D_MM,
    BOX_WALL_TAB_W_MM,
    CRANK_HOLE_COUNT,
    CRANK_HOLE_RADIUS_MM,
    CRANK_HOLE_RING_RADIUS_MM,
    DISC_RADIUS_MM,
    GEAR_CLEARANCE_MM,
    DRIVE_GEAR_TEETH,
    GEAR_MODULE,
    GEAR_PRESSURE_ANGLE_DEG,
    GEAR_RING_TEETH,
    MARKER_COUNT,
    MARKER_DICTIONARY,
    MARKER_IDS,
    MARKER_RING_RADIUS_MM,
    MARKER_SIZE_MM,
    MDF_THICKNESS_MM,
    PEG_DIAMETER_MM,
    SHEET_H_MM,
    SHEET_W_MM,
    SPACER_COUNT,
    SPACER_DIAMETER_MM,
    TAB_COUNT,
    TAB_DEPTH_MM,
    TAB_WIDTH_MM,
    WASHER_HOLE_R_MM,
    WASHER_OUTER_R_MM,
    marker_positions,
)

BITS = 4
BORDER_BITS = 1
CELL_COUNT = BITS + 2 * BORDER_BITS


def _marker_to_bitmap(marker_id: int) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(MARKER_DICTIONARY)
    img = cv2.aruco.generateImageMarker(dictionary, marker_id, CELL_COUNT)
    threshold = 128
    return (img[:CELL_COUNT, :CELL_COUNT] < threshold).astype(np.uint8)


def _draw_marker(dwg, group, bitmap: np.ndarray, cx: float, cy: float,
                 size_mm: float, angle_deg: float):
    cell = size_mm / CELL_COUNT
    ox = cx - size_mm / 2.0
    oy = cy - size_mm / 2.0

    marker_group = dwg.g(
        transform=f"rotate({angle_deg},{cx},{cy})"
    )

    marker_group.add(dwg.rect(
        insert=(ox, oy), size=(size_mm, size_mm), fill="white"
    ))

    for row in range(CELL_COUNT):
        for col in range(CELL_COUNT):
            if bitmap[row, col]:
                marker_group.add(dwg.rect(
                    insert=(ox + col * cell, oy + row * cell),
                    size=(cell, cell),
                    fill="black",
                ))

    group.add(marker_group)


def generate_disc_svg(output_path: str | Path) -> Path:
    """Generate the fiducial disc SVG and companion marker-positions JSON."""
    output_path = Path(output_path)
    margin = 5.0
    total = (DISC_RADIUS_MM + margin) * 2
    dwg = svgwrite.Drawing(
        str(output_path),
        size=(f"{total}mm", f"{total}mm"),
        viewBox=f"0 0 {total} {total}",
    )

    center = DISC_RADIUS_MM + margin

    disc = dwg.g(id="disc")

    disc.add(dwg.circle(
        center=(center, center), r=DISC_RADIUS_MM,
        stroke="black", stroke_width=0.5, fill="none",
    ))

    cross_len = 5.0
    disc.add(dwg.line(
        (center - cross_len, center), (center + cross_len, center),
        stroke="black", stroke_width=0.3,
    ))
    disc.add(dwg.line(
        (center, center - cross_len), (center, center + cross_len),
        stroke="black", stroke_width=0.3,
    ))

    scale_mark_len = 10.0
    for cardinal_deg in [0, 90, 180, 270]:
        rad = math.radians(cardinal_deg)
        sx = center + DISC_RADIUS_MM * math.cos(rad)
        sy = center + DISC_RADIUS_MM * math.sin(rad)
        ex = center + (DISC_RADIUS_MM - scale_mark_len) * math.cos(rad)
        ey = center + (DISC_RADIUS_MM - scale_mark_len) * math.sin(rad)
        disc.add(dwg.line((sx, sy), (ex, ey), stroke="black", stroke_width=0.3))

    positions = marker_positions()
    for marker_id in MARKER_IDS:
        mx, my = positions[marker_id]
        angle_deg = marker_id * (360.0 / MARKER_COUNT)
        bitmap = _marker_to_bitmap(marker_id)
        _draw_marker(
            dwg, disc, bitmap,
            cx=center + mx, cy=center + my,
            size_mm=MARKER_SIZE_MM, angle_deg=angle_deg,
        )

        label_r = MARKER_RING_RADIUS_MM + MARKER_SIZE_MM / 2.0 + 3.0
        rad = math.radians(angle_deg)
        lx = center + label_r * math.cos(rad)
        ly = center + label_r * math.sin(rad)
        disc.add(dwg.text(
            str(marker_id),
            insert=(lx, ly),
            font_size="3",
            text_anchor="middle",
            dominant_baseline="central",
            fill="gray",
        ))

    dwg.add(disc)
    dwg.save()

    json_path = output_path.with_suffix(".json")
    json_data = {str(k): list(v) for k, v in positions.items()}
    json_path.write_text(json.dumps(json_data, indent=2))

    return output_path


CUT_STROKE = {"stroke": "#FF0000", "stroke_width": 0.5, "fill": "none"}


def _gear_profile(teeth: int, module: float, pressure_deg: float,
                  ring_gear: bool = False) -> list[tuple[float, float]]:
    """Generate a closed spur/ring gear outline using boxes.py involute math."""
    from scanner.boxes_gears import gear_calculations, generate_spur_points

    circular_pitch = module * math.pi
    pitch_r, base_r, _add, _ded, outer_r, root_r, _thick = gear_calculations(
        teeth, circular_pitch, pressure_deg,
        clearance=0, ring_gear=ring_gear,
    )
    return generate_spur_points(
        teeth, base_r, pitch_r, outer_r, root_r,
        accuracy_involute=15, accuracy_circular=6,
    )


def _disc_with_slots(radius: float, cx: float, cy: float,
                     n_slots: int, slot_w: float, slot_d: float,
                     points_per_arc: int = 60) -> str:
    """Generate SVG path for a disc with rectangular slots around the edge."""
    slot_half_w_angle = math.asin(min(slot_w / 2.0 / radius, 1.0))
    slot_inner_r = radius - slot_d
    slot_angles = [i * 2 * math.pi / n_slots for i in range(n_slots)]

    parts = []
    first = True
    for idx in range(n_slots):
        a_center = slot_angles[idx]
        a_next = slot_angles[(idx + 1) % n_slots]
        if idx < n_slots - 1:
            a_next_actual = a_next
        else:
            a_next_actual = slot_angles[0] + 2 * math.pi

        arc_start = a_center + slot_half_w_angle
        arc_end = a_next_actual - slot_half_w_angle

        slot_a1 = a_center - slot_half_w_angle
        slot_a2 = a_center + slot_half_w_angle

        p1 = (cx + radius * math.cos(slot_a1), cy + radius * math.sin(slot_a1))
        p2 = (cx + slot_inner_r * math.cos(slot_a1), cy + slot_inner_r * math.sin(slot_a1))
        p3 = (cx + slot_inner_r * math.cos(slot_a2), cy + slot_inner_r * math.sin(slot_a2))
        p4 = (cx + radius * math.cos(slot_a2), cy + radius * math.sin(slot_a2))

        cmd = "M" if first else "L"
        first = False
        parts.append(f"{cmd}{p1[0]:.4f},{p1[1]:.4f}")
        parts.append(f"L{p2[0]:.4f},{p2[1]:.4f}")
        parts.append(f"L{p3[0]:.4f},{p3[1]:.4f}")
        parts.append(f"L{p4[0]:.4f},{p4[1]:.4f}")

        n_seg = max(4, int(points_per_arc * (arc_end - arc_start) / (2 * math.pi)))
        for i in range(1, n_seg + 1):
            a = arc_start + (arc_end - arc_start) * i / n_seg
            parts.append(f"L{cx + radius * math.cos(a):.4f},{cy + radius * math.sin(a):.4f}")

    parts.append("Z")
    return " ".join(parts)


def _gear_ring_profile(teeth: int, module: float, pressure_deg: float,
                       inner_r: float, tab_count: int, tab_w: float, tab_d: float,
                       ) -> tuple[str, str]:
    """Return (outer_path, inner_path) SVG path strings for a gear ring."""
    outer_pts = _gear_profile(teeth, module, pressure_deg)
    outer_path_parts = []
    for i, (x, y) in enumerate(outer_pts):
        cmd = "M" if i == 0 else "L"
        outer_path_parts.append(f"{cmd}{x:.4f},{y:.4f}")
    outer_path_parts.append("Z")
    outer_path = " ".join(outer_path_parts)

    tab_half_w_angle = math.asin(min(tab_w / 2.0 / inner_r, 1.0))
    tab_inner_r = inner_r - tab_d
    tab_angles = [i * 2 * math.pi / tab_count for i in range(tab_count)]

    inner_parts = []
    first = True
    for idx in range(tab_count):
        a_center = tab_angles[idx]
        a_next = tab_angles[(idx + 1) % tab_count]
        if idx < tab_count - 1:
            a_next_actual = a_next
        else:
            a_next_actual = tab_angles[0] + 2 * math.pi

        a1 = a_center - tab_half_w_angle
        a2 = a_center + tab_half_w_angle

        p1 = (inner_r * math.cos(a1), inner_r * math.sin(a1))
        p2 = (tab_inner_r * math.cos(a1), tab_inner_r * math.sin(a1))
        p3 = (tab_inner_r * math.cos(a2), tab_inner_r * math.sin(a2))
        p4 = (inner_r * math.cos(a2), inner_r * math.sin(a2))

        cmd = "M" if first else "L"
        first = False
        inner_parts.append(f"{cmd}{p1[0]:.4f},{p1[1]:.4f}")
        inner_parts.append(f"L{p2[0]:.4f},{p2[1]:.4f}")
        inner_parts.append(f"L{p3[0]:.4f},{p3[1]:.4f}")
        inner_parts.append(f"L{p4[0]:.4f},{p4[1]:.4f}")

        arc_start = a2
        arc_end = a_next_actual - tab_half_w_angle
        n_seg = max(4, int(30 * (arc_end - arc_start) / (2 * math.pi)))
        for i in range(1, n_seg + 1):
            a = arc_start + (arc_end - arc_start) * i / n_seg
            inner_parts.append(f"L{inner_r * math.cos(a):.4f},{inner_r * math.sin(a):.4f}")

    inner_parts.append("Z")
    inner_path = " ".join(inner_parts)

    return outer_path, inner_path


def _points_to_svg_path(points: list[tuple[float, float]], cx: float, cy: float) -> str:
    """Convert points to an SVG path string, offset by (cx, cy)."""
    parts = []
    for i, (x, y) in enumerate(points):
        cmd = "M" if i == 0 else "L"
        parts.append(f"{cmd}{cx + x:.4f},{cy + y:.4f}")
    parts.append("Z")
    return " ".join(parts)


def _clip_halfplane(pts: list[tuple[float, float]],
                    nx: float, ny: float) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip: keep points where nx*x + ny*y >= 0."""
    if not pts:
        return []
    result = []
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        d1 = nx * x1 + ny * y1
        d2 = nx * x2 + ny * y2
        if d1 >= 0:
            result.append((x1, y1))
        if (d1 >= 0) != (d2 >= 0):
            denom = d1 - d2
            if abs(denom) > 1e-12:
                t = d1 / denom
                result.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    return result


def _clip_to_quadrant(pts: list[tuple[float, float]],
                      quarter: int) -> list[tuple[float, float]]:
    """Clip a polygon to one quadrant (0=Q1 x>=0 y>=0, CCW)."""
    normals = [
        [(0, 1), (1, 0)],
        [(0, 1), (-1, 0)],
        [(0, -1), (-1, 0)],
        [(0, -1), (1, 0)],
    ]
    result = list(pts)
    for nx, ny in normals[quarter]:
        result = _clip_halfplane(result, nx, ny)
    return result


def _gear_ring_quarter_path(outer_pts: list[tuple[float, float]],
                            inner_pts: list[tuple[float, float]],
                            quarter: int,
                            cx: float, cy: float) -> str:
    """Build a single closed path for one gear ring quarter.

    Traces the clipped outer profile forward, then the clipped inner
    profile reversed, connected by straight radial lines.
    """
    outer_q = _clip_to_quadrant(outer_pts, quarter)
    inner_q = _clip_to_quadrant(inner_pts, quarter)
    if not outer_q:
        return ""
    parts = []
    for i, (x, y) in enumerate(outer_q):
        cmd = "M" if i == 0 else "L"
        parts.append(f"{cmd}{cx + x:.4f},{cy + y:.4f}")
    if inner_q:
        for x, y in reversed(inner_q):
            parts.append(f"L{cx + x:.4f},{cy + y:.4f}")
    parts.append("Z")
    return " ".join(parts)


def _inner_tab_points(inner_r: float, tab_count: int,
                      tab_w: float, tab_d: float) -> list[tuple[float, float]]:
    """Generate inner tab profile as (x, y) points (closed polygon)."""
    tab_half_w_angle = math.asin(min(tab_w / 2.0 / inner_r, 1.0))
    tab_inner_r = inner_r - tab_d
    tab_angles = [i * 2 * math.pi / tab_count for i in range(tab_count)]
    points: list[tuple[float, float]] = []
    for idx in range(tab_count):
        a_center = tab_angles[idx]
        a_next = tab_angles[(idx + 1) % tab_count]
        if idx < tab_count - 1:
            a_next_actual = a_next
        else:
            a_next_actual = tab_angles[0] + 2 * math.pi

        a1 = a_center - tab_half_w_angle
        a2 = a_center + tab_half_w_angle

        points.append((inner_r * math.cos(a1), inner_r * math.sin(a1)))
        points.append((tab_inner_r * math.cos(a1), tab_inner_r * math.sin(a1)))
        points.append((tab_inner_r * math.cos(a2), tab_inner_r * math.sin(a2)))
        points.append((inner_r * math.cos(a2), inner_r * math.sin(a2)))

        arc_start = a2
        arc_end = a_next_actual - tab_half_w_angle
        n_seg = max(4, int(30 * (arc_end - arc_start) / (2 * math.pi)))
        for i in range(1, n_seg + 1):
            a = arc_start + (arc_end - arc_start) * i / n_seg
            points.append((inner_r * math.cos(a), inner_r * math.sin(a)))
    return points


def _offset_path(path_str: str, cx: float, cy: float) -> str:
    """Offset all coordinates in an SVG path string."""
    result = []
    for token in path_str.split():
        if token == "Z":
            result.append("Z")
            continue
        cmd = token[0]
        coords = token[1:].split(",")
        x = float(coords[0]) + cx
        y = float(coords[1]) + cy
        result.append(f"{cmd}{x:.4f},{y:.4f}")
    return " ".join(result)


def _quarter_disc_path(quarter: int, outer_r: float, inner_r: float,
                       n_joints: int, tab_w: float, tab_d: float,
                       cx: float, cy: float,
                       points_per_arc: int = 40) -> str:
    """One quarter of an annular disc with rectangular interlocking joints.

    Tabs perpendicular to the radial edge, complementary between adjacent
    quarters: start-edge even positions are tabs, end-edge even are slots.
    """
    base_a = quarter * math.pi / 2.0
    end_a = base_a + math.pi / 2.0

    cos_s, sin_s = math.cos(base_a), math.sin(base_a)
    cos_e, sin_e = math.cos(end_a), math.sin(end_a)
    perp_sx, perp_sy = -sin_s, cos_s
    perp_ex, perp_ey = sin_e, -cos_e

    half_w = tab_w / 2.0
    centers = [inner_r + (outer_r - inner_r) * (i + 1) / (n_joints + 1)
               for i in range(n_joints)]

    pts: list[str] = []

    def add(x: float, y: float, first: bool = False):
        cmd = "M" if first else "L"
        pts.append(f"{cmd}{cx + x:.4f},{cy + y:.4f}")

    add(inner_r * cos_s, inner_r * sin_s, first=True)

    for j, rc in enumerate(centers):
        r_lo, r_hi = rc - half_w, rc + half_w
        sign = -1.0 if j % 2 == 0 else 1.0
        dx, dy = sign * perp_sx * tab_d, sign * perp_sy * tab_d
        add(r_lo * cos_s, r_lo * sin_s)
        add(r_lo * cos_s + dx, r_lo * sin_s + dy)
        add(r_hi * cos_s + dx, r_hi * sin_s + dy)
        add(r_hi * cos_s, r_hi * sin_s)

    add(outer_r * cos_s, outer_r * sin_s)

    for i in range(1, points_per_arc + 1):
        a = base_a + (end_a - base_a) * i / points_per_arc
        add(outer_r * math.cos(a), outer_r * math.sin(a))

    for j, rc in enumerate(reversed(centers)):
        orig_j = n_joints - 1 - j
        r_hi, r_lo = rc + half_w, rc - half_w
        sign = 1.0 if orig_j % 2 == 0 else -1.0
        dx, dy = sign * perp_ex * tab_d, sign * perp_ey * tab_d
        add(r_hi * cos_e, r_hi * sin_e)
        add(r_hi * cos_e + dx, r_hi * sin_e + dy)
        add(r_lo * cos_e + dx, r_lo * sin_e + dy)
        add(r_lo * cos_e, r_lo * sin_e)

    add(inner_r * cos_e, inner_r * sin_e)

    for i in range(1, points_per_arc + 1):
        a = end_a + (base_a - end_a) * i / points_per_arc
        add(inner_r * math.cos(a), inner_r * math.sin(a))

    pts.append("Z")
    return " ".join(pts)


def _make_svg(path: Path, size: float, margin: float = 5.0) -> svgwrite.Drawing:
    total = size + 2 * margin
    return svgwrite.Drawing(
        str(path),
        size=(f"{total:.1f}mm", f"{total:.1f}mm"),
        viewBox=f"0 0 {total:.1f} {total:.1f}",
    )


def _make_rect_svg(path: Path, w: float, h: float,
                   margin: float = 5.0) -> svgwrite.Drawing:
    tw, th = w + 2 * margin, h + 2 * margin
    return svgwrite.Drawing(
        str(path),
        size=(f"{tw:.1f}mm", f"{th:.1f}mm"),
        viewBox=f"0 0 {tw:.1f} {th:.1f}",
    )


def _rect_with_tabs(w: float, h: float, cx: float, cy: float,
                    tab_w: float, tab_d: float,
                    n_w: int, n_h: int) -> str:
    """Rectangle with outward tabs on all 4 edges for box joints."""
    pts: list[str] = []
    half_w, half_h = w / 2.0, h / 2.0

    def _edge(x0, y0, x1, y1, n_tabs, outward_nx, outward_ny):
        dx = (x1 - x0) / (2 * n_tabs + 1)
        dy = (y1 - y0) / (2 * n_tabs + 1)
        for i in range(2 * n_tabs + 1):
            sx = x0 + dx * i
            sy = y0 + dy * i
            ex = sx + dx
            ey = sy + dy
            if i % 2 == 1:
                pts.append(f"L{sx + outward_nx * tab_d:.4f},"
                           f"{sy + outward_ny * tab_d:.4f}")
                pts.append(f"L{ex + outward_nx * tab_d:.4f},"
                           f"{ey + outward_ny * tab_d:.4f}")
                pts.append(f"L{ex:.4f},{ey:.4f}")
            else:
                pts.append(f"L{ex:.4f},{ey:.4f}")

    x0, y0 = cx - half_w, cy - half_h
    pts.append(f"M{x0:.4f},{y0:.4f}")
    _edge(x0, y0, cx + half_w, y0, n_w, 0, -1)
    _edge(cx + half_w, y0, cx + half_w, cy + half_h, n_h, 1, 0)
    _edge(cx + half_w, cy + half_h, x0, cy + half_h, n_w, 0, 1)
    _edge(x0, cy + half_h, x0, y0, n_h, 0, 0)
    pts.append("Z")
    return " ".join(pts)


def _wall_path(length: float, height: float, cx: float, cy: float,
               n_tabs: int, tab_w: float, tab_d: float,
               cutout: tuple[float, float, float, float] | None = None) -> str:
    """Rectangular wall with bottom-edge tabs and optional cutout."""
    half_l, half_h = length / 2.0, height / 2.0
    pts: list[str] = []

    tl = (cx - half_l, cy - half_h)
    tr = (cx + half_l, cy - half_h)
    br = (cx + half_l, cy + half_h)
    bl = (cx - half_l, cy + half_h)

    pts.append(f"M{tl[0]:.4f},{tl[1]:.4f}")
    pts.append(f"L{tr[0]:.4f},{tr[1]:.4f}")

    if cutout:
        cx1, cy1, cx2, cy2 = cutout
        pts.append(f"L{cx2:.4f},{tr[1]:.4f}")
        pts.append(f"L{cx2:.4f},{cy1:.4f}")
        pts.append(f"L{cx1:.4f},{cy1:.4f}")
        pts.append(f"L{cx1:.4f},{tr[1]:.4f}")

    pts.append(f"L{br[0]:.4f},{br[1]:.4f}")

    seg = length / (2 * n_tabs + 1)
    for i in range(2 * n_tabs + 1):
        sx = br[0] - seg * i
        ex = br[0] - seg * (i + 1)
        if i % 2 == 1:
            pts.append(f"L{sx:.4f},{bl[1] + tab_d:.4f}")
            pts.append(f"L{ex:.4f},{bl[1] + tab_d:.4f}")
            pts.append(f"L{ex:.4f},{bl[1]:.4f}")
        else:
            pts.append(f"L{ex:.4f},{bl[1]:.4f}")

    pts.append("Z")
    return " ".join(pts)


def _add_crank_holes(dwg, cx: float, cy: float):
    for i in range(CRANK_HOLE_COUNT):
        a = i * 2 * math.pi / CRANK_HOLE_COUNT
        hx = cx + CRANK_HOLE_RING_RADIUS_MM * math.cos(a)
        hy = cy + CRANK_HOLE_RING_RADIUS_MM * math.sin(a)
        dwg.add(dwg.circle(center=(hx, hy), r=CRANK_HOLE_RADIUS_MM, **CUT_STROKE))


def generate_lasercut_svg(output_dir: str | Path) -> list[Path]:
    """Generate individual laser-cut SVGs for each part."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    ring_tip_r = (GEAR_RING_TEETH * GEAR_MODULE + 2 * GEAR_MODULE) / 2.0
    drive_tip_r = (DRIVE_GEAR_TEETH * GEAR_MODULE + 2 * GEAR_MODULE) / 2.0
    center_distance = (GEAR_RING_TEETH + DRIVE_GEAR_TEETH) * GEAR_MODULE / 2.0
    margin = 5.0
    joint_w = 8.0
    joint_d = MDF_THICKNESS_MM
    n_joints = 5
    disc_inner_r = PEG_DIAMETER_MM / 2.0 + 1.0

    gear_pts = _gear_profile(GEAR_RING_TEETH, GEAR_MODULE, GEAR_PRESSURE_ANGLE_DEG)

    path = output_dir / "gear_ring.svg"
    dwg = _make_svg(path, ring_tip_r * 2, margin)
    c = ring_tip_r + margin
    dwg.add(dwg.path(d=_points_to_svg_path(gear_pts, c, c), **CUT_STROKE))
    dwg.add(dwg.circle(center=(c, c), r=PEG_DIAMETER_MM / 2.0, **CUT_STROKE))
    dwg.save()
    generated.append(path)

    path = output_dir / "disc.svg"
    dwg = _make_svg(path, DISC_RADIUS_MM * 2, margin)
    c = DISC_RADIUS_MM + margin
    dwg.add(dwg.circle(center=(c, c), r=DISC_RADIUS_MM, **CUT_STROKE))
    dwg.add(dwg.circle(center=(c, c), r=PEG_DIAMETER_MM / 2.0, **CUT_STROKE))
    dwg.save()
    generated.append(path)

    path = output_dir / "drive_gear.svg"
    dwg = _make_svg(path, drive_tip_r * 2, margin)
    c = drive_tip_r + margin
    drive_profile = _gear_profile(DRIVE_GEAR_TEETH, GEAR_MODULE, GEAR_PRESSURE_ANGLE_DEG)
    dwg.add(dwg.path(d=_points_to_svg_path(drive_profile, c, c), **CUT_STROKE))
    dwg.add(dwg.circle(center=(c, c), r=PEG_DIAMETER_MM / 2.0, **CUT_STROKE))
    dwg.save()
    generated.append(path)

    t = MDF_THICKNESS_MM
    outer_w = BOX_INNER_W_MM + 2 * t
    outer_d = BOX_INNER_D_MM + 2 * t
    n_tabs_w = 4
    n_tabs_d = 5
    ring_tip_r_local = (GEAR_RING_TEETH * GEAR_MODULE + 2 * GEAR_MODULE) / 2.0
    disc_cx = t + ring_tip_r_local + 5.0
    disc_cy = t + BOX_DISC_OFFSET_Y_MM
    spacer_clearance_r = (SPACER_DIAMETER_MM + 2.0) / 2.0

    path = output_dir / "base_plate.svg"
    dwg = _make_rect_svg(path, outer_w, outer_d, margin)
    mx, my = margin, margin
    base_path = _rect_with_tabs(
        BOX_INNER_W_MM, BOX_INNER_D_MM,
        mx + outer_w / 2.0, my + outer_d / 2.0,
        BOX_WALL_TAB_W_MM, BOX_WALL_TAB_D_MM, n_tabs_w, n_tabs_d,
    )
    dwg.add(dwg.path(d=base_path, **CUT_STROKE))
    dwg.add(dwg.circle(
        center=(mx + disc_cx, my + disc_cy),
        r=spacer_clearance_r, **CUT_STROKE,
    ))
    dwg.add(dwg.circle(
        center=(mx + disc_cx + center_distance, my + disc_cy),
        r=PEG_DIAMETER_MM / 2.0, **CUT_STROKE,
    ))
    dwg.save()
    generated.append(path)

    wall_h = BOX_WALL_H_MM

    for name, length, n_tabs in [
        ("wall_front", outer_w, n_tabs_w),
        ("wall_back", outer_w, n_tabs_w),
        ("wall_left", BOX_INNER_D_MM, n_tabs_d),
    ]:
        path = output_dir / f"{name}.svg"
        dwg = _make_rect_svg(path, length, wall_h, margin)
        cx_w = margin + length / 2.0
        cy_w = margin + wall_h / 2.0
        wall = _wall_path(
            length, wall_h, cx_w, cy_w,
            n_tabs, BOX_WALL_TAB_W_MM, BOX_WALL_TAB_D_MM,
        )
        dwg.add(dwg.path(d=wall, **CUT_STROKE))
        dwg.save()
        generated.append(path)

    for i in range(2):
        path = output_dir / f"washer_{i + 1}.svg"
        dwg = _make_svg(path, WASHER_OUTER_R_MM * 2, margin)
        c = WASHER_OUTER_R_MM + margin
        dwg.add(dwg.circle(center=(c, c), r=WASHER_OUTER_R_MM, **CUT_STROKE))
        dwg.add(dwg.circle(center=(c, c), r=WASHER_HOLE_R_MM, **CUT_STROKE))
        dwg.save()
        generated.append(path)

    bridge_l = DISC_RADIUS_MM * 2.0
    bridge_w = 30.0
    path = output_dir / "gear_bridge.svg"
    dwg = _make_rect_svg(path, bridge_l, bridge_w, margin)
    dwg.add(dwg.rect(insert=(margin, margin), size=(bridge_l, bridge_w), **CUT_STROKE))
    dwg.add(dwg.circle(center=(margin + bridge_l / 2.0, margin + bridge_w / 2.0),
                        r=PEG_DIAMETER_MM / 2.0, **CUT_STROKE))
    dwg.save()
    generated.append(path)

    spacer_r = SPACER_DIAMETER_MM / 2.0
    path = output_dir / "spacer.svg"
    dwg = _make_svg(path, SPACER_DIAMETER_MM, margin)
    c = spacer_r + margin
    dwg.add(dwg.circle(center=(c, c), r=spacer_r, **CUT_STROKE))
    dwg.add(dwg.circle(center=(c, c), r=PEG_DIAMETER_MM / 2.0, **CUT_STROKE))
    dwg.save()
    generated.append(path)

    return generated


def generate_sheets_svg(output_dir: str | Path) -> list[Path]:
    """Nest all laser-cut parts onto 300x300 mm sheets (3 sheets)."""
    output_dir = Path(output_dir) / "sheets"
    output_dir.mkdir(parents=True, exist_ok=True)

    t = MDF_THICKNESS_MM
    ring_tip_r = (GEAR_RING_TEETH * GEAR_MODULE + 2 * GEAR_MODULE) / 2.0
    drive_tip_r = (DRIVE_GEAR_TEETH * GEAR_MODULE + 2 * GEAR_MODULE) / 2.0
    center_distance = (GEAR_RING_TEETH + DRIVE_GEAR_TEETH) * GEAR_MODULE / 2.0
    outer_w = BOX_INNER_W_MM + 2 * t
    outer_d = BOX_INNER_D_MM + 2 * t
    n_tabs_w, n_tabs_d = 4, 5
    disc_cx = t + ring_tip_r + 5.0
    disc_cy = t + BOX_DISC_OFFSET_Y_MM
    sw, sh = SHEET_W_MM, SHEET_H_MM
    pad = 5.0
    gap = 3.0
    wall_h = BOX_WALL_H_MM
    joint_w = 8.0
    joint_d = MDF_THICKNESS_MM
    n_joints = 5
    disc_inner_r = PEG_DIAMETER_MM / 2.0 + 1.0

    sheets: list[Path] = []

    def _new_sheet(idx: int) -> tuple[svgwrite.Drawing, Path]:
        p = output_dir / f"sheet_{idx}.svg"
        d = svgwrite.Drawing(
            str(p),
            size=(f"{sw:.1f}mm", f"{sh:.1f}mm"),
            viewBox=f"0 0 {sw:.1f} {sh:.1f}",
        )
        return d, p

    dwg, sp = _new_sheet(1)
    wall_cy_top = pad + wall_h / 2.0
    wall = _wall_path(outer_w, wall_h, sw / 2.0, wall_cy_top, n_tabs_w,
                      BOX_WALL_TAB_W_MM, BOX_WALL_TAB_D_MM)
    dwg.add(dwg.path(d=wall, **CUT_STROKE))
    plate_top = wall_cy_top + wall_h / 2.0 + BOX_WALL_TAB_D_MM + gap
    bp_cy = plate_top + outer_d / 2.0
    spacer_clearance_r = (SPACER_DIAMETER_MM + 2.0) / 2.0
    base_path = _rect_with_tabs(
        BOX_INNER_W_MM, BOX_INNER_D_MM,
        sw / 2.0, bp_cy,
        BOX_WALL_TAB_W_MM, BOX_WALL_TAB_D_MM, n_tabs_w, n_tabs_d,
    )
    bp_ox = (sw - outer_w) / 2.0
    bp_oy = bp_cy - outer_d / 2.0
    dwg.add(dwg.path(d=base_path, **CUT_STROKE))
    dwg.add(dwg.circle(
        center=(bp_ox + disc_cx, bp_oy + disc_cy),
        r=spacer_clearance_r, **CUT_STROKE))
    dwg.add(dwg.circle(
        center=(bp_ox + disc_cx + center_distance, bp_oy + disc_cy),
        r=PEG_DIAMETER_MM / 2.0, **CUT_STROKE))
    plate_bottom = bp_cy + outer_d / 2.0 + BOX_WALL_TAB_D_MM
    wall_cy_bot = plate_bottom + gap + wall_h / 2.0
    wall = _wall_path(outer_w, wall_h, sw / 2.0, wall_cy_bot, n_tabs_w,
                      BOX_WALL_TAB_W_MM, BOX_WALL_TAB_D_MM)
    dwg.add(dwg.path(d=wall, **CUT_STROKE))
    dwg.save()
    sheets.append(sp)

    dwg, sp = _new_sheet(2)
    g = dwg.g(transform=f"rotate(180,{sw / 2.0},{sh / 2.0})")
    gear_pts = _gear_profile(GEAR_RING_TEETH, GEAR_MODULE, GEAR_PRESSURE_ANGLE_DEG)
    gc = ring_tip_r
    g.add(dwg.path(d=_points_to_svg_path(gear_pts, gc, gc), **CUT_STROKE))
    g.add(dwg.circle(center=(gc, gc), r=PEG_DIAMETER_MM / 2.0, **CUT_STROKE))
    g.add(dwg.circle(center=(gc, gc), r=DISC_RADIUS_MM, **CUT_STROKE))
    gear_edge = ring_tip_r * 2
    col_x = gear_edge + gap
    col_cx = col_x + drive_tip_r
    ry = drive_tip_r
    drive_profile = _gear_profile(
        DRIVE_GEAR_TEETH, GEAR_MODULE, GEAR_PRESSURE_ANGLE_DEG)
    g.add(dwg.path(
        d=_points_to_svg_path(drive_profile, col_cx, ry), **CUT_STROKE))
    g.add(dwg.circle(
        center=(col_cx, ry), r=PEG_DIAMETER_MM / 2.0, **CUT_STROKE))
    ry += drive_tip_r + gap + WASHER_OUTER_R_MM
    for _ in range(2):
        g.add(dwg.circle(center=(col_cx, ry), r=WASHER_OUTER_R_MM, **CUT_STROKE))
        g.add(dwg.circle(center=(col_cx, ry), r=WASHER_HOLE_R_MM, **CUT_STROKE))
        ry += WASHER_OUTER_R_MM * 2 + gap
    spacer_r = SPACER_DIAMETER_MM / 2.0
    ry += spacer_r - WASHER_OUTER_R_MM
    for i in range(SPACER_COUNT):
        g.add(dwg.circle(center=(col_cx, ry), r=spacer_r, **CUT_STROKE))
        g.add(dwg.circle(center=(col_cx, ry), r=PEG_DIAMETER_MM / 2.0, **CUT_STROKE))
        ry += SPACER_DIAMETER_MM + gap
    bridge_l = DISC_RADIUS_MM * 2.0
    bridge_w = 30.0
    bridge_y = gear_edge + 1.0
    g.add(dwg.rect(insert=(0, bridge_y), size=(bridge_l, bridge_w), **CUT_STROKE))
    g.add(dwg.circle(center=(bridge_l / 2.0, bridge_y + bridge_w / 2.0),
                      r=PEG_DIAMETER_MM / 2.0, **CUT_STROKE))
    wall_cy = bridge_y + bridge_w + 1.0 + wall_h / 2.0
    wall = _wall_path(BOX_INNER_D_MM, wall_h, sw / 2.0, wall_cy, n_tabs_d,
                      BOX_WALL_TAB_W_MM, BOX_WALL_TAB_D_MM)
    g.add(dwg.path(d=wall, **CUT_STROKE))
    dwg.add(g)
    dwg.save()
    sheets.append(sp)

    return sheets


if __name__ == "__main__":
    out = generate_disc_svg("fiducial_disc.svg")
    print(f"Generated: {out} and {out.with_suffix('.json')}")
