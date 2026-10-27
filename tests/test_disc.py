"""Test fiducial disc generation."""

import json
import tempfile
from pathlib import Path

from scanner.config import MARKER_COUNT, DISC_RADIUS_MM, MARKER_RING_RADIUS_MM
from scanner.disc import generate_disc_svg


def test_disc_generates_svg_and_json():
    with tempfile.TemporaryDirectory() as tmp:
        svg_path = Path(tmp) / "disc.svg"
        result = generate_disc_svg(svg_path)
        assert result.exists()

        json_path = svg_path.with_suffix(".json")
        assert json_path.exists()

        positions = json.loads(json_path.read_text())
        assert len(positions) == MARKER_COUNT

        for marker_id, (x, y) in positions.items():
            r = (x**2 + y**2) ** 0.5
            assert abs(r - MARKER_RING_RADIUS_MM) < 0.01, (
                f"Marker {marker_id} at radius {r}, expected {MARKER_RING_RADIUS_MM}"
            )


def test_svg_contains_markers():
    with tempfile.TemporaryDirectory() as tmp:
        svg_path = Path(tmp) / "disc.svg"
        generate_disc_svg(svg_path)
        content = svg_path.read_text()
        assert f"{DISC_RADIUS_MM * 2 + 10}mm" in content
        assert "rotate(" in content
