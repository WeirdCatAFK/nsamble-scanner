"""Test upload filename sanitization."""

import pytest

from scanner.filenames import safe_upload_filename


def test_plain_name_is_unchanged():
    assert safe_upload_filename("clip.mp4") == "clip.mp4"


def test_parent_parts_are_stripped():
    assert safe_upload_filename("../evil.mp4") == "evil.mp4"
    assert safe_upload_filename("../../etc/cron") == "cron"
    assert safe_upload_filename("/abs/path/vid.mov") == "vid.mov"
    assert safe_upload_filename("sub/dir/vid.mov") == "vid.mov"


def test_empty_and_dot_names_raise():
    for bad in ["", ".", "..", None]:
        with pytest.raises(ValueError):
            safe_upload_filename(bad)
