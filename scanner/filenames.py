"""Sanitize client upload filenames."""

from pathlib import Path


def safe_upload_filename(filename: str | None) -> str:
    """Return the bare filename with no parent parts.

    Raises ValueError when nothing usable remains.
    """
    name = Path(filename or "").name
    if not name or name in (".", ".."):
        raise ValueError(f"Invalid upload filename: {filename!r}.")
    return name
