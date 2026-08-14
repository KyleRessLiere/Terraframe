"""Human-readable timestamps for run directories.

Run folders are read by people far more often than by machines, so they are
named for legibility: ``2026-08-13_11-27-00PM-EDT`` rather than ``20260813-232700``.

The zone abbreviation is derived, not hardcoded. US Eastern is EST for part of
the year and EDT for the rest, and stamping an August run "EST" would be simply
wrong.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["RUN_TIMEZONE", "newest_run", "run_stamp"]

#: Zone run folders are stamped in. Override with the TERRFRAME_TZ env var.
RUN_TIMEZONE = os.environ.get("TERRFRAME_TZ", "America/New_York")

#: strftime pattern: dashed date, dashed 12-hour clock, then the zone.
STAMP_FORMAT = "%Y-%m-%d_%I-%M-%S%p-%Z"


def _now() -> datetime:
    """Current time in :data:`RUN_TIMEZONE`, falling back to local time.

    ``zoneinfo`` needs the ``tzdata`` package on Windows, which ships no system
    tz database. A missing zone must not stop a render, so this degrades to the
    machine's own local time rather than raising.
    """
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(RUN_TIMEZONE))
    except Exception:
        return datetime.now(timezone.utc).astimezone()


def run_stamp(label: str = "") -> str:
    """Build a run-folder name, optionally suffixed with a label.

    Args:
        label: Free text appended after an underscore, e.g. ``"tuning"``.

    Returns:
        Something like ``2026-08-13_11-27-00PM-EDT_tuning``.
    """
    stamp = _now().strftime(STAMP_FORMAT)
    # %p is uppercase on some platforms and lowercase on others.
    stamp = stamp.replace("am-", "AM-").replace("pm-", "PM-")
    return f"{stamp}_{label}" if label else stamp


def newest_run(directory: str | os.PathLike[str]) -> Path | None:
    """Most recent run folder under ``directory``, by modification time.

    Deliberately not by name: a 12-hour clock does not sort lexically, since
    ``11-00AM`` orders after ``07-00PM``. Readability won for the folder name,
    so ordering uses the filesystem's own timestamps instead.

    Args:
        directory: Folder holding run directories.

    Returns:
        The newest run directory, or ``None`` if there are none.
    """
    root = Path(directory)
    if not root.is_dir():
        return None
    runs = [p for p in root.iterdir() if p.is_dir()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None
