"""Safe file handling: temp-then-commit writes, never overwriting a previous report."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_temp(final_path: Path, text: str) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_name(f".{final_path.name}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    return tmp


def non_clobbering(path: Path, suffix_hint: str) -> Path:
    """Return `path` if free, else 'name (suffix_hint).md', 'name (suffix_hint 2).md', ..."""
    if not path.exists():
        return path
    n = 1
    while True:
        label = suffix_hint if n == 1 else f"{suffix_hint} {n}"
        cand = path.with_name(f"{path.stem} ({label}){path.suffix}")
        if not cand.exists():
            return cand
        n += 1


def commit_new(tmp: Path, final_path: Path) -> None:
    """Move a validated temp file into place. Refuses to overwrite (preserve previous reports)."""
    if final_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report {final_path}")
    os.replace(tmp, final_path)


def commit_replace(tmp: Path, final_path: Path) -> Path | None:
    """Atomically replace a regenerated view (master archive), keeping a .bak of the previous version."""
    backup = None
    if final_path.exists():
        backup = final_path.with_name(final_path.name + ".bak")
        shutil.copy2(final_path, backup)
    os.replace(tmp, final_path)
    return backup


def restore_backup(final_path: Path, backup: Path | None) -> None:
    if backup and backup.exists():
        os.replace(backup, final_path)
