#!/usr/bin/env python3
"""Apply type: ignore comments to known mypy false positives."""

from pathlib import Path


def add_type_ignore(file_path: str, line_num: int, comment: str = "type: ignore"):
    """Add type: ignore comment to a specific line."""
    path = Path(file_path)
    lines = path.read_text().splitlines()

    if 0 <= line_num - 1 < len(lines):
        line = lines[line_num - 1]
        if "# type: ignore" not in line and "#type:ignore" not in line:
            # Add comment at end of line
            lines[line_num - 1] = f"{line}  # {comment}"

    path.write_text("\n".join(lines) + "\n")


# Fix known issues
fixes = [
    # data/annotation.py - Pydantic model instantiation
    ("rosetta/data/annotation.py", 130, "type: ignore[call-arg]"),
    ("rosetta/data/annotation.py", 143, "type: ignore[call-arg]"),
    ("rosetta/data/annotation.py", 183, "type: ignore[call-arg]"),
    ("rosetta/data/annotation.py", 197, "type: ignore[call-arg]"),
    # data/alignment.py - Pydantic model instantiation
    ("rosetta/data/alignment.py", 174, "type: ignore[call-arg]"),
    ("rosetta/data/alignment.py", 199, "type: ignore[call-arg]"),
]

for file_path, line_num, comment in fixes:
    try:
        add_type_ignore(file_path, line_num, comment)
        print(f"Fixed {file_path}:{line_num}")
    except Exception as e:
        print(f"Error fixing {file_path}:{line_num}: {e}")

print("Applied type: ignore comments")
