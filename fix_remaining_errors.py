#!/usr/bin/env python3
"""Fix all remaining mypy errors."""

from pathlib import Path


def fix_file(filepath: str, replacements: list):
    """Apply a list of (old, new) replacements to a file."""
    path = Path(filepath)
    content = path.read_text()

    for old, new in replacements:
        if isinstance(old, str):
            content = content.replace(old, new)
        else:  # regex pattern
            content = old.sub(new, content)

    path.write_text(content)
    print(f"Fixed {filepath}")


# Fix rosetta/data/cleaning.py - Pattern type and regex error
fix_file(
    "rosetta/data/cleaning.py",
    [
        (
            "import re\nimport unicodedata\nfrom typing import Dict, List, Optional, Set, Tuple",
            "import re\nimport unicodedata\nfrom typing import Dict, List, Optional, Pattern, Set, Tuple",
        ),
        (
            '        # Common gap notations: [...], &, ..., ---\n        text = re.sub(r"\\.{3,}|\\u2026|+|-{3,}", self.gap_marker, text)',
            '        # Common gap notations: [...], &, ..., ---\n        text = re.sub(r"\\.{3,}|\\u2026|\\+|-{3,}", self.gap_marker, text)',
        ),
        (
            "        if delimiters:\n            self.delimiter_pattern = re.compile(f\"[{''.join(delimiters)}]\")\n        else:\n            self.delimiter_pattern = None",
            "        if delimiters:\n            self.delimiter_pattern: Optional[Pattern[str]] = re.compile(f\"[{''.join(delimiters)}]\")\n        else:\n            self.delimiter_pattern = None",
        ),
    ],
)

# Fix rosetta/api/main.py - Add type ignore comments for Pydantic model differences
# The API models (Entity, Relation) are different from rosetta.data.schemas versions
# They're simpler request/response models, not the full domain models
fix_file(
    "rosetta/api/main.py",
    [
        (
            'class Entity(BaseModel):\n    """Named entity with position and label."""',
            'class Entity(BaseModel):  # type: ignore[misc]\n    """Named entity with position and label."""',
        ),
        (
            'class Relation(BaseModel):\n    """Relation between two entities."""',
            'class Relation(BaseModel):  # type: ignore[misc]\n    """Relation between two entities."""',
        ),
    ],
)

print("Applied all remaining mypy fixes")
