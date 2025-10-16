#!/usr/bin/env python3
"""Fix remaining critical mypy errors."""

from pathlib import Path


def fix_file(filepath: str, replacements: list):
    """Apply a list of (old, new) replacements to a file."""
    path = Path(filepath)
    content = path.read_text()

    for old, new in replacements:
        content = content.replace(old, new)

    path.write_text(content)
    print(f"Fixed {filepath}")


# Fix rosetta/models/base.py - best_metric type issues
fix_file(
    "rosetta/models/base.py",
    [
        (
            "        self.best_metric = None",
            "        self.best_metric: Optional[float] = None",
        ),
        (
            "        self.best_model_checkpoint = None",
            "        self.best_model_checkpoint: Optional[str] = None",
        ),
        (
            "            self.best_model_checkpoint: Optional[str] = checkpoint_dir",
            "            self.best_model_checkpoint = checkpoint_dir",
        ),
    ],
)

# Fix rosetta/models/token_tasks.py - CRF type
fix_file(
    "rosetta/models/token_tasks.py",
    [
        (
            "        if use_crf:\n            self.crf = ConditionalRandomField(num_labels)\n        else:\n            self.crf = None",
            "        if use_crf:\n            self.crf: Optional[ConditionalRandomField] = ConditionalRandomField(num_labels)\n        else:\n            self.crf: Optional[ConditionalRandomField] = None",
        ),
    ],
)

# Fix rosetta/data/annotation.py - Relation missing evidence_text
fix_file(
    "rosetta/data/annotation.py",
    [
        (
            "                        relation = Relation(\n                            head=head,\n                            tail=tail,\n                            label=relation_label,\n                            confidence=confidence,\n                        )",
            "                        relation = Relation(\n                            head=head,\n                            tail=tail,\n                            label=relation_label,\n                            confidence=confidence,\n                            evidence_text=None,\n                        )",
        ),
    ],
)

# Fix rosetta/evaluation/metrics.py - callable type
fix_file(
    "rosetta/evaluation/metrics.py",
    [
        (
            "from typing import Dict, List, Optional, Tuple, Union",
            "from typing import Callable, Dict, List, Optional, Tuple, Union",
        ),
        (
            "    compute_metrics: Optional[callable] = None,",
            "    compute_metrics: Optional[Callable] = None,",
        ),
    ],
)

print("Applied fixes for critical errors")
