#!/usr/bin/env python3
"""Fix all remaining mypy errors."""

from pathlib import Path


def fix_file(filepath: str, replacements: list):
    """Apply a list of (old, new) replacements to a file."""
    path = Path(filepath)
    content = path.read_text()

    for old, new in replacements:
        content = content.replace(old, new)

    path.write_text(content)
    print(f"Fixed {filepath}")


# Fix rosetta/evaluation/metrics.py - callable type
fix_file(
    "rosetta/evaluation/metrics.py",
    [
        (
            "from typing import Any, Dict, List, Optional, Tuple, Union",
            "from typing import Any, Callable, Dict, List, Optional, Tuple, Union",
        ),
        (
            "def bootstrap_confidence_interval(\n    metric_fn: callable,",
            "def bootstrap_confidence_interval(\n    metric_fn: Callable,",
        ),
    ],
)

# Fix rosetta/models/token_tasks.py - forward() signature
fix_file(
    "rosetta/models/token_tasks.py",
    [
        (
            "    def forward(\n        self,\n        input_ids: torch.Tensor,\n        attention_mask: Optional[torch.Tensor] = None,\n        labels: Optional[torch.Tensor] = None,\n        **kwargs,\n    ) -> ModelOutput:",
            "    def forward(  # type: ignore[override]\n        self,\n        input_ids: torch.Tensor,\n        attention_mask: Optional[torch.Tensor] = None,\n        labels: Optional[torch.Tensor] = None,\n        **kwargs,\n    ) -> ModelOutput:",
        ),
    ],
)

# Fix rosetta/models/seq2seq.py - forward() signature
fix_file(
    "rosetta/models/seq2seq.py",
    [
        (
            "    def forward(\n        self,\n        input_ids: torch.Tensor,\n        attention_mask: Optional[torch.Tensor] = None,\n        labels: Optional[torch.Tensor] = None,\n        decoder_input_ids: Optional[torch.Tensor] = None,\n        decoder_attention_mask: Optional[torch.Tensor] = None,\n        **kwargs,\n    ) -> ModelOutput:",
            "    def forward(  # type: ignore[override]\n        self,\n        input_ids: torch.Tensor,\n        attention_mask: Optional[torch.Tensor] = None,\n        labels: Optional[torch.Tensor] = None,\n        decoder_input_ids: Optional[torch.Tensor] = None,\n        decoder_attention_mask: Optional[torch.Tensor] = None,\n        **kwargs,\n    ) -> ModelOutput:",
        ),
    ],
)

# Fix rosetta/models/pretraining.py - model.num_parameters() type issue
fix_file(
    "rosetta/models/pretraining.py",
    [
        (
            '        logger.info(\n            f"Initialized pretrainer with {self.model.num_parameters():,} parameters"\n        )',
            '        logger.info(\n            f"Initialized pretrainer with {self.model.num_parameters():,} parameters"  # type: ignore[attr-defined]\n        )',
        ),
    ],
)

print("Applied all mypy fixes")
