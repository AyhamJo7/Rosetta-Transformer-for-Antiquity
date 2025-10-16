"""Human-in-the-loop feedback system for Rosetta Transformer.

This module provides:
- FeedbackStore: Store and manage user feedback
- Correction tracking for model predictions
- Active learning sample selection
- Export to retraining format
"""

import json
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from rosetta.utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Feedback Store
# ============================================================================


class FeedbackStore:
    """Store for managing user feedback and corrections.

    Supports both SQLite database and JSON file backend.
    Tracks corrections for active learning and model improvement.
    """

    def __init__(
        self,
        backend: str = "sqlite",
        db_path: Optional[str] = None,
        json_path: Optional[str] = None,
    ):
        """Initialize feedback store.

        Args:
            backend: Storage backend ('sqlite' or 'json')
            db_path: Path to SQLite database (for sqlite backend)
            json_path: Path to JSON file (for json backend)
        """
        self.backend = backend

        if backend == "sqlite":
            self.db_path = Path(db_path) if db_path else Path("data/feedback.db")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()
        elif backend == "json":
            self.json_path = (
                Path(json_path) if json_path else Path("data/feedback.json")
            )
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_json()
        else:
            raise ValueError(f"Unknown backend: {backend}")

        logger.info(f"Feedback store initialized with {backend} backend")

    def _init_sqlite(self) -> None:
        """Initialize SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Create feedback table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                input_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                correction TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                used_for_training INTEGER DEFAULT 0
            )
        """
        )

        # Create indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_task_type ON feedback(task_type)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_created_at ON feedback(created_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_used_for_training ON feedback(used_for_training)"
        )

        conn.commit()
        conn.close()

        logger.info(f"SQLite database initialized at {self.db_path}")

    def _init_json(self) -> None:
        """Initialize JSON file."""
        if not self.json_path.exists():
            with open(self.json_path, "w") as f:
                json.dump({"feedback": []}, f)

    async def add_feedback(
        self,
        task_type: str,
        input_text: str,
        prediction: Union[Dict, List, str],
        correction: Union[Dict, List, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add feedback entry.

        Args:
            task_type: Type of task (ner, relation, transliteration, translation)
            input_text: Original input text
            prediction: Model prediction
            correction: User correction
            metadata: Additional metadata

        Returns:
            Feedback ID
        """
        feedback_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        # Serialize prediction and correction
        prediction_str = (
            json.dumps(prediction) if not isinstance(prediction, str) else prediction
        )
        correction_str = (
            json.dumps(correction) if not isinstance(correction, str) else correction
        )
        metadata_str = json.dumps(metadata) if metadata else None

        if self.backend == "sqlite":
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO feedback (id, task_type, input_text, prediction, correction, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    feedback_id,
                    task_type,
                    input_text,
                    prediction_str,
                    correction_str,
                    metadata_str,
                    created_at,
                ),
            )

            conn.commit()
            conn.close()

        elif self.backend == "json":
            # Load existing data
            with open(self.json_path, "r") as f:
                data = json.load(f)

            # Add new feedback
            data["feedback"].append(
                {
                    "id": feedback_id,
                    "task_type": task_type,
                    "input_text": input_text,
                    "prediction": prediction,
                    "correction": correction,
                    "metadata": metadata,
                    "created_at": created_at,
                    "used_for_training": False,
                }
            )

            # Save
            with open(self.json_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Feedback added: {feedback_id} (task: {task_type})")

        return feedback_id

    async def get_feedback(
        self,
        task_type: Optional[str] = None,
        limit: Optional[int] = None,
        unused_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Get feedback entries.

        Args:
            task_type: Filter by task type
            limit: Maximum number of entries
            unused_only: Only get entries not yet used for training

        Returns:
            List of feedback entries
        """
        results: List[Dict[str, Any]] = []
        if self.backend == "sqlite":
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            query = "SELECT * FROM feedback WHERE 1=1"
            params = []

            if task_type:
                query += " AND task_type = ?"
                params.append(task_type)

            if unused_only:
                query += " AND used_for_training = 0"

            query += " ORDER BY created_at DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)

            columns = [desc[0] for desc in cursor.description]
            results = []

            for row in cursor.fetchall():
                entry = dict(zip(columns, row))

                # Parse JSON fields
                entry["prediction"] = json.loads(entry["prediction"])
                entry["correction"] = json.loads(entry["correction"])
                if entry["metadata"]:
                    entry["metadata"] = json.loads(entry["metadata"])

                results.append(entry)

            conn.close()

            return results

        elif self.backend == "json":
            with open(self.json_path, "r") as f:
                data = json.load(f)

            results = data["feedback"]

            # Apply filters
            if task_type:
                results = [r for r in results if r["task_type"] == task_type]

            if unused_only:
                results = [r for r in results if not r.get("used_for_training", False)]

            # Sort by created_at (descending)
            results.sort(key=lambda x: x["created_at"], reverse=True)

            # Limit
            if limit:
                results = results[:limit]

            return results

    async def get_statistics(self) -> Dict[str, int]:
        """Get feedback statistics.

        Returns:
            Dictionary with counts by task type
        """
        if self.backend == "sqlite":
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT task_type, COUNT(*) as count
                FROM feedback
                GROUP BY task_type
            """
            )

            stats = dict(cursor.fetchall())
            conn.close()

            return stats

        elif self.backend == "json":
            with open(self.json_path, "r") as f:
                data = json.load(f)

            stats = defaultdict(int)
            for entry in data["feedback"]:
                stats[entry["task_type"]] += 1

            return dict(stats)

        return {}

    async def mark_used_for_training(self, feedback_ids: List[str]) -> None:
        """Mark feedback entries as used for training.

        Args:
            feedback_ids: List of feedback IDs to mark
        """
        if self.backend == "sqlite":
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            placeholders = ",".join("?" * len(feedback_ids))
            cursor.execute(
                f"""
                UPDATE feedback
                SET used_for_training = 1
                WHERE id IN ({placeholders})
            """,
                feedback_ids,
            )

            conn.commit()
            conn.close()

        elif self.backend == "json":
            with open(self.json_path, "r") as f:
                data = json.load(f)

            for entry in data["feedback"]:
                if entry["id"] in feedback_ids:
                    entry["used_for_training"] = True

            with open(self.json_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Marked {len(feedback_ids)} feedback entries as used")

    async def export_for_training(
        self,
        output_dir: str,
        task_type: Optional[str] = None,
        format: str = "jsonl",
        unused_only: bool = True,
    ) -> Dict[str, str]:
        """Export feedback to training format.

        Args:
            output_dir: Output directory
            task_type: Filter by task type (exports all if None)
            format: Output format (jsonl, conll)
            unused_only: Only export unused feedback

        Returns:
            Dictionary mapping task types to output file paths
        """
        output_dir_path: Path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Get feedback
        feedback = await self.get_feedback(task_type=task_type, unused_only=unused_only)

        # Group by task type
        by_task = defaultdict(list)
        for entry in feedback:
            by_task[entry["task_type"]].append(entry)

        output_files: Dict[str, str] = {}

        # Export each task type
        for task, entries in by_task.items():
            output_file: Path
            if format == "jsonl":
                output_file = output_dir_path / f"{task}_feedback.jsonl"
                self._export_jsonl(entries, output_file, task)
            elif format == "conll":
                if task == "ner":
                    output_file = output_dir_path / f"{task}_feedback.conll"
                    self._export_conll_ner(entries, output_file)
                else:
                    logger.warning(f"CoNLL format not supported for task: {task}")
                    continue
            else:
                raise ValueError(f"Unknown format: {format}")

            output_files[task] = str(output_file)
            logger.info(f"Exported {len(entries)} {task} feedback to {output_file}")

        return output_files

    def _export_jsonl(
        self, entries: List[Dict[str, Any]], output_file: Path, task_type: str
    ) -> None:
        """Export feedback to JSONL format.

        Args:
            entries: Feedback entries
            output_file: Output file path
            task_type: Task type
        """
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in entries:
                # Create training example
                example = {
                    "text": entry["input_text"],
                    "prediction": entry["prediction"],
                    "label": entry["correction"],
                    "task": task_type,
                    "feedback_id": entry["id"],
                }

                f.write(json.dumps(example, ensure_ascii=False) + "\n")

    def _export_conll_ner(
        self, entries: List[Dict[str, Any]], output_file: Path
    ) -> None:
        """Export NER feedback to CoNLL format.

        Args:
            entries: NER feedback entries
            output_file: Output file path
        """
        with open(output_file, "w", encoding="utf-8") as f:
            for entry in entries:
                text = entry["input_text"]
                correction = entry["correction"]

                # Parse correction (should be list of entities)
                if isinstance(correction, str):
                    correction = json.loads(correction)

                # Convert to BIO tags
                tokens = text.split()  # Simple tokenization
                tags = ["O"] * len(tokens)

                # Apply entity tags
                for entity in correction:
                    # This is simplified - production would need proper alignment
                    entity_text = entity.get("text", "")
                    entity_label = entity.get("label", "MISC")

                    # Find entity in tokens
                    entity_tokens = entity_text.split()
                    for i in range(len(tokens) - len(entity_tokens) + 1):
                        if tokens[i : i + len(entity_tokens)] == entity_tokens:
                            tags[i] = f"B-{entity_label}"
                            for j in range(1, len(entity_tokens)):
                                tags[i + j] = f"I-{entity_label}"
                            break

                # Write in CoNLL format
                for token, tag in zip(tokens, tags):
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")  # Blank line between sentences

    async def select_active_learning_samples(
        self,
        task_type: str,
        strategy: str = "uncertainty",
        num_samples: int = 100,
    ) -> List[Dict[str, Any]]:
        """Select samples for active learning.

        Args:
            task_type: Task type
            strategy: Selection strategy (uncertainty, diversity, random)
            num_samples: Number of samples to select

        Returns:
            List of selected samples
        """
        # Get all feedback for task
        feedback = await self.get_feedback(task_type=task_type, unused_only=True)

        if len(feedback) <= num_samples:
            return feedback

        if strategy == "random":
            # Random sampling
            indices = np.random.choice(len(feedback), num_samples, replace=False)
            return [feedback[i] for i in indices]

        elif strategy == "uncertainty":
            # Select samples with high prediction uncertainty
            # This requires confidence scores in metadata
            samples_with_uncertainty = []

            for entry in feedback:
                metadata = entry.get("metadata", {})
                confidence = metadata.get("confidence")

                if confidence is not None:
                    # Lower confidence = higher uncertainty
                    uncertainty = 1.0 - confidence
                    samples_with_uncertainty.append((uncertainty, entry))

            if not samples_with_uncertainty:
                logger.warning("No confidence scores found, falling back to random")
                return await self.select_active_learning_samples(
                    task_type, "random", num_samples
                )

            # Sort by uncertainty (descending)
            samples_with_uncertainty.sort(key=lambda x: x[0], reverse=True)

            # Return top samples
            return [entry for _, entry in samples_with_uncertainty[:num_samples]]

        elif strategy == "diversity":
            # Select diverse samples (simplified - use embeddings in production)
            # For now, use random sampling
            logger.warning("Diversity sampling not fully implemented, using random")
            return await self.select_active_learning_samples(
                task_type, "random", num_samples
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    async def get_disagreement_rate(self, task_type: Optional[str] = None) -> float:
        """Calculate disagreement rate between predictions and corrections.

        Args:
            task_type: Filter by task type

        Returns:
            Disagreement rate (0-1)
        """
        feedback = await self.get_feedback(task_type=task_type)

        if not feedback:
            return 0.0

        disagreements = 0

        for entry in feedback:
            prediction = entry["prediction"]
            correction = entry["correction"]

            # Compare prediction and correction
            if prediction != correction:
                disagreements += 1

        return disagreements / len(feedback)

    def clear(self) -> None:
        """Clear all feedback data."""
        if self.backend == "sqlite":
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM feedback")
            conn.commit()
            conn.close()

        elif self.backend == "json":
            with open(self.json_path, "w") as f:
                json.dump({"feedback": []}, f)

        logger.warning("All feedback data cleared")


# ============================================================================
# Helper Functions
# ============================================================================


async def compute_feedback_metrics(
    feedback_store: FeedbackStore, task_type: Optional[str] = None
) -> Dict[str, Any]:
    """Compute metrics from feedback data.

    Args:
        feedback_store: Feedback store instance
        task_type: Filter by task type

    Returns:
        Dictionary of metrics
    """
    stats = await feedback_store.get_statistics()
    disagreement_rate = await feedback_store.get_disagreement_rate(task_type)

    total_feedback = sum(stats.values())

    metrics = {
        "total_feedback": total_feedback,
        "by_task": stats,
        "disagreement_rate": disagreement_rate,
    }

    return metrics


if __name__ == "__main__":
    import asyncio

    # Example usage
    async def main():
        # Initialize store
        store = FeedbackStore(backend="json", json_path="feedback_test.json")

        # Add feedback
        await store.add_feedback(
            task_type="ner",
            input_text="Alexander the Great conquered Persia.",
            prediction=[{"text": "Alexander", "label": "PER", "start": 0, "end": 9}],
            correction=[
                {"text": "Alexander the Great", "label": "PER", "start": 0, "end": 19},
                {"text": "Persia", "label": "LOC", "start": 30, "end": 36},
            ],
            metadata={"language": "en"},
        )

        # Get statistics
        stats = await store.get_statistics()
        print(f"Statistics: {stats}")

        # Export
        output_files = await store.export_for_training(
            output_dir="feedback_export", format="jsonl"
        )
        print(f"Exported to: {output_files}")

    asyncio.run(main())
