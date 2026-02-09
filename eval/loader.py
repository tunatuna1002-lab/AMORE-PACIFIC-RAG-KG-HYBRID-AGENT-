"""
Dataset Loader
==============
Load evaluation datasets from JSONL files.
"""

import json
import logging
from pathlib import Path

from eval.schemas import EvalItem

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path) -> list[EvalItem]:
    """
    Load evaluation dataset from JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of EvalItem objects

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If JSONL parsing fails

    Example:
        >>> items = load_dataset("eval/data/examples/chatbot_eval.jsonl")
        >>> print(len(items))
        10
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    items: list[EvalItem] = []
    errors: list[tuple[int, str]] = []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Skip comments (lines starting with #)
            if line.startswith("#"):
                continue

            try:
                data = json.loads(line)
                item = EvalItem.model_validate(data)
                items.append(item)
            except json.JSONDecodeError as e:
                errors.append((line_num, f"JSON parse error: {e}"))
            except Exception as e:
                errors.append((line_num, f"Validation error: {e}"))

    if errors:
        error_msg = "\n".join([f"  Line {ln}: {msg}" for ln, msg in errors[:5]])
        if len(errors) > 5:
            error_msg += f"\n  ... and {len(errors) - 5} more errors"
        logger.warning(f"Dataset loading had {len(errors)} errors:\n{error_msg}")

    logger.info(f"Loaded {len(items)} items from {path}")
    return items


def save_dataset(items: list[EvalItem], path: str | Path) -> None:
    """
    Save evaluation dataset to JSONL file.

    Args:
        items: List of EvalItem objects
        path: Output path for JSONL file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            line = item.model_dump_json(exclude_none=True)
            f.write(line + "\n")

    logger.info(f"Saved {len(items)} items to {path}")


def validate_dataset(
    items_or_path: list[EvalItem] | str | Path,
) -> list[str]:
    """
    Validate evaluation items and return errors.

    Args:
        items_or_path: List of EvalItem objects or path to JSONL file

    Returns:
        List of validation error messages (empty if valid)
    """
    # Handle path input
    if isinstance(items_or_path, str | Path):
        items = load_dataset(items_or_path)
    else:
        items = items_or_path

    errors: list[str] = []
    seen_ids: set[str] = set()

    for i, item in enumerate(items):
        # Check for empty ID
        if not item.id or not item.id.strip():
            errors.append(f"Item {i}: Empty or missing ID")

        # Check for duplicate IDs
        if item.id in seen_ids:
            errors.append(f"Item {i}: Duplicate ID '{item.id}'")
        seen_ids.add(item.id)

        # Check for empty question
        if not item.question or not item.question.strip():
            errors.append(f"Item {i} ({item.id}): Empty or missing question")

    return errors


def validate_dataset_file(path: str | Path) -> dict[str, int | list[str]]:
    """
    Validate dataset file and return statistics.

    Args:
        path: Path to JSONL file

    Returns:
        Dictionary with validation stats
    """
    path = Path(path)
    stats: dict[str, int | list[str]] = {
        "total_lines": 0,
        "valid_items": 0,
        "invalid_items": 0,
        "errors": [],
        "by_difficulty": {"easy": 0, "medium": 0, "hard": 0},
        "by_domain": {},
        "with_gold_answer": 0,
        "with_gold_kg_entities": 0,
        "with_gold_doc_chunks": 0,
    }

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            stats["total_lines"] = int(stats["total_lines"]) + 1

            try:
                data = json.loads(line)
                item = EvalItem.model_validate(data)
                stats["valid_items"] = int(stats["valid_items"]) + 1

                # Count by difficulty
                difficulty_counts = stats["by_difficulty"]
                if isinstance(difficulty_counts, dict):
                    diff = item.metadata.difficulty
                    difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

                # Count by domain
                domain_counts = stats["by_domain"]
                if isinstance(domain_counts, dict):
                    domain = item.metadata.domain
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1

                # Gold evidence stats
                if item.gold.answer:
                    stats["with_gold_answer"] = int(stats["with_gold_answer"]) + 1
                if item.gold.kg_entities:
                    stats["with_gold_kg_entities"] = int(stats["with_gold_kg_entities"]) + 1
                if item.gold.doc_chunk_ids:
                    stats["with_gold_doc_chunks"] = int(stats["with_gold_doc_chunks"]) + 1

            except Exception as e:
                stats["invalid_items"] = int(stats["invalid_items"]) + 1
                errors = stats["errors"]
                if isinstance(errors, list):
                    errors.append(f"Line {line_num}: {e}")

    return stats
