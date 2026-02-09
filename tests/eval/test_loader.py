"""Tests for dataset loader."""

from pathlib import Path

import pytest

from eval.loader import load_dataset, save_dataset, validate_dataset
from eval.schemas import EvalItem, GoldEvidence


class TestLoadDataset:
    """Tests for load_dataset function."""

    def test_load_valid_dataset(self, tmp_path: Path):
        """Test loading a valid JSONL dataset."""
        dataset_file = tmp_path / "test.jsonl"
        dataset_file.write_text(
            '{"id": "q001", "question": "Test question 1?"}\n'
            '{"id": "q002", "question": "Test question 2?"}\n'
        )

        items = load_dataset(dataset_file)
        assert len(items) == 2
        assert items[0].id == "q001"
        assert items[1].id == "q002"

    def test_load_with_gold_evidence(self, tmp_path: Path):
        """Test loading dataset with gold evidence."""
        dataset_file = tmp_path / "test.jsonl"
        dataset_file.write_text(
            '{"id": "q001", "question": "Test?", "gold": {"answer": "Test answer", "kg_entities": ["entity1"]}}\n'
        )

        items = load_dataset(dataset_file)
        assert items[0].gold.answer == "Test answer"
        assert items[0].gold.kg_entities == ["entity1"]

    def test_load_with_metadata(self, tmp_path: Path):
        """Test loading dataset with metadata."""
        dataset_file = tmp_path / "test.jsonl"
        dataset_file.write_text(
            '{"id": "q001", "question": "Test?", "metadata": {"requires_kg": false, "domain": "metric"}}\n'
        )

        items = load_dataset(dataset_file)
        assert items[0].metadata.requires_kg is False
        assert items[0].metadata.domain == "metric"

    def test_load_empty_lines(self, tmp_path: Path):
        """Test loading dataset with empty lines."""
        dataset_file = tmp_path / "test.jsonl"
        dataset_file.write_text(
            '{"id": "q001", "question": "Test 1?"}\n'
            "\n"
            '{"id": "q002", "question": "Test 2?"}\n'
            "\n"
        )

        items = load_dataset(dataset_file)
        assert len(items) == 2

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_dataset(Path("/nonexistent/path/file.jsonl"))

    def test_load_invalid_json(self, tmp_path: Path):
        """Test loading invalid JSON logs warning but continues."""
        dataset_file = tmp_path / "test.jsonl"
        dataset_file.write_text("not valid json\n")

        # Invalid JSON lines are skipped with warning
        items = load_dataset(dataset_file)
        assert len(items) == 0  # No valid items


class TestSaveDataset:
    """Tests for save_dataset function."""

    def test_save_dataset(self, tmp_path: Path):
        """Test saving dataset."""
        items = [
            EvalItem(id="q001", question="Test 1?"),
            EvalItem(id="q002", question="Test 2?"),
        ]

        output_path = tmp_path / "output.jsonl"
        save_dataset(items, output_path)

        # Verify file exists
        assert output_path.exists()

        # Verify content
        loaded = load_dataset(output_path)
        assert len(loaded) == 2
        assert loaded[0].id == "q001"

    def test_save_with_gold_evidence(self, tmp_path: Path):
        """Test saving dataset with gold evidence."""
        items = [
            EvalItem(
                id="q001",
                question="Test?",
                gold=GoldEvidence(answer="Answer", kg_entities=["e1"]),
            )
        ]

        output_path = tmp_path / "output.jsonl"
        save_dataset(items, output_path)

        loaded = load_dataset(output_path)
        assert loaded[0].gold.answer == "Answer"
        assert loaded[0].gold.kg_entities == ["e1"]


class TestValidateDataset:
    """Tests for validate_dataset function."""

    def test_validate_valid_dataset(self):
        """Test validation of valid dataset."""
        items = [
            EvalItem(id="q001", question="Test 1?"),
            EvalItem(id="q002", question="Test 2?"),
        ]

        errors = validate_dataset(items)
        assert len(errors) == 0

    def test_validate_duplicate_ids(self):
        """Test validation detects duplicate IDs."""
        items = [
            EvalItem(id="q001", question="Test 1?"),
            EvalItem(id="q001", question="Test 2?"),  # Duplicate ID
        ]

        errors = validate_dataset(items)
        assert len(errors) > 0
        assert any("duplicate" in e.lower() for e in errors)

    def test_validate_empty_question(self):
        """Test validation detects empty questions."""
        items = [
            EvalItem(id="q001", question=""),
        ]

        errors = validate_dataset(items)
        assert len(errors) > 0
        assert any("question" in e.lower() for e in errors)

    def test_validate_empty_id(self):
        """Test validation detects empty IDs."""
        items = [
            EvalItem(id="", question="Test?"),
        ]

        errors = validate_dataset(items)
        assert len(errors) > 0
        assert any("id" in e.lower() for e in errors)
