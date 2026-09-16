"""F9-4: the chat path (retrieval_strategy / hybrid_retriever) must not import owlready2."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_chat_path_does_not_import_owlready2() -> None:
    code = (
        "import src.rag.retrieval_strategy, src.rag.hybrid_retriever, sys; "
        "assert 'owlready2' not in sys.modules, 'owlready2 leaked into the chat path'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
