"""Bug D4: OWLRetrievalStrategy must accept docs_path (brain.py / container.py pass it)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag.retrieval_strategy import OWLRetrievalStrategy


def test_docs_path_is_forwarded_to_document_retriever(tmp_path):
    with (
        patch("src.rag.entity_linker.EntityLinker") as mock_el,
        patch("src.rag.confidence_fusion.ConfidenceFusion") as mock_cf,
        patch("src.rag.reranker.get_reranker") as mock_rr,
    ):
        mock_el.return_value = MagicMock()
        mock_cf.return_value = MagicMock()
        mock_rr.return_value = MagicMock()
        strategy = OWLRetrievalStrategy(knowledge_graph=None, docs_path=str(tmp_path))

    assert strategy.doc_retriever.docs_path == Path(tmp_path)


def test_docs_path_defaults_when_omitted():
    with (
        patch("src.rag.entity_linker.EntityLinker") as mock_el,
        patch("src.rag.confidence_fusion.ConfidenceFusion") as mock_cf,
        patch("src.rag.reranker.get_reranker") as mock_rr,
        patch("src.rag.retriever.DocumentRetriever") as mock_dr,
    ):
        mock_el.return_value = MagicMock()
        mock_cf.return_value = MagicMock()
        mock_rr.return_value = MagicMock()
        mock_dr.return_value = MagicMock()
        OWLRetrievalStrategy()

    assert "docs_path" not in mock_dr.call_args.kwargs


def test_injected_doc_retriever_wins_over_docs_path(tmp_path):
    injected = MagicMock()
    with (
        patch("src.rag.entity_linker.EntityLinker") as mock_el,
        patch("src.rag.confidence_fusion.ConfidenceFusion") as mock_cf,
        patch("src.rag.reranker.get_reranker") as mock_rr,
    ):
        mock_el.return_value = MagicMock()
        mock_cf.return_value = MagicMock()
        mock_rr.return_value = MagicMock()
        strategy = OWLRetrievalStrategy(doc_retriever=injected, docs_path=str(tmp_path))
    assert strategy.doc_retriever is injected
