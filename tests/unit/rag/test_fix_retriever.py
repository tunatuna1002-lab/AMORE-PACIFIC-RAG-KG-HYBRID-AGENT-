"""Bug D14: DocumentRetriever.search_bm25 results must carry a top-level "id"."""

import pytest

from src.rag.retriever import BM25_AVAILABLE, DocumentRetriever

CHUNKS = [
    {
        "id": "c1",
        "doc_id": "d1",
        "title": "LANEIGE",
        "content": "LANEIGE Lip Sleeping Mask is great",
        "keywords": ["laneige"],
        "description": "test",
    },
    {
        "id": "c2",
        "doc_id": "d1",
        "title": "COSRX",
        "content": "COSRX Snail Mucin is popular",
        "keywords": ["cosrx"],
        "description": "test",
    },
    # rank_bm25's BM25Okapi yields an IDF of 0 for a term that appears in >= half of the
    # corpus, so a 2-chunk corpus returns no hits at all (see TestSearchBM25). Pad the corpus
    # with unrelated chunks, exactly like the existing BM25 tests do.
    *[
        {
            "id": f"filler{i}",
            "doc_id": "d2",
            "title": f"Filler {i}",
            "content": f"unrelated filler document number {i} about market analysis",
            "keywords": ["filler"],
            "description": "test",
        }
        for i in range(3)
    ],
]


@pytest.mark.skipif(not BM25_AVAILABLE, reason="rank_bm25 not installed")
def test_search_bm25_results_have_id(tmp_path):
    r = DocumentRetriever(docs_path=str(tmp_path))
    r.chunks = [dict(c) for c in CHUNKS]
    r._chunk_index = {c["id"]: c for c in r.chunks}

    results = r.search_bm25("LANEIGE", top_k=5)

    assert results, "expected at least one BM25 hit"
    assert results[0]["id"] == "c1"
    assert all("id" in res for res in results)
    # existing contract kept
    assert results[0]["source"] == "bm25"
    assert results[0]["metadata"]["id"] == "c1"
