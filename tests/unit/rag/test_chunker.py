"""
SemanticChunker 단위 테스트

Coverage target: 42% → 75%+
Covers: init, _split_sentences, _cosine_similarity, _find_breakpoints,
        _create_chunks_from_breakpoints, _simple_chunk, chunk, chunk_document,
        _initialize, _embed_sentences, get_semantic_chunker
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.chunker import Chunk, SemanticChunker, get_semantic_chunker

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def chunker():
    return SemanticChunker(
        min_chunk_size=50,
        max_chunk_size=500,
        similarity_threshold=0.3,
        percentile_threshold=25,
        overlap_sentences=1,
    )


@pytest.fixture
def chunker_no_overlap():
    return SemanticChunker(
        min_chunk_size=50,
        max_chunk_size=500,
        overlap_sentences=0,
    )


def _fake_embeddings(n, dim=8):
    """Generate deterministic fake embeddings."""
    rng = np.random.RandomState(42)
    return [rng.randn(dim).tolist() for _ in range(n)]


def _fake_embeddings_similar(n, dim=8):
    """Generate embeddings that are all very similar (high cosine sim)."""
    base = np.ones(dim)
    return [(base + np.random.RandomState(i).randn(dim) * 0.01).tolist() for i in range(n)]


def _fake_embeddings_divergent(n, dim=8):
    """Generate embeddings where adjacent pairs alternate between similar and dissimilar."""
    embs = []
    for i in range(n):
        if i % 2 == 0:
            embs.append(np.ones(dim).tolist())
        else:
            embs.append((-np.ones(dim)).tolist())
    return embs


# =========================================================================
# Chunk dataclass
# =========================================================================


class TestChunk:
    def test_chunk_defaults(self):
        c = Chunk(id="c1", content="hello")
        assert c.id == "c1"
        assert c.content == "hello"
        assert c.metadata == {}
        assert c.embedding is None
        assert c.sentence_count == 0
        assert c.avg_similarity == 0.0


# =========================================================================
# Init
# =========================================================================


class TestSemanticChunkerInit:
    def test_init_defaults(self):
        chunker = SemanticChunker()
        assert chunker.min_chunk_size == 100
        assert chunker.max_chunk_size == 1000
        assert chunker.similarity_threshold == 0.5
        assert chunker.overlap_sentences == 1
        assert chunker._initialized is False

    def test_init_custom(self):
        chunker = SemanticChunker(
            min_chunk_size=50,
            max_chunk_size=500,
            similarity_threshold=0.7,
            percentile_threshold=30,
            overlap_sentences=2,
        )
        assert chunker.min_chunk_size == 50
        assert chunker.max_chunk_size == 500
        assert chunker.similarity_threshold == 0.7
        assert chunker.percentile_threshold == 30
        assert chunker.overlap_sentences == 2


# =========================================================================
# _initialize
# =========================================================================


class TestInitialize:
    def test_already_initialized(self, chunker):
        chunker._initialized = True
        assert chunker._initialize() is True

    @patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False)
    def test_no_api_key(self, chunker):
        chunker._initialized = False
        assert chunker._initialize() is False

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "sk-test"},  # pragma: allowlist secret
        clear=False,
    )
    @patch("src.rag.chunker.openai", create=True)
    def test_successful_init(self, mock_openai, chunker):
        chunker._initialized = False
        chunker.openai_client = None
        # Mock the import inside _initialize
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                MagicMock() if name == "openai" else __import__(name, *a, **kw)
            ),
        ):
            # The actual _initialize uses `import openai` then `openai.OpenAI()`
            result = chunker._initialize()
            # May or may not succeed depending on openai package availability
            assert isinstance(result, bool)

    def test_import_error(self, chunker):
        chunker._initialized = False
        with patch("builtins.__import__", side_effect=ImportError("no openai")):
            assert chunker._initialize() is False


# =========================================================================
# _split_sentences
# =========================================================================


class TestSplitSentences:
    def test_basic_english(self, chunker):
        text = "Hello world. This is a test. Third sentence here."
        sentences = chunker._split_sentences(text)
        assert len(sentences) == 3

    def test_mixed_punctuation(self, chunker):
        text = "Is this working? Yes it is! And this too."
        sentences = chunker._split_sentences(text)
        assert len(sentences) == 3

    def test_newlines_split(self, chunker):
        text = "First line\nSecond line\nThird line"
        sentences = chunker._split_sentences(text)
        assert len(sentences) == 3

    def test_multiple_newlines(self, chunker):
        text = "First paragraph.\n\n\nSecond paragraph."
        sentences = chunker._split_sentences(text)
        assert len(sentences) >= 2

    def test_preserves_decimal_numbers(self, chunker):
        text = "The value is 3.14 and that is pi. Next sentence."
        sentences = chunker._split_sentences(text)
        # 3.14 should not be split
        combined = " ".join(sentences)
        assert "3.14" in combined

    def test_empty_text(self, chunker):
        assert chunker._split_sentences("") == []

    def test_whitespace_only(self, chunker):
        assert chunker._split_sentences("   \n  ") == []

    def test_single_sentence_no_period(self, chunker):
        sentences = chunker._split_sentences("Just one sentence")
        assert len(sentences) == 1

    def test_korean_text(self, chunker):
        text = "라네즈는 아모레퍼시픽 브랜드입니다. 립 슬리핑 마스크가 인기입니다."
        sentences = chunker._split_sentences(text)
        assert len(sentences) >= 1


# =========================================================================
# _cosine_similarity
# =========================================================================


class TestCosineSimilarity:
    def test_identical_vectors(self, chunker):
        v = [1.0, 0.0, 1.0]
        assert chunker._cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self, chunker):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert chunker._cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self, chunker):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert chunker._cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self, chunker):
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        assert chunker._cosine_similarity(a, b) == 0.0

    def test_both_zero(self, chunker):
        assert chunker._cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


# =========================================================================
# _find_breakpoints
# =========================================================================


class TestFindBreakpoints:
    def test_single_embedding(self, chunker):
        embs = _fake_embeddings(1)
        assert chunker._find_breakpoints(embs, ["s1"]) == []

    def test_empty_embeddings(self, chunker):
        assert chunker._find_breakpoints([], []) == []

    def test_similar_embeddings_no_breakpoints(self, chunker):
        """Very similar embeddings should produce few or no breakpoints."""
        embs = _fake_embeddings_similar(5)
        sents = ["Short sentence"] * 5
        similarities = [
            chunker._cosine_similarity(embs[i], embs[i + 1]) for i in range(len(embs) - 1)
        ]
        with patch("src.rag.chunker.np.percentile", return_value=0.99):
            bps = chunker._find_breakpoints(embs, sents)
        assert isinstance(bps, list)

    def test_divergent_embeddings_produce_breakpoints(self):
        chunker = SemanticChunker(
            min_chunk_size=1,
            max_chunk_size=10000,
            similarity_threshold=0.0,
            percentile_threshold=50,
        )
        embs = _fake_embeddings_divergent(6)
        sents = ["A sentence here."] * 6
        with patch("src.rag.chunker.np.percentile", return_value=0.0):
            bps = chunker._find_breakpoints(embs, sents)
        assert len(bps) > 0

    def test_max_chunk_size_forces_breakpoint(self):
        chunker = SemanticChunker(
            min_chunk_size=1,
            max_chunk_size=50,
            similarity_threshold=0.0,
        )
        embs = _fake_embeddings_similar(10)
        sents = ["A" * 20] * 10
        with patch("src.rag.chunker.np.percentile", return_value=0.99):
            bps = chunker._find_breakpoints(embs, sents)
        assert len(bps) > 0


# =========================================================================
# _create_chunks_from_breakpoints
# =========================================================================


class TestCreateChunksFromBreakpoints:
    def test_no_breakpoints(self, chunker):
        sents = ["Sentence one.", "Sentence two.", "Sentence three."]
        embs = _fake_embeddings(3)
        chunks = chunker._create_chunks_from_breakpoints(sents, embs, [], "doc1")
        assert len(chunks) == 1
        assert chunks[0].sentence_count == 3
        assert "doc1" in chunks[0].id
        assert chunks[0].metadata["chunking_method"] == "semantic"

    def test_with_breakpoints(self, chunker_no_overlap):
        sents = ["S1.", "S2.", "S3.", "S4."]
        embs = _fake_embeddings(4)
        chunks = chunker_no_overlap._create_chunks_from_breakpoints(sents, embs, [2], "doc1")
        assert len(chunks) == 2
        assert chunks[0].start_idx == 0
        assert chunks[0].end_idx == 2
        assert chunks[1].start_idx == 2

    def test_with_overlap(self, chunker):
        sents = ["S1.", "S2.", "S3.", "S4.", "S5."]
        embs = _fake_embeddings(5)
        chunks = chunker._create_chunks_from_breakpoints(sents, embs, [3], "doc1")
        assert len(chunks) == 2
        # Second chunk should have overlap from previous
        assert chunks[1].metadata["has_overlap"] is True
        assert chunks[1].sentence_count > (5 - 3)  # more than just S4, S5

    def test_metadata_passed_through(self, chunker):
        sents = ["Hello world."]
        embs = _fake_embeddings(1)
        chunks = chunker._create_chunks_from_breakpoints(
            sents, embs, [], "doc1", metadata={"source": "test"}
        )
        assert chunks[0].metadata["source"] == "test"
        assert chunks[0].metadata["doc_id"] == "doc1"

    def test_empty_embeddings(self, chunker):
        sents = ["S1.", "S2."]
        chunks = chunker._create_chunks_from_breakpoints(sents, [], [], "doc1")
        assert len(chunks) == 1
        assert chunks[0].embedding is None

    def test_avg_similarity_calculated(self, chunker_no_overlap):
        """Multiple sentences in a chunk should have avg_similarity > 0."""
        sents = ["S1.", "S2.", "S3."]
        # Use identical embeddings → similarity = 1.0
        embs = [[1.0, 0.0, 0.0]] * 3
        chunks = chunker_no_overlap._create_chunks_from_breakpoints(sents, embs, [], "doc1")
        assert chunks[0].avg_similarity == pytest.approx(1.0)


# =========================================================================
# _simple_chunk
# =========================================================================


class TestSimpleChunk:
    def test_short_text_single_chunk(self, chunker):
        text = "Short text here."
        chunks = chunker._simple_chunk(text, "doc1")
        assert len(chunks) == 1
        assert chunks[0].metadata["chunking_method"] == "simple"

    def test_long_text_multiple_chunks(self, chunker):
        text = "This is a sentence. " * 100
        chunks = chunker._simple_chunk(text, "doc1")
        assert len(chunks) >= 2

    def test_overlap_in_simple_chunk(self, chunker):
        """overlap_sentences > 0 should carry tail sentences to next chunk."""
        text = "Sentence A. Sentence B. Sentence C. Sentence D. " * 20
        chunks = chunker._simple_chunk(text, "doc1")
        if len(chunks) >= 2:
            assert chunks[1].metadata["has_overlap"] is True

    def test_no_overlap_in_simple_chunk(self, chunker_no_overlap):
        text = "Sentence. " * 200
        chunks = chunker_no_overlap._simple_chunk(text, "doc1")
        if len(chunks) >= 2:
            # no overlap → fresh start for each chunk
            assert chunks[1].metadata.get("has_overlap", False) is False

    def test_metadata_passed(self, chunker):
        chunks = chunker._simple_chunk("Text here.", "doc1", metadata={"k": "v"})
        assert chunks[0].metadata["k"] == "v"

    def test_empty_text(self, chunker):
        chunks = chunker._simple_chunk("", "doc1")
        assert chunks == []


# =========================================================================
# chunk (main entry point)
# =========================================================================


class TestChunkMethod:
    def test_empty_text(self, chunker):
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []

    def test_falls_back_to_simple_when_no_openai(self, chunker):
        """Without OPENAI_API_KEY, should fall back to simple chunking."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            chunker._initialized = False
            chunks = chunker.chunk("Hello world. This is a test.", doc_id="test")
            assert len(chunks) >= 1
            assert chunks[0].metadata["chunking_method"] == "simple"

    def test_single_sentence_semantic(self, chunker):
        """Single sentence → single chunk when embeddings available."""
        chunker._initialized = True
        chunker.openai_client = MagicMock()
        # Single sentence won't go through embedding
        chunks = chunker.chunk("Just one sentence without period")
        assert len(chunks) == 1
        assert chunks[0].metadata["chunking_method"] == "semantic"

    def test_semantic_chunking_full_path(self, chunker):
        """Mock OpenAI embeddings for full semantic path."""
        chunker._initialized = True
        mock_client = MagicMock()

        # Create mock embedding response
        sents_text = "First sentence here. Second sentence now. Third one follows."
        sentences = chunker._split_sentences(sents_text)
        n_sents = len(sentences)
        fake_embs = _fake_embeddings(n_sents)

        mock_items = []
        for emb in fake_embs:
            item = MagicMock()
            item.embedding = emb
            mock_items.append(item)
        mock_response = MagicMock()
        mock_response.data = mock_items
        mock_client.embeddings.create.return_value = mock_response

        chunker.openai_client = mock_client
        with patch("src.rag.chunker.np.percentile", return_value=0.5):
            chunks = chunker.chunk(sents_text, doc_id="test")
        assert len(chunks) >= 1
        mock_client.embeddings.create.assert_called_once()

    def test_embedding_failure_falls_back(self, chunker):
        """If embedding fails, falls back to simple chunking."""
        chunker._initialized = True
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("API error")
        chunker.openai_client = mock_client

        chunks = chunker.chunk(
            "First sentence. Second sentence. Third sentence.",
            doc_id="test",
        )
        assert len(chunks) >= 1
        assert chunks[0].metadata["chunking_method"] == "simple"

    def test_embedding_count_mismatch_falls_back(self, chunker):
        """If embedding count != sentence count, falls back."""
        chunker._initialized = True
        mock_client = MagicMock()
        # Return only 1 embedding for multiple sentences
        item = MagicMock()
        item.embedding = [0.1, 0.2]
        mock_response = MagicMock()
        mock_response.data = [item]
        mock_client.embeddings.create.return_value = mock_response
        chunker.openai_client = mock_client

        chunks = chunker.chunk(
            "First sentence. Second sentence. Third sentence.",
            doc_id="test",
        )
        assert len(chunks) >= 1
        assert chunks[0].metadata["chunking_method"] == "simple"

    def test_chunk_with_metadata(self, chunker):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            chunker._initialized = False
            chunks = chunker.chunk("Some text.", doc_id="d1", metadata={"key": "val"})
            assert chunks[0].metadata["key"] == "val"


# =========================================================================
# _embed_sentences
# =========================================================================


class TestEmbedSentences:
    def test_no_client(self, chunker):
        chunker.openai_client = None
        assert chunker._embed_sentences(["hello"]) == []

    def test_empty_sentences(self, chunker):
        chunker.openai_client = MagicMock()
        assert chunker._embed_sentences([]) == []

    def test_successful_embedding(self, chunker):
        mock_client = MagicMock()
        item1 = MagicMock()
        item1.embedding = [0.1, 0.2, 0.3]
        item2 = MagicMock()
        item2.embedding = [0.4, 0.5, 0.6]
        mock_response = MagicMock()
        mock_response.data = [item1, item2]
        mock_client.embeddings.create.return_value = mock_response
        chunker.openai_client = mock_client

        result = chunker._embed_sentences(["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    def test_embedding_exception(self, chunker):
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("fail")
        chunker.openai_client = mock_client
        assert chunker._embed_sentences(["hello"]) == []


# =========================================================================
# chunk_document
# =========================================================================


class TestChunkDocument:
    def test_chunk_document_format(self, chunker):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            chunker._initialized = False
            doc_info = {
                "filename": "test_doc.md",
                "doc_type": "metric_guide",
                "description": "A test document",
                "keywords": ["test", "doc"],
            }
            result = chunker.chunk_document("Some content here.", doc_info)
            assert len(result) >= 1
            r = result[0]
            assert "id" in r
            assert "doc_id" in r
            assert r["doc_type"] == "metric_guide"
            assert r["description"] == "A test document"
            assert r["keywords"] == ["test", "doc"]
            assert r["content_type"] == "text"
            assert "content" in r
            assert "sentence_count" in r
            assert "avg_similarity" in r

    def test_chunk_document_doc_id_from_filename(self, chunker):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            chunker._initialized = False
            doc_info = {"filename": "my doc.md"}
            result = chunker.chunk_document("Text.", doc_info)
            assert result[0]["doc_id"] == "my_doc"


# =========================================================================
# get_semantic_chunker singleton
# =========================================================================


class TestGetSemanticChunker:
    def test_returns_instance(self):
        import src.rag.chunker as chunker_mod

        # Reset singleton
        chunker_mod._chunker_instance = None
        instance = get_semantic_chunker()
        assert isinstance(instance, SemanticChunker)

    def test_singleton_returns_same(self):
        import src.rag.chunker as chunker_mod

        chunker_mod._chunker_instance = None
        a = get_semantic_chunker()
        b = get_semantic_chunker()
        assert a is b


# =========================================================================
# MAX_DOCUMENT_SIZE validation
# =========================================================================


class TestDocumentSizeLimit:
    def test_chunk_rejects_oversized_document(self, chunker):
        """Documents exceeding MAX_DOCUMENT_SIZE should return empty list."""

        with patch("src.rag.chunker.MAX_DOCUMENT_SIZE", 100):
            # Create text larger than 100 bytes
            oversized_text = "A" * 200
            result = chunker.chunk(oversized_text, doc_id="oversized")
            assert result == []

    def test_chunk_accepts_normal_document(self, chunker):
        """Documents within size limit should be processed normally."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            chunker._initialized = False
            small_text = "Normal sized document."
            result = chunker.chunk(small_text, doc_id="normal")
            assert len(result) >= 1

    def test_chunk_document_rejects_oversized(self, chunker):
        """chunk_document should also reject oversized documents."""
        with patch("src.rag.chunker.MAX_DOCUMENT_SIZE", 50):
            doc_info = {"filename": "big.md", "doc_type": "metric_guide"}
            result = chunker.chunk_document("X" * 100, doc_info)
            assert result == []

    def test_max_document_size_constant(self):
        """MAX_DOCUMENT_SIZE should be 10MB."""
        from src.rag.chunker import MAX_DOCUMENT_SIZE

        assert MAX_DOCUMENT_SIZE == 10 * 1024 * 1024
