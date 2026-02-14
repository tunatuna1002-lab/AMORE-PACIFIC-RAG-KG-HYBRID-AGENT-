"""
Unit tests for SourceProvider
"""

import pytest

from src.agents.source_provider import SourceProvider
from src.domain.entities.relations import InferenceResult, InsightType
from src.rag.hybrid_retriever import HybridContext


class TestSourceProvider:
    """Test SourceProvider functionality"""

    @pytest.fixture
    def mock_kg(self):
        """Create mock KnowledgeGraph"""

        class MockKG:
            def get_category_hierarchy(self, category):
                return {
                    "name": category,
                    "level": 2,
                    "url": f"https://amazon.com/category/{category}",
                    "ancestors": [{"name": "Beauty"}],
                }

        return MockKG()

    @pytest.fixture
    def provider(self, mock_kg):
        """Create SourceProvider instance"""
        return SourceProvider(knowledge_graph=mock_kg)

    @pytest.fixture
    def mock_hybrid_context(self):
        """Create mock HybridContext"""
        return HybridContext(
            query="test query",
            entities={"brands": ["LANEIGE"], "categories": ["Lip Care"]},
            ontology_facts=[
                {"subject": "LANEIGE", "predicate": "competes_with", "object": "Innisfree"}
            ],
            inferences=[
                InferenceResult(
                    insight="Test insight",
                    insight_type=InsightType.COMPETITIVE_THREAT,
                    rule_name="test_rule",
                    confidence=0.9,
                    evidence=["test"],
                    recommendation="Test recommendation",
                )
            ],
            rag_chunks=[
                {
                    "text": "Test chunk",
                    "score": 0.85,
                    "metadata": {
                        "doc_id": "doc1",
                        "title": "Test Document",
                        "file_path": "/path/to/doc.md",
                        "section": "Introduction",
                    },
                }
            ],
            combined_context="Test context",
        )

    @pytest.fixture
    def mock_current_data(self):
        """Create mock current data"""
        return {
            "metadata": {"data_date": "2026-01-27"},
            "categories": {
                "Lip Care": {
                    "rank_records": [
                        {
                            "asin": "B001234567",  # pragma: allowlist secret
                            "brand": "LANEIGE",
                            "product_name": "Lip Sleeping Mask",
                            "rank": 4,
                        }
                    ]
                }
            },
        }

    @pytest.fixture
    def mock_external_signals(self):
        """Create mock external signals"""

        class MockSignal:
            def __init__(self, title, source, url, published_at):
                self.title = title
                self.source = source
                self.url = url
                self.published_at = published_at
                self.relevance_score = 0.8
                self.metadata = {"reliability_score": 0.9}
                self.content = "Test content for signal"
                self.tier = "tier_1"

        return [MockSignal("Test News", "Tavily News", "https://example.com", "2026-01-27")]

    def test_extract_sources_from_rag_chunks(
        self, provider, mock_hybrid_context, mock_current_data
    ):
        """Test source extraction from RAG chunks"""
        sources = provider.extract_sources(
            hybrid_context=mock_hybrid_context,
            current_data=mock_current_data,
        )

        # Should have crawled_data, KG, inference, RAG, AI model
        assert isinstance(sources, list)
        assert len(sources) >= 5

        # Check RAG document source exists
        rag_sources = [s for s in sources if s["type"] == "rag_document"]
        assert len(rag_sources) == 1
        assert rag_sources[0]["description"] == "Test Document"
        assert rag_sources[0]["relevance_score"] == 0.85

    def test_extract_sources_from_kg_triples(
        self, provider, mock_hybrid_context, mock_current_data
    ):
        """Test source extraction from KG triples"""
        sources = provider.extract_sources(
            hybrid_context=mock_hybrid_context,
            current_data=mock_current_data,
        )

        kg_sources = [s for s in sources if s["type"] == "knowledge_graph"]
        assert len(kg_sources) == 1
        assert kg_sources[0]["fact_count"] == 1
        assert "LANEIGE" in kg_sources[0]["entities"]

    def test_perplexity_style_formatting(self, provider, mock_hybrid_context, mock_current_data):
        """Test Perplexity-style markdown formatting"""
        sources = provider.extract_sources(
            hybrid_context=mock_hybrid_context,
            current_data=mock_current_data,
        )

        formatted = provider.format_sources_for_display(sources)

        assert isinstance(formatted, str)
        assert "---" in formatted  # Separator
        assert "**📚 출처 및 참고자료:**" in formatted
        assert "📅 **데이터 기준:" in formatted
        assert "2026-01-27" in formatted

    def test_source_deduplication(self, provider, mock_hybrid_context, mock_current_data):
        """Test deduplication of sources"""
        # Add duplicate RAG chunks
        mock_hybrid_context.rag_chunks.append(
            {
                "text": "Another chunk",
                "score": 0.75,
                "metadata": {
                    "doc_id": "doc1",  # Same doc_id
                    "title": "Test Document",
                    "file_path": "/path/to/doc.md",
                    "section": "Conclusion",
                },
            }
        )

        sources = provider.extract_sources(
            hybrid_context=mock_hybrid_context,
            current_data=mock_current_data,
        )

        # Should only have 1 RAG source (highest score wins)
        rag_sources = [s for s in sources if s["type"] == "rag_document"]
        assert len(rag_sources) == 1
        assert rag_sources[0]["relevance_score"] == 0.85  # Higher score

    def test_empty_sources(self, provider):
        """Test handling of empty sources"""
        empty_context = HybridContext(
            query="test query",
            entities={},
            ontology_facts=[],
            inferences=[],
            rag_chunks=[],
            combined_context="",
        )

        sources = provider.extract_sources(
            hybrid_context=empty_context,
        )

        # Should at least have AI model source
        assert len(sources) >= 1
        ai_sources = [s for s in sources if s["type"] == "ai_model"]
        assert len(ai_sources) == 1

    def test_external_signal_formatting(
        self, provider, mock_hybrid_context, mock_current_data, mock_external_signals
    ):
        """Test external signal source formatting"""
        sources = provider.extract_sources(
            hybrid_context=mock_hybrid_context,
            current_data=mock_current_data,
            external_signals=mock_external_signals,
        )

        news_sources = [s for s in sources if s["type"] == "external_news"]
        assert len(news_sources) == 1
        assert news_sources[0]["description"] == "Test News"
        assert news_sources[0]["reliability_score"] == 0.9

    def test_extract_entity_names(self, provider):
        """Test entity name extraction from KG facts"""
        facts = [
            {"subject": "LANEIGE", "predicate": "competes_with", "object": "Innisfree"},
            {"subject": "LANEIGE", "predicate": "has_product", "object": "Lip Mask"},
        ]

        entities = provider._extract_entity_names(facts)

        assert isinstance(entities, list)
        assert "LANEIGE" in entities
        assert "Innisfree" in entities or "Lip Mask" in entities
        assert len(entities) <= 5  # Max 5

    def test_extract_relation_types(self, provider):
        """Test relation type extraction from KG facts"""
        facts = [
            {"subject": "LANEIGE", "predicate": "competes_with", "object": "Innisfree"},
            {"subject": "LANEIGE", "predicate": "has_product", "object": "Lip Mask"},
        ]

        relations = provider._extract_relation_types(facts)

        assert isinstance(relations, list)
        assert "competes_with" in relations
        assert "has_product" in relations

    def test_extract_mentioned_asins(self, provider, mock_hybrid_context, mock_current_data):
        """Test ASIN extraction from mentioned products"""
        categories = mock_current_data["categories"]

        asins = provider._extract_mentioned_asins(mock_hybrid_context, categories)

        assert isinstance(asins, list)
        assert len(asins) >= 1
        assert asins[0]["asin"] == "B001234567"  # pragma: allowlist secret
        assert asins[0]["brand"] == "LANEIGE"
        assert asins[0]["rank"] == 4

    def test_format_sources_empty(self, provider):
        """Test formatting with empty sources"""
        formatted = provider.format_sources_for_display([])
        assert formatted == ""

    def test_ontology_inference_source(self, provider, mock_hybrid_context, mock_current_data):
        """Test ontology inference source extraction"""
        sources = provider.extract_sources(
            hybrid_context=mock_hybrid_context,
            current_data=mock_current_data,
        )

        inference_sources = [s for s in sources if s["type"] == "ontology_inference"]
        assert len(inference_sources) == 1
        assert inference_sources[0]["rule_name"] == "test_rule"
        assert inference_sources[0]["confidence"] == 0.9

    def test_crawled_data_source_with_asins(self, provider, mock_hybrid_context, mock_current_data):
        """Test crawled data source includes ASIN information"""
        sources = provider.extract_sources(
            hybrid_context=mock_hybrid_context,
            current_data=mock_current_data,
        )

        crawled = [s for s in sources if s["type"] == "crawled_data"]
        assert len(crawled) == 1
        assert "mentioned_products" in crawled[0]
        assert len(crawled[0]["mentioned_products"]) >= 1
