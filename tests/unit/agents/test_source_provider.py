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

    def test_external_signal_reddit(self, provider, mock_hybrid_context, mock_current_data):
        """Test Reddit external signal source extraction"""

        class RedditSignal:
            def __init__(self):
                self.title = "Reddit Discussion"
                self.source = "Reddit"
                self.url = "https://reddit.com/test"
                self.published_at = "2026-01-27"
                self.relevance_score = 0.6
                self.metadata = {"reliability_score": 0.5}
                self.content = "Reddit content"
                self.tier = "tier_2"

        sources = provider.extract_sources(
            hybrid_context=mock_hybrid_context,
            current_data=mock_current_data,
            external_signals=[RedditSignal()],
        )

        social_sources = [s for s in sources if s["type"] == "social_media"]
        assert len(social_sources) == 1
        assert social_sources[0]["icon"] == "💬"
        assert social_sources[0]["reliability_score"] == 0.5

    def test_external_signal_rss(self, provider, mock_hybrid_context, mock_current_data):
        """Test RSS feed external signal source extraction"""

        class RSSSignal:
            def __init__(self):
                self.title = "RSS Feed Article"
                self.source = "RSS Feed"
                self.url = "https://example.com/rss"
                self.published_at = "2026-01-27"
                self.relevance_score = 0.8
                self.metadata = {"reliability_score": 0.85}
                self.content = "RSS content"
                self.tier = "tier_1"

        sources = provider.extract_sources(
            hybrid_context=mock_hybrid_context,
            current_data=mock_current_data,
            external_signals=[RSSSignal()],
        )

        rss_sources = [s for s in sources if s["type"] == "rss_feed"]
        assert len(rss_sources) == 1
        assert rss_sources[0]["icon"] == "📡"

    def test_external_signal_youtube(self, provider, mock_hybrid_context, mock_current_data):
        """Test YouTube external signal source extraction"""

        class YouTubeSignal:
            def __init__(self):
                self.title = "YouTube Video"
                self.source = "YouTube"
                self.url = "https://youtube.com/watch?v=test"
                self.published_at = "2026-01-27"
                self.relevance_score = 0.7
                self.metadata = {"reliability_score": 0.6}
                self.content = "YouTube content"
                self.tier = "tier_2"

        sources = provider.extract_sources(
            hybrid_context=mock_hybrid_context,
            current_data=mock_current_data,
            external_signals=[YouTubeSignal()],
        )

        social_sources = [s for s in sources if s["type"] == "social_media"]
        assert len(social_sources) == 1
        assert social_sources[0]["icon"] == "📺"

    def test_external_signal_unknown(self, provider, mock_hybrid_context, mock_current_data):
        """Test unknown external signal source extraction"""

        class UnknownSignal:
            def __init__(self):
                self.title = "Unknown Source"
                self.source = "Unknown Platform"
                self.url = "https://unknown.com/test"
                self.published_at = "2026-01-27"
                self.relevance_score = 0.5
                self.metadata = None
                self.content = "Unknown content"
                self.tier = "tier_3"

        sources = provider.extract_sources(
            hybrid_context=mock_hybrid_context,
            current_data=mock_current_data,
            external_signals=[UnknownSignal()],
        )

        external_sources = [s for s in sources if s["type"] == "external_source"]
        assert len(external_sources) == 1
        assert external_sources[0]["icon"] == "🌐"
        assert external_sources[0]["reliability_score"] == 0.7  # Default

    def test_format_sources_external_news(self, provider):
        """Test formatting for external news sources"""
        sources = [
            {
                "type": "external_news",
                "icon": "📰",
                "description": "Beauty Trends 2026",
                "source": "Allure Magazine",
                "url": "https://allure.com/test",
                "published_at": "2026-01-27",
                "reliability_score": 0.9,
            }
        ]

        formatted = provider.format_sources_for_display(sources)

        assert "📰" in formatted
        assert "Beauty Trends 2026" in formatted
        assert "신뢰도: 90%" in formatted
        assert "Allure Magazine" in formatted
        assert "2026-01-27" in formatted

    def test_format_sources_social_media(self, provider):
        """Test formatting for social media sources"""
        sources = [
            {
                "type": "social_media",
                "icon": "💬",
                "description": "Reddit Discussion",
                "source": "Reddit",
                "url": "https://reddit.com/test",
                "published_at": "2026-01-27",
                "reliability_score": 0.5,
                "relevance_score": 0.75,
            }
        ]

        formatted = provider.format_sources_for_display(sources)

        assert "💬" in formatted
        assert "Reddit Discussion" in formatted
        assert "신뢰도: 50%" in formatted
        assert "플랫폼: Reddit" in formatted
        assert "관련도: 0.75" in formatted

    def test_extract_entity_names_single_dict(self, provider):
        """Test entity extraction from single dict fact"""
        fact = {"subject": "LANEIGE", "predicate": "has_product", "object": "Lip Mask"}

        entities = provider._extract_entity_names(fact)

        assert isinstance(entities, list)
        assert "LANEIGE" in entities
        assert "Lip Mask" in entities

    def test_extract_entity_names_with_empty_values(self, provider):
        """Test entity extraction with empty/None values"""
        facts = [
            {"subject": "LANEIGE", "predicate": "competes_with", "object": ""},
            {"subject": "", "predicate": "has_product", "object": None},
            {"subject": "Innisfree", "predicate": "competes_with", "object": "LANEIGE"},
        ]

        entities = provider._extract_entity_names(facts)

        assert isinstance(entities, list)
        assert "LANEIGE" in entities
        assert "Innisfree" in entities
        assert "" not in entities
        assert None not in entities

    def test_extract_relation_types_single_dict(self, provider):
        """Test relation extraction from single dict fact"""
        fact = {"subject": "LANEIGE", "predicate": "competes_with", "object": "Innisfree"}

        relations = provider._extract_relation_types(fact)

        assert isinstance(relations, list)
        assert "competes_with" in relations

    def test_extract_relation_types_with_empty_values(self, provider):
        """Test relation extraction with empty/None values"""
        facts = [
            {"subject": "LANEIGE", "predicate": "", "object": "Innisfree"},
            {"subject": "LANEIGE", "predicate": "has_product", "object": "Lip Mask"},
            {"subject": "Brand", "predicate": None, "object": "Product"},
        ]

        relations = provider._extract_relation_types(facts)

        assert isinstance(relations, list)
        assert "has_product" in relations
        assert "" not in relations
        assert None not in relations

    def test_extract_mentioned_asins_no_matching_brands(self, provider, mock_current_data):
        """Test ASIN extraction with no matching brands"""
        context = HybridContext(
            query="test query",
            entities={"brands": ["NonExistentBrand"]},  # Brand not in data
            ontology_facts=[],
            inferences=[],
            rag_chunks=[],
            combined_context="",
        )

        asins = provider._extract_mentioned_asins(context, mock_current_data["categories"])

        assert isinstance(asins, list)
        assert len(asins) == 0

    def test_extract_mentioned_asins_max_five(self, provider, mock_hybrid_context):
        """Test ASIN extraction returns max 5 products"""
        categories = {
            "Lip Care": {
                "rank_records": [
                    {
                        "asin": f"B00000000{i}",  # pragma: allowlist secret
                        "brand": "LANEIGE",
                        "product_name": f"Product {i}",
                        "rank": i,
                    }
                    for i in range(1, 10)  # 9 products
                ]
            }
        }

        asins = provider._extract_mentioned_asins(mock_hybrid_context, categories)

        assert len(asins) <= 5

    def test_extract_mentioned_asins_sorted_by_rank(self, provider, mock_hybrid_context):
        """Test ASIN extraction sorts by rank"""
        categories = {
            "Lip Care": {
                "rank_records": [
                    {
                        "asin": "B003",  # pragma: allowlist secret
                        "brand": "LANEIGE",
                        "product_name": "Product C",
                        "rank": 30,
                    },
                    {
                        "asin": "B001",  # pragma: allowlist secret
                        "brand": "LANEIGE",
                        "product_name": "Product A",
                        "rank": 10,
                    },
                    {
                        "asin": "B002",  # pragma: allowlist secret
                        "brand": "LANEIGE",
                        "product_name": "Product B",
                        "rank": 20,
                    },
                ]
            }
        }

        asins = provider._extract_mentioned_asins(mock_hybrid_context, categories)

        assert len(asins) == 3
        assert asins[0]["rank"] == 10
        assert asins[1]["rank"] == 20
        assert asins[2]["rank"] == 30

    def test_crawled_data_source_no_categories(self, provider, mock_hybrid_context):
        """Test crawled data source with no categories"""
        current_data = {
            "metadata": {"data_date": "2026-01-27"},
            "categories": {},
        }

        sources = provider.extract_sources(
            hybrid_context=mock_hybrid_context,
            current_data=current_data,
        )

        crawled = [s for s in sources if s["type"] == "crawled_data"]
        assert len(crawled) == 1
        assert crawled[0]["details"]["total_products"] == 0

    def test_category_hierarchy_source(self, provider, mock_hybrid_context, mock_current_data):
        """Test category hierarchy source extraction"""
        sources = provider.extract_sources(
            hybrid_context=mock_hybrid_context,
            current_data=mock_current_data,
        )

        hierarchy_sources = [s for s in sources if s["type"] == "category_hierarchy"]
        assert len(hierarchy_sources) >= 1
        assert hierarchy_sources[0]["icon"] == "🗂️"
        assert "path" in hierarchy_sources[0]
        assert "level" in hierarchy_sources[0]

    def test_format_sources_category_hierarchy(self, provider):
        """Test formatting for category hierarchy sources"""
        sources = [
            {
                "type": "category_hierarchy",
                "icon": "🗂️",
                "description": "카테고리 계층 구조",
                "path": ["Beauty", "Skin Care", "Lip Care"],
                "level": 2,
                "url": "https://amazon.com/category/lip-care",
            }
        ]

        formatted = provider.format_sources_for_display(sources)

        assert "🗂️" in formatted
        assert "Beauty > Skin Care > Lip Care" in formatted
        assert "레벨: 2" in formatted
        assert "https://amazon.com/category/lip-care" in formatted

    def test_format_sources_all_types(self, provider, mock_current_data):
        """Test formatting with all source types"""
        sources = [
            {
                "type": "crawled_data",
                "icon": "📊",
                "description": "Amazon Data",
                "collected_at": "2026-01-27",
                "url": "https://amazon.com",
                "details": {"total_products": 100},
                "mentioned_products": [
                    {
                        "asin": "B001",  # pragma: allowlist secret
                        "name": "Product",
                        "rank": 1,
                        "category": "Lip Care",
                    }
                ],
            },
            {
                "type": "knowledge_graph",
                "icon": "🔗",
                "description": "KG Data",
                "fact_count": 10,
                "entities": ["LANEIGE"],
                "relations": ["competes_with"],
            },
            {
                "type": "ontology_inference",
                "icon": "🧠",
                "description": "Inference",
                "rule_name": "test_rule",
                "confidence": 0.9,
            },
            {
                "type": "rag_document",
                "icon": "📄",
                "description": "Document",
                "file_path": "/path/to/doc.md",
                "section": "Intro",
                "relevance_score": 0.85,
            },
        ]

        formatted = provider.format_sources_for_display(sources)

        assert "📊" in formatted
        assert "🔗" in formatted
        assert "🧠" in formatted
        assert "📄" in formatted
        assert "2026-01-27" in formatted
        assert "LANEIGE" in formatted
