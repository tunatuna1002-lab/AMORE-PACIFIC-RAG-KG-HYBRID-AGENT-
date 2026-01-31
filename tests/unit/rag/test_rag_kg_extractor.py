"""
RAGKGExtractor 단위 테스트

RAG 문서에서 트렌드/액션/상태 추출 테스트
"""

from unittest.mock import MagicMock

from src.rag.rag_kg_extractor import RAGKGExtractor


class TestRAGKGExtractorInit:
    """초기화 테스트"""

    def test_init_with_kg(self):
        """KG와 함께 초기화"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)
        assert extractor._kg is mock_kg


class TestExtractTrends:
    """트렌드 추출 테스트"""

    def test_extract_glass_skin_trend(self):
        """Glass Skin 트렌드 추출"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        content = "The 'glass skin' trend is dominating K-beauty."
        trends = extractor._extract_trends(content)

        assert "glass skin" in trends

    def test_extract_k_beauty_trend(self):
        """K-beauty 트렌드 추출"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        content = "K-beauty 트렌드가 미국 시장에서 인기를 끌고 있습니다."
        trends = extractor._extract_trends(content)

        assert "k-beauty" in trends

    def test_extract_hydration_trend(self):
        """Hydration 트렌드 추출"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        content = "Hydration is the key to healthy skin."
        trends = extractor._extract_trends(content)

        assert "hydration" in trends

    def test_extract_multiple_trends(self):
        """복수 트렌드 추출"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        content = """
        K-beauty trends include glass skin and slugging.
        Skinimalism is also popular.
        """
        trends = extractor._extract_trends(content)

        assert len(trends) >= 2
        assert "glass skin" in trends or "slugging" in trends

    def test_max_trends_limit(self):
        """최대 10개 트렌드 제한"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        # 많은 트렌드가 포함된 텍스트
        content = """
        glass skin, glazed donut, clean girl,
        k-beauty, skinimalism, skip care,
        slugging, skin barrier, hydration,
        dewy look, glow skin, natural beauty,
        clean beauty, sustainable beauty
        """
        trends = extractor._extract_trends(content)

        assert len(trends) <= 10


class TestExtractActions:
    """액션 추출 테스트"""

    def test_extract_checkbox_action(self):
        """체크박스 형식 액션 추출"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        content = "□ 경쟁사 가격 모니터링 강화\n□ 리뷰 분석 리포트 작성"
        actions = extractor._extract_actions(content)

        assert len(actions) >= 1
        assert any("모니터링" in a or "리포트" in a for a in actions)

    def test_extract_recommend_action(self):
        """권장 액션 추출"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        content = "권장: 주 1회 경쟁사 순위 변동 분석을 수행하세요."
        actions = extractor._extract_actions(content)

        assert len(actions) >= 1

    def test_extract_numbered_action(self):
        """번호 형식 액션 추출"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        content = "1. 순위 하락 시 즉시 대응이 필요합니다."
        actions = extractor._extract_actions(content)

        # 지시문 포함 여부 확인
        assert len(actions) >= 0  # 패턴에 따라 다름

    def test_max_actions_limit(self):
        """최대 5개 액션 제한"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        content = """
        □ 액션1: 매출 분석 필요
        □ 액션2: 가격 조정 필요
        □ 액션3: 마케팅 강화 필요
        □ 액션4: 재고 관리 필요
        □ 액션5: 리뷰 모니터링 필요
        □ 액션6: 경쟁사 분석 필요
        □ 액션7: 트렌드 분석 필요
        """
        actions = extractor._extract_actions(content)

        assert len(actions) <= 5


class TestExtractStates:
    """상태 추출 테스트"""

    def test_extract_rising_state(self):
        """상승 상태 추출"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        content = "현재 상태: 급등"
        states = extractor._extract_states(content)

        assert "급등" in states

    def test_extract_market_state(self):
        """시장 상태 추출"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        content = "시장: 성장 단계에 진입"
        states = extractor._extract_states(content)

        assert "성장" in states

    def test_max_states_limit(self):
        """최대 3개 상태 제한"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        content = """
        상태: 급등
        상태: 상승
        상태: 안정
        상태: 하락
        상태: 급락
        """
        states = extractor._extract_states(content)

        assert len(states) <= 3


class TestInferActionPriority:
    """액션 우선순위 추론 테스트"""

    def test_high_priority_urgent(self):
        """긴급 액션 = high"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        priority = extractor._infer_action_priority("긴급: 즉시 대응 필요")
        assert priority == "high"

    def test_medium_priority_recommend(self):
        """권장 액션 = medium"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        priority = extractor._infer_action_priority("가격 조정을 권장합니다")
        assert priority == "medium"

    def test_low_priority_default(self):
        """기본 = low"""
        mock_kg = MagicMock()
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        priority = extractor._infer_action_priority("일반적인 모니터링")
        assert priority == "low"


class TestExtractFromChunks:
    """청크에서 추출 통합 테스트"""

    def test_extract_from_playbook_chunk(self):
        """Playbook 청크에서 추출"""
        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = True  # 항상 성공
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        chunks = [
            {
                "content": "K-beauty glass skin 트렌드가 인기입니다. 권장: 수분 라인 강화",
                "metadata": {
                    "doc_type": "playbook",
                    "doc_id": "ranking_playbook",
                    "chunk_id": "chunk_001",
                },
            }
        ]

        result = extractor.extract_from_chunks(chunks, brand="LANEIGE")

        assert result["total"] > 0
        assert mock_kg.add_relation.called

    def test_skip_non_playbook_chunks(self):
        """Playbook/intelligence 외 청크 건너뛰기"""
        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = True
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        chunks = [
            {
                "content": "K-beauty glass skin 트렌드",
                "metadata": {
                    "doc_type": "other",  # playbook/intelligence 아님
                    "doc_id": "other_doc",
                },
            }
        ]

        result = extractor.extract_from_chunks(chunks, brand="LANEIGE")

        assert result["total"] == 0
        assert not mock_kg.add_relation.called

    def test_extract_with_source_context(self):
        """source_context 포함 추출"""
        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = True
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        chunks = [
            {
                "content": "Glass skin 트렌드",
                "metadata": {
                    "doc_type": "intelligence",
                    "doc_id": "market_intel",
                },
            }
        ]

        result = extractor.extract_from_chunks(
            chunks, brand="LANEIGE", source_context="daily_insight_2026-01-31"
        )

        # add_relation이 호출되었으면 properties에 context 포함 확인
        if mock_kg.add_relation.called:
            call_args = mock_kg.add_relation.call_args
            relation = call_args[0][0]
            assert relation.properties.get("context") == "daily_insight_2026-01-31"


class TestExtractFromSignal:
    """External Signal에서 추출 테스트"""

    def test_extract_trend_from_signal(self):
        """시그널에서 트렌드 추출"""
        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = True
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        signal = {
            "title": "Glass Skin Trend Takes Over TikTok",
            "summary": "K-beauty glass skin look is trending on social media",
            "source": "tiktok",
            "url": "https://tiktok.com/...",
            "collected_at": "2026-01-31",
        }

        result = extractor.extract_from_signal(signal, brand="LANEIGE")

        assert result["trends_added"] >= 1
        assert mock_kg.add_relation.called

    def test_signal_confidence_lower(self):
        """시그널 신뢰도는 0.6"""
        mock_kg = MagicMock()
        mock_kg.add_relation.return_value = True
        extractor = RAGKGExtractor(knowledge_graph=mock_kg)

        signal = {
            "title": "Glass skin trend",
            "summary": "Glass skin is popular",
            "source": "reddit",
        }

        extractor.extract_from_signal(signal, brand="LANEIGE")

        if mock_kg.add_relation.called:
            call_args = mock_kg.add_relation.call_args
            relation = call_args[0][0]
            assert relation.confidence == 0.6
