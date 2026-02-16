"""
OWL Reasoner 단위 테스트
========================
owlready2 미설치 환경(fallback 모드)에서의 동작을 테스트합니다.
owlready2 의존 없이 모든 공개 메서드의 fallback 경로를 검증합니다.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.ontology.owl_reasoner import OWLReasoner, get_owl_reasoner

# =========================================================================
# Fixture: owlready2 unavailable 환경 시뮬레이션
# =========================================================================


@pytest.fixture
def reasoner_no_owl():
    """owlready2 미설치 상태의 OWLReasoner (fallback 모드)"""
    with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
        mock_fallback = MagicMock()
        r = OWLReasoner(fallback_reasoner=mock_fallback)
        assert r.onto is None
        yield r


@pytest.fixture
def reasoner_no_owl_auto_fallback():
    """owlready2 미설치 상태에서 자동 fallback reasoner 생성"""
    with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
        with patch(
            "src.ontology.owl_reasoner.OWLReasoner.__init__", return_value=None
        ) as mock_init:
            # Manually create instance to test auto-fallback logic
            pass
    # Instead, test with explicit fallback
    with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
        mock_fallback = MagicMock()
        r = OWLReasoner(fallback_reasoner=mock_fallback)
        yield r


# =========================================================================
# 초기화 테스트
# =========================================================================


class TestOWLReasonerInit:
    """OWLReasoner 초기화 테스트"""

    def test_init_without_owlready2_uses_fallback(self, reasoner_no_owl):
        """owlready2 미설치 시 fallback reasoner 사용"""
        assert reasoner_no_owl.onto is None
        assert reasoner_no_owl.fallback_reasoner is not None

    def test_init_with_explicit_fallback(self):
        """명시적 fallback reasoner 전달"""
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
            mock_fb = MagicMock()
            r = OWLReasoner(fallback_reasoner=mock_fb)
            assert r.fallback_reasoner is mock_fb

    def test_init_auto_creates_fallback_when_none(self):
        """fallback_reasoner=None이면 OntologyReasoner를 자동 생성"""
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
            with patch("src.ontology.reasoner.OntologyReasoner") as MockReasoner:
                MockReasoner.return_value = MagicMock()
                r = OWLReasoner(fallback_reasoner=None)
                assert r.fallback_reasoner is not None

    def test_init_stores_owl_file_path(self):
        """owl_file 경로가 Path로 저장"""
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
            r = OWLReasoner(owl_file="/tmp/test.owl", fallback_reasoner=MagicMock())
            from pathlib import Path

            assert r.owl_file == Path("/tmp/test.owl")

    def test_init_stores_reasoner_type(self):
        """reasoner_type 저장"""
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
            r = OWLReasoner(reasoner_type="hermit", fallback_reasoner=MagicMock())
            assert r.reasoner_type == "hermit"

    def test_init_default_reasoner_type_is_pellet(self, reasoner_no_owl):
        """기본 reasoner_type은 pellet"""
        assert reasoner_no_owl.reasoner_type == "pellet"

    def test_init_none_owl_file_stores_none(self, reasoner_no_owl):
        """owl_file=None이면 None으로 저장"""
        assert reasoner_no_owl.owl_file is None


# =========================================================================
# 비동기 초기화 테스트
# =========================================================================


class TestOWLReasonerInitializeAsync:
    """비동기 초기화 (호환성) 테스트"""

    @pytest.mark.asyncio
    async def test_initialize_returns_none(self, reasoner_no_owl):
        """initialize()는 로깅만 하고 정상 반환"""
        result = await reasoner_no_owl.initialize()
        assert result is None


# =========================================================================
# 엔티티 추가 (fallback 모드) 테스트
# =========================================================================


class TestOWLReasonerAddBrandFallback:
    """add_brand fallback 모드 테스트"""

    def test_add_brand_returns_false_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 add_brand는 False 반환"""
        result = reasoner_no_owl.add_brand("LANEIGE", sos=0.25)
        assert result is False

    def test_add_brand_with_all_params_returns_false(self, reasoner_no_owl):
        """모든 파라미터 전달해도 False"""
        result = reasoner_no_owl.add_brand("LANEIGE", sos=0.25, avg_rank=15.5, product_count=5)
        assert result is False


class TestOWLReasonerAddProductFallback:
    """add_product fallback 모드 테스트"""

    def test_add_product_returns_false_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 add_product는 False 반환"""
        result = reasoner_no_owl.add_product(
            asin="B08XYZ123", brand="LANEIGE", category="lip_care", rank=10
        )
        assert result is False

    def test_add_product_with_optional_params(self, reasoner_no_owl):
        """선택 파라미터 포함해도 False"""
        result = reasoner_no_owl.add_product(
            asin="B08XYZ123",
            brand="LANEIGE",
            category="lip_care",
            rank=10,
            price=22.0,
            rating=4.5,
        )
        assert result is False


class TestOWLReasonerAddCompetitorFallback:
    """add_competitor_relation fallback 모드 테스트"""

    def test_add_competitor_returns_false_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 add_competitor_relation는 False 반환"""
        result = reasoner_no_owl.add_competitor_relation("LANEIGE", "COSRX")
        assert result is False


class TestOWLReasonerAddTrendFallback:
    """add_trend fallback 모드 테스트"""

    def test_add_trend_returns_false_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 add_trend는 False 반환"""
        result = reasoner_no_owl.add_trend("K-Beauty", ["LANEIGE", "COSRX"])
        assert result is False


# =========================================================================
# 추론 실행 (fallback 모드) 테스트
# =========================================================================


class TestOWLReasonerRunReasonerFallback:
    """run_reasoner fallback 모드 테스트"""

    def test_run_reasoner_returns_false_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 run_reasoner는 False 반환"""
        result = reasoner_no_owl.run_reasoner()
        assert result is False


class TestOWLReasonerInferMarketPositionsFallback:
    """infer_market_positions fallback 모드 테스트"""

    def test_infer_returns_empty_dict_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 infer_market_positions는 빈 dict 반환"""
        result = reasoner_no_owl.infer_market_positions()
        assert result == {}


class TestOWLReasonerGetInferredFactsFallback:
    """get_inferred_facts fallback 모드 테스트"""

    def test_get_inferred_facts_returns_empty_list(self, reasoner_no_owl):
        """owlready2 미설치 시 get_inferred_facts는 빈 리스트 반환"""
        result = reasoner_no_owl.get_inferred_facts()
        assert result == []


# =========================================================================
# 쿼리 기능 (fallback 모드) 테스트
# =========================================================================


class TestOWLReasonerQueryFallback:
    """쿼리 기능 fallback 모드 테스트"""

    def test_query_sparql_returns_empty_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 query_sparql는 빈 리스트 반환"""
        result = reasoner_no_owl.query_sparql("SELECT ?x WHERE { ?x a :Brand }")
        assert result == []

    def test_get_brand_info_returns_none_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 get_brand_info는 None 반환"""
        result = reasoner_no_owl.get_brand_info("LANEIGE")
        assert result is None

    def test_get_competitors_returns_empty_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 get_competitors는 빈 리스트 반환"""
        result = reasoner_no_owl.get_competitors("LANEIGE")
        assert result == []

    def test_get_all_brands_returns_empty_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 get_all_brands는 빈 리스트 반환"""
        result = reasoner_no_owl.get_all_brands()
        assert result == []

    def test_get_category_brands_returns_empty_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 get_category_brands는 빈 리스트 반환"""
        result = reasoner_no_owl.get_category_brands("lip_care")
        assert result == []

    def test_get_brand_market_position_returns_none_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 get_brand_market_position는 None 반환"""
        result = reasoner_no_owl.get_brand_market_position("LANEIGE")
        assert result is None


# =========================================================================
# 데이터 마이그레이션 (fallback 모드) 테스트
# =========================================================================


class TestOWLReasonerImportFallback:
    """데이터 마이그레이션 fallback 모드 테스트"""

    def test_import_from_knowledge_graph_returns_zero(self, reasoner_no_owl):
        """owlready2 미설치 시 import_from_knowledge_graph는 0 반환"""
        mock_kg = MagicMock()
        result = reasoner_no_owl.import_from_knowledge_graph(mock_kg)
        assert result == 0

    def test_import_from_metrics_returns_zero(self, reasoner_no_owl):
        """owlready2 미설치 시 import_from_metrics는 0 반환"""
        result = reasoner_no_owl.import_from_metrics({"brand_metrics": []})
        assert result == 0

    def test_import_from_metrics_with_data_returns_zero(self, reasoner_no_owl):
        """데이터가 있어도 owlready2 미설치 시 0 반환"""
        metrics_data = {
            "brand_metrics": [
                {
                    "brand_name": "LANEIGE",
                    "share_of_shelf": 0.25,
                    "avg_rank": 15.0,
                    "product_count": 5,
                },
            ]
        }
        result = reasoner_no_owl.import_from_metrics(metrics_data)
        assert result == 0


# =========================================================================
# 영속화 (fallback 모드) 테스트
# =========================================================================


class TestOWLReasonerPersistenceFallback:
    """영속화 fallback 모드 테스트"""

    def test_save_returns_false_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 save는 False 반환"""
        result = reasoner_no_owl.save()
        assert result is False

    def test_save_with_path_returns_false(self, reasoner_no_owl):
        """경로 지정해도 False"""
        result = reasoner_no_owl.save(path="/tmp/test.owl")
        assert result is False

    def test_load_returns_false_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 load는 False 반환"""
        result = reasoner_no_owl.load("/tmp/test.owl")
        assert result is False


# =========================================================================
# 유틸리티 (fallback 모드) 테스트
# =========================================================================


class TestOWLReasonerUtilsFallback:
    """유틸리티 fallback 모드 테스트"""

    def test_get_stats_returns_error_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 get_stats는 에러 dict 반환"""
        result = reasoner_no_owl.get_stats()
        assert "error" in result

    def test_clear_does_not_raise_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 clear는 예외 없이 반환"""
        reasoner_no_owl.clear()  # should not raise

    def test_repr_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 repr 반환"""
        result = repr(reasoner_no_owl)
        assert "OWLReasoner" in result


# =========================================================================
# 싱글톤 패턴 테스트
# =========================================================================


class TestGetOWLReasonerSingleton:
    """get_owl_reasoner 싱글톤 테스트"""

    def test_get_owl_reasoner_returns_instance(self):
        """get_owl_reasoner는 OWLReasoner 인스턴스 반환"""
        import src.ontology.owl_reasoner as mod

        # Reset singleton state
        mod._owl_reasoner_instance = None
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
            with patch("src.ontology.reasoner.OntologyReasoner", return_value=MagicMock()):
                instance = get_owl_reasoner()
                assert isinstance(instance, OWLReasoner)
                # Cleanup
                mod._owl_reasoner_instance = None

    def test_get_owl_reasoner_returns_same_instance(self):
        """get_owl_reasoner는 동일 인스턴스 반환 (싱글톤)"""
        import src.ontology.owl_reasoner as mod

        mod._owl_reasoner_instance = None
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
            with patch("src.ontology.reasoner.OntologyReasoner", return_value=MagicMock()):
                inst1 = get_owl_reasoner()
                inst2 = get_owl_reasoner()
                assert inst1 is inst2
                mod._owl_reasoner_instance = None

    def test_get_owl_reasoner_accepts_params(self):
        """get_owl_reasoner는 owl_file, reasoner_type 파라미터 수용"""
        import src.ontology.owl_reasoner as mod

        mod._owl_reasoner_instance = None
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
            with patch("src.ontology.reasoner.OntologyReasoner", return_value=MagicMock()):
                instance = get_owl_reasoner(owl_file="/tmp/test.owl", reasoner_type="hermit")
                assert instance.reasoner_type == "hermit"
                mod._owl_reasoner_instance = None


# =========================================================================
# _define_ontology_structure fallback 테스트
# =========================================================================


class TestOWLReasonerDefineStructureFallback:
    """_define_ontology_structure fallback 테스트"""

    def test_define_structure_noop_without_owlready2(self, reasoner_no_owl):
        """owlready2 미설치 시 _define_ontology_structure는 noop"""
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
            # Should not raise
            reasoner_no_owl._define_ontology_structure()

    def test_define_structure_noop_when_onto_is_none(self, reasoner_no_owl):
        """onto=None이면 _define_ontology_structure는 noop"""
        reasoner_no_owl.onto = None
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", True):
            reasoner_no_owl._define_ontology_structure()


# =========================================================================
# _get_brand_position 테스트
# =========================================================================


class TestOWLReasonerGetBrandPosition:
    """_get_brand_position 내부 메서드 테스트"""

    def test_get_brand_position_dominant(self):
        """DominantBrand 포지션 추출"""
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
            r = OWLReasoner(fallback_reasoner=MagicMock())
            # Mock onto with brand subclass check
            mock_onto = MagicMock()
            mock_brand = MagicMock()
            mock_brand.is_a = [mock_onto.DominantBrand]
            r.onto = mock_onto

            result = r._get_brand_position(mock_brand)
            assert result == "DominantBrand"

    def test_get_brand_position_strong(self):
        """StrongBrand 포지션 추출"""
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
            r = OWLReasoner(fallback_reasoner=MagicMock())
            mock_onto = MagicMock()
            mock_brand = MagicMock()
            # DominantBrand is NOT in is_a, but StrongBrand is
            mock_brand.is_a = [mock_onto.StrongBrand]
            # Make DominantBrand not match
            mock_onto.DominantBrand = MagicMock()
            r.onto = mock_onto

            result = r._get_brand_position(mock_brand)
            assert result == "StrongBrand"

    def test_get_brand_position_niche(self):
        """NicheBrand 포지션 추출"""
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
            r = OWLReasoner(fallback_reasoner=MagicMock())
            mock_onto = MagicMock()
            mock_brand = MagicMock()
            mock_brand.is_a = [mock_onto.NicheBrand]
            mock_onto.DominantBrand = MagicMock()
            mock_onto.StrongBrand = MagicMock()
            r.onto = mock_onto

            result = r._get_brand_position(mock_brand)
            assert result == "NicheBrand"

    def test_get_brand_position_none(self):
        """포지션 없는 브랜드"""
        with patch("src.ontology.owl_reasoner.OWLREADY2_AVAILABLE", False):
            r = OWLReasoner(fallback_reasoner=MagicMock())
            mock_onto = MagicMock()
            mock_brand = MagicMock()
            mock_brand.is_a = []  # No subclass
            mock_onto.DominantBrand = MagicMock()
            mock_onto.StrongBrand = MagicMock()
            mock_onto.NicheBrand = MagicMock()
            r.onto = mock_onto

            result = r._get_brand_position(mock_brand)
            assert result is None
