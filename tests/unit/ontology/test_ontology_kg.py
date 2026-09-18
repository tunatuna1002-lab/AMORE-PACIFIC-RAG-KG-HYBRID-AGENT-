"""
OntologyKnowledgeGraph 단위 테스트
"""

import pytest

from src.ontology.ontology_knowledge_graph import OntologyKnowledgeGraph


class TestOntologyKnowledgeGraph:
    """OntologyKnowledgeGraph 기본 테스트"""

    def test_instantiation(self):
        """인스턴스 생성"""
        okg = OntologyKnowledgeGraph(knowledge_graph=None)
        assert okg is not None

    def test_has_required_methods(self):
        """필수 메서드 존재 확인"""
        okg = OntologyKnowledgeGraph(knowledge_graph=None)
        assert hasattr(okg, "initialize")
        assert hasattr(okg, "add_validated_relation")
        assert hasattr(okg, "sync_owl_inferences")
        assert hasattr(okg, "check_consistency")

    def test_owl_class_mapping(self):
        """OWL 클래스 매핑 확인"""
        from src.ontology.ontology_knowledge_graph import OWL_CLASS_MAPPING

        assert isinstance(OWL_CLASS_MAPPING, dict)
        assert len(OWL_CLASS_MAPPING) > 0
        assert "Brand" in OWL_CLASS_MAPPING or "brand" in OWL_CLASS_MAPPING

    @pytest.mark.asyncio
    async def test_initialize(self):
        """비동기 초기화"""
        okg = OntologyKnowledgeGraph(knowledge_graph=None)
        await okg.initialize()
        assert okg._initialized is True

    @pytest.mark.asyncio
    async def test_double_initialize(self):
        """이중 초기화 방지"""
        okg = OntologyKnowledgeGraph(knowledge_graph=None)
        await okg.initialize()
        await okg.initialize()  # Should not raise
        assert okg._initialized is True
