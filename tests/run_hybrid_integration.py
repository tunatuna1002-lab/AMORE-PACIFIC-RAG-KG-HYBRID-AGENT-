"""
통합 테스트: Ontology-RAG Hybrid System
이 테스트는 전체 하이브리드 시스템의 기능을 검증합니다.

테스트 범위:
1. Ontology 컴포넌트 (relations, knowledge_graph, reasoner, business_rules)
2. Hybrid RAG 컴포넌트 (hybrid_retriever, context_builder)
3. Hybrid Agents (hybrid_insight_agent, hybrid_chatbot_agent)
4. Orchestrator 통합
"""

import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestResult:
    """테스트 결과 추적"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: list[str] = []

    def record_pass(self, test_name: str):
        self.passed += 1
        print(f"  ✅ {test_name}")

    def record_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"  ❌ {test_name}: {error}")

    def summary(self) -> str:
        total = self.passed + self.failed
        return f"\n{'=' * 60}\n총 {total}개 테스트: ✅ {self.passed} 성공, ❌ {self.failed} 실패\n{'=' * 60}"


def test_ontology_relations(results: TestResult):
    """Phase 1.1: 관계 타입 테스트"""
    print("\n📋 테스트 1: Ontology Relations")

    try:
        from ontology.relations import (
            InferenceResult,
            InsightType,
            Relation,
            RelationType,
            create_brand_product_relation,
            create_competition_relation,
        )

        results.record_pass("Import relations module")

        # RelationType 검증
        assert len(RelationType) >= 10, "관계 타입 10개 이상 필요"
        results.record_pass("RelationType enum 정의")

        # Relation 생성
        relation = Relation(
            subject="Brand_A", predicate=RelationType.HAS_PRODUCT, object="Product_X"
        )
        assert relation.subject == "Brand_A"
        results.record_pass("Relation dataclass 생성")

        # Helper 함수 테스트 - 실제 시그니처에 맞게 수정
        brand_rel = create_brand_product_relation(
            brand="TestBrand", product_asin="B08XYZ123", product_name="TestProduct"
        )
        assert brand_rel.predicate == RelationType.HAS_PRODUCT
        results.record_pass("create_brand_product_relation 함수")

        # competition relation - 실제 시그니처에 맞게 수정
        comp_rel = create_competition_relation(
            brand1="Brand_A", brand2="Brand_B", category="lip_care", competition_type="direct"
        )
        assert comp_rel.predicate == RelationType.DIRECT_COMPETITOR
        results.record_pass("create_competition_relation 함수")

        # InferenceResult 테스트 - 실제 시그니처에 맞게 수정
        inf_result = InferenceResult(
            rule_name="test_rule",
            insight_type=InsightType.MARKET_POSITION,
            insight="테스트 인사이트입니다",
            confidence=0.9,
            evidence={"hhi": 0.2},
        )
        assert inf_result.rule_name == "test_rule"
        results.record_pass("InferenceResult dataclass 생성")

    except Exception as e:
        results.record_fail("Ontology Relations", str(e))


def test_knowledge_graph(results: TestResult):
    """Phase 1.2: 지식 그래프 테스트"""
    print("\n📋 테스트 2: Knowledge Graph")

    try:
        from ontology.knowledge_graph import KnowledgeGraph
        from ontology.relations import Relation, RelationType

        kg = KnowledgeGraph()
        results.record_pass("KnowledgeGraph 인스턴스 생성")

        # 관계 추가 - Relation 객체로 추가
        rel1 = Relation(subject="Brand_A", predicate=RelationType.HAS_PRODUCT, object="Product_X")
        rel2 = Relation(subject="Brand_A", predicate=RelationType.HAS_PRODUCT, object="Product_Y")
        rel3 = Relation(subject="Brand_B", predicate=RelationType.COMPETES_WITH, object="Brand_A")
        kg.add_relation(rel1)
        kg.add_relation(rel2)
        kg.add_relation(rel3)
        results.record_pass("관계 추가")

        # 쿼리 테스트
        products = kg.query(subject="Brand_A", predicate=RelationType.HAS_PRODUCT)
        assert len(products) == 2, f"Expected 2 products, got {len(products)}"
        results.record_pass("관계 쿼리")

        # 이웃 노드 검색
        neighbors = kg.get_neighbors("Brand_A")
        outgoing = neighbors.get("outgoing", [])
        assert len(outgoing) > 0, "No outgoing neighbors found"
        results.record_pass("이웃 노드 검색")

        # 통계 확인
        stats = kg.get_stats()
        assert stats["total_triples"] == 3
        results.record_pass("통계 확인")

    except Exception as e:
        results.record_fail("Knowledge Graph", str(e))


def test_reasoner(results: TestResult):
    """Phase 1.3: 추론 엔진 테스트"""
    print("\n📋 테스트 3: Ontology Reasoner")

    try:
        from ontology.reasoner import InferenceRule, OntologyReasoner, RuleCondition
        from ontology.relations import InsightType

        reasoner = OntologyReasoner()
        results.record_pass("OntologyReasoner 인스턴스 생성")

        # 테스트 규칙 생성 - 실제 RuleCondition 시그니처에 맞게 수정
        test_condition = RuleCondition(
            name="hhi_low", check=lambda ctx: ctx.get("hhi", 1) <= 0.3, description="HHI 0.3 이하"
        )

        def conclusion_func(ctx):
            return {
                "insight": f"시장이 분산되어 있습니다 (HHI: {ctx.get('hhi', 0)})",
                "recommendation": "다양한 세그먼트 공략 필요",
            }

        test_rule = InferenceRule(
            name="test_rule_001",
            description="HHI가 0.3 이하면 시장 분산됨",
            conditions=[test_condition],
            conclusion=conclusion_func,
            insight_type=InsightType.MARKET_POSITION,
            priority=1,
        )

        reasoner.register_rule(test_rule)
        results.record_pass("규칙 등록")

        # 추론 실행
        test_context = {"hhi": 0.25, "category": "test"}
        inferences = reasoner.infer(test_context)
        assert len(inferences) >= 1, "추론 결과가 없습니다"
        results.record_pass("추론 실행")

        # 설명 생성
        explanation = reasoner.explain_inference(inferences[0])
        assert "규칙" in explanation or "조건" in explanation
        results.record_pass("추론 설명 생성")

    except Exception as e:
        results.record_fail("Ontology Reasoner", str(e))


def test_business_rules(results: TestResult):
    """Phase 1.4: 비즈니스 규칙 테스트"""
    print("\n📋 테스트 4: Business Rules")

    try:
        from ontology.business_rules import ALL_BUSINESS_RULES, register_all_rules
        from ontology.reasoner import OntologyReasoner

        # 규칙 수 확인
        assert (
            len(ALL_BUSINESS_RULES) >= 10
        ), f"최소 10개 규칙 필요, 현재 {len(ALL_BUSINESS_RULES)}개"
        results.record_pass(f"비즈니스 규칙 {len(ALL_BUSINESS_RULES)}개 정의됨")

        # 규칙 등록 테스트
        reasoner = OntologyReasoner()
        register_all_rules(reasoner)

        rule_count = len(reasoner.rules)
        assert rule_count >= 10
        results.record_pass(f"Reasoner에 {rule_count}개 규칙 등록됨")

        # 특정 시나리오 테스트
        # 시장 분산 시나리오
        test_context_fragmented = {"hhi": 0.15, "top1_sos": 0.10, "sos": 0.10}
        inferences1 = reasoner.infer(test_context_fragmented)
        results.record_pass(f"시장 분산 시나리오: {len(inferences1)}개 인사이트 생성")

        # 시장 지배 시나리오
        test_context_dominant = {"hhi": 0.45, "top1_sos": 0.60, "sos": 0.60}
        inferences2 = reasoner.infer(test_context_dominant)
        results.record_pass(f"시장 지배 시나리오: {len(inferences2)}개 인사이트 생성")

    except Exception as e:
        results.record_fail("Business Rules", str(e))


def test_hybrid_retriever(results: TestResult):
    """Phase 2.1: 하이브리드 검색기 테스트"""
    print("\n📋 테스트 5: Hybrid Retriever")

    try:
        from ontology.business_rules import register_all_rules
        from ontology.knowledge_graph import KnowledgeGraph
        from ontology.reasoner import OntologyReasoner
        from rag.hybrid_retriever import EntityExtractor, HybridContext, HybridRetriever

        # EntityExtractor 테스트
        extractor = EntityExtractor()
        entities = extractor.extract("LG 브랜드의 시장 점유율 분석해줘")
        results.record_pass("EntityExtractor 동작")

        # HybridRetriever 인스턴스 생성 - doc_retriever 파라미터 사용
        kg = KnowledgeGraph()
        reasoner = OntologyReasoner()
        register_all_rules(reasoner)

        retriever = HybridRetriever(
            knowledge_graph=kg,
            reasoner=reasoner,
            doc_retriever=None,  # RAG 없이 테스트
        )
        results.record_pass("HybridRetriever 인스턴스 생성")

        # HybridContext 구조 확인
        context = HybridContext(query="테스트 쿼리")
        assert hasattr(context, "inferences")
        assert hasattr(context, "rag_chunks")
        results.record_pass("HybridContext 구조 검증")

    except Exception as e:
        results.record_fail("Hybrid Retriever", str(e))


def test_context_builder(results: TestResult):
    """Phase 2.2: 컨텍스트 빌더 테스트"""
    print("\n📋 테스트 6: Context Builder")

    try:
        from ontology.relations import InferenceResult, InsightType
        from rag.context_builder import CompactContextBuilder, ContextBuilder
        from rag.hybrid_retriever import HybridContext

        # ContextBuilder 인스턴스 생성
        builder = ContextBuilder(max_tokens=4000)
        results.record_pass("ContextBuilder 인스턴스 생성")

        # 테스트용 InferenceResult 생성 - 실제 시그니처에 맞게 수정
        test_inference = InferenceResult(
            rule_name="test_001",
            insight_type=InsightType.MARKET_POSITION,
            insight="테스트 결론입니다",
            confidence=0.9,
            evidence={"hhi": 0.2},
        )

        hybrid_context = HybridContext(
            query="시장 분석",
            inferences=[test_inference],
            rag_chunks=[{"content": "문서 1 내용", "metadata": {"title": "Test"}}],
        )

        # 시스템 프롬프트 생성
        system_prompt = builder.build_system_prompt(hybrid_context)
        assert len(system_prompt) > 100
        results.record_pass("시스템 프롬프트 생성")

        # 유저 프롬프트 생성
        user_prompt = builder.build_user_prompt("시장 분석해줘", hybrid_context)
        assert len(user_prompt) > 0
        results.record_pass("유저 프롬프트 생성")

        # CompactContextBuilder 테스트
        compact_builder = CompactContextBuilder()
        compact_prompt = compact_builder.build(hybrid_context)
        assert len(compact_prompt) > 0
        results.record_pass("CompactContextBuilder 동작")

    except Exception as e:
        results.record_fail("Context Builder", str(e))


def test_hybrid_insight_agent(results: TestResult):
    """Phase 3.1: 하이브리드 인사이트 에이전트 테스트"""
    print("\n📋 테스트 7: Hybrid Insight Agent")

    try:
        from agents.hybrid_insight_agent import HybridInsightAgent
        from ontology.knowledge_graph import KnowledgeGraph
        from ontology.reasoner import OntologyReasoner

        # 에이전트 인스턴스 생성
        kg = KnowledgeGraph()
        reasoner = OntologyReasoner()

        agent = HybridInsightAgent(knowledge_graph=kg, reasoner=reasoner)
        results.record_pass("HybridInsightAgent 인스턴스 생성")

        # 에이전트 속성 확인 - 실제 구현은 kg, reasoner로 저장
        assert hasattr(agent, "kg")
        assert hasattr(agent, "reasoner")
        results.record_pass("HybridInsightAgent 속성 검증")

    except Exception as e:
        results.record_fail("Hybrid Insight Agent", str(e))


def test_hybrid_chatbot_agent(results: TestResult):
    """Phase 3.2: 하이브리드 챗봇 에이전트 테스트"""
    print("\n📋 테스트 8: Hybrid Chatbot Agent")

    try:
        from agents.hybrid_chatbot_agent import HybridChatbotAgent, HybridChatbotSession
        from ontology.knowledge_graph import KnowledgeGraph
        from ontology.reasoner import OntologyReasoner

        # 에이전트 인스턴스 생성
        kg = KnowledgeGraph()
        reasoner = OntologyReasoner()

        agent = HybridChatbotAgent(knowledge_graph=kg, reasoner=reasoner)
        results.record_pass("HybridChatbotAgent 인스턴스 생성")

        # 세션 테스트
        session = HybridChatbotSession(agent)
        results.record_pass("HybridChatbotSession 생성")

        # 속성 확인
        assert hasattr(agent, "explain_last_response")
        results.record_pass("explain_last_response 메서드 존재")

    except Exception as e:
        results.record_fail("Hybrid Chatbot Agent", str(e))


def test_orchestrator_integration(results: TestResult):
    """Phase 3.3: 오케스트레이터 통합 테스트"""
    print("\n📋 테스트 9: Orchestrator Integration")

    try:
        from orchestrator import Orchestrator, WorkflowStep

        # Orchestrator 인스턴스 생성 (hybrid 모드)
        orchestrator = Orchestrator(use_hybrid=True)
        results.record_pass("Orchestrator 인스턴스 생성 (hybrid=True)")

        # use_hybrid 플래그 확인
        assert orchestrator.use_hybrid == True
        results.record_pass("use_hybrid 플래그 활성화")

        # UPDATE_KG 워크플로우 스텝 확인
        assert hasattr(WorkflowStep, "UPDATE_KG")
        results.record_pass("UPDATE_KG 워크플로우 스텝 정의됨")

        # Knowledge Graph 속성 확인 (property 접근)
        kg = orchestrator.knowledge_graph
        assert kg is not None
        results.record_pass("knowledge_graph 속성 존재")

        # Reasoner 속성 확인 (property 접근)
        reasoner = orchestrator.reasoner
        assert reasoner is not None
        results.record_pass("reasoner 속성 존재")

        # 하이브리드 에이전트 속성 확인
        hybrid_insight = orchestrator.hybrid_insight
        hybrid_chatbot = orchestrator.hybrid_chatbot
        assert hybrid_insight is not None
        assert hybrid_chatbot is not None
        results.record_pass("hybrid agent 속성 존재")

        # 통계 메서드 확인
        assert hasattr(orchestrator, "get_knowledge_graph_stats")
        assert hasattr(orchestrator, "get_inference_stats")
        results.record_pass("통계 메서드 존재")

    except Exception as e:
        results.record_fail("Orchestrator Integration", str(e))


def test_end_to_end_workflow(results: TestResult):
    """전체 워크플로우 E2E 테스트"""
    print("\n📋 테스트 10: End-to-End Workflow")

    try:
        from ontology.business_rules import register_all_rules
        from ontology.knowledge_graph import KnowledgeGraph
        from ontology.reasoner import OntologyReasoner
        from ontology.relations import Relation, RelationType
        from rag.context_builder import ContextBuilder
        from rag.hybrid_retriever import HybridRetriever

        # 1. Knowledge Graph 구축
        kg = KnowledgeGraph()
        kg.add_relation(
            Relation(subject="LG", predicate=RelationType.HAS_PRODUCT, object="LG_TV_001")
        )
        kg.add_relation(
            Relation(subject="Samsung", predicate=RelationType.HAS_PRODUCT, object="Samsung_TV_001")
        )
        kg.add_relation(
            Relation(subject="LG", predicate=RelationType.COMPETES_WITH, object="Samsung")
        )
        kg.add_relation(
            Relation(subject="LG_TV_001", predicate=RelationType.BELONGS_TO_CATEGORY, object="TV")
        )
        results.record_pass("E2E: Knowledge Graph 구축")

        # 2. Reasoner 설정
        reasoner = OntologyReasoner()
        register_all_rules(reasoner)
        results.record_pass("E2E: Reasoner 규칙 등록")

        # 3. Hybrid Retrieval
        retriever = HybridRetriever(knowledge_graph=kg, reasoner=reasoner)

        metrics_context = {
            "hhi": 0.22,
            "sos": 0.35,
            "top1_sos": 0.35,
            "brand": "LG",
            "category": "TV",
        }

        # 추론 직접 테스트
        inferences = reasoner.infer(metrics_context)
        results.record_pass(f"E2E: {len(inferences)}개 인사이트 추론 완료")

        # 4. Context Building
        from rag.hybrid_retriever import HybridContext

        context = HybridContext(query="LG의 시장 점유율은?", inferences=inferences)

        builder = ContextBuilder()
        system_prompt = builder.build_system_prompt(context)
        user_prompt = builder.build_user_prompt("LG의 시장 점유율은?", context)
        results.record_pass("E2E: Context 빌드 완료")

        # 5. 최종 출력 검증
        assert len(system_prompt) > 50
        assert len(user_prompt) > 10
        results.record_pass("E2E: 전체 파이프라인 성공")

    except Exception as e:
        results.record_fail("End-to-End Workflow", str(e))


def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 60)
    print("🧪 Ontology-RAG Hybrid System 통합 테스트")
    print(f"   실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = TestResult()

    # Phase 1: Ontology 컴포넌트
    print("\n" + "=" * 40)
    print("📦 Phase 1: Ontology Components")
    print("=" * 40)
    test_ontology_relations(results)
    test_knowledge_graph(results)
    test_reasoner(results)
    test_business_rules(results)

    # Phase 2: Hybrid RAG
    print("\n" + "=" * 40)
    print("📦 Phase 2: Hybrid RAG Components")
    print("=" * 40)
    test_hybrid_retriever(results)
    test_context_builder(results)

    # Phase 3: Hybrid Agents
    print("\n" + "=" * 40)
    print("📦 Phase 3: Hybrid Agents")
    print("=" * 40)
    test_hybrid_insight_agent(results)
    test_hybrid_chatbot_agent(results)
    test_orchestrator_integration(results)

    # Phase 4: E2E
    print("\n" + "=" * 40)
    print("📦 Phase 4: End-to-End Test")
    print("=" * 40)
    test_end_to_end_workflow(results)

    # 결과 출력
    print(results.summary())

    if results.errors:
        print("\n❌ 실패한 테스트 상세:")
        for error in results.errors:
            print(f"   - {error}")

    return results.failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
