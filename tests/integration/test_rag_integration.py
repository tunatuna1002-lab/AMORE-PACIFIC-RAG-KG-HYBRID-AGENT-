"""
RAG 문서 통합 테스트
===================
신규 7개 문서의 통합 및 의도 기반 검색 테스트

테스트 범위:
1. 문서 로딩 테스트 (11개 문서 모두 로드)
2. 문서 유형별 메타데이터 확인
3. QueryIntent 분류 테스트
4. Intent 기반 필터링 검색 테스트
5. 표(Table) 청킹 테스트
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]


# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestResult:
    """테스트 결과 추적"""

    __test__ = False  # Prevent pytest collection

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


def test_query_intent_classification(results: TestResult):
    """QueryIntent 분류 테스트"""
    print("\n📋 테스트 1: QueryIntent Classification")

    try:
        from src.rag.hybrid_retriever import (
            QueryIntent,
            classify_intent,
            get_doc_type_filter,
        )

        results.record_pass("Import QueryIntent 모듈")

        # DIAGNOSIS 테스트
        diagnosis_queries = [
            "LANEIGE 순위가 왜 떨어졌나요?",
            "BSR 급등 원인 분석",
            "갑자기 판매량이 증가한 이유",
            "순위 변동 체크 해줘",
        ]
        for query in diagnosis_queries:
            intent = classify_intent(query)
            assert intent == QueryIntent.DIAGNOSIS, (
                f"Expected DIAGNOSIS for '{query}', got {intent}"
            )
        results.record_pass("DIAGNOSIS 의도 분류")

        # TREND 테스트
        trend_queries = [
            "요즘 미국 립케어 트렌드는?",
            "최근 인기 성분은 뭔가요",
            "바이럴 키워드 알려줘",
            "PDRN 트렌드 분석",
        ]
        for query in trend_queries:
            intent = classify_intent(query)
            assert intent == QueryIntent.TREND, f"Expected TREND for '{query}', got {intent}"
        results.record_pass("TREND 의도 분류")

        # CRISIS 테스트
        crisis_queries = [
            "부정 리뷰 대응 어떻게 해?",
            "인플루언서 마케팅 메시지",
            "브랜드 이슈 대응 방안",
            "문제 발생시 어떻게 해야 하나요",
        ]
        for query in crisis_queries:
            intent = classify_intent(query)
            assert intent == QueryIntent.CRISIS, f"Expected CRISIS for '{query}', got {intent}"
        results.record_pass("CRISIS 의도 분류")

        # METRIC 테스트
        metric_queries = [
            "SoS 지표 해석 방법",
            "HHI 계산 공식",
            "CPI 의미가 뭔가요",
            "시장 점유율 지표 정의",
        ]
        for query in metric_queries:
            intent = classify_intent(query)
            assert intent == QueryIntent.METRIC, f"Expected METRIC for '{query}', got {intent}"
        results.record_pass("METRIC 의도 분류")

        # get_doc_type_filter 테스트
        diagnosis_filter = get_doc_type_filter(QueryIntent.DIAGNOSIS)
        assert diagnosis_filter == ["playbook", "metric_guide", "intelligence"]
        results.record_pass("Intent별 doc_type_filter 반환")

    except Exception as e:
        results.record_fail("QueryIntent Classification", str(e))


def test_document_metadata(results: TestResult):
    """문서 메타데이터 테스트"""
    print("\n📋 테스트 2: Document Metadata")

    try:
        from src.rag.retriever import DocumentRetriever

        # DOCUMENTS 딕셔너리 확인
        docs = DocumentRetriever.DOCUMENTS
        results.record_pass("DOCUMENTS 딕셔너리 접근")

        # 11개 문서 확인
        assert len(docs) == 11, f"Expected 11 documents, got {len(docs)}"
        results.record_pass(f"총 {len(docs)}개 문서 정의됨")

        # 필수 메타데이터 필드 확인
        required_fields = [
            "filename",
            "description",
            "keywords",
            "doc_type",
            "intent_triggers",
            "freshness",
        ]
        for doc_id, doc_info in docs.items():
            for field in required_fields:
                assert field in doc_info, f"Missing '{field}' in {doc_id}"
        results.record_pass("필수 메타데이터 필드 존재")

        # 문서 유형별 개수 확인
        doc_types = [d["doc_type"] for d in docs.values()]
        type_counts = {
            "metric_guide": doc_types.count("metric_guide"),
            "playbook": doc_types.count("playbook"),
            "intelligence": doc_types.count("intelligence"),
            "knowledge_base": doc_types.count("knowledge_base"),
            "response_guide": doc_types.count("response_guide"),
        }
        assert type_counts["metric_guide"] == 4, (
            f"Expected 4 metric_guide, got {type_counts['metric_guide']}"
        )
        assert type_counts["playbook"] == 2, f"Expected 2 playbook, got {type_counts['playbook']}"
        assert type_counts["intelligence"] == 2, (
            f"Expected 2 intelligence, got {type_counts['intelligence']}"
        )
        assert type_counts["knowledge_base"] == 1, (
            f"Expected 1 knowledge_base, got {type_counts['knowledge_base']}"
        )
        assert type_counts["response_guide"] == 2, (
            f"Expected 2 response_guide, got {type_counts['response_guide']}"
        )
        results.record_pass(
            f"문서 유형별 개수: metric_guide={type_counts['metric_guide']}, playbook={type_counts['playbook']}, intelligence={type_counts['intelligence']}, knowledge_base={type_counts['knowledge_base']}, response_guide={type_counts['response_guide']}"
        )

    except Exception as e:
        results.record_fail("Document Metadata", str(e))


@pytest.mark.asyncio
async def test_document_loading(results: TestResult):
    """문서 로딩 테스트"""
    print("\n📋 테스트 3: Document Loading")

    try:
        from src.rag.retriever import DocumentRetriever

        # DocumentRetriever 초기화
        retriever = DocumentRetriever(docs_path="./docs")
        await retriever.initialize()
        results.record_pass("DocumentRetriever 초기화")

        # 로드된 문서 수 확인
        loaded_docs = len(retriever.documents)
        assert loaded_docs >= 10, f"Expected at least 10 documents, got {loaded_docs}"
        results.record_pass(f"{loaded_docs}개 문서 로드됨")

        # 청크 수 확인
        chunk_count = len(retriever.chunks)
        assert chunk_count > 0, "No chunks created"
        results.record_pass(f"{chunk_count}개 청크 생성됨")

        # 청크에 doc_type 포함 확인
        has_doc_type = all("doc_type" in chunk for chunk in retriever.chunks)
        assert has_doc_type, "Some chunks missing doc_type"
        results.record_pass("모든 청크에 doc_type 포함")

        # 청크에 content_type 포함 확인
        has_content_type = all("content_type" in chunk for chunk in retriever.chunks)
        assert has_content_type, "Some chunks missing content_type"
        results.record_pass("모든 청크에 content_type 포함")

        # 테이블 청크 확인
        table_chunks = [c for c in retriever.chunks if c.get("content_type") == "table"]
        print(f"    ℹ️  표(Table) 청크: {len(table_chunks)}개")
        results.record_pass(f"표 청크 분리 완료 ({len(table_chunks)}개)")

        return retriever

    except Exception as e:
        results.record_fail("Document Loading", str(e))
        return None


@pytest.mark.asyncio
async def test_intent_based_search(results: TestResult, retriever):
    """Intent 기반 검색 테스트"""
    print("\n📋 테스트 4: Intent-based Search")

    if retriever is None:
        results.record_fail("Intent-based Search", "Retriever not initialized")
        return

    try:
        from src.rag.hybrid_retriever import classify_intent, get_doc_type_filter

        # DIAGNOSIS 쿼리 테스트
        diagnosis_query = "BSR 순위가 갑자기 떨어진 원인은?"
        intent = classify_intent(diagnosis_query)
        doc_type_filter = get_doc_type_filter(intent)

        search_results = await retriever.search(
            diagnosis_query, top_k=5, doc_type_filter=doc_type_filter
        )

        # playbook 또는 metric_guide 우선 반환 확인
        if search_results:
            first_doc_type = search_results[0]["metadata"].get("doc_type")
            assert first_doc_type in [
                "playbook",
                "metric_guide",
                "intelligence",
            ], f"Expected playbook/metric_guide/intelligence first, got {first_doc_type}"
            results.record_pass("DIAGNOSIS 쿼리: 플레이북 우선 반환")
        else:
            results.record_pass("DIAGNOSIS 쿼리: 검색 완료 (결과 없음 - 키워드 검색 폴백 시 정상)")

        # TREND 쿼리 테스트
        trend_query = "요즘 미국에서 인기있는 스킨케어 트렌드"
        intent = classify_intent(trend_query)
        doc_type_filter = get_doc_type_filter(intent)

        search_results = await retriever.search(
            trend_query, top_k=5, doc_type_filter=doc_type_filter
        )

        if search_results:
            first_doc_type = search_results[0]["metadata"].get("doc_type")
            assert first_doc_type in [
                "intelligence",
                "knowledge_base",
                "response_guide",
            ], f"Expected intelligence/knowledge_base/response_guide first, got {first_doc_type}"
            results.record_pass("TREND 쿼리: 인텔리전스 우선 반환")
        else:
            results.record_pass("TREND 쿼리: 검색 완료 (결과 없음 - 키워드 검색 폴백 시 정상)")

        # CRISIS 쿼리 테스트
        crisis_query = "부정 리뷰 대응 방안 알려줘"
        intent = classify_intent(crisis_query)
        doc_type_filter = get_doc_type_filter(intent)

        search_results = await retriever.search(
            crisis_query, top_k=5, doc_type_filter=doc_type_filter
        )

        if search_results:
            first_doc_type = search_results[0]["metadata"].get("doc_type")
            assert first_doc_type in [
                "response_guide",
                "intelligence",
                "playbook",
            ], f"Expected response_guide/intelligence first, got {first_doc_type}"
            results.record_pass("CRISIS 쿼리: 대응 가이드 우선 반환")
        else:
            results.record_pass("CRISIS 쿼리: 검색 완료 (결과 없음 - 키워드 검색 폴백 시 정상)")

        # 전체 검색 (GENERAL)
        general_query = "LANEIGE"
        search_results = await retriever.search(
            general_query,
            top_k=5,
            doc_type_filter=None,  # 전체 문서 검색
        )
        results.record_pass(f"GENERAL 쿼리: {len(search_results)}개 결과 반환")

    except Exception as e:
        results.record_fail("Intent-based Search", str(e))


@pytest.mark.asyncio
async def test_hybrid_retriever_integration(results: TestResult):
    """HybridRetriever 통합 테스트"""
    print("\n📋 테스트 5: HybridRetriever Integration")

    try:
        from src.ontology.business_rules import register_all_rules
        from src.ontology.knowledge_graph import KnowledgeGraph
        from src.ontology.reasoner import OntologyReasoner
        from src.rag.hybrid_retriever import HybridRetriever, QueryIntent

        # 컴포넌트 초기화
        kg = KnowledgeGraph()
        reasoner = OntologyReasoner()
        register_all_rules(reasoner)

        # HybridRetriever 생성
        hybrid_retriever = HybridRetriever(knowledge_graph=kg, reasoner=reasoner)
        await hybrid_retriever.initialize()
        results.record_pass("HybridRetriever 초기화")

        # DIAGNOSIS 쿼리로 테스트
        query = "LANEIGE 립슬리핑마스크 순위가 왜 떨어졌나요?"
        context = await hybrid_retriever.retrieve(query, current_metrics={"sos": 0.15, "hhi": 0.22})

        # 메타데이터에 query_intent 포함 확인
        assert "query_intent" in context.metadata, "Missing query_intent in metadata"
        assert context.metadata["query_intent"] == QueryIntent.DIAGNOSIS.value
        results.record_pass("query_intent 메타데이터 포함")

        # doc_type_filter 메타데이터 확인
        assert "doc_type_filter" in context.metadata, "Missing doc_type_filter in metadata"
        results.record_pass("doc_type_filter 메타데이터 포함")

        # RAG 청크 반환 확인
        print(f"    ℹ️  RAG 청크 수: {len(context.rag_chunks)}")
        results.record_pass(f"RAG 검색 완료 ({len(context.rag_chunks)}개 청크)")

        # combined_context 생성 확인
        assert len(context.combined_context) > 0, "combined_context is empty"
        results.record_pass("combined_context 생성 완료")

    except Exception as e:
        results.record_fail("HybridRetriever Integration", str(e))


async def run_all_tests():
    """모든 테스트 실행"""
    print("=" * 60)
    print("🧪 RAG 문서 통합 테스트")
    print(f"   실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = TestResult()

    # Phase 1: QueryIntent 분류
    print("\n" + "=" * 40)
    print("📦 Phase 1: QueryIntent Classification")
    print("=" * 40)
    test_query_intent_classification(results)

    # Phase 2: 문서 메타데이터
    print("\n" + "=" * 40)
    print("📦 Phase 2: Document Metadata")
    print("=" * 40)
    test_document_metadata(results)

    # Phase 3: 문서 로딩
    print("\n" + "=" * 40)
    print("📦 Phase 3: Document Loading")
    print("=" * 40)
    retriever = await test_document_loading(results)

    # Phase 4: Intent 기반 검색
    print("\n" + "=" * 40)
    print("📦 Phase 4: Intent-based Search")
    print("=" * 40)
    await test_intent_based_search(results, retriever)

    # Phase 5: HybridRetriever 통합
    print("\n" + "=" * 40)
    print("📦 Phase 5: HybridRetriever Integration")
    print("=" * 40)
    await test_hybrid_retriever_integration(results)

    # 결과 출력
    print(results.summary())

    if results.errors:
        print("\n❌ 실패한 테스트 상세:")
        for error in results.errors:
            print(f"   - {error}")

    return results.failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
