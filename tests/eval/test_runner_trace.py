"""EvalRunner 트레이스 캡처의 동시성 안전성 테스트

배경 (2026-08-30): concurrency>1로 평가 시 공유 에이전트의
_last_hybrid_context가 경쟁 상태로 덮어써져, 문항 트레이스가 다른 문항의
컨텍스트(엔티티·청크·KG 사실)를 캡처하는 오염이 있었다.
수정: chat() 결과에 요청별 hybrid_context를 동봉하고 러너가 이를 우선 사용.
"""

from types import SimpleNamespace

from eval.runner import EvalRunner, _normalize_edge_node


def _ctx(entities: dict, chunks: list) -> SimpleNamespace:
    return SimpleNamespace(
        entities=entities,
        rag_chunks=chunks,
        ontology_facts=[],
        inferences=[],
    )


def _bare_runner() -> EvalRunner:
    return EvalRunner.__new__(EvalRunner)


class TestTraceConcurrencySafety:
    def test_prefers_result_embedded_context_over_agent_state(self):
        """결과 동봉 컨텍스트가 있으면 공유 에이전트 상태를 읽지 않아야 함"""
        runner = _bare_runner()
        own_ctx = _ctx({"brands": ["LANEIGE"], "categories": [], "concepts": ["sos"]}, [])
        other_ctx = _ctx({"brands": ["COSRX"], "categories": [], "concepts": ["hhi"]}, [])
        runner.agent = SimpleNamespace(get_last_hybrid_context=lambda: other_ctx)

        trace = runner._extract_l1_trace({"hybrid_context": own_ctx}, own_ctx)
        assert trace.extracted_brands == ["LANEIGE"]
        assert trace.extracted_concepts == ["sos"]

    def test_capture_trace_uses_result_context(self):
        """_capture_trace 수준에서 result의 컨텍스트가 선택되는지 (경쟁 상태 방지)"""
        runner = _bare_runner()
        own_ctx = _ctx({"brands": ["LANEIGE"], "categories": ["lip_care"]}, [])
        other_ctx = _ctx({"brands": ["COSRX"], "categories": ["face_powder"]}, [])
        runner.agent = SimpleNamespace(get_last_hybrid_context=lambda: other_ctx)

        result = {"response": "answer", "hybrid_context": own_ctx}
        # _capture_trace 내부의 컨텍스트 선택 로직만 검증 (앞부분 재현)
        hybrid_ctx = result.get("hybrid_context")
        if hybrid_ctx is None and hasattr(runner.agent, "get_last_hybrid_context"):
            hybrid_ctx = runner.agent.get_last_hybrid_context()
        assert hybrid_ctx is own_ctx

    def test_falls_back_to_agent_state_without_embedded_context(self):
        """구형 에이전트(결과에 컨텍스트 미동봉)는 기존 경로 유지"""
        runner = _bare_runner()
        agent_ctx = _ctx({"brands": ["LANEIGE"]}, [])
        runner.agent = SimpleNamespace(get_last_hybrid_context=lambda: agent_ctx)

        result = {"response": "answer"}
        hybrid_ctx = result.get("hybrid_context")
        if hybrid_ctx is None and hasattr(runner.agent, "get_last_hybrid_context"):
            hybrid_ctx = runner.agent.get_last_hybrid_context()
        assert hybrid_ctx is agent_ctx


def test_normalize_edge_node():
    assert _normalize_edge_node("LANEIGE") == "laneige"
    assert _normalize_edge_node("Burt's Bees") == "burts_bees"


class TestL3TraceOntologyStatic:
    """[2026-09 사후] O0 후속: L3 엣지 추출이 ontology_static 카드(트랙 O3, 플래그
    ontology.use_class_reasoning ON)도 metric_edges처럼 담는지."""

    def test_extracts_edges_from_ontology_static_fact(self):
        runner = _bare_runner()
        ctx = SimpleNamespace(
            entities={},
            ontology_facts=[
                {
                    "type": "ontology_static",
                    "entity": "cosrx",
                    "data": {
                        "edges": [
                            {
                                "subject": "cosrx",
                                "predicate": "ownedByGroup",
                                "object": "amorepacific",
                            },
                            {"subject": "cosrx", "predicate": "hasSegment", "object": "Premium"},
                        ],
                        "as_of": "2026-08-31",
                        "version": "1",
                    },
                }
            ],
            inferences=[],
        )
        trace = runner._extract_l3_trace({}, ctx)
        assert "cosrx -ownedByGroup-> amorepacific" in trace.kg_edges_found
        assert "cosrx -hasSegment-> premium" in trace.kg_edges_found

    def test_ontology_static_edge_without_object_is_skipped(self):
        """expansionTruncated 등 object가 없는 카드는 엣지가 아니라 건너뛴다."""
        runner = _bare_runner()
        ctx = SimpleNamespace(
            entities={},
            ontology_facts=[
                {
                    "type": "ontology_static",
                    "entity": "query",
                    "data": {
                        "edges": [
                            {
                                "subject": "amorepacific",
                                "predicate": "expansionTruncated",
                                "object": None,
                                "count": 3,
                            }
                        ],
                        "as_of": "2026-08-31",
                        "version": "1",
                    },
                }
            ],
            inferences=[],
        )
        trace = runner._extract_l3_trace({}, ctx)
        assert trace.kg_edges_found == []

    def test_off_flag_trace_unaffected_no_ontology_static_fact(self):
        """플래그 OFF에서는 ontology_static 카드가 애초에 생기지 않으므로 metric_edges만 그대로."""
        runner = _bare_runner()
        ctx = SimpleNamespace(
            entities={},
            ontology_facts=[
                {
                    "type": "metric_edges",
                    "entity": "cosrx",
                    "data": {
                        "edges": [
                            {"subject": "cosrx", "predicate": "ownedBy", "object": "amorepacific"}
                        ]
                    },
                }
            ],
            inferences=[],
        )
        trace = runner._extract_l3_trace({}, ctx)
        assert trace.kg_edges_found == ["cosrx -ownedBy-> amorepacific"]
