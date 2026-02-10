"""
ReAct Agent 통합 검증 테스트
brain.py의 신뢰도 라우팅과 ReAct 에이전트 연동 검증
"""

import inspect

from src.core.models import Context


class TestReActIntegration:
    """ReAct Agent가 brain.py와 올바르게 연동되는지 검증"""

    def test_react_agent_importable(self):
        """ReAct Agent import 가능"""
        from src.core.react_agent import ReActAgent

        assert ReActAgent is not None

    def test_react_agent_instantiation(self):
        """ReAct Agent 인스턴스 생성"""
        from src.core.react_agent import ReActAgent

        agent = ReActAgent()
        assert agent is not None

    def test_react_agent_has_process_method(self):
        """ReAct Agent에 run 메서드 존재"""
        from src.core.react_agent import ReActAgent

        agent = ReActAgent()
        # ReActAgent는 run() 메서드 사용
        assert hasattr(agent, "run")

    def test_brain_complex_query_detection(self):
        """brain.py의 복잡한 질문 감지 로직"""
        from src.core.brain import UnifiedBrain

        brain = UnifiedBrain.__new__(UnifiedBrain)

        # _is_complex_query가 존재하는지 확인
        assert hasattr(brain, "_is_complex_query")

    def test_brain_has_react_agent_field(self):
        """brain.py에 _process_with_react 메서드 존재"""
        from src.core.brain import UnifiedBrain

        # _process_with_react 메서드 존재 확인
        assert hasattr(UnifiedBrain, "_process_with_react")

    def test_confidence_routing_preserves_react_path(self):
        """신뢰도 라우팅이 ReAct 경로를 보존하는지 확인

        MEDIUM/LOW 신뢰도에서 복잡한 질문은 여전히 ReAct로 가야 함
        (3.1: process_query가 QueryGraph에 위임, 라우팅은 QueryGraph에서 처리)
        """
        from src.core.query_graph import QueryGraph

        # QueryGraph의 라우팅에서 ReAct 경로 확인
        route_source = inspect.getsource(QueryGraph._route_after_confidence)
        assert "_is_complex_query" in route_source

        # QueryGraph.run에서 ReAct 노드 호출 확인
        run_source = inspect.getsource(QueryGraph.run)
        assert "_node_react" in run_source

    def test_high_confidence_skips_react(self):
        """HIGH 신뢰도에서는 ReAct를 건너뛰는지 확인

        HIGH 신뢰도 → direct response (ReAct 불필요)
        (3.1: 라우팅 로직이 QueryGraph._route_after_confidence로 이동)
        """
        from src.core.query_graph import QueryGraph

        source = inspect.getsource(QueryGraph._route_after_confidence)
        # HIGH confidence path (should_skip_llm_decision) should come before ReAct check
        high_pos = source.find("should_skip_llm_decision")
        react_pos = source.find("_is_complex_query")
        if high_pos >= 0 and react_pos >= 0:
            # HIGH confidence check should appear before ReAct
            assert high_pos < react_pos, "HIGH confidence should be checked before ReAct"


class TestComplexQueryDetection:
    """복잡한 질문 감지 로직 단위 테스트"""

    def test_analysis_keyword_is_complex(self):
        """분석 키워드 포함 → 복잡"""
        from src.core.brain import UnifiedBrain

        brain = UnifiedBrain.__new__(UnifiedBrain)
        brain._react_agent = True  # Mock

        context = Context(query="분석해줘")
        context.rag_docs = []

        result = brain._is_complex_query("LANEIGE 경쟁사 대비 분석해줘", context)
        assert result is True

    def test_simple_query_not_complex(self):
        """단순 질문 → 비복잡"""
        from src.core.brain import UnifiedBrain

        brain = UnifiedBrain.__new__(UnifiedBrain)
        brain._react_agent = True

        context = Context(query="순위")
        context.rag_docs = [
            {"content": "doc1"},
            {"content": "doc2"},
            {"content": "doc3"},
        ]

        result = brain._is_complex_query("LANEIGE 순위", context)
        assert result is False

    def test_multi_step_query_is_complex(self):
        """다단계 질문 → 복잡"""
        from src.core.brain import UnifiedBrain

        brain = UnifiedBrain.__new__(UnifiedBrain)
        brain._react_agent = True

        context = Context(query="test")
        context.rag_docs = []

        result = brain._is_complex_query("LANEIGE 순위는? 그리고 경쟁사 대비 어때?", context)
        assert result is True
