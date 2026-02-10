"""Tests for ConversationMemory (3.3)"""

from src.memory.conversation_memory import ConversationContext, ConversationMemory, ConversationTurn


class TestConversationTurn:
    def test_create_turn(self):
        turn = ConversationTurn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.content == "hello"

    def test_to_dict(self):
        turn = ConversationTurn(role="user", content="test")
        d = turn.to_dict()
        assert d["role"] == "user"
        assert "timestamp" in d


class TestConversationContext:
    def test_to_prompt_text_empty(self):
        ctx = ConversationContext()
        assert ctx.to_prompt_text() == ""

    def test_to_prompt_text_with_turns(self):
        ctx = ConversationContext(recent_turns=[{"role": "user", "content": "hello"}])
        text = ctx.to_prompt_text()
        assert "사용자" in text
        assert "hello" in text

    def test_to_prompt_text_with_summary(self):
        ctx = ConversationContext(summary="이전 대화 내용")
        text = ctx.to_prompt_text()
        assert "이전 대화 요약" in text

    def test_to_prompt_text_with_entities(self):
        ctx = ConversationContext(tracked_entities={"brands": ["LANEIGE", "COSRX"]})
        text = ctx.to_prompt_text()
        assert "엔티티" in text
        assert "LANEIGE" in text

    def test_to_prompt_text_long_content_truncation(self):
        long_content = "x" * 500
        ctx = ConversationContext(recent_turns=[{"role": "user", "content": long_content}])
        text = ctx.to_prompt_text()
        assert "..." in text
        assert len(text) < len(long_content)


class TestConversationMemory:
    def test_add_and_get(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "LANEIGE 순위 알려줘")
        mem.add_turn("s1", "assistant", "현재 3위입니다")
        ctx = mem.get_context("s1")
        assert ctx.total_turns == 2
        assert len(ctx.recent_turns) == 2

    def test_recent_turns_limit(self):
        mem = ConversationMemory(max_recent_turns=2)
        for i in range(5):
            mem.add_turn("s1", "user", f"message {i}")
        ctx = mem.get_context("s1")
        assert len(ctx.recent_turns) == 2
        assert ctx.total_turns == 5

    def test_entity_tracking(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "LANEIGE 점유율 알려줘")
        ctx = mem.get_context("s1")
        assert "LANEIGE" in ctx.tracked_entities.get("brands", [])

    def test_entity_accumulation(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "LANEIGE 분석")
        mem.add_turn("s1", "user", "COSRX 비교")
        ctx = mem.get_context("s1")
        brands = ctx.tracked_entities.get("brands", [])
        assert "LANEIGE" in brands
        assert "COSRX" in brands

    def test_clear_session(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "test")
        mem.clear_session("s1")
        ctx = mem.get_context("s1")
        assert ctx.total_turns == 0

    def test_lru_eviction(self):
        mem = ConversationMemory(max_sessions=2)
        mem.add_turn("s1", "user", "first")
        mem.add_turn("s2", "user", "second")
        mem.add_turn("s3", "user", "third")  # s1 evicted
        assert "s1" not in mem._sessions
        assert "s2" in mem._sessions
        assert "s3" in mem._sessions

    def test_max_turns_per_session(self):
        mem = ConversationMemory(max_turns_per_session=3)
        for i in range(5):
            mem.add_turn("s1", "user", f"message {i}")
        assert len(mem._sessions["s1"]) <= 3
        # Old turns should be summarized
        assert "s1" in mem._summaries

    def test_get_history(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "hello")
        mem.add_turn("s1", "assistant", "hi")
        history = mem.get_history("s1")
        assert len(history) == 2

    def test_empty_session_context(self):
        mem = ConversationMemory()
        ctx = mem.get_context("nonexistent")
        assert ctx.total_turns == 0
        assert ctx.recent_turns == []

    def test_stats(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "a")
        mem.add_turn("s2", "user", "b")
        stats = mem.get_stats()
        assert stats["active_sessions"] == 2
        assert stats["total_turns"] == 2

    def test_custom_entities(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "test", entities={"brands": ["CustomBrand"]})
        ctx = mem.get_context("s1")
        assert "CustomBrand" in ctx.tracked_entities.get("brands", [])

    def test_indicator_extraction(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "SoS와 HHI 비교해줘")
        ctx = mem.get_context("s1")
        indicators = ctx.tracked_entities.get("indicators", [])
        assert "SoS" in indicators
        assert "HHI" in indicators

    def test_korean_brand_extraction(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "라네즈 제품 좋아요")
        ctx = mem.get_context("s1")
        assert "LANEIGE" in ctx.tracked_entities.get("brands", [])

    def test_multiple_brands_in_one_message(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "LANEIGE와 COSRX 비교")
        ctx = mem.get_context("s1")
        brands = ctx.tracked_entities.get("brands", [])
        assert "LANEIGE" in brands
        assert "COSRX" in brands

    def test_summary_generation(self):
        mem = ConversationMemory(max_turns_per_session=2)
        mem.add_turn("s1", "user", "first message")
        mem.add_turn("s1", "assistant", "first response")
        mem.add_turn("s1", "user", "second message")  # Triggers summarization
        assert "s1" in mem._summaries
        assert len(mem._summaries["s1"]) > 0

    def test_summary_length_limit(self):
        mem = ConversationMemory(max_turns_per_session=2)
        long_content = "x" * 1000
        for i in range(10):
            mem.add_turn("s1", "user", long_content)
        # Summary should be truncated to 2000 chars
        assert len(mem._summaries.get("s1", "")) <= 2000

    def test_lru_touch_on_get_context(self):
        mem = ConversationMemory(max_sessions=2)
        mem.add_turn("s1", "user", "first")
        mem.add_turn("s2", "user", "second")
        # Touch s1 by getting context
        mem.get_context("s1")
        # Add s3, which should evict s2 (not s1)
        mem.add_turn("s3", "user", "third")
        assert "s1" in mem._sessions
        assert "s2" not in mem._sessions
        assert "s3" in mem._sessions

    def test_session_id_in_context(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "test")
        ctx = mem.get_context("s1")
        assert ctx.session_id == "s1"

    def test_entity_deduplication(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "LANEIGE")
        mem.add_turn("s1", "user", "LANEIGE")
        ctx = mem.get_context("s1")
        brands = ctx.tracked_entities.get("brands", [])
        # Should only appear once
        assert brands.count("LANEIGE") == 1

    def test_empty_entities_not_in_prompt(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "random text without brands")
        ctx = mem.get_context("s1")
        text = ctx.to_prompt_text()
        # Should not include empty entity section
        if not ctx.tracked_entities.get("brands") and not ctx.tracked_entities.get("indicators"):
            assert "엔티티" not in text

    def test_conversation_turn_metadata(self):
        turn = ConversationTurn(role="user", content="test", metadata={"source": "api"})
        assert turn.metadata["source"] == "api"

    def test_conversation_turn_entities(self):
        turn = ConversationTurn(role="user", content="test", entities={"brands": ["LANEIGE"]})
        assert turn.entities["brands"] == ["LANEIGE"]

    def test_get_history_empty_session(self):
        mem = ConversationMemory()
        history = mem.get_history("nonexistent")
        assert history == []

    def test_multiple_sessions_independence(self):
        mem = ConversationMemory()
        mem.add_turn("s1", "user", "LANEIGE")
        mem.add_turn("s2", "user", "COSRX")
        ctx1 = mem.get_context("s1")
        ctx2 = mem.get_context("s2")
        assert "LANEIGE" in ctx1.tracked_entities.get("brands", [])
        assert "COSRX" in ctx2.tracked_entities.get("brands", [])
        assert "COSRX" not in ctx1.tracked_entities.get("brands", [])
        assert "LANEIGE" not in ctx2.tracked_entities.get("brands", [])

    def test_stats_empty_memory(self):
        mem = ConversationMemory()
        stats = mem.get_stats()
        assert stats["active_sessions"] == 0
        assert stats["total_turns"] == 0
        assert stats["sessions_with_summary"] == 0

    def test_clear_nonexistent_session(self):
        mem = ConversationMemory()
        # Should not raise error
        mem.clear_session("nonexistent")
        stats = mem.get_stats()
        assert stats["active_sessions"] == 0
