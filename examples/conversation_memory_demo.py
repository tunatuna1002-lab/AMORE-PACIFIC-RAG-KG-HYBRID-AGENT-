"""
ConversationMemory Demo
=======================
Demonstrates the ConversationMemory module functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.memory.conversation_memory import ConversationMemory


def demo_basic_usage():
    """Basic conversation tracking"""
    print("=== Basic Usage ===\n")

    memory = ConversationMemory()

    # Add conversation turns
    memory.add_turn("session1", "user", "LANEIGE 점유율 알려줘")
    memory.add_turn("session1", "assistant", "현재 LANEIGE의 SoS는 12.5%입니다.")
    memory.add_turn("session1", "user", "COSRX와 비교하면 어때?")
    memory.add_turn("session1", "assistant", "COSRX는 8.3%로 LANEIGE보다 낮습니다.")

    # Get context
    ctx = memory.get_context("session1")
    print(f"Total turns: {ctx.total_turns}")
    print(f"Recent turns: {len(ctx.recent_turns)}")
    print(f"Tracked brands: {ctx.tracked_entities.get('brands', [])}")
    print(f"Tracked indicators: {ctx.tracked_entities.get('indicators', [])}")
    print()


def demo_prompt_generation():
    """LLM prompt generation"""
    print("=== Prompt Generation ===\n")

    memory = ConversationMemory(max_recent_turns=2)

    memory.add_turn("session2", "user", "LANEIGE Lip Sleeping Mask 순위는?")
    memory.add_turn("session2", "assistant", "현재 3위입니다.")
    memory.add_turn("session2", "user", "가격은?")
    memory.add_turn("session2", "assistant", "$24입니다.")

    ctx = memory.get_context("session2")
    prompt_text = ctx.to_prompt_text()

    print("Generated prompt context:")
    print("-" * 60)
    print(prompt_text)
    print("-" * 60)
    print()


def demo_lru_eviction():
    """LRU session eviction"""
    print("=== LRU Eviction ===\n")

    memory = ConversationMemory(max_sessions=2)

    memory.add_turn("s1", "user", "First session")
    memory.add_turn("s2", "user", "Second session")
    print(f"Active sessions: {memory.get_stats()['active_sessions']}")

    memory.add_turn("s3", "user", "Third session (s1 evicted)")
    print(f"Active sessions after s3: {memory.get_stats()['active_sessions']}")
    print(f"s1 exists: {'s1' in memory._sessions}")
    print(f"s2 exists: {'s2' in memory._sessions}")
    print(f"s3 exists: {'s3' in memory._sessions}")
    print()


def demo_turn_summarization():
    """Turn summarization on overflow"""
    print("=== Turn Summarization ===\n")

    memory = ConversationMemory(max_turns_per_session=3)

    for i in range(5):
        memory.add_turn("session3", "user", f"Message {i}")

    ctx = memory.get_context("session3")
    print(f"Total turns recorded: {ctx.total_turns}")
    print(f"Summary exists: {bool(ctx.summary)}")
    if ctx.summary:
        print(f"Summary preview: {ctx.summary[:100]}...")
    print()


def demo_entity_tracking():
    """Entity extraction and tracking"""
    print("=== Entity Tracking ===\n")

    memory = ConversationMemory()

    memory.add_turn("session4", "user", "LANEIGE와 COSRX 비교해줘")
    memory.add_turn("session4", "user", "SoS와 HHI 지표로")
    memory.add_turn("session4", "user", "설화수도 추가해줘")

    ctx = memory.get_context("session4")
    print(f"Tracked brands: {ctx.tracked_entities.get('brands', [])}")
    print(f"Tracked indicators: {ctx.tracked_entities.get('indicators', [])}")
    print()


def demo_multiple_sessions():
    """Multiple independent sessions"""
    print("=== Multiple Sessions ===\n")

    memory = ConversationMemory()

    # Session A: LANEIGE focus
    memory.add_turn("sessionA", "user", "LANEIGE 분석해줘")
    memory.add_turn("sessionA", "assistant", "LANEIGE는 현재...")

    # Session B: COSRX focus
    memory.add_turn("sessionB", "user", "COSRX 어때?")
    memory.add_turn("sessionB", "assistant", "COSRX는...")

    ctx_a = memory.get_context("sessionA")
    ctx_b = memory.get_context("sessionB")

    print(f"Session A brands: {ctx_a.tracked_entities.get('brands', [])}")
    print(f"Session B brands: {ctx_b.tracked_entities.get('brands', [])}")
    print(f"Sessions are independent: {ctx_a.tracked_entities != ctx_b.tracked_entities}")
    print()


def demo_stats():
    """Memory statistics"""
    print("=== Memory Statistics ===\n")

    memory = ConversationMemory()

    memory.add_turn("s1", "user", "Hello")
    memory.add_turn("s1", "assistant", "Hi")
    memory.add_turn("s2", "user", "Test")

    stats = memory.get_stats()
    print(f"Active sessions: {stats['active_sessions']}")
    print(f"Total turns: {stats['total_turns']}")
    print(f"Sessions with summary: {stats['sessions_with_summary']}")
    print()


if __name__ == "__main__":
    demo_basic_usage()
    demo_prompt_generation()
    demo_lru_eviction()
    demo_turn_summarization()
    demo_entity_tracking()
    demo_multiple_sessions()
    demo_stats()

    print("✅ All demos completed successfully!")
