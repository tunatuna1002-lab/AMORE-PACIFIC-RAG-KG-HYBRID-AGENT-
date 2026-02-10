"""
ReAct Agent Demo
================
ReAct Self-Reflection 패턴 시연

Usage:
    python examples/react_agent_demo.py
"""

import asyncio
import json
import logging

from src.core.models import ToolResult
from src.core.react_agent import ReActAgent

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class DemoToolExecutor:
    """데모용 도구 실행기"""

    def __init__(self):
        self.data = {
            "laneige_rank": 5,
            "laneige_sos": 12.5,
            "competitors": ["CeraVe", "Neutrogena", "La Roche-Posay"],
            "market_trend": "성장 중",
        }

    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        """도구 실행 시뮬레이션"""
        logger.info(f"🔧 Tool: {tool_name} | Params: {params}")

        if tool_name == "query_data":
            query_type = params.get("query_type", "brand_metrics")

            if query_type == "brand_metrics":
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    data={
                        "brand": "LANEIGE",
                        "rank": self.data["laneige_rank"],
                        "sos": self.data["laneige_sos"],
                    },
                )
            elif query_type == "competitor_analysis":
                return ToolResult(
                    tool_name=tool_name,
                    success=True,
                    data={
                        "competitors": self.data["competitors"],
                        "trend": self.data["market_trend"],
                    },
                )

        elif tool_name == "query_knowledge_graph":
            entity = params.get("entity", "LANEIGE")
            return ToolResult(
                tool_name=tool_name,
                success=True,
                data={
                    "entity": entity,
                    "relations": [
                        {"type": "competes_with", "target": "CeraVe"},
                        {"type": "category", "target": "Lip Care"},
                    ],
                },
            )

        elif tool_name == "final_answer":
            return ToolResult(tool_name=tool_name, success=True, data={"completed": True})

        return ToolResult(tool_name=tool_name, success=False, error=f"Unknown tool: {tool_name}")


async def demo_simple_query():
    """간단한 질문 시연"""
    print("\n" + "=" * 80)
    print("🔹 Demo 1: 간단한 질문 (단일 도구 호출)")
    print("=" * 80)

    agent = ReActAgent(max_iterations=3)
    agent.set_tool_executor(DemoToolExecutor())

    query = "LANEIGE의 현재 순위는?"
    context = "최근 데이터: Amazon Lip Care 카테고리 Top 100"

    print(f"\n📝 질문: {query}")
    print(f"📄 컨텍스트: {context}")

    result = await agent.run(query, context)

    print(f"\n✅ 최종 답변: {result.final_answer}")
    print(f"🔁 반복 횟수: {result.iterations}")
    print(f"📊 신뢰도: {result.confidence:.2f}")

    print("\n📋 실행 단계:")
    for i, step in enumerate(result.steps, 1):
        print(f"\n  Step {i}:")
        print(f"    💭 Thought: {step.thought[:80]}...")
        if step.action:
            print(f"    🎬 Action: {step.action}")
        if step.observation:
            print(f"    👁️  Observation: {step.observation[:80]}...")


async def demo_complex_query():
    """복잡한 질문 시연"""
    print("\n" + "=" * 80)
    print("🔹 Demo 2: 복잡한 질문 (다중 도구 호출)")
    print("=" * 80)

    agent = ReActAgent(max_iterations=5)
    agent.set_tool_executor(DemoToolExecutor())

    query = "LANEIGE가 경쟁사 대비 어떤 위치에 있는지 분석해줘"
    context = """
최근 수집된 데이터:
- LANEIGE Lip Sleeping Mask: 5위
- 카테고리: Lip Care (Skin Care 하위)
- 경쟁 브랜드: CeraVe, Neutrogena 등
"""

    print(f"\n📝 질문: {query}")
    print(f"📄 컨텍스트: {context}")

    result = await agent.run(query, context)

    print(f"\n✅ 최종 답변: {result.final_answer}")
    print(f"🔁 반복 횟수: {result.iterations}")
    print(f"📊 신뢰도: {result.confidence:.2f}")
    print(f"⚠️  개선 필요: {result.needs_improvement}")

    print("\n📋 실행 단계:")
    for i, step in enumerate(result.steps, 1):
        print(f"\n  Step {i}:")
        print(f"    💭 Thought: {step.thought[:100]}...")
        if step.action:
            print(f"    🎬 Action: {step.action}")
            if step.action_input:
                print(f"    📥 Input: {json.dumps(step.action_input, ensure_ascii=False)}")
        if step.observation:
            obs = step.observation[:150]
            print(f"    👁️  Observation: {obs}...")


async def demo_reflection():
    """Self-Reflection 시연"""
    print("\n" + "=" * 80)
    print("🔹 Demo 3: Self-Reflection (품질 평가)")
    print("=" * 80)

    agent = ReActAgent(max_iterations=2)
    agent.set_tool_executor(DemoToolExecutor())

    query = "LANEIGE의 전략을 추천해줘"
    context = "제한된 컨텍스트"

    print(f"\n📝 질문: {query}")
    print(f"📄 컨텍스트: {context}")

    result = await agent.run(query, context)

    print(f"\n✅ 최종 답변: {result.final_answer}")
    print(f"📊 신뢰도 (Self-Reflection): {result.confidence:.2f}")

    if result.confidence < 0.7:
        print("⚠️  낮은 신뢰도 감지: 추가 정보가 필요합니다")

    if result.needs_improvement:
        print("⚠️  개선 필요: 응답 품질이 기준에 미달입니다")


async def main():
    """메인 실행 함수"""
    print("\n" + "=" * 80)
    print("🤖 ReAct Self-Reflection Agent Demo")
    print("=" * 80)

    try:
        await demo_simple_query()
        await asyncio.sleep(1)

        await demo_complex_query()
        await asyncio.sleep(1)

        await demo_reflection()

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)

    print("\n" + "=" * 80)
    print("✨ Demo 완료!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
