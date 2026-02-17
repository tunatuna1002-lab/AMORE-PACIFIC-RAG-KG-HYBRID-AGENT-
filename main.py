"""
AMORE RAG-Ontology Hybrid Agent System
메인 진입점

일일 Amazon 베스트셀러 크롤링 및 LANEIGE 분석 시스템
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from src.core.brain import get_brain  # 챗봇용
from src.core.orchestrator import Orchestrator  # 워크플로우용
from src.monitoring.logger import AgentLogger

# 환경 변수 로드
load_dotenv()


async def run_daily_workflow(
    categories: list | None = None, spreadsheet_id: str | None = None
) -> dict:
    """
    일일 워크플로우 실행

    Args:
        categories: 크롤링할 카테고리 (None이면 전체)
        spreadsheet_id: Google Sheets ID

    Returns:
        실행 결과
    """
    logger = AgentLogger("main")
    logger.info("=" * 50)
    logger.info("AMORE RAG-Ontology Hybrid Agent")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    # 환경 변수 확인
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return {"status": "failed", "error": "OPENAI_API_KEY not configured"}

    # Spreadsheet ID
    sheet_id = spreadsheet_id or os.getenv("GOOGLE_SPREADSHEET_ID")

    # 오케스트레이터 초기화
    orchestrator = Orchestrator(
        config_path="./config/thresholds.json", spreadsheet_id=sheet_id, model="gpt-4.1-mini"
    )

    try:
        # 워크플로우 실행
        result = await orchestrator.run_daily_workflow(categories)

        # 결과 출력
        logger.info("=" * 50)
        logger.info("Workflow Complete")
        logger.info(f"Status: {result.get('status')}")

        summary = result.get("summary", {})
        logger.info(f"Products crawled: {summary.get('products_crawled', 0)}")
        logger.info(f"LANEIGE tracked: {summary.get('laneige_tracked', 0)}")
        logger.info(f"Alerts: {summary.get('alerts', 0)}")
        logger.info(f"Action items: {summary.get('action_items', 0)}")

        if result.get("status") == "completed":
            logger.info("\n📊 Daily Insight Preview:")
            insight = summary.get("daily_insight", "")
            if insight:
                logger.info(insight[:500])

        return result

    except KeyboardInterrupt:
        logger.warning("Workflow interrupted by user")
        return {"status": "interrupted"}

    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}

    finally:
        await orchestrator.cleanup()


async def run_chatbot(spreadsheet_id: str | None = None) -> None:
    """
    챗봇 인터랙티브 모드 (UnifiedBrain 사용)

    Args:
        spreadsheet_id: Google Sheets ID (데이터 로드용)
    """
    import json

    logger = AgentLogger("chatbot")
    logger.info("=" * 50)
    logger.info("LANEIGE Amazon Insight Chatbot")
    logger.info("Type 'exit' to quit, 'help' for commands")
    logger.info("=" * 50)

    # UnifiedBrain 사용
    brain = get_brain()

    # 현재 데이터 로드
    current_metrics = None
    try:
        with open("./data/dashboard_data.json", encoding="utf-8") as f:
            current_metrics = json.load(f)
    except FileNotFoundError:
        logger.warning("Dashboard data not found, starting without data context")

    print("\n💬 Chatbot ready. Ask me about LANEIGE Amazon performance!\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                if user_input.lower() == "exit":
                    print("Goodbye!")
                    break

                if user_input.lower() == "help":
                    print_help()
                    continue

                if user_input.lower() == "status":
                    stats = brain.get_stats() if hasattr(brain, "get_stats") else {}
                    print(f"\n📊 Status: {stats}\n")
                    continue

                if user_input.lower() == "errors":
                    errors = (
                        brain.get_recent_errors(limit=5)
                        if hasattr(brain, "get_recent_errors")
                        else []
                    )
                    if errors:
                        print("\n⚠️ Recent Errors:")
                        for err in errors:
                            print(
                                f"   - [{err.get('agent', 'unknown')}] {err.get('message', 'unknown error')}"
                            )
                    else:
                        print("\n✅ No recent errors")
                    print()
                    continue

                # UnifiedBrain으로 응답 생성
                response = await brain.process_query(
                    query=user_input, current_metrics=current_metrics
                )

                # 응답 출력
                response_dict = response.to_dict() if hasattr(response, "to_dict") else response
                print(
                    f"\n🤖 Assistant: {response_dict.get('text', response_dict.get('content', 'No response'))}"
                )

                # 도구 호출 정보
                tools_called = response_dict.get(
                    "tools_called", response_dict.get("tools_used", [])
                )
                if tools_called:
                    print(f"   [도구 사용: {', '.join(tools_called)}]")

                # 후속 질문 제안
                suggestions = response_dict.get("suggestions", [])
                if suggestions:
                    print("\n💡 Related questions:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"   {i}. {suggestion}")

                print()

            except EOFError:
                print("\nGoodbye!")
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")


def print_help():
    """도움말 출력"""
    help_text = """
Available Commands:
  exit    - Exit the chatbot
  help    - Show this help message
  status  - Show orchestrator stats
  errors  - Show recent errors

Example Questions:
  - SoS란 무엇인가요?
  - 오늘 LANEIGE 제품 순위는 어떤가요?
  - Lip Care 카테고리에서 LANEIGE 포지션은?
  - 순위가 하락한 제품이 있나요?
  - HHI 지수가 높으면 어떤 의미인가요?
"""
    print(help_text)


async def run_single_category(category: str) -> dict:
    """단일 카테고리 크롤링"""
    return await run_daily_workflow(categories=[category])


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="AMORE RAG-Ontology Hybrid Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full daily workflow
  python main.py

  # Run specific categories only
  python main.py --categories lip_care face_moisturizer

  # Start interactive chatbot
  python main.py --chat

  # Specify Google Sheets ID
  python main.py --spreadsheet-id YOUR_SPREADSHEET_ID
        """,
    )

    parser.add_argument("--chat", action="store_true", help="Start interactive chatbot mode")

    parser.add_argument(
        "--categories", nargs="+", help="Specific categories to crawl (default: all)"
    )

    parser.add_argument("--spreadsheet-id", type=str, help="Google Sheets spreadsheet ID")

    parser.add_argument(
        "--dry-run", action="store_true", help="Run without saving to Google Sheets"
    )

    args = parser.parse_args()

    # 드라이런 모드
    if args.dry_run:
        os.environ["DRY_RUN"] = "true"

    # 실행
    if args.chat:
        asyncio.run(run_chatbot(args.spreadsheet_id))
    else:
        result = asyncio.run(
            run_daily_workflow(categories=args.categories, spreadsheet_id=args.spreadsheet_id)
        )

        # 종료 코드 설정
        if result.get("status") == "completed":
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
