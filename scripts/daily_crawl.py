#!/usr/bin/env python3
"""
Daily Crawl Script (Standalone)
================================
launchd에서 매일 실행되는 독립형 크롤링 스크립트.
FastAPI 서버 없이 **단일 배치 파이프라인**(``BatchWorkflow.run_daily_workflow``)을
그대로 실행합니다: crawl → store → kg → metrics → insight → alert → export.
(스케줄러/CrawlManager와 동일한 진입점 — 별도의 크롤링/저장 복사본은 없습니다.)

Usage:
    # 전체 파이프라인
    python3 scripts/daily_crawl.py

    # 크롤링만 (저장/내보내기 건너뛰기)
    python3 scripts/daily_crawl.py --crawl-only

    # 드라이런 (실제 크롤링 없이 파이프라인 테스트)
    python3 scripts/daily_crawl.py --dry-run
"""

import asyncio
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# 프로젝트 루트를 sys.path에 추가
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# .env 로드
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# 로깅 설정
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"daily_crawl_{datetime.now().strftime('%Y-%m-%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("daily_crawl")


# ── macOS 알림 ──────────────────────────────────────────────
def notify_macos(title: str, message: str, sound: str = "Glass"):
    """macOS 데스크탑 알림 전송"""
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message}" with title "{title}" sound name "{sound}"',
            ],
            timeout=5,
            check=False,
        )
    except Exception:
        pass  # 알림 실패는 무시


# ── 드라이런용 크롤러 ─────────────────────────────────────────
class DryRunCrawler:
    """실제 크롤링 없이 빈 결과를 반환하는 CrawlerAgent 대체물"""

    def __init__(self, snapshot_date: str):
        self.snapshot_date = snapshot_date

    async def execute(self, categories: list[str] | None = None) -> dict[str, Any]:
        logger.info("[DRY RUN] 크롤링 건너뜀")
        return {
            "status": "completed",
            "snapshot_date": self.snapshot_date,
            "total_products": 0,
            "laneige_count": 0,
            "categories": {},
            "laneige_products": [],
            "errors": [],
        }

    async def close(self) -> None:
        return None


# ── 크롤링 파이프라인 ───────────────────────────────────────
async def run_pipeline(crawl_only: bool = False, dry_run: bool = False) -> dict:
    """
    전체 파이프라인 실행 (BatchWorkflow.run_daily_workflow 위임)

    Returns:
        실행 결과 dict (status / products_collected / categories_done / errors ...)
    """
    from src.application.workflows.batch_workflow import WorkflowDependencies
    from src.infrastructure.container import Container
    from src.shared.constants import KST

    start_time = time.time()
    kst_now = datetime.now(KST)
    snapshot_date = kst_now.strftime("%Y-%m-%d")

    result = {
        "status": "started",
        "snapshot_date": snapshot_date,
        "started_at": kst_now.isoformat(),
        "products_collected": 0,
        "categories_done": 0,
        "errors": [],
    }

    logger.info("=" * 60)
    logger.info("  AMORE Daily Crawl (Standalone)")
    logger.info(f"  Date: {snapshot_date} (KST)")
    logger.info(f"  Mode: {'DRY RUN' if dry_run else 'CRAWL ONLY' if crawl_only else 'FULL'}")
    logger.info("=" * 60)

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set — insight 생성은 실패할 수 있음 (partial)")

    deps = WorkflowDependencies(crawler=DryRunCrawler(snapshot_date)) if dry_run else None
    workflow = Container.get_batch_workflow(
        config_path=str(PROJECT_ROOT / "config" / "thresholds.json"),
        spreadsheet_id=os.getenv("GOOGLE_SPREADSHEET_ID"),
        deps=deps,
        data_dir=str(PROJECT_ROOT / "data"),
    )

    def on_step(step: str, payload: dict[str, Any]) -> None:
        status = payload.get("status")
        if status == "completed":
            logger.info(f"[{step}] 완료")
        else:
            logger.warning(f"[{step}] 실패: {payload.get('error')}")

    try:
        wf_result = await workflow.run_daily_workflow(
            crawl_only=crawl_only, progress_callback=on_step
        )
    except Exception as e:
        logger.error(f"파이프라인 실패: {e}", exc_info=True)
        result["errors"].append(f"workflow: {e}")
        result["status"] = "failed"
        return result
    finally:
        try:
            await workflow.cleanup()
        except Exception as e:
            logger.warning(f"cleanup 실패 (무시): {e}")

    crawl_result = ((wf_result.get("steps") or {}).get("crawl") or {}).get("result") or {}
    result["products_collected"] = crawl_result.get("total_products", 0)
    result["categories_done"] = len(crawl_result.get("categories", {}))
    result["errors"] = list(wf_result.get("errors", []))
    result["summary"] = wf_result.get("summary", {})

    wf_status = wf_result.get("status")
    if wf_status == "completed":
        result["status"] = "completed"
    elif wf_status == "partial":
        result["status"] = "completed_with_warnings"
    else:
        result["status"] = "failed"
        if wf_result.get("error"):
            result["error"] = wf_result["error"]
    if crawl_only:
        result["mode"] = "crawl_only"

    elapsed = time.time() - start_time
    result["completed_at"] = datetime.now(KST).isoformat()
    result["elapsed_seconds"] = round(elapsed, 1)

    logger.info("=" * 60)
    logger.info(f"  {'완료!' if result['status'] != 'failed' else '실패'} ({elapsed:.0f}초 소요)")
    logger.info(f"  제품: {result['products_collected']}개")
    logger.info(f"  카테고리: {result['categories_done']}개")
    if result["errors"]:
        logger.warning(f"  경고/오류: {len(result['errors'])}건 — {result['errors']}")
    logger.info("=" * 60)

    return result


# ── 엔트리포인트 ─────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AMORE Daily Crawl (Standalone)")
    parser.add_argument(
        "--crawl-only", action="store_true", help="크롤링만 수행 (저장/내보내기 건너뛰기)"
    )
    parser.add_argument("--dry-run", action="store_true", help="실제 크롤링 없이 파이프라인 테스트")
    args = parser.parse_args()

    result = asyncio.run(run_pipeline(crawl_only=args.crawl_only, dry_run=args.dry_run))

    status = result.get("status", "unknown")
    products = result.get("products_collected", 0)

    if status in ("completed", "completed_with_warnings"):
        notify_macos(
            "AMORE 크롤링 완료 ✅",
            f"{products}개 제품 수집 완료 ({result.get('elapsed_seconds', 0)}초)",
        )
        sys.exit(0)
    else:
        errors = result.get("errors", [])
        notify_macos(
            "AMORE 크롤링 실패 ❌",
            f"오류: {errors[0] if errors else result.get('error', 'unknown')}",
            sound="Basso",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
