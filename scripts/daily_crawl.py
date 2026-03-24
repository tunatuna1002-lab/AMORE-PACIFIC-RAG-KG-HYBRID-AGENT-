#!/usr/bin/env python3
"""
Daily Crawl Script (Standalone)
================================
launchd에서 매일 실행되는 독립형 크롤링 스크립트.
FastAPI 서버 없이 직접 크롤링 → 저장 → 대시보드 갱신을 수행합니다.

Usage:
    # 전체 파이프라인
    python3 scripts/daily_crawl.py

    # 크롤링만 (저장/내보내기 건너뛰기)
    python3 scripts/daily_crawl.py --crawl-only

    # 드라이런 (실제 크롤링 없이 파이프라인 테스트)
    python3 scripts/daily_crawl.py --dry-run
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# .env 로드
from dotenv import load_dotenv

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


# ── 크롤링 파이프라인 ───────────────────────────────────────
async def run_pipeline(crawl_only: bool = False, dry_run: bool = False) -> dict:
    """
    전체 크롤링 파이프라인 실행

    Steps:
        1. Amazon BSR 크롤링 (5개 카테고리 × Top 100)
        2. SQLite 저장
        3. Google Sheets 저장
        4. Dashboard JSON 생성

    Returns:
        실행 결과 dict
    """
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

    # ── STEP 0: 환경 확인 ──
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.warning("OPENAI_API_KEY not set — insight 생성 건너뜀")

    # ── STEP 1: 크롤링 ──
    logger.info("[1/4] Amazon BSR 크롤링 시작...")

    if dry_run:
        logger.info("[DRY RUN] 크롤링 건너뜀")
        crawl_result = _make_dry_run_result(snapshot_date)
    else:
        try:
            from src.infrastructure.container import Container

            crawler = Container.get_crawler_agent()
            await crawler.scraper.initialize()

            try:
                crawl_result = await crawler.execute()
            finally:
                await crawler.scraper.close()

            if crawl_result.get("status") == "failed":
                raise RuntimeError(f"Crawl failed: {crawl_result.get('errors', 'unknown')}")

            result["products_collected"] = crawl_result.get("total_products", 0)
            result["categories_done"] = len(crawl_result.get("categories", {}))

            logger.info(
                f"[1/4] 완료: {result['products_collected']}개 제품, "
                f"{result['categories_done']}개 카테고리"
            )

        except Exception as e:
            logger.error(f"[1/4] 크롤링 실패: {e}", exc_info=True)
            result["errors"].append(f"crawl: {e}")
            result["status"] = "failed"
            return result

    # 크롤링 원본 JSON 저장 (항상)
    _save_crawl_json(crawl_result, snapshot_date)

    if crawl_only:
        result["status"] = "completed"
        result["mode"] = "crawl_only"
        return result

    # ── STEP 2: SQLite 저장 ──
    logger.info("[2/4] SQLite 저장 시작...")
    try:
        sqlite_count = await _save_to_sqlite(crawl_result, snapshot_date)
        logger.info(f"[2/4] SQLite 저장 완료: {sqlite_count}건")
    except Exception as e:
        logger.error(f"[2/4] SQLite 저장 실패: {e}", exc_info=True)
        result["errors"].append(f"sqlite: {e}")

    # ── STEP 3: Google Sheets 저장 ──
    logger.info("[3/4] Google Sheets 저장 시작...")
    try:
        sheets_count = await _save_to_sheets(crawl_result)
        logger.info(f"[3/4] Google Sheets 저장 완료: {sheets_count}건")
    except Exception as e:
        logger.warning(f"[3/4] Google Sheets 저장 실패 (non-fatal): {e}")
        result["errors"].append(f"sheets: {e}")

    # ── STEP 4: Dashboard 데이터 생성 ──
    logger.info("[4/4] Dashboard 데이터 생성 시작...")
    try:
        await _export_dashboard()
        logger.info("[4/4] Dashboard 데이터 생성 완료")
    except Exception as e:
        logger.warning(f"[4/4] Dashboard 생성 실패 (non-fatal): {e}")
        result["errors"].append(f"dashboard: {e}")

    # ── 완료 ──
    elapsed = time.time() - start_time
    result["status"] = "completed" if not result["errors"] else "completed_with_warnings"
    result["completed_at"] = datetime.now(KST).isoformat()
    result["elapsed_seconds"] = round(elapsed, 1)

    logger.info("=" * 60)
    logger.info(f"  완료! ({elapsed:.0f}초 소요)")
    logger.info(f"  제품: {result['products_collected']}개")
    logger.info(f"  카테고리: {result['categories_done']}개")
    if result["errors"]:
        logger.warning(f"  경고: {len(result['errors'])}건")
    logger.info("=" * 60)

    return result


# ── 헬퍼 함수들 ─────────────────────────────────────────────


def _save_crawl_json(crawl_result: dict, snapshot_date: str):
    """크롤링 결과를 JSON 파일로 저장"""
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    def json_serializer(obj):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if hasattr(obj, "__str__"):
            return str(obj)
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    # latest_crawl_result.json
    try:
        with open(data_dir / "latest_crawl_result.json", "w", encoding="utf-8") as f:
            json.dump(crawl_result, f, ensure_ascii=False, indent=2, default=json_serializer)
    except Exception as e:
        logger.error(f"latest_crawl_result.json 저장 실패: {e}")

    # 날짜별 히스토리
    try:
        raw_dir = data_dir / "raw_products"
        raw_dir.mkdir(parents=True, exist_ok=True)

        all_products = []
        for cat_id, cat_data in crawl_result.get("categories", {}).items():
            for product in cat_data.get("products", []):
                product["category_id"] = cat_id
                all_products.append(product)

        with open(raw_dir / f"{snapshot_date}.json", "w", encoding="utf-8") as f:
            json.dump(all_products, f, ensure_ascii=False, indent=2, default=json_serializer)

        logger.info(
            f"히스토리 저장: {len(all_products)}개 제품 → raw_products/{snapshot_date}.json"
        )
    except Exception as e:
        logger.error(f"히스토리 저장 실패: {e}")


async def _save_to_sqlite(crawl_result: dict, snapshot_date: str) -> int:
    """크롤링 결과를 SQLite에 저장 (append_rank_records 사용)"""
    from src.tools.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage()
    await storage.initialize()

    try:
        # 전체 카테고리의 제품을 레코드 리스트로 변환
        records = []
        for cat_id, cat_data in crawl_result.get("categories", {}).items():
            for product in cat_data.get("products", []):
                records.append(
                    {
                        "snapshot_date": snapshot_date,
                        "category_id": cat_id,
                        "rank": product.get("rank", 0),
                        "asin": product.get("asin", ""),
                        "product_name": product.get("title", product.get("product_name", "")),
                        "brand": product.get("brand", "Unknown"),
                        "price": product.get("price"),
                        "list_price": product.get("list_price"),
                        "discount_percent": product.get("discount_percent"),
                        "rating": product.get("rating"),
                        "reviews_count": product.get("reviews_count"),
                        "badge": product.get("badge"),
                        "coupon_text": product.get("coupon_text"),
                        "is_subscribe_save": product.get("is_subscribe_save", False),
                        "promo_badges": product.get("promo_badges", []),
                        "product_url": product.get("product_url", ""),
                    }
                )

        result = await storage.append_rank_records(records)

        if not result.get("success"):
            raise RuntimeError(f"SQLite 저장 실패: {result.get('error')}")

        return result.get("rows_added", 0)
    finally:
        if hasattr(storage, "close"):
            await storage.close()


async def _save_to_sheets(crawl_result: dict) -> int:
    """크롤링 결과를 Google Sheets에 저장"""
    from src.infrastructure.container import Container

    storage = Container.get_storage_agent()
    storage_result = await storage.execute(crawl_result)

    if storage_result.get("errors"):
        logger.warning(f"Sheets 저장 경고: {storage_result['errors']}")

    return storage_result.get("raw_records", 0)


async def _export_dashboard():
    """대시보드 JSON 데이터 생성"""
    from src.tools.exporters.dashboard_exporter import DashboardExporter

    data_file = str(PROJECT_ROOT / "data" / "dashboard_data.json")
    exporter = DashboardExporter()
    await exporter.initialize()
    await exporter.export_dashboard_data(data_file)


def _make_dry_run_result(snapshot_date: str) -> dict:
    """드라이런용 가짜 크롤링 결과"""
    return {
        "status": "success",
        "snapshot_date": snapshot_date,
        "total_products": 0,
        "categories": {},
        "laneige_products": [],
        "errors": [],
    }


# ── 엔트리포인트 ─────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AMORE Daily Crawl (Standalone)")
    parser.add_argument(
        "--crawl-only", action="store_true", help="크롤링만 수행 (저장/내보내기 건너뛰기)"
    )
    parser.add_argument("--dry-run", action="store_true", help="실제 크롤링 없이 파이프라인 테스트")
    args = parser.parse_args()

    # 파이프라인 실행
    result = asyncio.run(run_pipeline(crawl_only=args.crawl_only, dry_run=args.dry_run))

    # macOS 알림
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
            f"오류: {errors[0] if errors else 'unknown'}",
            sound="Basso",
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
