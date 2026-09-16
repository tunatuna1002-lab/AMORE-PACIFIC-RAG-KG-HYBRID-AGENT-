"""
Dashboard Data Service
======================
The ONE place that knows where dashboard data lives and how to load it (F6-1).

Resolution order for the data directory (``resolve_data_dir``):
    1. ``DATA_DIR`` environment variable
    2. ``/data`` when running on Railway (``RAILWAY_ENVIRONMENT`` set) or the volume exists
    3. ``./data`` (local development, relative to CWD)

Resolution order for dashboard data (``get_dashboard_data``):
    JSON cache (``dashboard_data.json``) → SQLite fallback → empty skeleton.

Every loader annotates ``metadata`` with ``_cache_age_hours`` / ``_is_stale`` /
``_source`` so consumers can tell how fresh the payload is.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from src.domain.brand import TARGET_BRAND, is_target_brand

logger = logging.getLogger(__name__)

DASHBOARD_JSON = "dashboard_data.json"
LATEST_CRAWL_JSON = "latest_crawl_result.json"
STALE_AFTER_HOURS = 24.0

EMPTY_DASHBOARD_MESSAGE = "데이터가 없습니다. 크롤링을 실행하여 데이터를 수집하세요."


def resolve_data_dir() -> Path:
    """Data directory: env ``DATA_DIR`` > ``/data`` (Railway / volume present) > ``./data``."""
    env_dir = os.environ.get("DATA_DIR")
    if env_dir:
        return Path(env_dir)
    if os.environ.get("RAILWAY_ENVIRONMENT") or Path("/data").exists():
        return Path("/data")
    return Path("./data")


def empty_dashboard_skeleton() -> dict[str, Any]:
    """Shape the frontend can render when nothing has been crawled yet."""
    return {
        "metadata": {
            "data_date": None,
            "total_products": 0,
            "_is_stale": False,
            "_is_empty": True,
            "_message": EMPTY_DASHBOARD_MESSAGE,
        },
        "home": {"action_items": [], "status": {}, "summary": {}},
        "brand": {"kpis": {}, "competitors": []},
        "products": {},
        "categories": {},
        "charts": {},
    }


class DashboardDataService:
    """
    Loads dashboard data from the resolved data directory.

    ``data_dir=None`` means "resolve on every access" (env/CWD may change between
    calls - tests chdir, Railway sets env at boot), which keeps the service safe to
    hold as a module-level singleton.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        stale_after_hours: float = STALE_AFTER_HOURS,
        sqlite_factory: Callable[[], Any] | None = None,
    ):
        self._data_dir = Path(data_dir) if data_dir else None
        self.stale_after_hours = stale_after_hours
        self._sqlite_factory = sqlite_factory

    # ------------------------------------------------------------------ paths
    @property
    def data_dir(self) -> Path:
        return self._data_dir if self._data_dir is not None else resolve_data_dir()

    @property
    def dashboard_json_path(self) -> Path:
        return self.data_dir / DASHBOARD_JSON

    @property
    def latest_crawl_json_path(self) -> Path:
        return self.data_dir / LATEST_CRAWL_JSON

    def path_for(self, *parts: str) -> Path:
        """A path inside the data directory (``service.path_for("exports", name)``)."""
        return self.data_dir.joinpath(*parts)

    # ------------------------------------------------------------------ JSON
    def load_dashboard_json(self) -> dict[str, Any]:
        """``dashboard_data.json`` with staleness metadata; ``{}`` when missing/corrupt."""
        path = self.dashboard_json_path
        if not path.exists():
            logger.warning(f"Dashboard data file not found: {path} (data_dir={self.data_dir})")
            return {}

        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > self.stale_after_hours:
            logger.warning(
                f"Dashboard data is stale: {age_hours:.1f} hours old. "
                f"Consider running a crawl or calling /api/data/refresh."
            )

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted dashboard data file: {e}")
            return {}
        if not isinstance(data, dict):
            logger.warning(f"Dashboard data file is not a JSON object: {path}")
            return {}

        metadata = data.setdefault("metadata", {})
        metadata["_cache_age_hours"] = round(age_hours, 1)
        metadata["_is_stale"] = age_hours > self.stale_after_hours
        metadata["_source"] = "json"
        return data

    def load_latest_crawl_json(self) -> dict[str, Any] | None:
        """``latest_crawl_result.json`` (raw crawl snapshot) or None."""
        path = self.latest_crawl_json_path
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted crawl result file {path}: {e}")
            return None
        return data if isinstance(data, dict) else None

    # ---------------------------------------------------------------- SQLite
    def _sqlite(self) -> Any:
        if self._sqlite_factory is not None:
            return self._sqlite_factory()
        from src.tools.storage.sqlite_storage import get_sqlite_storage

        return get_sqlite_storage()

    async def from_sqlite_fallback(self) -> dict[str, Any] | None:
        """
        Minimal dashboard payload built from the latest SQLite snapshot.

        Same top-level shape as the exporter JSON (products keyed by ASIN) so every
        consumer can treat both sources alike. None when SQLite has no rows or fails.
        """
        try:
            sqlite = self._sqlite()
            await sqlite.initialize()
            records = await sqlite.get_latest_data()
        except Exception as e:
            logger.error(f"SQLite fallback generation failed: {e}")
            return None
        if not records:
            return None
        try:
            return self._build_from_records(records)
        except Exception as e:
            logger.error(f"SQLite fallback generation failed: {e}")
            return None

    @staticmethod
    def _rank_of(record: dict[str, Any], default: int = 0) -> int:
        return int(record.get("rank", default)) if record.get("rank") else default

    def _build_from_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        latest_date = records[0].get("snapshot_date", "") if records else ""
        target_products = [
            r
            for r in records
            if is_target_brand(r.get("brand")) or is_target_brand(r.get("product_name"))
        ]
        total = len(records)

        action_items = []
        for p in target_products[:8]:
            rank = self._rank_of(p)
            action_items.append(
                {
                    "asin": p.get("asin", ""),
                    "product_name": p.get("product_name", "Unknown"),
                    "brand_variant": TARGET_BRAND,
                    "rank": rank,
                    "rank_change": 0,
                    "signal": f"순위 #{rank}",
                    "signal_detail": "",
                    "action_tag": "Monitor" if rank <= 10 else "Watch",
                    "priority": "P1" if rank <= 5 else ("P2" if rank <= 20 else "P3"),
                }
            )

        brand_counts: dict[str, dict[str, Any]] = {}
        for r in records:
            b = r.get("brand", "Unknown")
            entry = brand_counts.setdefault(b, {"ranks": [], "count": 0})
            entry["ranks"].append(self._rank_of(r))
            entry["count"] += 1

        competitors = []
        for b_name, b_data in sorted(
            brand_counts.items(), key=lambda x: x[1]["count"], reverse=True
        )[:15]:
            ranks = b_data["ranks"]
            competitors.append(
                {
                    "brand": b_name,
                    "sos": round(b_data["count"] / max(total, 100) * 100, 2),
                    "avg_rank": round(sum(ranks) / len(ranks), 1) if ranks else 0,
                    "product_count": b_data["count"],
                }
            )

        target_ranks: list[int] = []
        for b_name, b_data in brand_counts.items():
            if is_target_brand(b_name):
                target_ranks = b_data["ranks"]
                break

        best_rank = min((self._rank_of(p, 100) for p in target_products), default=None)
        now_iso = datetime.now().isoformat()

        return {
            "metadata": {
                "generated_at": now_iso,
                "data_date": latest_date,
                "total_products": total,
                "laneige_products": len(target_products),
                "_source": "sqlite_fallback",
                "_cache_age_hours": 0,
                "_is_stale": False,
            },
            "data_source": {
                "platform": "Amazon US Best Sellers",
                "collected_at": now_iso,
                "snapshot_date": latest_date,
                "disclaimer": "SQLite 최신 스냅샷에서 실시간 생성 (JSON 캐시 없음)",
                "url": "https://www.amazon.com/gp/bestsellers/beauty",
            },
            "home": {
                "insight_message": (
                    f"SQLite 데이터 기준 ({latest_date}). JSON 캐시가 없어 실시간 생성되었습니다."
                ),
                "status": {
                    "exposure": "N/A",
                    "position": f"Top {best_rank}" if best_rank is not None else "N/A",
                    "warning_count": 0,
                },
                "action_items": action_items,
            },
            "brand": {
                "kpis": {
                    "sos": round(len(target_products) / max(total, 100) * 100, 2),
                    "top10_count": sum(1 for r in target_products if self._rank_of(r, 100) <= 10),
                    "avg_rank": round(sum(target_ranks) / len(target_ranks), 1)
                    if target_ranks
                    else 0,
                    "hhi": "N/A",
                },
                "competitors": competitors,
            },
            "categories": {},
            "products": {
                p.get("asin", ""): {
                    "name": p.get("product_name", "Unknown"),
                    "rank": self._rank_of(p),
                    "rank_delta": "N/A",
                    "rating": p.get("rating", 0),
                    "rating_delta": "N/A",
                    "volatility": 0,
                    "volatility_status": "N/A",
                    "category": p.get("category_id", ""),
                }
                for p in target_products
                if p.get("asin")
            },
            "charts": {},
        }

    # ------------------------------------------------------------- composite
    async def get_dashboard_data(self) -> dict[str, Any]:
        """JSON cache → SQLite fallback → empty skeleton (never raises, never empty)."""
        data = self.load_dashboard_json()
        if data:
            return data

        logger.warning("Dashboard JSON cache missing, attempting SQLite fallback")
        fallback = await self.from_sqlite_fallback()
        if fallback:
            return fallback

        logger.warning("No dashboard data available, returning empty dashboard structure")
        return empty_dashboard_skeleton()


_default_service: DashboardDataService | None = None


def get_dashboard_data_service() -> DashboardDataService:
    """Process-wide default service (resolves the data dir on every access)."""
    global _default_service
    if _default_service is None:
        _default_service = DashboardDataService()
    return _default_service


def load_dashboard_data() -> dict[str, Any]:
    """Convenience: ``get_dashboard_data_service().load_dashboard_json()``."""
    return get_dashboard_data_service().load_dashboard_json()
