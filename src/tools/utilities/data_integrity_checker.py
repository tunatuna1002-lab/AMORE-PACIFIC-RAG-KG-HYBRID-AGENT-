"""
Data Integrity Checker
=======================
Google Sheets와 SQLite 간 데이터 정합성 검사

사용법:
    from src.tools.utilities.data_integrity_checker import DataIntegrityChecker

    checker = DataIntegrityChecker()
    await checker.initialize()

    # 동기화 상태 확인
    status = await checker.check_sync_status()
    print(f"Synced: {status['is_synced']}, Gap: {status['gap']}")

    # 누락 날짜 확인
    missing = await checker.get_missing_dates()
    print(f"Missing dates in SQLite: {missing}")
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# 기본 Spreadsheet ID
DEFAULT_SPREADSHEET_ID = os.environ.get(
    "GOOGLE_SHEETS_SPREADSHEET_ID", "1cNr3E2WSSbO83XXh_9V92jwc6nfsxjAogcswlHcjV9w"
)


class DataIntegrityChecker:
    """Google Sheets와 SQLite 간 데이터 정합성 검사"""

    def __init__(
        self, spreadsheet_id: str | None = None, sqlite_path: str = "./data/amore_data.db"
    ):
        """
        Args:
            spreadsheet_id: Google Sheets ID (None이면 환경변수 또는 기본값 사용)
            sqlite_path: SQLite 데이터베이스 경로
        """
        self.spreadsheet_id = spreadsheet_id or DEFAULT_SPREADSHEET_ID
        self.sqlite_path = sqlite_path
        self._sheets = None
        self._sqlite = None
        self._initialized = False

    async def initialize(self) -> bool:
        """초기화"""
        try:
            from src.tools.storage.sheets_writer import SheetsWriter
            from src.tools.storage.sqlite_storage import SQLiteStorage

            self._sheets = SheetsWriter(spreadsheet_id=self.spreadsheet_id)
            self._sqlite = SQLiteStorage(db_path=self.sqlite_path)

            sheets_ok = await self._sheets.initialize()
            sqlite_ok = await self._sqlite.initialize()

            self._initialized = sheets_ok and sqlite_ok

            if not self._initialized:
                logger.warning(f"Initialization incomplete: Sheets={sheets_ok}, SQLite={sqlite_ok}")

            return self._initialized

        except Exception as e:
            logger.error(f"Failed to initialize DataIntegrityChecker: {e}")
            return False

    async def check_sync_status(self, days: int = 30) -> dict[str, Any]:
        """
        Google Sheets와 SQLite 동기화 상태 확인

        Args:
            days: 검사할 기간 (일)

        Returns:
            {
                "sheets_count": int,
                "sqlite_count": int,
                "is_synced": bool,
                "gap": int,
                "sheets_date_range": {"min": str, "max": str},
                "sqlite_date_range": {"min": str, "max": str},
                "checked_at": str
            }
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Google Sheets 데이터 조회
            sheets_records = await self._sheets.get_rank_history(days=days)
            sheets_count = len(sheets_records) if sheets_records else 0

            sheets_dates = set()
            if sheets_records:
                sheets_dates = {
                    r.get("snapshot_date", "") for r in sheets_records if r.get("snapshot_date")
                }

            # SQLite 데이터 조회
            sqlite_stats = self._sqlite.get_stats()
            sqlite_count = sqlite_stats.get("raw_data_count", 0)
            sqlite_date_range = sqlite_stats.get("date_range", {})

            # 동기화 상태 판단 (레코드 수 차이 100 이하면 동기화됨)
            gap = sheets_count - sqlite_count
            is_synced = abs(gap) < 100

            return {
                "sheets_count": sheets_count,
                "sqlite_count": sqlite_count,
                "is_synced": is_synced,
                "gap": gap,
                "sheets_date_range": {
                    "min": min(sheets_dates) if sheets_dates else None,
                    "max": max(sheets_dates) if sheets_dates else None,
                },
                "sqlite_date_range": sqlite_date_range,
                "checked_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to check sync status: {e}")
            return {"error": str(e), "is_synced": False, "checked_at": datetime.now().isoformat()}

    async def get_missing_dates(self, days: int = 30) -> list[str]:
        """
        SQLite에 누락된 날짜 목록 조회

        Args:
            days: 검사할 기간 (일)

        Returns:
            누락된 날짜 목록 (YYYY-MM-DD 형식)
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Google Sheets 날짜 목록
            sheets_records = await self._sheets.get_rank_history(days=days)
            sheets_dates: set[str] = set()

            if sheets_records:
                sheets_dates = {
                    r.get("snapshot_date", "") for r in sheets_records if r.get("snapshot_date")
                }

            # SQLite 날짜 목록 (start_date 계산)
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            sqlite_raw = await self._sqlite.get_raw_data(start_date=start_date, limit=50000)
            sqlite_dates: set[str] = set()

            if sqlite_raw:
                sqlite_dates = {
                    r.get("snapshot_date", "") for r in sqlite_raw if r.get("snapshot_date")
                }

            # 차집합 (Sheets에는 있으나 SQLite에는 없는 날짜)
            missing = sheets_dates - sqlite_dates

            return sorted(missing)

        except Exception as e:
            logger.error(f"Failed to get missing dates: {e}")
            return []

    async def get_date_record_counts(self, days: int = 14) -> dict[str, dict[str, int]]:
        """
        날짜별 레코드 수 비교

        Args:
            days: 검사할 기간 (일)

        Returns:
            {
                "2026-01-25": {"sheets": 500, "sqlite": 500, "diff": 0},
                "2026-01-24": {"sheets": 500, "sqlite": 480, "diff": 20},
                ...
            }
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Google Sheets 데이터
            sheets_records = await self._sheets.get_rank_history(days=days)
            sheets_by_date: dict[str, int] = {}

            if sheets_records:
                for r in sheets_records:
                    date = r.get("snapshot_date", "")
                    if date:
                        sheets_by_date[date] = sheets_by_date.get(date, 0) + 1

            # SQLite 데이터 (start_date 계산)
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            sqlite_raw = await self._sqlite.get_raw_data(start_date=start_date, limit=50000)
            sqlite_by_date: dict[str, int] = {}

            if sqlite_raw:
                for r in sqlite_raw:
                    date = r.get("snapshot_date", "")
                    if date:
                        sqlite_by_date[date] = sqlite_by_date.get(date, 0) + 1

            # 결과 병합
            all_dates = set(sheets_by_date.keys()) | set(sqlite_by_date.keys())
            result = {}

            for date in sorted(all_dates, reverse=True):
                sheets_cnt = sheets_by_date.get(date, 0)
                sqlite_cnt = sqlite_by_date.get(date, 0)
                result[date] = {
                    "sheets": sheets_cnt,
                    "sqlite": sqlite_cnt,
                    "diff": sheets_cnt - sqlite_cnt,
                }

            return result

        except Exception as e:
            logger.error(f"Failed to get date record counts: {e}")
            return {}

    async def run_full_check(self) -> dict[str, Any]:
        """
        전체 정합성 검사 실행

        Returns:
            종합 검사 결과
        """
        if not self._initialized:
            await self.initialize()

        sync_status = await self.check_sync_status(days=30)
        missing_dates = await self.get_missing_dates(days=14)
        date_counts = await self.get_date_record_counts(days=7)

        # 심각도 판단
        severity = "OK"
        if len(missing_dates) > 0:
            severity = "WARNING"
        if len(missing_dates) > 3 or abs(sync_status.get("gap", 0)) > 500:
            severity = "CRITICAL"

        return {
            "severity": severity,
            "sync_status": sync_status,
            "missing_dates": missing_dates,
            "date_counts": date_counts,
            "recommendations": self._generate_recommendations(sync_status, missing_dates),
            "checked_at": datetime.now().isoformat(),
        }

    def _generate_recommendations(
        self, sync_status: dict[str, Any], missing_dates: list[str]
    ) -> list[str]:
        """개선 권고사항 생성"""
        recommendations = []

        gap = sync_status.get("gap", 0)

        if gap > 100:
            recommendations.append(
                f"SQLite에 {gap}개 레코드 누락. `python scripts/sync_sheets_to_sqlite.py` 실행 권장"
            )

        if len(missing_dates) > 0:
            recommendations.append(
                f"SQLite에 {len(missing_dates)}일 데이터 누락: "
                f"{', '.join(missing_dates[:5])}{'...' if len(missing_dates) > 5 else ''}"
            )

        if not sync_status.get("is_synced", True):
            recommendations.append("Google Sheets → SQLite 자동 동기화 설정 확인 필요")

        if not recommendations:
            recommendations.append("데이터 정합성 정상")

        return recommendations


# 편의 함수
async def check_data_integrity() -> dict[str, Any]:
    """데이터 정합성 검사 실행 (편의 함수)"""
    checker = DataIntegrityChecker()
    await checker.initialize()
    return await checker.run_full_check()


if __name__ == "__main__":
    # CLI 실행
    async def main():
        print("=" * 60)
        print("Data Integrity Check")
        print("=" * 60)

        result = await check_data_integrity()

        print(f"\nSeverity: {result['severity']}")
        print("\nSync Status:")
        status = result["sync_status"]
        print(f"  - Sheets: {status.get('sheets_count', 0)} records")
        print(f"  - SQLite: {status.get('sqlite_count', 0)} records")
        print(f"  - Gap: {status.get('gap', 0)}")
        print(f"  - Synced: {status.get('is_synced', False)}")

        if result["missing_dates"]:
            print(f"\nMissing dates in SQLite: {result['missing_dates']}")

        print("\nRecommendations:")
        for rec in result["recommendations"]:
            print(f"  - {rec}")

    asyncio.run(main())
