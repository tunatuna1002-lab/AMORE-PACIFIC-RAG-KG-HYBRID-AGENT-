"""
Storage Protocol
================
SQLiteStorage에 대한 추상 인터페이스

구현체:
- SQLiteStorage (src/tools/sqlite_storage.py)
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StorageProtocol(Protocol):
    """
    Storage Protocol

    SQLite 기반 데이터 저장소 인터페이스.
    제품 데이터, 메트릭, 경쟁사 정보, 딜 정보 등을 저장하고 조회합니다.

    Methods:
        initialize: 데이터베이스 초기화
        append_rank_records: 순위 레코드 추가
        get_raw_data: 원본 데이터 조회
        get_latest_data: 최신 데이터 조회
        get_historical_data: 히스토리 데이터 조회
        save_brand_metrics: 브랜드 메트릭 저장
        save_market_metrics: 시장 메트릭 저장
        save_competitor_products: 경쟁사 제품 저장
        get_competitor_products: 경쟁사 제품 조회
        get_data_date: 데이터 날짜 조회
        get_stats: 통계 조회
        export_to_excel: 엑셀 내보내기
    """

    async def initialize(self) -> bool:
        """
        데이터베이스를 초기화합니다.

        테이블 생성, 인덱스 생성 등을 수행합니다.

        Returns:
            초기화 성공 여부
        """
        ...

    async def append_rank_records(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """
        순위 레코드를 추가합니다.

        중복 데이터는 자동으로 무시됩니다.

        Args:
            records: 순위 레코드 리스트
                [
                    {
                        "snapshot_date": "2026-01-27",
                        "category_id": "lip_care",
                        "rank": 1,
                        "asin": "B001GXRQW0",
                        "product_name": "...",
                        "brand": "LANEIGE",
                        "price": 24.0,
                        ...
                    },
                    ...
                ]

        Returns:
            결과 딕셔너리
            {
                "inserted": int,
                "duplicates": int,
                "errors": int,
            }
        """
        ...

    async def get_raw_data(
        self,
        category_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        원본 데이터를 조회합니다.

        Args:
            category_id: 카테고리 ID 필터 (선택)
            start_date: 시작 날짜 (선택)
            end_date: 종료 날짜 (선택)
            limit: 최대 조회 수 (선택)

        Returns:
            레코드 리스트
        """
        ...

    async def get_latest_data(self, category_id: str | None = None) -> list[dict[str, Any]]:
        """
        최신 데이터를 조회합니다.

        Args:
            category_id: 카테고리 ID 필터 (선택)

        Returns:
            최신 레코드 리스트
        """
        ...

    async def get_historical_data(
        self,
        asin: str,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """
        특정 제품의 히스토리 데이터를 조회합니다.

        Args:
            asin: 제품 ASIN
            days: 조회 기간 (일)

        Returns:
            히스토리 레코드 리스트
        """
        ...

    async def save_brand_metrics(self, metrics: list[dict[str, Any]]) -> int:
        """
        브랜드 메트릭을 저장합니다.

        Args:
            metrics: 브랜드 메트릭 리스트
                [
                    {
                        "snapshot_date": "2026-01-27",
                        "category_id": "lip_care",
                        "brand": "LANEIGE",
                        "sos": 5.2,
                        "brand_avg_rank": 42.5,
                        "product_count": 3,
                        "cpi": 105.0,
                        "avg_rating_gap": 0.2,
                    },
                    ...
                ]

        Returns:
            저장된 레코드 수
        """
        ...

    async def save_market_metrics(self, metrics: list[dict[str, Any]]) -> int:
        """
        시장 메트릭을 저장합니다.

        Args:
            metrics: 시장 메트릭 리스트
                [
                    {
                        "snapshot_date": "2026-01-27",
                        "category_id": "lip_care",
                        "hhi": 0.08,
                        "avg_price": 22.5,
                        "avg_rating": 4.3,
                        "top_brands": ["LANEIGE", "Burt's Bees", ...],
                    },
                    ...
                ]

        Returns:
            저장된 레코드 수
        """
        ...

    async def save_competitor_products(self, products: list[dict[str, Any]]) -> dict[str, Any]:
        """
        경쟁사 제품을 저장합니다.

        Args:
            products: 경쟁사 제품 리스트

        Returns:
            저장 결과 딕셔너리
        """
        ...

    async def get_competitor_products(
        self,
        category_id: str | None = None,
        brands: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        경쟁사 제품을 조회합니다.

        Args:
            category_id: 카테고리 ID 필터 (선택)
            brands: 브랜드 필터 (선택)

        Returns:
            경쟁사 제품 리스트
        """
        ...

    def get_data_date(self) -> str | None:
        """
        가장 최근 데이터 날짜를 반환합니다.

        Returns:
            날짜 문자열 (YYYY-MM-DD) 또는 None
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """
        통계를 반환합니다.

        Returns:
            통계 딕셔너리
            {
                "total_records": int,
                "total_products": int,
                "total_brands": int,
                "categories": [...],
                "date_range": {"start": str, "end": str},
                ...
            }
        """
        ...

    def export_to_excel(
        self,
        output_path: str,
        category_id: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """
        데이터를 엑셀 파일로 내보냅니다.

        Args:
            output_path: 출력 파일 경로
            category_id: 카테고리 ID 필터 (선택)
            start_date: 시작 날짜 (선택)
            end_date: 종료 날짜 (선택)

        Returns:
            내보내기 결과 딕셔너리
        """
        ...
