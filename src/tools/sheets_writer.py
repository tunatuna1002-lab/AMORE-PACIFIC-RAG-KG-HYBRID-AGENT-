"""
Google Sheets Writer
Google Sheets API를 통한 데이터 저장/조회

사용법:
    writer = SheetsWriter()
    await writer.initialize()
    await writer.append_rank_records(records)

환경 변수:
    GOOGLE_SHEETS_CREDENTIALS_JSON: Google 서비스 계정 credentials JSON 문자열 (Railway 배포용)
    GOOGLE_SHEETS_CREDENTIALS_PATH: Google 서비스 계정 credentials 파일 경로 (로컬 개발용)
    GOOGLE_SHEETS_SPREADSHEET_ID: Google Sheets 스프레드시트 ID
"""

import os
import json
import logging
from datetime import date, datetime
from typing import List, Dict, Any, Optional
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from src.ontology.schema import RankRecord

logger = logging.getLogger(__name__)


class SheetsWriter:
    """Google Sheets 데이터 관리 클래스"""

    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

    # 시트 구조 정의
    SHEETS_CONFIG = {
        "raw_data": {
            "name": "RawData",
            "headers": [
                "snapshot_date", "category_id", "rank", "asin", "product_name",
                "brand", "price", "list_price", "discount_percent", "rating",
                "reviews_count", "badge", "coupon_text", "is_subscribe_save",
                "promo_badges", "product_url"
            ]
        },
        "products": {
            "name": "Products",
            "headers": [
                "asin", "product_name", "brand", "first_seen_date", "launch_date", "product_url"
            ]
        },
        "brand_metrics": {
            "name": "BrandMetrics",
            "headers": [
                "snapshot_date", "category_id", "brand", "sos", "brand_avg_rank",
                "product_count", "cpi", "avg_rating_gap"
            ]
        },
        "product_metrics": {
            "name": "ProductMetrics",
            "headers": [
                "snapshot_date", "category_id", "asin", "rank_volatility", "rank_shock",
                "rank_change", "streak_days", "rating_trend", "best_rank"
            ]
        },
        "market_metrics": {
            "name": "MarketMetrics",
            "headers": [
                "snapshot_date", "category_id", "hhi", "churn_rate",
                "category_avg_price", "category_avg_rating"
            ]
        }
    }

    def __init__(self, credentials_path: Optional[str] = None, spreadsheet_id: Optional[str] = None):
        """
        Args:
            credentials_path: Google 서비스 계정 인증 파일 경로
            spreadsheet_id: Google Sheets 스프레드시트 ID
        """
        self.credentials_path = credentials_path or os.getenv(
            "GOOGLE_SHEETS_CREDENTIALS_PATH",
            "./config/google_credentials.json"
        )
        # 환경 변수에서 credentials JSON 문자열 (Railway 배포용)
        self.credentials_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON")
        # Spreadsheet ID에서 줄바꿈/공백 제거 (환경변수 입력 오류 방지)
        raw_spreadsheet_id = spreadsheet_id or os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID") or ""
        self.spreadsheet_id = raw_spreadsheet_id.strip()
        self.service = None
        self._initialized = False
        # 제품 목록 캐시 (API 호출 최소화)
        self._products_cache: Optional[Dict[str, Dict[str, Any]]] = None

    def _get_credentials(self) -> Credentials:
        """
        Credentials 객체 생성

        우선순위:
        1. GOOGLE_SHEETS_CREDENTIALS_JSON 환경 변수 (Railway 배포용)
        2. credentials_path 파일 (로컬 개발용)

        Returns:
            Google Credentials 객체
        """
        if self.credentials_json:
            # 환경 변수에서 JSON 문자열로 로드 (Railway 배포용)
            logger.info("Loading credentials from GOOGLE_SHEETS_CREDENTIALS_JSON environment variable")
            try:
                credentials_info = json.loads(self.credentials_json)
                return Credentials.from_service_account_info(
                    credentials_info,
                    scopes=self.SCOPES
                )
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GOOGLE_SHEETS_CREDENTIALS_JSON: {e}")
                raise ValueError(f"Invalid JSON in GOOGLE_SHEETS_CREDENTIALS_JSON: {e}")
        else:
            # 파일에서 로드 (로컬 개발용)
            logger.info(f"Loading credentials from file: {self.credentials_path}")
            return Credentials.from_service_account_file(
                self.credentials_path,
                scopes=self.SCOPES
            )

    async def initialize(self) -> bool:
        """
        Google Sheets API 초기화

        Returns:
            초기화 성공 여부
        """
        try:
            creds = self._get_credentials()
            self.service = build("sheets", "v4", credentials=creds)
            self._initialized = True

            # 시트 구조 확인 및 생성
            await self._ensure_sheets_exist()

            logger.info("Google Sheets API initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Google Sheets 초기화 실패: {e}")
            print(f"Google Sheets 초기화 실패: {e}")
            return False

    async def _ensure_sheets_exist(self) -> None:
        """필요한 시트가 없으면 생성"""
        if not self.service or not self.spreadsheet_id:
            return

        try:
            # 현재 시트 목록 조회
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id
            ).execute()

            existing_sheets = {
                sheet["properties"]["title"]
                for sheet in spreadsheet.get("sheets", [])
            }

            # 없는 시트 생성
            requests = []
            for sheet_config in self.SHEETS_CONFIG.values():
                if sheet_config["name"] not in existing_sheets:
                    requests.append({
                        "addSheet": {
                            "properties": {"title": sheet_config["name"]}
                        }
                    })

            if requests:
                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body={"requests": requests}
                ).execute()

                # 헤더 추가
                for sheet_key, sheet_config in self.SHEETS_CONFIG.items():
                    if sheet_config["name"] not in existing_sheets:
                        await self._write_headers(sheet_config["name"], sheet_config["headers"])

        except HttpError as e:
            print(f"시트 생성 오류: {e}")

    async def _write_headers(self, sheet_name: str, headers: List[str]) -> None:
        """시트에 헤더 작성"""
        range_name = f"{sheet_name}!A1"
        self.service.spreadsheets().values().update(
            spreadsheetId=self.spreadsheet_id,
            range=range_name,
            valueInputOption="RAW",
            body={"values": [headers]}
        ).execute()

    async def append_rank_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        순위 기록 추가

        Args:
            records: RankRecord 딕셔너리 리스트

        Returns:
            {
                "success": True,
                "rows_added": 100,
                "sheet": "RawData"
            }
        """
        if not self._initialized:
            await self.initialize()

        sheet_config = self.SHEETS_CONFIG["raw_data"]
        headers = sheet_config["headers"]

        # 데이터 행 생성
        rows = []
        for record in records:
            row = []
            for h in headers:
                val = record.get(h, "")
                # date/datetime 객체를 문자열로 변환
                if hasattr(val, 'isoformat'):
                    val = val.isoformat() if hasattr(val, 'isoformat') else str(val)
                elif val is None:
                    val = ""
                row.append(str(val) if val else "")
            rows.append(row)

        try:
            range_name = f"{sheet_config['name']}!A:P"
            logger.info(f"Appending {len(rows)} rows to {range_name} in spreadsheet {self.spreadsheet_id[:20]}...")
            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body={"values": rows}
            ).execute()

            updates = result.get("updates", {})
            logger.info(f"Successfully appended rows: {updates}")
            return {
                "success": True,
                "rows_added": len(rows),
                "sheet": sheet_config["name"],
                "updates": updates
            }
        except HttpError as e:
            logger.error(f"Google Sheets API error: {e}")
            return {
                "success": False,
                "error": str(e),
                "sheet": sheet_config["name"]
            }
        except Exception as e:
            logger.error(f"Unexpected error appending to sheets: {e}")
            return {
                "success": False,
                "error": str(e),
                "sheet": sheet_config["name"]
            }

    async def append_brand_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """브랜드 지표 추가"""
        if not self._initialized:
            await self.initialize()

        sheet_config = self.SHEETS_CONFIG["brand_metrics"]
        headers = sheet_config["headers"]

        rows = []
        for metric in metrics:
            row = [metric.get(h, "") for h in headers]
            rows.append(row)

        try:
            range_name = f"{sheet_config['name']}!A:H"
            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body={"values": rows}
            ).execute()

            return {"success": True, "rows_added": len(rows)}
        except HttpError as e:
            return {"success": False, "error": str(e)}

    async def append_product_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """제품 지표 추가"""
        if not self._initialized:
            await self.initialize()

        sheet_config = self.SHEETS_CONFIG["product_metrics"]
        headers = sheet_config["headers"]

        rows = []
        for metric in metrics:
            row = [metric.get(h, "") for h in headers]
            rows.append(row)

        try:
            range_name = f"{sheet_config['name']}!A:I"
            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body={"values": rows}
            ).execute()

            return {"success": True, "rows_added": len(rows)}
        except HttpError as e:
            return {"success": False, "error": str(e)}

    async def append_market_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """시장 지표 추가"""
        if not self._initialized:
            await self.initialize()

        sheet_config = self.SHEETS_CONFIG["market_metrics"]
        headers = sheet_config["headers"]

        rows = []
        for metric in metrics:
            row = [metric.get(h, "") for h in headers]
            rows.append(row)

        try:
            range_name = f"{sheet_config['name']}!A:F"
            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body={"values": rows}
            ).execute()

            return {"success": True, "rows_added": len(rows)}
        except HttpError as e:
            return {"success": False, "error": str(e)}

    async def get_rank_history(
        self,
        category_id: Optional[str] = None,
        asin: Optional[str] = None,
        brand: Optional[str] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        순위 히스토리 조회

        Args:
            category_id: 카테고리 필터
            asin: 제품 ASIN 필터
            brand: 브랜드 필터
            days: 조회 일수

        Returns:
            순위 기록 리스트
        """
        if not self._initialized:
            await self.initialize()

        try:
            sheet_config = self.SHEETS_CONFIG["raw_data"]
            range_name = f"{sheet_config['name']}!A:P"

            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()

            values = result.get("values", [])
            if not values:
                return []

            headers = values[0]
            records = []

            for row in values[1:]:
                record = dict(zip(headers, row + [""] * (len(headers) - len(row))))

                # 필터 적용
                if category_id and record.get("category_id") != category_id:
                    continue
                if asin and record.get("asin") != asin:
                    continue
                if brand and record.get("brand", "").lower() != brand.lower():
                    continue

                records.append(record)

            # 날짜 기준 정렬 및 최근 N일만
            records.sort(key=lambda x: x.get("snapshot_date", ""), reverse=True)

            # days 필터링
            if days > 0:
                cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
                records = [r for r in records if r.get("snapshot_date", "") >= cutoff_date]

            return records

        except HttpError as e:
            print(f"데이터 조회 오류: {e}")
            return []

    async def get_latest_snapshot(self, category_id: str) -> List[Dict[str, Any]]:
        """최신 스냅샷 데이터 조회"""
        records = await self.get_rank_history(category_id=category_id, days=1)
        return records

    async def get_raw_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category_id: Optional[str] = None,
        brand: Optional[str] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        원본 데이터 조회 (SQLiteStorage와 동일한 인터페이스)

        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            category_id: 카테고리 필터
            brand: 브랜드 필터
            days: 조회 일수 (start_date/end_date가 없을 때 사용)

        Returns:
            데이터 딕셔너리 리스트
        """
        if not self._initialized:
            await self.initialize()

        try:
            sheet_config = self.SHEETS_CONFIG["raw_data"]
            range_name = f"{sheet_config['name']}!A:P"

            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()

            values = result.get("values", [])
            if not values:
                return []

            headers = values[0]
            records = []

            # 날짜 범위 계산
            if start_date and end_date:
                date_start = start_date
                date_end = end_date
            else:
                date_end = datetime.now().strftime("%Y-%m-%d")
                date_start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

            for row in values[1:]:
                record = dict(zip(headers, row + [""] * (len(headers) - len(row))))

                # 날짜 필터
                snapshot_date = record.get("snapshot_date", "")
                if snapshot_date < date_start or snapshot_date > date_end:
                    continue

                # 카테고리 필터
                if category_id and record.get("category_id") != category_id:
                    continue

                # 브랜드 필터
                if brand and record.get("brand", "").lower() != brand.lower():
                    continue

                records.append(record)

            # 날짜 기준 정렬
            records.sort(key=lambda x: x.get("snapshot_date", ""), reverse=True)

            return records

        except HttpError as e:
            logger.error(f"Google Sheets 데이터 조회 오류: {e}")
            return []

    async def _load_products_cache(self) -> Dict[str, Dict[str, Any]]:
        """제품 목록을 캐시로 로드 (한 번만 API 호출)"""
        if self._products_cache is not None:
            return self._products_cache

        if not self._initialized:
            await self.initialize()

        self._products_cache = {}

        try:
            sheet_config = self.SHEETS_CONFIG["products"]
            range_name = f"{sheet_config['name']}!A:F"

            result = self.service.spreadsheets().values().get(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()

            values = result.get("values", [])
            if not values:
                return self._products_cache

            headers = values[0]
            for row in values[1:]:
                record = dict(zip(headers, row + [""] * (len(headers) - len(row))))
                asin = record.get("asin", "")
                if asin:
                    self._products_cache[asin] = record

            logger.info(f"Loaded {len(self._products_cache)} products into cache")
            return self._products_cache

        except HttpError as e:
            logger.error(f"제품 목록 로드 오류: {e}")
            return self._products_cache

    async def get_product_info(self, asin: str) -> Optional[Dict[str, Any]]:
        """제품 정보 조회 (캐시 활용)"""
        cache = await self._load_products_cache()
        return cache.get(asin)

    async def upsert_product(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """제품 정보 추가/업데이트 (first_seen_date 관리) - 캐시 활용"""
        if not self._initialized:
            await self.initialize()

        cache = await self._load_products_cache()
        asin = product.get("asin", "")
        existing = cache.get(asin)

        if existing:
            # 기존 제품 - first_seen_date 유지
            product["first_seen_date"] = existing.get("first_seen_date", "")
            return {"success": True, "action": "updated", "asin": asin}
        else:
            # 신규 제품 - first_seen_date 설정
            product["first_seen_date"] = date.today().isoformat()

            sheet_config = self.SHEETS_CONFIG["products"]
            headers = sheet_config["headers"]
            row = [product.get(h, "") for h in headers]

            try:
                range_name = f"{sheet_config['name']}!A:F"
                self.service.spreadsheets().values().append(
                    spreadsheetId=self.spreadsheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    insertDataOption="INSERT_ROWS",
                    body={"values": [row]}
                ).execute()

                # 캐시 업데이트
                self._products_cache[asin] = product

                return {"success": True, "action": "created", "asin": asin}
            except HttpError as e:
                return {"success": False, "error": str(e)}

    async def upsert_products_batch(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        제품 정보 일괄 추가 (API 호출 최소화)

        Args:
            products: 제품 정보 리스트

        Returns:
            {"success": True, "created": 10, "updated": 50}
        """
        if not self._initialized:
            await self.initialize()

        cache = await self._load_products_cache()

        new_products = []
        updated_count = 0

        for product in products:
            asin = product.get("asin", "")
            if not asin:
                continue

            existing = cache.get(asin)
            if existing:
                # 기존 제품
                updated_count += 1
            else:
                # 신규 제품
                product["first_seen_date"] = date.today().isoformat()
                new_products.append(product)
                # 캐시 업데이트
                cache[asin] = product

        # 신규 제품만 일괄 추가 (한 번의 API 호출)
        created_count = 0
        if new_products:
            sheet_config = self.SHEETS_CONFIG["products"]
            headers = sheet_config["headers"]
            rows = []
            for product in new_products:
                row = [product.get(h, "") for h in headers]
                rows.append(row)

            try:
                range_name = f"{sheet_config['name']}!A:F"
                self.service.spreadsheets().values().append(
                    spreadsheetId=self.spreadsheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    insertDataOption="INSERT_ROWS",
                    body={"values": rows}
                ).execute()
                created_count = len(new_products)
                logger.info(f"Batch created {created_count} new products")
            except HttpError as e:
                logger.error(f"제품 일괄 추가 오류: {e}")
                return {"success": False, "error": str(e)}

        return {
            "success": True,
            "created": created_count,
            "updated": updated_count
        }


# 임포트 추가
from datetime import timedelta
