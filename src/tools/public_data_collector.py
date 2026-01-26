"""
Public Data Collector
=====================
한국 공공데이터 API를 수집하는 모듈

## 지원 API
1. 관세청 수출입통계 (HS Code 3304 화장품)
2. 식약처 기능성화장품 정보
3. KOSIS 국가통계포털

## 사용 예시
```python
collector = PublicDataCollector(api_key="YOUR_DATA_GO_KR_KEY")
await collector.initialize()

# 화장품 수출입 통계
export_data = await collector.fetch_cosmetics_trade(
    year="2025",
    month="01",
    trade_type="export"
)

# 미국 대상 수출 통계
us_exports = await collector.fetch_trade_by_country(
    country_code="US",
    year="2025"
)
```

## 데이터 출력 (인사이트용)
```
■ 거시경제/무역 데이터:
• 화장품 대미 수출: $12.3B (+12% YoY) - 관세청 2025.01
• 기능성화장품 등록: 1,234건 (전월 대비 +5%) - 식약처 2025.01
```
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


logger = logging.getLogger(__name__)

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


class DataSourceType(Enum):
    """데이터 소스 유형"""
    CUSTOMS_TRADE = "customs_trade"       # 관세청 수출입통계
    MFDS_COSMETICS = "mfds_cosmetics"    # 식약처 기능성화장품
    KOSIS_STATS = "kosis_stats"          # KOSIS 국가통계


class ReliabilityScore(Enum):
    """신뢰도 점수"""
    GOVERNMENT = 0.95  # 정부 공식 데이터
    RESEARCH = 0.9     # 전문기관 연구
    NEWS = 0.7         # 뉴스 매체


@dataclass
class TradeData:
    """
    수출입 통계 데이터

    Attributes:
        data_id: 고유 ID
        hs_code: HS 코드 (3304: 화장품)
        trade_type: 수출(export) / 수입(import)
        year: 연도
        month: 월
        country_code: 국가 코드 (선택)
        amount_usd: 금액 (USD)
        amount_krw: 금액 (KRW)
        quantity: 수량
        unit: 단위
        yoy_change: 전년 동기 대비 변동률 (%)
        collected_at: 수집 시각
        source_url: 출처 URL
        metadata: 추가 메타데이터
    """
    data_id: str
    hs_code: str
    trade_type: str  # "export" or "import"
    year: str
    month: str
    country_code: Optional[str] = None
    country_name: Optional[str] = None
    amount_usd: Optional[float] = None
    amount_krw: Optional[float] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    yoy_change: Optional[float] = None
    collected_at: str = ""
    source_url: str = ""
    reliability_score: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.collected_at:
            self.collected_at = datetime.now(KST).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_insight_format(self) -> str:
        """인사이트용 자연어 포맷"""
        period = f"{self.year}.{self.month}"
        trade_type_kr = "수출" if self.trade_type == "export" else "수입"

        amount_str = ""
        if self.amount_usd:
            if self.amount_usd >= 1_000_000_000:
                amount_str = f"${self.amount_usd/1_000_000_000:.1f}B"
            elif self.amount_usd >= 1_000_000:
                amount_str = f"${self.amount_usd/1_000_000:.1f}M"
            else:
                amount_str = f"${self.amount_usd:,.0f}"

        yoy_str = ""
        if self.yoy_change is not None:
            sign = "+" if self.yoy_change >= 0 else ""
            yoy_str = f" ({sign}{self.yoy_change:.1f}% YoY)"

        country_str = ""
        if self.country_name:
            country_str = f" 대{self.country_name}"

        return f"• 화장품{country_str} {trade_type_kr}: {amount_str}{yoy_str} - 관세청 {period}"


@dataclass
class CosmeticsProduct:
    """
    식약처 기능성화장품 데이터

    Attributes:
        product_id: 제품 ID
        product_name: 제품명
        company_name: 업체명
        functional_type: 기능성 유형
        approval_date: 허가일
    """
    product_id: str
    product_name: str
    company_name: str
    functional_type: str
    approval_date: str
    collected_at: str = ""
    reliability_score: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.collected_at:
            self.collected_at = datetime.now(KST).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# API 엔드포인트 설정
API_ENDPOINTS = {
    DataSourceType.CUSTOMS_TRADE: {
        "base_url": "https://apis.data.go.kr/1220000/natImexpTrdStt/getNatImexpTrdSttList",
        "description": "관세청 국가별 수출입 통계",
        "update_frequency": "monthly"
    },
    DataSourceType.MFDS_COSMETICS: {
        "base_url": "http://apis.data.go.kr/1471000/FtnltCosmRptPrdlstInfoService/getRptPrdlstInq",
        "description": "식약처 기능성화장품 보고대상 품목정보",
        "update_frequency": "daily"
    }
}

# 화장품 HS 코드
HS_CODE_COSMETICS = "3304"  # 화장품류 (메이크업, 스킨케어 등)
HS_CODE_SKINCARE = "330499"  # 기타 화장품 (스킨케어 포함)

# 주요 국가 코드 (관세청 API용)
COUNTRY_CODES = {
    "US": {"code": "US", "name": "미국", "name_en": "United States"},
    "CN": {"code": "CN", "name": "중국", "name_en": "China"},
    "JP": {"code": "JP", "name": "일본", "name_en": "Japan"},
    "VN": {"code": "VN", "name": "베트남", "name_en": "Vietnam"},
    "HK": {"code": "HK", "name": "홍콩", "name_en": "Hong Kong"},
    "TH": {"code": "TH", "name": "태국", "name_en": "Thailand"},
    "RU": {"code": "RU", "name": "러시아", "name_en": "Russia"},
}


class PublicDataCollector:
    """
    공공데이터 수집기

    기능:
    1. 관세청 수출입 통계 조회
    2. 식약처 기능성화장품 조회
    3. 데이터 저장 및 캐싱
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        data_dir: str = "./data/public_data"
    ):
        """
        Args:
            api_key: data.go.kr API 키 (환경변수 PUBLIC_DATA_API_KEY로도 설정 가능)
            data_dir: 데이터 저장 디렉토리
        """
        self.api_key = api_key or os.getenv("PUBLIC_DATA_API_KEY", "")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 수집된 데이터 저장
        self.trade_data: List[TradeData] = []
        self.cosmetics_products: List[CosmeticsProduct] = []

        # HTTP 세션
        self._session: Optional[aiohttp.ClientSession] = None

        # 데이터 ID 시퀀스
        self._data_seq = 0

    async def initialize(self) -> None:
        """비동기 초기화"""
        if AIOHTTP_AVAILABLE and not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
                headers={
                    "Accept": "application/json",
                    "User-Agent": "AmorePacificMarketIntelligence/1.0"
                }
            )

        # 기존 데이터 로드
        self._load_data()

        logger.info(f"PublicDataCollector initialized. API key configured: {bool(self.api_key)}")

    async def close(self) -> None:
        """세션 종료"""
        if self._session:
            await self._session.close()
            self._session = None

    # =========================================================================
    # 관세청 수출입 통계 API
    # =========================================================================

    async def fetch_cosmetics_trade(
        self,
        year: str,
        month: Optional[str] = None,
        trade_type: str = "export",
        hs_code: str = HS_CODE_COSMETICS
    ) -> List[TradeData]:
        """
        화장품 수출입 통계 조회

        Args:
            year: 조회 연도 (예: "2025")
            month: 조회 월 (예: "01", 없으면 연간)
            trade_type: "export" 또는 "import"
            hs_code: HS 코드 (기본: 3304 화장품류)

        Returns:
            TradeData 리스트
        """
        if not self.api_key:
            logger.warning("API key not configured. Set PUBLIC_DATA_API_KEY environment variable.")
            return []

        if not self._session:
            await self.initialize()

        endpoint = API_ENDPOINTS[DataSourceType.CUSTOMS_TRADE]["base_url"]

        # 파라미터 설정
        params = {
            "serviceKey": self.api_key,
            "type": "json",
            "hsSgn": hs_code,
            "searchYear": year,
            "numOfRows": "100",
            "pageNo": "1"
        }

        if month:
            params["searchMon"] = month

        # 수출/수입 구분
        if trade_type == "export":
            params["searchImexpDt"] = "1"  # 수출
        else:
            params["searchImexpDt"] = "2"  # 수입

        try:
            logger.info(f"Fetching customs trade data: {trade_type} {year}/{month or 'all'}")

            async with self._session.get(endpoint, params=params) as response:
                if response.status != 200:
                    logger.error(f"Customs API error: {response.status}")
                    return []

                data = await response.json()

            # 응답 파싱
            items = self._parse_customs_response(data)

            trade_data_list = []
            for item in items:
                trade = TradeData(
                    data_id=self._generate_data_id("TRADE"),
                    hs_code=hs_code,
                    trade_type=trade_type,
                    year=year,
                    month=month or "00",
                    country_code=item.get("natCd"),
                    country_name=item.get("natEngNm") or item.get("natNm"),
                    amount_usd=self._safe_float(item.get("expDlr") or item.get("impDlr")),
                    amount_krw=self._safe_float(item.get("expWgt") or item.get("impWgt")),
                    quantity=self._safe_float(item.get("expQty") or item.get("impQty")),
                    unit=item.get("unitCd"),
                    source_url=endpoint,
                    metadata={
                        "raw_response": item
                    }
                )
                trade_data_list.append(trade)

            # 저장
            self.trade_data.extend(trade_data_list)
            self._save_data()

            logger.info(f"Fetched {len(trade_data_list)} trade records")
            return trade_data_list

        except Exception as e:
            logger.error(f"Failed to fetch customs trade data: {e}")
            return []

    async def fetch_trade_by_country(
        self,
        country_code: str,
        year: str,
        month: Optional[str] = None,
        trade_type: str = "export"
    ) -> Optional[TradeData]:
        """
        특정 국가 대상 수출입 통계 조회

        Args:
            country_code: 국가 코드 (예: "US", "CN")
            year: 조회 연도
            month: 조회 월
            trade_type: "export" 또는 "import"

        Returns:
            TradeData 또는 None
        """
        if not self.api_key:
            logger.warning("API key not configured.")
            return None

        if not self._session:
            await self.initialize()

        endpoint = API_ENDPOINTS[DataSourceType.CUSTOMS_TRADE]["base_url"]

        params = {
            "serviceKey": self.api_key,
            "type": "json",
            "hsSgn": HS_CODE_COSMETICS,
            "searchYear": year,
            "natCd": country_code,
            "numOfRows": "10",
            "pageNo": "1"
        }

        if month:
            params["searchMon"] = month

        if trade_type == "export":
            params["searchImexpDt"] = "1"
        else:
            params["searchImexpDt"] = "2"

        try:
            logger.info(f"Fetching {trade_type} to {country_code} for {year}/{month or 'all'}")

            async with self._session.get(endpoint, params=params) as response:
                if response.status != 200:
                    logger.error(f"Customs API error: {response.status}")
                    return None

                data = await response.json()

            items = self._parse_customs_response(data)

            if not items:
                logger.info(f"No data found for {country_code}")
                return None

            item = items[0]  # 첫 번째 결과

            country_info = COUNTRY_CODES.get(country_code, {})

            trade = TradeData(
                data_id=self._generate_data_id("TRADE"),
                hs_code=HS_CODE_COSMETICS,
                trade_type=trade_type,
                year=year,
                month=month or "00",
                country_code=country_code,
                country_name=country_info.get("name", item.get("natEngNm")),
                amount_usd=self._safe_float(item.get("expDlr") or item.get("impDlr")),
                amount_krw=self._safe_float(item.get("expWgt") or item.get("impWgt")),
                quantity=self._safe_float(item.get("expQty") or item.get("impQty")),
                unit=item.get("unitCd"),
                source_url=endpoint,
                metadata={
                    "raw_response": item,
                    "country_info": country_info
                }
            )

            self.trade_data.append(trade)
            self._save_data()

            return trade

        except Exception as e:
            logger.error(f"Failed to fetch trade data for {country_code}: {e}")
            return None

    async def fetch_us_cosmetics_export(
        self,
        year: str,
        month: Optional[str] = None
    ) -> Optional[TradeData]:
        """
        미국 대상 화장품 수출 통계 조회 (편의 메서드)

        Args:
            year: 조회 연도
            month: 조회 월 (선택)

        Returns:
            TradeData 또는 None
        """
        return await self.fetch_trade_by_country(
            country_code="US",
            year=year,
            month=month,
            trade_type="export"
        )

    # =========================================================================
    # 식약처 기능성화장품 API
    # =========================================================================

    async def fetch_functional_cosmetics(
        self,
        company_name: Optional[str] = None,
        product_name: Optional[str] = None,
        num_of_rows: int = 100
    ) -> List[CosmeticsProduct]:
        """
        기능성화장품 정보 조회

        Args:
            company_name: 업체명 검색 (예: "아모레퍼시픽")
            product_name: 제품명 검색
            num_of_rows: 조회 건수

        Returns:
            CosmeticsProduct 리스트
        """
        if not self.api_key:
            logger.warning("API key not configured.")
            return []

        if not self._session:
            await self.initialize()

        endpoint = API_ENDPOINTS[DataSourceType.MFDS_COSMETICS]["base_url"]

        params = {
            "serviceKey": self.api_key,
            "type": "json",
            "numOfRows": str(num_of_rows),
            "pageNo": "1"
        }

        if company_name:
            params["entrpsNm"] = company_name
        if product_name:
            params["prdlstNm"] = product_name

        try:
            logger.info(f"Fetching functional cosmetics: company={company_name}, product={product_name}")

            async with self._session.get(endpoint, params=params) as response:
                if response.status != 200:
                    logger.error(f"MFDS API error: {response.status}")
                    return []

                data = await response.json()

            # 응답 파싱
            items = self._parse_mfds_response(data)

            products = []
            for item in items:
                product = CosmeticsProduct(
                    product_id=self._generate_data_id("COSM"),
                    product_name=item.get("PRDLST_NM", ""),
                    company_name=item.get("ENTRPS_NM", ""),
                    functional_type=item.get("FNCT_CTGRY_NM", ""),
                    approval_date=item.get("RPT_DT", ""),
                    metadata={
                        "raw_response": item
                    }
                )
                products.append(product)

            self.cosmetics_products.extend(products)
            self._save_data()

            logger.info(f"Fetched {len(products)} cosmetics products")
            return products

        except Exception as e:
            logger.error(f"Failed to fetch functional cosmetics: {e}")
            return []

    # =========================================================================
    # 데이터 집계 및 인사이트 생성
    # =========================================================================

    def get_trade_summary(
        self,
        year: str,
        trade_type: str = "export"
    ) -> Dict[str, Any]:
        """
        수출입 통계 요약

        Args:
            year: 조회 연도
            trade_type: "export" 또는 "import"

        Returns:
            요약 딕셔너리
        """
        filtered = [
            t for t in self.trade_data
            if t.year == year and t.trade_type == trade_type
        ]

        if not filtered:
            return {"error": "No data found"}

        # 국가별 집계
        by_country = {}
        total_usd = 0

        for trade in filtered:
            if trade.country_code:
                if trade.country_code not in by_country:
                    by_country[trade.country_code] = {
                        "country_name": trade.country_name,
                        "amount_usd": 0,
                        "records": 0
                    }
                by_country[trade.country_code]["amount_usd"] += (trade.amount_usd or 0)
                by_country[trade.country_code]["records"] += 1

            total_usd += (trade.amount_usd or 0)

        # 상위 5개국
        top_countries = sorted(
            by_country.items(),
            key=lambda x: x[1]["amount_usd"],
            reverse=True
        )[:5]

        return {
            "year": year,
            "trade_type": trade_type,
            "total_amount_usd": total_usd,
            "total_records": len(filtered),
            "top_countries": [
                {
                    "country_code": code,
                    "country_name": info["country_name"],
                    "amount_usd": info["amount_usd"]
                }
                for code, info in top_countries
            ],
            "generated_at": datetime.now(KST).isoformat()
        }

    def generate_insight_section(self) -> str:
        """
        인사이트 보고서용 섹션 생성

        Returns:
            인사이트 섹션 문자열
        """
        sections = ["■ 거시경제/무역 데이터:"]

        # 최근 수출입 데이터
        recent_trades = sorted(
            self.trade_data,
            key=lambda x: (x.year, x.month),
            reverse=True
        )[:5]

        for trade in recent_trades:
            sections.append(trade.to_insight_format())

        # 기능성화장품 통계
        if self.cosmetics_products:
            recent_products = len([
                p for p in self.cosmetics_products
                if p.approval_date >= (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            ])
            if recent_products > 0:
                sections.append(f"• 최근 30일 기능성화장품 등록: {recent_products}건 - 식약처")

        return "\n".join(sections) if len(sections) > 1 else "수집된 공공데이터가 없습니다."

    # =========================================================================
    # Source 객체 생성 (출처 관리 시스템 연동용)
    # =========================================================================

    def create_source_reference(
        self,
        trade_data: Optional[TradeData] = None,
        product_data: Optional[CosmeticsProduct] = None
    ) -> Dict[str, Any]:
        """
        출처 참조 객체 생성 (SourceManager 연동용)

        Args:
            trade_data: 수출입 데이터
            product_data: 화장품 데이터

        Returns:
            Source 형식 딕셔너리
        """
        if trade_data:
            period = f"{trade_data.year}.{trade_data.month}" if trade_data.month != "00" else trade_data.year
            return {
                "title": f"품목별 수출입통계 (HS Code {trade_data.hs_code})",
                "publisher": "관세청",
                "date": period,
                "url": trade_data.source_url,
                "source_type": "government",
                "reliability_score": trade_data.reliability_score
            }

        if product_data:
            return {
                "title": f"기능성화장품 보고대상 품목정보 - {product_data.product_name}",
                "publisher": "식품의약품안전처",
                "date": product_data.approval_date,
                "url": API_ENDPOINTS[DataSourceType.MFDS_COSMETICS]["base_url"],
                "source_type": "government",
                "reliability_score": product_data.reliability_score
            }

        return {}

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def _parse_customs_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """관세청 API 응답 파싱"""
        try:
            # 응답 구조: response > body > items > item
            body = data.get("response", {}).get("body", {})
            items = body.get("items", {})

            if isinstance(items, dict):
                item_list = items.get("item", [])
                if isinstance(item_list, dict):
                    return [item_list]
                return item_list

            return []
        except Exception as e:
            logger.error(f"Failed to parse customs response: {e}")
            return []

    def _parse_mfds_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """식약처 API 응답 파싱"""
        try:
            # 응답 구조: body > items > item
            body = data.get("body", {})
            items = body.get("items", [])

            if isinstance(items, list):
                return items

            return []
        except Exception as e:
            logger.error(f"Failed to parse MFDS response: {e}")
            return []

    def _safe_float(self, value: Any) -> Optional[float]:
        """안전한 float 변환"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _generate_data_id(self, prefix: str) -> str:
        """데이터 ID 생성"""
        self._data_seq += 1
        date_str = datetime.now(KST).strftime("%Y%m%d")
        return f"{prefix}-{date_str}-{self._data_seq:04d}"

    def _save_data(self) -> None:
        """데이터 저장"""
        # 수출입 데이터 저장
        trade_file = self.data_dir / "trade_data.json"
        trade_data = {
            "trade_records": [t.to_dict() for t in self.trade_data],
            "updated_at": datetime.now(KST).isoformat(),
            "count": len(self.trade_data)
        }
        with open(trade_file, "w", encoding="utf-8") as f:
            json.dump(trade_data, f, ensure_ascii=False, indent=2)

        # 화장품 데이터 저장
        cosmetics_file = self.data_dir / "cosmetics_data.json"
        cosmetics_data = {
            "products": [p.to_dict() for p in self.cosmetics_products],
            "updated_at": datetime.now(KST).isoformat(),
            "count": len(self.cosmetics_products)
        }
        with open(cosmetics_file, "w", encoding="utf-8") as f:
            json.dump(cosmetics_data, f, ensure_ascii=False, indent=2)

    def _load_data(self) -> None:
        """데이터 로드"""
        # 수출입 데이터 로드
        trade_file = self.data_dir / "trade_data.json"
        if trade_file.exists():
            try:
                with open(trade_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.trade_data = [
                    TradeData(**t) for t in data.get("trade_records", [])
                ]
                logger.info(f"Loaded {len(self.trade_data)} trade records")
            except Exception as e:
                logger.warning(f"Failed to load trade data: {e}")

        # 화장품 데이터 로드
        cosmetics_file = self.data_dir / "cosmetics_data.json"
        if cosmetics_file.exists():
            try:
                with open(cosmetics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.cosmetics_products = [
                    CosmeticsProduct(**p) for p in data.get("products", [])
                ]
                logger.info(f"Loaded {len(self.cosmetics_products)} cosmetics products")
            except Exception as e:
                logger.warning(f"Failed to load cosmetics data: {e}")

        # 시퀀스 업데이트
        self._data_seq = len(self.trade_data) + len(self.cosmetics_products)

    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        return {
            "trade_records": len(self.trade_data),
            "cosmetics_products": len(self.cosmetics_products),
            "api_key_configured": bool(self.api_key),
            "data_dir": str(self.data_dir),
            "last_updated": datetime.now(KST).isoformat()
        }
