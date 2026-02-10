"""
Source Manager
==============
인사이트 보고서의 출처를 관리하는 모듈

## 기능
1. 출처 등록 및 관리
2. 인용 번호 자동 할당 ([1], [2], ...)
3. 참고자료 섹션 생성
4. 신뢰도 점수 기반 정렬

## 출처 유형
- ir: IR 보고서 (아모레퍼시픽 Earnings Release)
- government: 정부 공식 데이터 (관세청, 식약처)
- research: 전문기관 연구 (KCII, KHIDI, KPMG)
- analyst: 증권사 리포트
- news: 뉴스 매체
- sns: SNS (Reddit, TikTok)

## 사용 예시
```python
manager = SourceManager()

# 출처 추가
ref1 = manager.add_source(
    title="3Q 2025 Earnings Release",
    publisher="아모레퍼시픽 IR",
    date="2025-11-06",
    url="https://www.apgroup.com/...",
    source_type="ir"
)

# 인용 표시 생성
citation = manager.cite(ref1)  # "[1]"

# 참고자료 섹션 생성
references = manager.generate_references_section()
```

## 출력 형식
```markdown
## 참고자료
[1] 아모레퍼시픽 IR, "3Q 2025 Earnings Release", 2025.11.06
    https://www.apgroup.com/...
[2] 관세청, "품목별 수출입통계 (HS Code 3304)", 2025.01
```
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


class SourceType(Enum):
    """출처 유형"""

    IR = "ir"  # IR 보고서 (최고 신뢰도)
    GOVERNMENT = "government"  # 정부 공식 데이터
    RESEARCH = "research"  # 전문기관 연구
    ANALYST = "analyst"  # 증권사 리포트
    NEWS = "news"  # 뉴스 매체
    SNS = "sns"  # SNS


# 출처 유형별 기본 신뢰도 점수
RELIABILITY_SCORES = {
    SourceType.IR.value: 1.0,  # IR/공시는 최고 신뢰도
    SourceType.GOVERNMENT.value: 0.95,  # 정부 공식 데이터
    SourceType.RESEARCH.value: 0.9,  # 전문기관 연구
    SourceType.ANALYST.value: 0.85,  # 증권사 리포트
    SourceType.NEWS.value: 0.7,  # 뉴스 매체
    SourceType.SNS.value: 0.5,  # SNS
}

# 매체별 신뢰도 점수 오버라이드
PUBLISHER_RELIABILITY = {
    # IR/공시
    "아모레퍼시픽 IR": 1.0,
    "아모레퍼시픽": 1.0,
    # 정부 기관
    "관세청": 0.95,
    "식품의약품안전처": 0.95,
    "식약처": 0.95,
    "한국은행": 0.95,
    "통계청": 0.95,
    "KOSIS": 0.95,
    # 전문기관
    "대한화장품산업연구원": 0.9,
    "KCII": 0.9,
    "한국보건산업진흥원": 0.9,
    "KHIDI": 0.9,
    "삼정KPMG": 0.9,
    "KPMG": 0.9,
    "KDI": 0.9,
    "한국무역협회": 0.9,
    "KITA": 0.9,
    # 증권사
    "메리츠증권": 0.85,
    "상상인증권": 0.85,
    "DB금융투자": 0.85,
    # 전문 매체
    "WWD": 0.85,
    "Cosmetics Design Asia": 0.85,
    "Cosmetics Design Europe": 0.85,
    "Allure": 0.8,
    "Byrdie": 0.8,
    "Refinery29": 0.75,
    "KEDGlobal": 0.8,
    "Korea Herald": 0.8,
    "Reuters": 0.85,
    # 일반 뉴스
    "Bloomberg": 0.8,
    "CNBC": 0.75,
    # SNS
    "Reddit": 0.5,
    "TikTok": 0.5,
    "TikTok Creative Center": 0.6,
    "Instagram": 0.5,
    "YouTube": 0.6,
    "X": 0.4,
    "Twitter": 0.4,
}


@dataclass
class Source:
    """
    출처 데이터

    Attributes:
        source_id: 고유 ID
        citation_number: 인용 번호 (1, 2, 3, ...)
        title: 제목/보고서명
        publisher: 발행 기관/매체
        date: 발행일
        url: 원본 URL (선택)
        source_type: 출처 유형
        reliability_score: 신뢰도 점수 (0.0 ~ 1.0)
        metadata: 추가 메타데이터
    """

    source_id: str
    citation_number: int
    title: str
    publisher: str
    date: str
    url: str | None = None
    source_type: str = "news"
    reliability_score: float = 0.7
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(KST).isoformat()

        # 신뢰도 점수 자동 설정
        if self.reliability_score == 0.7:  # 기본값이면
            # 매체별 오버라이드 확인
            if self.publisher in PUBLISHER_RELIABILITY:
                self.reliability_score = PUBLISHER_RELIABILITY[self.publisher]
            # 유형별 기본값
            elif self.source_type in RELIABILITY_SCORES:
                self.reliability_score = RELIABILITY_SCORES[self.source_type]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_citation(self) -> str:
        """인용 표시 생성 (예: [1])"""
        return f"[{self.citation_number}]"

    def to_reference_line(self, include_url: bool = True) -> str:
        """
        참고자료 라인 생성

        Args:
            include_url: URL 포함 여부

        Returns:
            참고자료 라인 문자열
        """
        # 날짜 포맷팅
        date_str = self.date
        if len(date_str) == 10:  # YYYY-MM-DD
            parts = date_str.split("-")
            date_str = f"{parts[0]}.{parts[1]}.{parts[2]}"
        elif len(date_str) == 7:  # YYYY-MM
            parts = date_str.split("-")
            date_str = f"{parts[0]}.{parts[1]}"

        line = f'[{self.citation_number}] {self.publisher}, "{self.title}", {date_str}'

        if include_url and self.url:
            line += f"\n    {self.url}"

        return line

    def to_inline_reference(self) -> str:
        """인라인 참조 생성 (예: (관세청, 2025.01))"""
        date_short = self.date[:7].replace("-", ".") if len(self.date) >= 7 else self.date
        return f"({self.publisher}, {date_short})"


class SourceManager:
    """
    출처 관리자

    기능:
    1. 출처 등록 및 관리
    2. 인용 번호 자동 할당
    3. 중복 출처 감지
    4. 참고자료 섹션 생성
    """

    def __init__(self, data_dir: str = "./data/sources"):
        """
        Args:
            data_dir: 출처 데이터 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 현재 세션의 출처 목록
        self.sources: dict[str, Source] = {}

        # 인용 번호 카운터
        self._citation_counter = 0

        # 출처 ID 시퀀스
        self._source_seq = 0

    def add_source(
        self,
        title: str,
        publisher: str,
        date: str,
        url: str | None = None,
        source_type: str = "news",
        reliability_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Source:
        """
        출처 추가

        Args:
            title: 제목/보고서명
            publisher: 발행 기관/매체
            date: 발행일 (YYYY-MM-DD 또는 YYYY-MM)
            url: 원본 URL
            source_type: 출처 유형
            reliability_score: 신뢰도 점수 (자동 설정됨)
            metadata: 추가 메타데이터

        Returns:
            생성된 Source 객체
        """
        # 중복 체크
        existing = self._find_duplicate(title, publisher, date)
        if existing:
            logger.debug(f"Duplicate source found: {existing.source_id}")
            return existing

        # 인용 번호 할당
        self._citation_counter += 1
        citation_number = self._citation_counter

        # 출처 ID 생성
        source_id = self._generate_source_id()

        # 신뢰도 점수
        score = reliability_score
        if score is None:
            if publisher in PUBLISHER_RELIABILITY:
                score = PUBLISHER_RELIABILITY[publisher]
            elif source_type in RELIABILITY_SCORES:
                score = RELIABILITY_SCORES[source_type]
            else:
                score = 0.7

        source = Source(
            source_id=source_id,
            citation_number=citation_number,
            title=title,
            publisher=publisher,
            date=date,
            url=url,
            source_type=source_type,
            reliability_score=score,
            metadata=metadata or {},
        )

        self.sources[source_id] = source
        logger.debug(f"Added source: {source_id} as [{citation_number}]")

        return source

    def add_source_from_dict(self, data: dict[str, Any]) -> Source:
        """
        딕셔너리에서 출처 추가

        Args:
            data: 출처 정보 딕셔너리

        Returns:
            생성된 Source 객체
        """
        return self.add_source(
            title=data.get("title", ""),
            publisher=data.get("publisher", ""),
            date=data.get("date", ""),
            url=data.get("url"),
            source_type=data.get("source_type", "news"),
            reliability_score=data.get("reliability_score"),
            metadata=data.get("metadata"),
        )

    def cite(self, source: Source) -> str:
        """
        인용 표시 생성

        Args:
            source: Source 객체

        Returns:
            인용 표시 문자열 (예: "[1]")
        """
        return source.to_citation()

    def cite_by_id(self, source_id: str) -> str:
        """
        ID로 인용 표시 생성

        Args:
            source_id: 출처 ID

        Returns:
            인용 표시 문자열
        """
        source = self.sources.get(source_id)
        if source:
            return source.to_citation()
        return ""

    def get_source(self, source_id: str) -> Source | None:
        """
        출처 조회

        Args:
            source_id: 출처 ID

        Returns:
            Source 객체 또는 None
        """
        return self.sources.get(source_id)

    def get_source_by_citation(self, citation_number: int) -> Source | None:
        """
        인용 번호로 출처 조회

        Args:
            citation_number: 인용 번호

        Returns:
            Source 객체 또는 None
        """
        for source in self.sources.values():
            if source.citation_number == citation_number:
                return source
        return None

    def get_all_sources(self, sort_by: str = "citation_number") -> list[Source]:
        """
        모든 출처 조회

        Args:
            sort_by: 정렬 기준 ("citation_number", "reliability", "date")

        Returns:
            Source 리스트
        """
        sources = list(self.sources.values())

        if sort_by == "citation_number":
            sources.sort(key=lambda s: s.citation_number)
        elif sort_by == "reliability":
            sources.sort(key=lambda s: s.reliability_score, reverse=True)
        elif sort_by == "date":
            sources.sort(key=lambda s: s.date, reverse=True)

        return sources

    def generate_references_section(
        self, include_urls: bool = True, sort_by: str = "citation_number"
    ) -> str:
        """
        참고자료 섹션 생성

        Args:
            include_urls: URL 포함 여부
            sort_by: 정렬 기준

        Returns:
            참고자료 섹션 문자열
        """
        if not self.sources:
            return ""

        lines = ["## 참고자료", ""]

        sources = self.get_all_sources(sort_by)

        for source in sources:
            lines.append(source.to_reference_line(include_urls))

        return "\n".join(lines)

    def generate_compact_references(self) -> str:
        """
        간략한 참고자료 생성 (URL 없이)

        Returns:
            참고자료 문자열
        """
        if not self.sources:
            return ""

        lines = ["---", "**참고자료**"]

        sources = self.get_all_sources("citation_number")

        for source in sources:
            date_short = source.date[:7].replace("-", ".")
            lines.append(f"[{source.citation_number}] {source.publisher}, {date_short}")

        return "\n".join(lines)

    def clear(self) -> None:
        """모든 출처 초기화"""
        self.sources.clear()
        self._citation_counter = 0
        logger.debug("Source manager cleared")

    def reset_session(self) -> None:
        """
        세션 초기화 (새 인사이트 생성 시)

        인용 번호와 출처 목록을 초기화합니다.
        """
        self.clear()

    # =========================================================================
    # 유틸리티 메서드
    # =========================================================================

    def _find_duplicate(self, title: str, publisher: str, date: str) -> Source | None:
        """중복 출처 찾기"""
        for source in self.sources.values():
            if source.title == title and source.publisher == publisher and source.date == date:
                return source
        return None

    def _generate_source_id(self) -> str:
        """출처 ID 생성"""
        self._source_seq += 1
        return f"SRC-{datetime.now(KST).strftime('%Y%m%d')}-{self._source_seq:04d}"

    def save(self) -> None:
        """출처 데이터 저장"""
        filepath = self.data_dir / "sources.json"
        data = {
            "sources": [s.to_dict() for s in self.sources.values()],
            "updated_at": datetime.now(KST).isoformat(),
            "count": len(self.sources),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """출처 데이터 로드"""
        filepath = self.data_dir / "sources.json"
        if not filepath.exists():
            return

        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            for source_data in data.get("sources", []):
                source = Source(**source_data)
                self.sources[source.source_id] = source

                # 카운터 업데이트
                if source.citation_number > self._citation_counter:
                    self._citation_counter = source.citation_number

            logger.info(f"Loaded {len(self.sources)} sources")

        except Exception as e:
            logger.warning(f"Failed to load sources: {e}")

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        by_type = {}
        for source in self.sources.values():
            by_type[source.source_type] = by_type.get(source.source_type, 0) + 1

        return {
            "total_sources": len(self.sources),
            "by_type": by_type,
            "citation_counter": self._citation_counter,
        }


class InsightSourceBuilder:
    """
    인사이트 출처 빌더

    인사이트 생성 시 사용되는 헬퍼 클래스입니다.
    텍스트에 인용을 삽입하고 참고자료를 자동 생성합니다.
    """

    def __init__(self):
        self.manager = SourceManager()
        self._text_parts: list[str] = []

    def add_text(self, text: str) -> "InsightSourceBuilder":
        """텍스트 추가"""
        self._text_parts.append(text)
        return self

    def add_cited_text(self, text: str, source_info: dict[str, Any]) -> "InsightSourceBuilder":
        """
        인용과 함께 텍스트 추가

        Args:
            text: 텍스트
            source_info: 출처 정보 딕셔너리

        Returns:
            self (체이닝용)
        """
        source = self.manager.add_source_from_dict(source_info)
        cited_text = f"{text} {source.to_citation()}"
        self._text_parts.append(cited_text)
        return self

    def build(self, include_references: bool = True) -> str:
        """
        최종 텍스트 생성

        Args:
            include_references: 참고자료 섹션 포함 여부

        Returns:
            완성된 인사이트 텍스트
        """
        text = "\n".join(self._text_parts)

        if include_references:
            refs = self.manager.generate_references_section()
            if refs:
                text += f"\n\n{refs}"

        return text

    def reset(self) -> None:
        """빌더 초기화"""
        self.manager.clear()
        self._text_parts.clear()


# 편의 함수
def create_source_manager() -> SourceManager:
    """SourceManager 인스턴스 생성"""
    return SourceManager()


def create_insight_builder() -> InsightSourceBuilder:
    """InsightSourceBuilder 인스턴스 생성"""
    return InsightSourceBuilder()
