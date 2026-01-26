"""
참고자료 추적기
==============
보고서에 사용된 모든 출처를 추적하고 형식화

## 출처 유형
- DATA: 데이터 출처 (Amazon, IR, Google Sheets)
- ARTICLE: 기사 (Allure, WWD, People 등)
- REPORT: 리서치 보고서 (증권사, 시장조사기관)
- SOCIAL: 소셜 미디어 (Reddit, TikTok)
- ACADEMIC: 학술 자료 (논문)

## 사용 예시

```python
tracker = ReferenceTracker()

# 데이터 출처
tracker.add_data_source(
    title="Skin Care",
    source="Amazon Best Sellers",
    date_range="2026-01-14 ~ 2026-01-25"
)

# 기사
tracker.add_article(
    title="2026 Beauty Trends",
    source="Allure",
    date="2026-01-10",
    url="https://allure.com/..."
)

# 소셜 미디어
tracker.add_social(
    title="Best lip products for winter",
    platform="Reddit",
    date="2026-01-15",
    subreddit="SkincareAddiction",
    upvotes=2400
)

# 참고자료 섹션 생성
print(tracker.format_section())
```
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import json


class ReferenceType(Enum):
    """참고자료 유형"""
    DATA = "data"           # 데이터 출처
    ARTICLE = "article"     # 기사
    REPORT = "report"       # 리서치 보고서
    SOCIAL = "social"       # 소셜 미디어
    ACADEMIC = "academic"   # 학술 자료


@dataclass
class Reference:
    """단일 참고자료"""
    ref_type: ReferenceType
    title: str
    source: str               # 출처 기관/사이트
    date: Optional[str] = None      # YYYY-MM-DD or YYYY-MM
    url: Optional[str] = None
    author: Optional[str] = None

    # 추가 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 내부 ID (자동 생성)
    ref_id: str = ""

    def __post_init__(self):
        if not self.ref_id:
            # 유형별 접두사: D=Data, A=Article, R=Report, S=Social, P=Paper
            prefix_map = {
                ReferenceType.DATA: "D",
                ReferenceType.ARTICLE: "A",
                ReferenceType.REPORT: "R",
                ReferenceType.SOCIAL: "S",
                ReferenceType.ACADEMIC: "P"
            }
            self.ref_id = f"{prefix_map.get(self.ref_type, 'X')}{id(self) % 1000:03d}"

    def format_citation(self) -> str:
        """형식화된 인용문 반환"""
        parts = [f"[{self.ref_id}]"]

        if self.ref_type == ReferenceType.DATA:
            # [D1] Amazon Best Sellers - Skin Care, 2026-01-14 ~ 2026-01-25
            parts.append(f" {self.source} - {self.title}")
            if self.date:
                parts.append(f", {self.date}")

        elif self.ref_type == ReferenceType.ARTICLE:
            # [A1] "Title", Source, Date
            #      URL
            parts.append(f' "{self.title}", {self.source}')
            if self.date:
                parts.append(f", {self.date}")

        elif self.ref_type == ReferenceType.REPORT:
            # [R1] "Title", Source, Date
            parts.append(f' "{self.title}", {self.source}')
            if self.date:
                parts.append(f", {self.date}")

        elif self.ref_type == ReferenceType.SOCIAL:
            # [S1] r/subreddit, "Title", Date, engagement
            if "subreddit" in self.metadata:
                parts.append(f" r/{self.metadata['subreddit']}, ")
            elif "platform" in self.metadata:
                parts.append(f" {self.metadata['platform']}, ")
            else:
                parts.append(" ")
            parts.append(f'"{self.title}"')
            if self.date:
                parts.append(f", {self.date}")
            if "upvotes" in self.metadata:
                parts.append(f", {self.metadata['upvotes']} upvotes")
            if "views" in self.metadata:
                parts.append(f", {self.metadata['views']} views")

        elif self.ref_type == ReferenceType.ACADEMIC:
            # [P1] Author et al., "Title", Journal, Year
            if self.author:
                parts.append(f" {self.author}, ")
            else:
                parts.append(" ")
            parts.append(f'"{self.title}"')
            if self.source:
                parts.append(f", {self.source}")
            if self.date:
                parts.append(f", {self.date[:4]}")  # Year only

        return "".join(parts)


class ReferenceTracker:
    """참고자료 추적기"""

    def __init__(self):
        self.references: List[Reference] = []
        self._id_counter = {t: 0 for t in ReferenceType}

    def add_reference(
        self,
        ref_type: ReferenceType,
        title: str,
        source: str,
        date: Optional[str] = None,
        url: Optional[str] = None,
        author: Optional[str] = None,
        **metadata
    ) -> Reference:
        """참고자료 추가"""
        self._id_counter[ref_type] += 1
        prefix_map = {
            ReferenceType.DATA: "D",
            ReferenceType.ARTICLE: "A",
            ReferenceType.REPORT: "R",
            ReferenceType.SOCIAL: "S",
            ReferenceType.ACADEMIC: "P"
        }
        ref_id = f"{prefix_map[ref_type]}{self._id_counter[ref_type]}"

        ref = Reference(
            ref_type=ref_type,
            title=title,
            source=source,
            date=date,
            url=url,
            author=author,
            metadata=metadata,
            ref_id=ref_id
        )
        self.references.append(ref)
        return ref

    # Convenience methods for each type
    def add_data_source(
        self,
        title: str,
        source: str,
        date_range: str = None,
        **kwargs
    ) -> Reference:
        """데이터 출처 추가"""
        return self.add_reference(
            ReferenceType.DATA, title, source, date=date_range, **kwargs
        )

    def add_article(
        self,
        title: str,
        source: str,
        date: str,
        url: str = None,
        **kwargs
    ) -> Reference:
        """기사 추가"""
        return self.add_reference(
            ReferenceType.ARTICLE, title, source, date=date, url=url, **kwargs
        )

    def add_report(
        self,
        title: str,
        source: str,
        date: str = None,
        **kwargs
    ) -> Reference:
        """리서치 보고서 추가"""
        return self.add_reference(
            ReferenceType.REPORT, title, source, date=date, **kwargs
        )

    def add_social(
        self,
        title: str,
        platform: str,
        date: str = None,
        url: str = None,
        **kwargs
    ) -> Reference:
        """소셜 미디어 추가"""
        return self.add_reference(
            ReferenceType.SOCIAL, title, platform, date=date, url=url, **kwargs
        )

    def add_academic(
        self,
        title: str,
        journal: str,
        author: str = None,
        year: str = None,
        **kwargs
    ) -> Reference:
        """학술 자료 추가"""
        return self.add_reference(
            ReferenceType.ACADEMIC, title, journal, date=year, author=author, **kwargs
        )

    def get_by_type(self, ref_type: ReferenceType) -> List[Reference]:
        """유형별 참고자료 조회"""
        return [r for r in self.references if r.ref_type == ref_type]

    def get_by_id(self, ref_id: str) -> Optional[Reference]:
        """ID로 참고자료 조회"""
        for r in self.references:
            if r.ref_id == ref_id:
                return r
        return None

    def format_section(self) -> str:
        """전체 참고자료 섹션 형식화"""
        sections = []

        # 8.1 데이터 출처
        data_refs = self.get_by_type(ReferenceType.DATA)
        if data_refs:
            lines = ["8.1 데이터 출처 (Data Sources)"]
            for ref in data_refs:
                lines.append(f"  {ref.format_citation()}")
                if ref.url:
                    lines.append(f"       {ref.url}")
            sections.append("\n".join(lines))

        # 8.2 기사
        article_refs = self.get_by_type(ReferenceType.ARTICLE)
        if article_refs:
            lines = ["8.2 기사 (News Articles)"]
            for ref in article_refs:
                lines.append(f"  {ref.format_citation()}")
                if ref.url:
                    lines.append(f"       {ref.url}")
            sections.append("\n".join(lines))

        # 8.3 리서치 보고서
        report_refs = self.get_by_type(ReferenceType.REPORT)
        if report_refs:
            lines = ["8.3 리서치 보고서 (Research Reports)"]
            for ref in report_refs:
                lines.append(f"  {ref.format_citation()}")
                if ref.url:
                    lines.append(f"       {ref.url}")
            sections.append("\n".join(lines))

        # 8.4 소셜 미디어
        social_refs = self.get_by_type(ReferenceType.SOCIAL)
        if social_refs:
            lines = ["8.4 소셜 미디어 (Social Media)"]
            for ref in social_refs:
                lines.append(f"  {ref.format_citation()}")
                if ref.url:
                    lines.append(f"       {ref.url}")
            sections.append("\n".join(lines))

        # 8.5 학술 자료
        academic_refs = self.get_by_type(ReferenceType.ACADEMIC)
        if academic_refs:
            lines = ["8.5 학술 자료 (Academic Papers)"]
            for ref in academic_refs:
                lines.append(f"  {ref.format_citation()}")
                if ref.url:
                    lines.append(f"       {ref.url}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return {
            "references": [
                {
                    "ref_id": r.ref_id,
                    "type": r.ref_type.value,
                    "title": r.title,
                    "source": r.source,
                    "date": r.date,
                    "url": r.url,
                    "author": r.author,
                    "metadata": r.metadata
                }
                for r in self.references
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceTracker":
        """딕셔너리에서 복원"""
        tracker = cls()
        for ref_data in data.get("references", []):
            ref_type = ReferenceType(ref_data["type"])
            tracker.add_reference(
                ref_type=ref_type,
                title=ref_data["title"],
                source=ref_data["source"],
                date=ref_data.get("date"),
                url=ref_data.get("url"),
                author=ref_data.get("author"),
                **ref_data.get("metadata", {})
            )
        return tracker

    def auto_add_amazon_sources(
        self,
        start_date: str,
        end_date: str,
        categories: List[str] = None
    ):
        """Amazon 데이터 소스 자동 추가"""
        categories = categories or [
            "Beauty & Personal Care",
            "Skin Care",
            "Lip Care",
            "Lip Makeup",
            "Face Powder"
        ]
        for cat in categories:
            self.add_data_source(
                title=cat,
                source="Amazon Best Sellers",
                date_range=f"{start_date} ~ {end_date}"
            )

    def auto_add_ir_report(self, quarter: str, year: str = "2025"):
        """IR 보고서 자동 추가"""
        self.add_data_source(
            title=f"아모레퍼시픽 {year}년 {quarter} 실적 발표",
            source="AMOREPACIFIC IR",
            date_range=f"{year}-{quarter}"
        )

    def add_external_signals(self, signals: List[Any]) -> int:
        """
        외부 신호를 참고자료에 자동 추가

        ExternalSignal 리스트를 받아서 Tier에 따라 적절한 ReferenceType으로 변환하여 추가합니다.

        Tier → ReferenceType 매핑:
        - tier3_authority → ARTICLE (뉴스)
        - tier2_validation → SOCIAL (Reddit/YouTube)
        - tier1_viral → SOCIAL (TikTok/Instagram)
        - tavily_news → ARTICLE

        Args:
            signals: ExternalSignal 리스트

        Returns:
            추가된 참고자료 수
        """
        added = 0

        # Tier/Source → ReferenceType 매핑
        tier_to_type = {
            "tier3_authority": ReferenceType.ARTICLE,
            "tier2_validation": ReferenceType.SOCIAL,
            "tier1_viral": ReferenceType.SOCIAL,
            "tier4_pr": ReferenceType.ARTICLE,
        }

        source_to_type = {
            "tavily_news": ReferenceType.ARTICLE,
            "rss": ReferenceType.ARTICLE,
            "reddit": ReferenceType.SOCIAL,
            "youtube": ReferenceType.SOCIAL,
            "tiktok": ReferenceType.SOCIAL,
            "instagram": ReferenceType.SOCIAL,
            "twitter": ReferenceType.SOCIAL,
            "x": ReferenceType.SOCIAL,
        }

        for signal in signals:
            # URL 중복 체크
            signal_url = getattr(signal, 'url', '')
            if signal_url and self._is_duplicate(signal_url):
                continue

            # Tier 또는 Source에서 ReferenceType 결정
            tier = getattr(signal, 'tier', 'unknown')
            source = getattr(signal, 'source', 'unknown').lower()

            ref_type = tier_to_type.get(tier)
            if not ref_type:
                # Tier에서 못 찾으면 source로 판단
                for key, rtype in source_to_type.items():
                    if key in source:
                        ref_type = rtype
                        break

            if not ref_type:
                # 기본값: ARTICLE
                ref_type = ReferenceType.ARTICLE

            # 메타데이터에서 추가 정보 추출
            metadata = getattr(signal, 'metadata', {}) or {}
            reliability_score = metadata.get('reliability_score')

            # 소셜 미디어 특화 메타데이터
            extra_metadata = {}
            if ref_type == ReferenceType.SOCIAL:
                if 'reddit' in source:
                    # Reddit 게시물
                    if 'subreddit' in metadata:
                        extra_metadata['subreddit'] = metadata['subreddit']
                    extra_metadata['platform'] = 'Reddit'
                elif 'youtube' in source:
                    extra_metadata['platform'] = 'YouTube'
                    if 'views' in metadata:
                        extra_metadata['views'] = metadata['views']
                elif 'tiktok' in source:
                    extra_metadata['platform'] = 'TikTok'
                    if 'views' in metadata:
                        extra_metadata['views'] = metadata['views']
                else:
                    extra_metadata['platform'] = source.replace('_', ' ').title()

            if reliability_score is not None:
                extra_metadata['reliability_score'] = reliability_score

            # 참고자료 추가
            self.add_reference(
                ref_type=ref_type,
                title=getattr(signal, 'title', 'Unknown'),
                source=source.replace('_', ' ').title(),
                date=getattr(signal, 'published_at', None),
                url=signal_url,
                **extra_metadata
            )
            added += 1

        return added

    def _is_duplicate(self, url: str) -> bool:
        """URL 중복 여부 확인"""
        if not url:
            return False
        for ref in self.references:
            if ref.url and ref.url == url:
                return True
        return False


# Example usage
if __name__ == "__main__":
    tracker = ReferenceTracker()

    # Add data sources
    tracker.add_data_source(
        title="Skin Care",
        source="Amazon Best Sellers",
        date_range="2026-01-14 ~ 2026-01-25"
    )

    # Add article
    tracker.add_article(
        title="2026 Beauty Trends: What to Watch",
        source="Allure",
        date="2026-01-10",
        url="https://www.allure.com/story/2026-beauty-trends"
    )

    # Add social media
    tracker.add_social(
        title="Best lip products for winter",
        platform="Reddit",
        date="2026-01-15",
        subreddit="SkincareAddiction",
        upvotes=2400,
        url="https://reddit.com/r/SkincareAddiction/..."
    )

    # Auto-add Amazon sources
    tracker.auto_add_amazon_sources("2026-01-01", "2026-01-25")

    # Print formatted section
    print(tracker.format_section())

    # Export to JSON
    print("\n" + "="*60 + "\n")
    print(json.dumps(tracker.to_dict(), indent=2, ensure_ascii=False))
