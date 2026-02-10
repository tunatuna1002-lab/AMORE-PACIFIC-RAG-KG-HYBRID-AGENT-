"""
AMOREPACIFIC 스타일 인사이트 포맷터
===================================

AMOREPACIFIC 경영실적 발표 자료 디자인 가이드를 기반으로
AI 생성 인사이트의 가독성을 개선하는 포맷터 모듈

주요 기능:
- 브랜드명 한글화 (LANEIGE → 라네즈)
- 섹션 헤더 스타일링 (좌측 바 + 볼드)
- 숫자 강조 (색상 힌트)
- 불릿 포인트 정리
"""

import re
from dataclasses import dataclass


@dataclass
class FormattedInsight:
    """포맷팅된 인사이트 결과"""

    markdown: str
    html: str
    stats: dict[str, int]


class AmorepacificInsightFormatter:
    """
    AMOREPACIFIC 디자인 가이드 기반 인사이트 포맷터

    디자인 원칙:
    1. 좌측 바 스타일 섹션 헤더 (▎)
    2. 브랜드명 한글 우선 표기
    3. 수치 변화 강조 (색상 힌트)
    4. 계층적 불릿 포인트 (• > -)
    5. 깔끔한 여백과 정렬
    """

    # 브랜드명 한글화 매핑 (영문 → 한글)
    BRAND_NAMES: dict[str, str] = {
        # 아모레퍼시픽 그룹
        "LANEIGE": "라네즈",
        "SULWHASOO": "설화수",
        "INNISFREE": "이니스프리",
        "ETUDE": "에뛰드",
        "ETUDE HOUSE": "에뛰드",
        "HERA": "헤라",
        "AESTURA": "에스트라",
        "PRIMERA": "프리메라",
        "IOPE": "아이오페",
        "MAMONDE": "마몽드",
        "ESPOIR": "에스쁘아",
        "AMOREPACIFIC": "아모레퍼시픽",
        # 경쟁사
        "COSRX": "코스알엑스",
        "TIRTIR": "티르티르",
        "ANUA": "아누아",
        "BEAUTY OF JOSEON": "조선미녀",
        "SKIN1004": "스킨1004",
        "MISSHA": "미샤",
        "SOME BY MI": "썸바이미",
        "TORRIDEN": "토리든",
        "NUMBUZIN": "넘버즈인",
        "MEDICUBE": "메디큐브",
        "VT": "VT",
        "RARE BEAUTY": "레어뷰티",
        "FENTY BEAUTY": "펜티뷰티",
        "CLINIQUE": "클리니크",
        "ESTEE LAUDER": "에스티로더",
        "LANCOME": "랑콤",
        "SHISEIDO": "시세이도",
    }

    # 섹션 헤더 스타일 변환 패턴
    SECTION_PATTERNS = [
        # ## 📌 오늘의 핵심 → ▎오늘의 핵심
        (r"^##\s*📌?\s*(.+)$", r"▎**\1**"),
        (r"^##\s*🔍?\s*(.+)$", r"▎**\1**"),
        (r"^##\s*⚠️?\s*(.+)$", r"▎**\1**"),
        (r"^##\s*💡?\s*(.+)$", r"▎**\1**"),
        (r"^##\s*📚?\s*(.+)$", r"▎**\1**"),
        # ### Layer N: 제목 → **Layer N: 제목**
        (r"^###\s*(Layer\s*\d+:\s*.+)$", r"**\1**"),
        # 기타 ## 헤더
        (r"^##\s+(.+)$", r"▎**\1**"),
        # ### 소제목 → **소제목**
        (r"^###\s+(.+)$", r"**\1**"),
    ]

    # 숫자 포맷팅 패턴
    NUMBER_PATTERNS = [
        # +숫자% → **+숫자%**
        (r"(?<!\*)\+(\d+\.?\d*)\s*%", r"**+\1%**"),
        (r"(?<!\*)-(\d+\.?\d*)\s*%", r"**-\1%**"),
        # +숫자%p (percentage point)
        (r"(?<!\*)\+(\d+\.?\d*)\s*%p", r"**+\1%p**"),
        (r"(?<!\*)-(\d+\.?\d*)\s*%p", r"**-\1%p**"),
        # #순위 → **#순위**
        (r"#(\d+)", r"**#\1**"),
        # 억원 단위
        (r"(\d+(?:,\d{3})*)\s*억원", r"**\1억원**"),
    ]

    def __init__(self, enable_korean_brands: bool = True):
        """
        Args:
            enable_korean_brands: 브랜드명 한글화 활성화 여부
        """
        self.enable_korean_brands = enable_korean_brands
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """정규식 패턴 컴파일"""
        self._section_re = [(re.compile(p, re.MULTILINE), r) for p, r in self.SECTION_PATTERNS]
        self._number_re = [(re.compile(p), r) for p, r in self.NUMBER_PATTERNS]

        # 브랜드명 패턴 (대소문자 무시)
        self._brand_re = {}
        for en, ko in self.BRAND_NAMES.items():
            # 단어 경계로 정확히 매칭
            pattern = re.compile(rf"\b{re.escape(en)}\b", re.IGNORECASE)
            self._brand_re[pattern] = ko

    def format_markdown(self, content: str) -> str:
        """
        마크다운 텍스트를 AMOREPACIFIC 스타일로 포맷팅

        Args:
            content: 원본 마크다운 텍스트

        Returns:
            포맷팅된 마크다운 텍스트
        """
        result = content

        # 1. 섹션 헤더 변환
        for pattern, replacement in self._section_re:
            result = pattern.sub(replacement, result)

        # 2. 브랜드명 한글화
        if self.enable_korean_brands:
            result = self._convert_brand_names(result)

        # 3. 숫자 강조
        result = self._format_numbers(result)

        # 4. 불릿 포인트 정리
        result = self._format_bullets(result)

        return result

    def _convert_brand_names(self, content: str) -> str:
        """브랜드명을 한글로 변환"""
        result = content
        for pattern, korean in self._brand_re.items():
            result = pattern.sub(f"**{korean}**", result)
        return result

    def _format_numbers(self, content: str) -> str:
        """숫자 강조 포맷팅"""
        result = content
        for pattern, replacement in self._number_re:
            result = pattern.sub(replacement, result)
        return result

    def _format_bullets(self, content: str) -> str:
        """불릿 포인트 정리"""
        lines = content.split("\n")
        result_lines = []

        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            # 첫 번째 레벨 불릿 (- 또는 *)
            if stripped.startswith("- ") or stripped.startswith("* "):
                # 인덴트에 따라 불릿 스타일 결정
                if indent == 0:
                    # 최상위: • 사용
                    result_lines.append("• " + stripped[2:])
                else:
                    # 하위: - 유지 (들여쓰기 포함)
                    result_lines.append("  - " + stripped[2:])
            else:
                result_lines.append(line)

        return "\n".join(result_lines)

    def format_html(self, markdown: str) -> str:
        """
        포맷팅된 마크다운을 HTML로 변환

        Dashboard 렌더링용 HTML 생성

        Args:
            markdown: 포맷팅된 마크다운

        Returns:
            HTML 문자열
        """
        html = markdown

        # 섹션 헤더: ▎**제목** → <div class="insight-section-header">제목</div>
        html = re.sub(r"▎\*\*(.+?)\*\*", r'<div class="insight-section-header">\1</div>', html)

        # Layer 헤더: **Layer N: 제목** → <div class="insight-layer-header">Layer N: 제목</div>
        html = re.sub(
            r"\*\*(Layer\s*\d+:\s*[^*]+)\*\*", r'<div class="insight-layer-header">\1</div>', html
        )

        # 긍정 수치: **+숫자** → <span class="insight-metric-positive">+숫자</span>
        html = re.sub(
            r"\*\*(\+\d+\.?\d*%?p?)\*\*", r'<span class="insight-metric-positive">\1</span>', html
        )

        # 부정 수치: **-숫자** → <span class="insight-metric-negative">-숫자</span>
        html = re.sub(
            r"\*\*(-\d+\.?\d*%?p?)\*\*", r'<span class="insight-metric-negative">\1</span>', html
        )

        # 순위: **#숫자** → <span class="insight-rank">#숫자</span>
        html = re.sub(r"\*\*(#\d+)\*\*", r'<span class="insight-rank">\1</span>', html)

        # 브랜드명 볼드: **한글브랜드** → <span class="insight-brand-name">한글브랜드</span>
        korean_brands = "|".join(self.BRAND_NAMES.values())
        html = re.sub(
            rf"\*\*({korean_brands})\*\*", r'<span class="insight-brand-name">\1</span>', html
        )

        # 나머지 볼드: **텍스트** → <strong>텍스트</strong>
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)

        # 불릿 포인트: • → <span class="insight-bullet">•</span>
        html = re.sub(r"^• ", '<span class="insight-bullet">• </span>', html, flags=re.MULTILINE)

        # 서브 불릿: - → <span class="insight-sub-bullet">-</span>
        html = re.sub(
            r"^  - ", '<span class="insight-sub-bullet">  - </span>', html, flags=re.MULTILINE
        )

        # 줄바꿈 → <br>
        html = html.replace("\n", "<br>\n")

        return html

    def format(self, content: str) -> FormattedInsight:
        """
        전체 포맷팅 수행

        Args:
            content: 원본 마크다운 텍스트

        Returns:
            FormattedInsight 객체 (markdown, html, stats)
        """
        markdown = self.format_markdown(content)
        html = self.format_html(markdown)

        # 통계 수집
        stats = {
            "brands_converted": len(re.findall(r'class="insight-brand-name"', html)),
            "metrics_highlighted": len(re.findall(r'class="insight-metric-', html)),
            "sections_formatted": len(re.findall(r'class="insight-section-header"', html)),
            "bullets_formatted": len(re.findall(r'class="insight-bullet"', html)),
        }

        return FormattedInsight(markdown=markdown, html=html, stats=stats)

    @staticmethod
    def get_css() -> str:
        """
        Dashboard에서 사용할 CSS 스타일 반환

        Returns:
            CSS 문자열
        """
        return """
/* AMOREPACIFIC 인사이트 스타일 */
.insight-container {
    font-family: 'Arita Dotum', 'Noto Sans KR', sans-serif;
    line-height: 1.8;
    color: var(--pacific-blue, #001C58);
    font-size: 14px;
}

.insight-section-header {
    display: flex;
    align-items: center;
    margin: 24px 0 16px 0;
    font-size: 16px;
    font-weight: 700;
    color: var(--pacific-blue, #001C58);
}

.insight-section-header::before {
    content: '';
    width: 4px;
    height: 20px;
    background: var(--pacific-blue, #001C58);
    margin-right: 12px;
    border-radius: 2px;
    flex-shrink: 0;
}

.insight-layer-header {
    font-size: 14px;
    font-weight: 600;
    color: var(--amore-blue, #1F5795);
    margin: 16px 0 8px 0;
    padding-left: 16px;
    border-left: 2px solid var(--amore-blue, #1F5795);
}

.insight-metric-positive {
    color: var(--amore-blue, #1F5795);
    font-weight: 700;
}

.insight-metric-negative {
    color: var(--danger, #8B2635);
    font-weight: 700;
}

.insight-rank {
    color: var(--pacific-blue, #001C58);
    font-weight: 700;
}

.insight-brand-name {
    font-weight: 700;
    color: var(--pacific-blue, #001C58);
}

.insight-bullet {
    color: var(--pacific-blue, #001C58);
}

.insight-sub-bullet {
    color: var(--text-secondary, #7D7D7D);
}

/* 인사이트 카드 컨테이너 */
.insight-card {
    background: var(--bg-card, #FFFFFF);
    border-radius: 8px;
    padding: 24px;
    box-shadow: var(--shadow-sm, 0 1px 3px rgba(0, 28, 88, 0.08));
}

/* 액션 아이템 스타일 */
.insight-action-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 8px 12px;
    background: rgba(31, 87, 149, 0.04);
    border-radius: 4px;
    margin-bottom: 8px;
}

.insight-action-priority {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: 500;
    flex-shrink: 0;
}

.insight-action-priority.high {
    background: rgba(139, 38, 53, 0.1);
    color: #8B2635;
}

.insight-action-priority.medium {
    background: rgba(196, 169, 98, 0.15);
    color: #8B7355;
}

.insight-action-priority.low {
    background: rgba(31, 87, 149, 0.1);
    color: #1F5795;
}

/* 참고자료 스타일 */
.insight-reference {
    font-size: 12px;
    color: var(--text-secondary, #7D7D7D);
    padding-left: 16px;
    margin: 4px 0;
}

.insight-reference a {
    color: var(--amore-blue, #1F5795);
    text-decoration: none;
}

.insight-reference a:hover {
    text-decoration: underline;
}
"""


# 싱글톤 인스턴스
_formatter: AmorepacificInsightFormatter | None = None


def get_formatter() -> AmorepacificInsightFormatter:
    """포맷터 싱글톤 인스턴스 반환"""
    global _formatter
    if _formatter is None:
        _formatter = AmorepacificInsightFormatter()
    return _formatter


def format_insight(content: str) -> str:
    """
    인사이트 텍스트를 AMOREPACIFIC 스타일로 포맷팅

    Args:
        content: 원본 마크다운 텍스트

    Returns:
        포맷팅된 마크다운 텍스트
    """
    return get_formatter().format_markdown(content)


def format_insight_html(content: str) -> str:
    """
    인사이트 텍스트를 HTML로 변환

    Args:
        content: 원본 마크다운 텍스트

    Returns:
        HTML 문자열
    """
    formatter = get_formatter()
    markdown = formatter.format_markdown(content)
    return formatter.format_html(markdown)
