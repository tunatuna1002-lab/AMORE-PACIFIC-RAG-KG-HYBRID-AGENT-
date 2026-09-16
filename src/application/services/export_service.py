"""
Export Application Service
==========================
The work behind the export endpoints that is *not* HTTP and *not* document
rendering (F6):

- reference-line filtering of LLM output
- assembling everything an analyst report needs (period analysis, insights,
  charts, references) into one :class:`AnalystReportContext`

Nothing here imports FastAPI or ``src.api``; the heavy tool imports are deferred
to call time so importing this module stays cheap.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from src.application.services.external_signals_service import (
    DEFAULT_SIGNAL_BRANDS,
    get_external_signals,
)

logger = logging.getLogger(__name__)

ANALYST_REPORT_TOC = [
    "1. Executive Summary",
    "2. LANEIGE 심층 분석",
    "3. 경쟁 환경 분석",
    "4. 시장 동향",
    "5. 외부 신호 분석",
    "6. 리스크 및 기회 요인",
    "7. 전략 제언",
    "8. 참고자료 (References)",
]


# 참고자료가 본문에 포함된 경우 필터링할 패턴
REFERENCE_LINE_PATTERNS = [
    r"^\s*참고\s*자료\s*:?\s*$",  # "참고자료:" 단독 라인
    r"^\s*참고\s*:?\s*$",  # "참고:" 단독 라인
    r"^\s*References?\s*:?\s*$",  # "Reference(s):" 단독 라인
    r"^\s*출처\s*:?\s*$",  # "출처:" 단독 라인
    r"^\s*\[\d+\]\s*https?://",  # "[1] https://..." 형태
    r"^\s*-\s*\[\d+\]\s*https?://",  # "- [1] https://..." 형태
    r"^\s*\d+\.\s*https?://",  # "1. https://..." 형태
    r"^\s*•\s*https?://",  # "• https://..." 형태
    r"^\s*-\s*https?://",  # "- https://..." 형태
    r"^\s*https?://\S+\s*$",  # URL만 있는 라인
]

_REFERENCE_SECTION_RE = re.compile(r"^\s*(참고\s*자료|References?|출처)\s*:?\s*$", re.IGNORECASE)


def filter_reference_lines(content: str) -> str:
    """
    LLM 출력에서 참고자료 관련 라인을 필터링

    Args:
        content: LLM 생성 텍스트

    Returns:
        참고자료 라인이 제거된 텍스트
    """
    filtered_lines = []
    skip_until_section = False

    for line in content.split("\n"):
        # 참고자료 섹션 시작 감지
        if _REFERENCE_SECTION_RE.match(line):
            skip_until_section = True
            continue

        # 새 섹션 시작 시 스킵 해제
        if skip_until_section and re.match(r"^[■\d]", line.strip()):
            skip_until_section = False

        if skip_until_section:
            continue

        # 개별 참고자료 라인 패턴 체크
        is_reference_line = any(
            re.match(pattern, line, re.IGNORECASE) for pattern in REFERENCE_LINE_PATTERNS
        )

        if not is_reference_line:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


# --------------------------------------------------------------- analyst report
ProgressCallback = Callable[[int, str], Awaitable[None]]


class NoDataForPeriod(ValueError):
    """
    요청 기간에 스냅샷이 없다 (호출자가 404/작업 실패로 변환).

    ValueError 하위 타입인 이유: 비동기 작업 큐가 기존에 받던 예외 타입
    (``ValueError("No data found for period ...")``)을 그대로 유지한다.
    """


@dataclass
class AnalystReportContext:
    """Everything the analyst report document needs, already gathered."""

    start_date: str
    end_date: str
    analysis: Any
    report: Any
    tracker: Any
    chart_paths: dict[str, Any] = field(default_factory=dict)
    temp_dir: str | None = None
    external_signals: dict[str, Any] | None = None

    @property
    def filename(self) -> str:
        return f"AMORE_Analyst_Report_{self.start_date}_{self.end_date}.docx"

    def cleanup_charts(self) -> None:
        """생성된 임시 차트 파일 정리 (실패는 무시)."""
        if not self.temp_dir:
            return
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


async def build_analyst_report_context(
    start_date: str,
    end_date: str,
    include_charts: bool = True,
    include_external_signals: bool = True,
    progress: ProgressCallback | None = None,
) -> AnalystReportContext:
    """
    기간 분석 -> 외부 신호 -> LLM 인사이트 -> 차트 -> 참고자료 순으로 모아온다.

    Raises:
        NoDataForPeriod: 해당 기간에 데이터가 없음
    """
    import tempfile

    from src.infrastructure.container import Container
    from src.tools.calculators.period_analyzer import PeriodAnalyzer
    from src.tools.exporters.chart_generator import ChartGenerator
    from src.tools.utilities.reference_tracker import ReferenceTracker

    async def _progress(pct: int, message: str) -> None:
        if progress is not None:
            await progress(pct, message)

    await _progress(5, "기간 분석 중...")

    # 1. Period Analysis
    analyzer = PeriodAnalyzer()
    analysis = await analyzer.analyze(start_date, end_date)

    if analysis.total_days == 0:
        raise NoDataForPeriod(f"No data found for period {start_date} ~ {end_date}")

    await _progress(15, "데이터 처리 중...")

    # 2. External Signals (optional) - Tavily 뉴스 포함 + 3-Tier 분류
    external_signals = None
    external_signals_list: list = []
    if include_external_signals:
        try:
            signals_result = await get_external_signals(
                days=analysis.total_days,
                brands=list(DEFAULT_SIGNAL_BRANDS),
                include_tavily=True,
                start_date=start_date,  # 3-Tier 분류용
                end_date=end_date,  # 3-Tier 분류용
            )
            if signals_result.get("signals"):
                external_signals = signals_result
                external_signals_list = signals_result.get("signals", [])
                classified = signals_result.get("classified", {})
                logger.info(
                    f"Collected {len(external_signals_list)} signals: "
                    f"TIER1={len(classified.get('tier1_core', []))}, "
                    f"TIER2={len(classified.get('tier2_background', []))}, "
                    f"TIER3={len(classified.get('tier3_archive', []))}"
                )
        except Exception as e:
            logger.warning(f"External signal collection failed: {e}")

    await _progress(30, "AI 인사이트 생성 중...")

    # 3. Generate Insights (via Container to avoid a direct agents dependency)
    insight_agent = Container.get_period_insight_agent()
    report = await insight_agent.generate_report(analysis, external_signals=external_signals)

    await _progress(50, "차트 생성 중...")

    # 4. Generate Charts
    chart_paths: dict[str, Any] = {}
    temp_dir = None
    if include_charts:
        temp_dir = tempfile.mkdtemp()
        chart_gen = ChartGenerator(output_dir=temp_dir)
        chart_paths = chart_gen.generate_all_charts(analysis)
        logger.info(f"Generated {len(chart_paths)} charts in {temp_dir}")

    await _progress(60, "참고자료 추적 중...")

    # 5. Reference Tracker
    tracker = ReferenceTracker()
    tracker.auto_add_amazon_sources(start_date=start_date, end_date=end_date)
    if external_signals_list:
        added_refs = tracker.add_external_signals(external_signals_list)
        logger.info(f"Added {added_refs} external signal references")

    return AnalystReportContext(
        start_date=start_date,
        end_date=end_date,
        analysis=analysis,
        report=report,
        tracker=tracker,
        chart_paths=chart_paths,
        temp_dir=temp_dir,
        external_signals=external_signals,
    )
