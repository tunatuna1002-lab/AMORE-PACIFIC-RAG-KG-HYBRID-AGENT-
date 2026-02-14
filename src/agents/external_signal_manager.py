"""
External Signal Manager
외부 데이터 신호 수집 관리자

Lazy initialization for collectors:
- Tavily (뉴스)
- RSS (업계 피드)
- Reddit (소셜 미디어)
"""

from typing import Any


class ExternalSignalManager:
    """Manages external data signal collection (Tavily, RSS, Reddit).

    Uses lazy initialization for collectors.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config or {}
        self._collector = None
        self._failed_collectors: list[str] = []

    async def collect(
        self,
        query: str,
        entities: dict[str, list[str]] | None = None,
        signal_types: list[str] | None = None,
    ) -> list[Any]:
        """Collect external signals for a query.

        Args:
            query: 사용자 질문
            entities: 추출된 엔티티 (브랜드, 카테고리 등)
            signal_types: 수집할 신호 유형 리스트 (None이면 모두 수집)

        Returns:
            ExternalSignal 리스트
        """
        try:
            # 외부 신호 수집기 lazy initialization
            if self._collector is None:
                try:
                    from src.tools.collectors.external_signal_collector import (
                        ExternalSignalCollector,
                    )

                    self._collector = ExternalSignalCollector()
                    await self._collector.initialize()
                except ImportError:
                    self._failed_collectors.append("ExternalSignalCollector (import error)")
                    return []
                except Exception as e:
                    self._failed_collectors.append(f"ExternalSignalCollector ({str(e)})")
                    return []

            # 엔티티에서 브랜드/토픽 추출
            brands = []
            topics = []

            if entities:
                brands = entities.get("brands", [])
                categories = entities.get("categories", [])
                # 카테고리를 토픽으로 변환
                topics = [cat.replace("_", " ") for cat in categories]

            # 기본값 설정
            if not brands:
                brands = ["LANEIGE", "K-Beauty"]
            if not topics:
                topics = ["skincare trends", "beauty news"]

            # 신호 수집
            all_signals = []

            # Tavily 뉴스 검색 (비동기) - 검색 기간 확장
            if signal_types is None or "tavily" in signal_types:
                try:
                    tavily_signals = await self._collector.fetch_tavily_news(
                        brands=brands[:3],  # 최대 3개 브랜드
                        topics=topics[:2],  # 최대 2개 토픽
                        days=14,  # 2주로 확장 (더 많은 뉴스 수집)
                        max_results=8,  # 최대 8개로 증가
                    )
                    all_signals.extend(tavily_signals)
                except Exception as e:
                    self._failed_collectors.append(f"Tavily ({str(e)})")

            # RSS 피드 수집 (선택적)
            if signal_types is None or "rss" in signal_types:
                try:
                    keywords = brands + topics
                    rss_signals = await self._collector.fetch_all_rss_feeds(keywords=keywords[:5])
                    # 상위 3개만 추가
                    all_signals.extend(rss_signals[:3])
                except Exception:
                    # RSS는 선택적이므로 실패해도 경고만
                    pass

            # 신뢰도 * 관련성 점수로 정렬하여 상위 8개 반환
            all_signals.sort(
                key=lambda s: (
                    getattr(s, "metadata", {}).get("reliability_score", 0.7)
                    * getattr(s, "relevance_score", 0.5)
                ),
                reverse=True,
            )

            return all_signals[:8]

        except Exception as e:
            self._failed_collectors.append(f"Collection error: {str(e)}")
            return []

    def get_failed_collectors(self) -> list[str]:
        """Return list of unavailable signal collectors.

        Returns:
            실패한 수집기 이름 리스트
        """
        failed = []

        # 수집기가 초기화되지 않았으면
        if self._collector is None:
            try:
                import importlib.util

                if (
                    importlib.util.find_spec("src.tools.collectors.external_signal_collector")
                    is None
                ):
                    failed.append("External Signals (Tavily/RSS/Reddit)")
            except ImportError:
                failed.append("External Signals (Tavily/RSS/Reddit)")

        # 실행 중 실패한 수집기 추가
        failed.extend(self._failed_collectors)

        return list(set(failed))  # 중복 제거
