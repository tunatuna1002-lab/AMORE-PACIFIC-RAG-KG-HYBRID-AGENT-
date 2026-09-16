"""Document registry
=================

The static catalogue of the 14 markdown documents the RAG corpus is built from
(id → filename, doc_type, keywords, freshness), the folders they are looked up
in, and the RAG block of ``config/thresholds.json``.

Moved verbatim out of ``retriever.py`` (F3 split). ``DocumentRetriever.DOCUMENTS``
is an alias of :data:`DOCUMENTS`, so existing readers are unaffected.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 설정 파일 경로 (프로젝트 루트 기준)
CONFIG_PATH = "config/thresholds.json"

# 기본 검색 결과 캐시 TTL (초)
DEFAULT_CACHE_TTL = 300

# 문서를 찾아볼 폴더 (우선순위 순). docs/guides → docs/market → docs/ir → docs → 루트
DOC_SUBDIRS = ("guides", "market", "ir")


def load_rag_config() -> dict:
    """설정 파일에서 RAG 관련 설정 로드"""
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / CONFIG_PATH

    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
                return config.get("system", {}).get("rag", {})
        except Exception:
            logger.warning("Suppressed Exception", exc_info=True)

    return {}  # 설정 없으면 기본값 사용


def candidate_paths(docs_path: Path, filename: str) -> list[Path]:
    """한 문서를 찾아볼 경로들 (docs/guides, docs/market, docs/ir, docs, 루트 순)."""
    return [docs_path / sub / filename for sub in DOC_SUBDIRS] + [
        docs_path / filename,
        docs_path.parent / filename,
    ]


# 문서 메타데이터
DOCUMENTS: dict[str, dict[str, Any]] = {
    # ========== Type D: 기존 지표 가이드 (docs/guides/) ==========
    "strategic_indicators": {
        "filename": "Strategic Indicators Definition.md",
        "description": "지표 정의 및 산출식",
        "doc_type": "metric_guide",
        "keywords": ["정의", "산출식", "SoS", "HHI", "CPI", "계산", "공식"],
        "intent_triggers": ["정의", "공식", "계산", "산출"],
        "freshness": "static",
    },
    "metric_interpretation": {
        "filename": "Metric Interpretation Guide.md",
        "description": "지표 해석 가이드",
        "doc_type": "metric_guide",
        "keywords": ["해석", "의미", "높음", "낮음", "주의사항", "함께 봐야"],
        "intent_triggers": ["의미", "해석", "뜻"],
        "freshness": "static",
    },
    "indicator_combination": {
        "filename": "Indicator Combination Playbook.md",
        "description": "지표 조합 해석 플레이북",
        "doc_type": "metric_guide",
        "keywords": ["조합", "시나리오", "액션", "전략", "상승", "하락"],
        "intent_triggers": ["조합", "같이", "함께", "시나리오"],
        "freshness": "static",
    },
    "home_insight_rules": {
        "filename": "Home Page Insight Rules.md",
        "description": "인사이트 생성 규칙",
        "doc_type": "metric_guide",
        "keywords": ["인사이트", "요약", "문구", "템플릿", "톤", "안전장치"],
        "intent_triggers": ["인사이트", "요약", "규칙"],
        "freshness": "static",
    },
    # ========== Type A: 분석 플레이북 (docs/market/) ==========
    "amazon_ranking_diagnosis": {
        "filename": "아마존 랭킹 급등 원인 역추적 보고서.md",
        "description": "BSR 급변 원인 진단 체크리스트 및 If-Then 가설 트리",
        "doc_type": "playbook",
        "keywords": [
            "순위",
            "BSR",
            "급등",
            "급락",
            "원인",
            "분석",
            "체크리스트",
            "가설",
            "재고",
            "광고",
            "프로모션",
            "리뷰",
            "가격",
        ],
        "intent_triggers": ["왜", "원인", "갑자기", "급변", "떨어", "올라", "변동"],
        "freshness": "quarterly",
    },
    "amazon_algorithm_guide": {
        "filename": "아마존 랭킹 변동 원인 분석 가이드.md",
        "description": "COSMO/Rufus 알고리즘 대응 및 심층 진단",
        "doc_type": "playbook",
        "keywords": [
            "알고리즘",
            "COSMO",
            "Rufus",
            "A10",
            "검색",
            "억제",
            "외부트래픽",
            "틱톡",
            "바이럴",
            "지식그래프",
            "BSR",
        ],
        "intent_triggers": ["알고리즘", "검색", "노출", "억제", "틱톡", "자세히"],
        "freshness": "quarterly",
    },
    # ========== Type B: 시장 인텔리전스 (docs/market/) ==========
    "kbeauty_industry": {
        "filename": "(1) K-뷰티 초격차의 서막 [풀영상] _ 창 534회 (KBS 26.1.20.) - YouTube.md",
        "description": "K-뷰티 산업 배경 (ODM, 글로벌 확장, 중국 위협)",
        "doc_type": "knowledge_base",
        "keywords": [
            "K-뷰티",
            "ODM",
            "글로벌",
            "중국",
            "미용기기",
            "맞춤화장품",
            "콘텐츠",
            "편집숍",
            "아마존",
            "초격차",
            "한국 화장품",
        ],
        "intent_triggers": ["K-뷰티", "한국 화장품", "산업", "배경", "ODM"],
        "freshness": "static",
    },
    "us_beauty_trends_weekly": {
        "filename": "미국 뷰티 트렌드 레이더.md",
        "description": "미국 주간 뷰티 트렌드 Top 10 및 LANEIGE 연결 가설",
        "doc_type": "intelligence",
        "keywords": [
            "트렌드",
            "펩타이드",
            "PDRN",
            "립케어",
            "글래스스킨",
            "세라마이드",
            "스네일뮤신",
            "나이아신아마이드",
            "키워드",
            "TikTok",
        ],
        "intent_triggers": ["트렌드", "요즘", "최근", "인기", "바이럴", "키워드"],
        "freshness": "weekly",
        "valid_period": "2025-12-21 ~ 2026-01-20",
    },
    "laneige_strategy_2026": {
        "filename": "뷰티 트렌드 분석 및 판매 전략 제안.md",
        "description": "2026년 1월 LANEIGE 아마존 판매 전략 (모닝쉐드, PDRN, 립케어)",
        "doc_type": "intelligence",
        "keywords": [
            "전략",
            "판매",
            "모닝쉐드",
            "슬리핑마스크",
            "번들",
            "립베이스팅",
            "핑크펩타이드",
            "워터뱅크",
            "크림스킨",
            "LANEIGE",
        ],
        "intent_triggers": ["전략", "어떻게", "제안", "추천", "LANEIGE"],
        "freshness": "monthly",
        "target_brand": "laneige",
    },
    # ========== Type C: 대응 가이드 (docs/market/) ==========
    "negative_issue_response": {
        "filename": "부정 이슈 조기경보 및 대응 프롬프트.md",
        "description": "브랜드별 부정 이슈 분석 및 대응 문구 (라운드랩, 아누아, 티르티르)",
        "doc_type": "response_guide",
        "keywords": [
            "부정",
            "위기",
            "리뷰",
            "대응",
            "라운드랩",
            "아누아",
            "티르티르",
            "가품",
            "리포뮬레이션",
            "끈적임",
            "산화",
            "트러블",
        ],
        "intent_triggers": ["부정", "문제", "이슈", "대응", "어떻게 해", "위기"],
        "freshness": "monthly",
        "brands_covered": ["round_lab", "anua", "tirtir", "beef_tallow"],
    },
    "laneige_influencer_map": {
        "filename": "인플루언서 맵 & 메시지 맵 생성.md",
        "description": "LANEIGE 채널별 인플루언서 분류 및 크리에이티브 훅 5선",
        "doc_type": "response_guide",
        "keywords": [
            "인플루언서",
            "틱톡",
            "유튜브",
            "레딧",
            "인스타그램",
            "메시지",
            "크리에이티브",
            "훅",
            "리스크",
            "LANEIGE",
            "마케팅",
        ],
        "intent_triggers": ["인플루언서", "마케팅", "메시지", "콘텐츠", "크리에이터"],
        "freshness": "monthly",
        "target_brand": "laneige",
    },
    # ========== Type E: IR 분기 실적 보고서 (docs/ir/) ==========
    "ir_2025_q1": {
        "filename": "AP_1Q25_EN.md",
        "description": "아모레퍼시픽 2025 Q1 실적 (COSRX 편입, Americas +102%)",
        "doc_type": "ir_report",
        "keywords": [
            "매출",
            "영업이익",
            "Revenue",
            "Operating Profit",
            "Americas",
            "COSRX",
            "LANEIGE",
            "Sulwhasoo",
            "Prime Day",
            "Western Region",
            "Greater China",
            "Q1",
            "1분기",
            "실적",
            "아모레퍼시픽",
            "Amorepacific",
            "IR",
            "earnings",
        ],
        "intent_triggers": [
            "Q1",
            "1분기",
            "2025",
            "실적",
            "매출",
            "Americas",
            "COSRX 편입",
            "IR",
            "분기",
            "earnings",
        ],
        "freshness": "quarterly",
        "quarter": "2025-Q1",
        "parent_company": "amorepacific",
    },
    "ir_2025_q2": {
        "filename": "AP_2Q25_EN.md",
        "description": "아모레퍼시픽 2025 Q2 실적 (Greater China 턴어라운드, OP +1673%)",
        "doc_type": "ir_report",
        "keywords": [
            "매출",
            "영업이익",
            "Revenue",
            "Operating Profit",
            "Americas",
            "Greater China",
            "턴어라운드",
            "LANEIGE",
            "Neo Cushion",
            "Aestura",
            "Q2",
            "2분기",
            "실적",
            "아모레퍼시픽",
            "Amorepacific",
            "IR",
            "earnings",
            "중국",
        ],
        "intent_triggers": [
            "Q2",
            "2분기",
            "2025",
            "중국",
            "Greater China",
            "실적",
            "IR",
            "분기",
            "earnings",
            "턴어라운드",
        ],
        "freshness": "quarterly",
        "quarter": "2025-Q2",
        "parent_company": "amorepacific",
    },
    "ir_2025_q3": {
        "filename": "AP_3Q25_EN.md",
        "description": "아모레퍼시픽 2025 Q3 실적 (Prime Day 2배, Americas +6.9%)",
        "doc_type": "ir_report",
        "keywords": [
            "매출",
            "영업이익",
            "Revenue",
            "Operating Profit",
            "Americas",
            "Prime Day",
            "아마존",
            "Amazon",
            "LANEIGE",
            "Illiyoon",
            "Mise-en-scène",
            "Q3",
            "3분기",
            "실적",
            "아모레퍼시픽",
            "Amorepacific",
            "IR",
            "earnings",
        ],
        "intent_triggers": [
            "Q3",
            "3분기",
            "2025",
            "Prime Day",
            "아마존",
            "실적",
            "IR",
            "분기",
            "earnings",
            "최근",
        ],
        "freshness": "quarterly",
        "quarter": "2025-Q3",
        "parent_company": "amorepacific",
    },
}
