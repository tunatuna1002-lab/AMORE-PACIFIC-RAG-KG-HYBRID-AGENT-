#!/usr/bin/env python3
"""골든셋 172문항을 골드 수치의 **검증 근거**로 분류한다 (멱등, --dry-run 지원).

배경 (docs/eval/rag-eval-review-2026-09-06.md §2-c)
--------------------------------------------------
데이터형 문항의 골드 수치가 실제 크롤 DB와 맞지 않는다(LANEIGE Lip Care SoS
골드 5.2% vs 실측 2.0% 등). 원인은 골든셋이 "문서에서 나온 정적인 답"과
"특정 시점의 DB 수치"와 "도메인 추정치"를 구분하지 않은 채 한 층에 섞어 둔 것이다.
구분해야 각 층에 맞는 채점을 할 수 있다.

분류 기준
---------
document
    답이 코퍼스 문서(정의·해석·전략·IR 원문)에서 나오고 수치가 시간에 따라
    변하지 않는다. 예: lg041 "SoS란?", IR 12문항.
snapshot
    답이 크롤 DB(brand_metrics·market_metrics·raw_data·product_metrics)의
    특정 시점 수치다. 골드 답이 **현재 시점의 DB 확인 가능 수치를 단정**하면
    질문이 전략형이어도 snapshot으로 본다 — 그 단정은 데이터와 맞아야 한다.
    예: lg048 "현재 SoS", lg071 "현재 순위".
domain_expectation
    답이 DB에도 문서에도 없다. 시장 규모·다년 추이·성장률·반사실 예측·상관계수 등.
    예: lg126 "2025년 시장 규모", lg189 "5년 후 전망", lg183 TIRTIR Face Powder.

snapshot 문항에는 `as_of = 2026-08-31`(brand_metrics·market_metrics의 최신
스냅샷)을 단다. 4단계(`scripts/refresh_golden_snapshot_values.py`)가 이 날짜의
DB에서 `expected_values`를 생성하며, **DB에 값이 없는 문항은 그 단계에서
domain_expectation으로 강등하거나 "해당 기간 데이터 없음"으로 바꾼다.**
이 스크립트는 값을 만들지 않는다 — 층만 나눈다.

실행 순서
--------
이 스크립트 → `scripts/refresh_golden_snapshot_values.py`. 4단계에서 DB에 관측이
없어 강등된 문항은 `DEMOTED_BY_DATA`에 기록돼 있어, 어느 순서로 다시 돌려도
결과가 같다(둘 다 멱등).

사용법:
    python3 scripts/classify_golden_sources.py --dry-run
    python3 scripts/classify_golden_sources.py
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "eval" / "data" / "golden" / "laneige_golden_v2.jsonl"

# brand_metrics·market_metrics의 최신 스냅샷 날짜
#   sqlite3 data/amore_data.db "select max(snapshot_date) from brand_metrics"  -> 2026-08-31
AS_OF = "2026-08-31"

D = "document"
S = "snapshot"
X = "domain_expectation"

# 문항별 분류와 근거. 근거는 "왜 그 층인가"만 적는다 — 수치 검증은 4단계가 한다.
CLASSIFICATION: dict[str, tuple[str, str]] = {
    # ---- metric (30) ------------------------------------------------------
    "lg041": (D, "SoS 정의 — Strategic Indicators Definition.md, 시간 불변"),
    "lg042": (D, "HHI 산출식 — 정의"),
    "lg043": (D, "CPI 정의·계산법 — 정의"),
    "lg044": (D, "'SoS 5%는 좋은가' — Metric Interpretation Guide의 해석 기준"),
    "lg045": (D, "HHI 0.15는 정의상 임계값(중간 집중도 하한), 관측치가 아님"),
    "lg046": (D, "SoS·HHI 조합 해석 — Indicator Combination Playbook"),
    "lg047": (D, "CPI·SoS 동반 하락 대응 전략 — 플레이북"),
    "lg048": (S, "현재 LANEIGE lip_care SoS — brand_metrics"),
    "lg049": (S, "lip_care HHI 현황 — market_metrics"),
    "lg050": (S, "SoS·HHI·CPI 현재 수치를 종합해 단정"),
    "lg051": (S, "LANEIGE CPI — brand_metrics.cpi (NULL이면 4단계에서 재분류)"),
    "lg052": (D, "Churn Rate의 의미 — 지표 해석"),
    "lg053": (D, "SoS·HHI 통합 해석 프레임 — 플레이북"),
    "lg054": (X, "SoS 1%→매출 2~3% 상관. 매출 데이터가 시스템에 없어 검증 불가"),
    "lg055": (S, "lip_care vs lip_makeup HHI 비교 — 둘 다 market_metrics에 있음"),
    "lg056": (D, "CPI<1.0일 때 전략 — 플레이북"),
    "lg057": (S, "BIODANCE의 lip_care 점유율 — DB에서 부재/존재를 확인 가능"),
    "lg058": (D, "카테고리별 변동성 특성의 개념 설명"),
    "lg059": (D, "'3개월 연속 하락이라면' 가정형 원인 진단 — 플레이북"),
    "lg060": (S, "beauty 카테고리 SoS 상위 브랜드 — brand_metrics"),
    "lg061": (D, "이탈률→SoS 메커니즘 설명"),
    "lg062": (X, "TIRTIR의 face_powder HHI 기여(반사실). DB에 해당 조합 데이터 없음"),
    "lg063": (X, "SoS-리뷰 수 상관관계. 상관 자체는 DB로 검증 불가"),
    "lg064": (D, "HHI 계산 입력 데이터 — 정의"),
    "lg065": (D, "CPI 인하 시 효과 — 반사실 전략 해석"),
    "lg066": (S, "Beauty of Joseon skin_care SoS — brand_metrics"),
    "lg067": (X, "ANUA 진입이 HHI에 미친 영향 — 시계열 반사실"),
    "lg068": (S, "face_powder 브랜드별 SoS 현황 — brand_metrics"),
    "lg069": (D, "이탈률 공식·모니터링 주기 — 정의"),
    "lg070": (S, "MEDICUBE skin_care SoS — brand_metrics"),
    # ---- product (30) -----------------------------------------------------
    "lg071": (S, "Lip Sleeping Mask 현재 순위 — raw_data.rank"),
    "lg072": (S, "lip_care Top 10 목록 — raw_data"),
    "lg073": (S, "LANEIGE 제품 평균 가격 — raw_data.price"),
    "lg074": (S, "경쟁 제품 대비 가격 — raw_data.price"),
    "lg075": (S, "평점 4.5 이상 제품 — raw_data.rating"),
    "lg076": (S, "리뷰 수 최다 제품 — raw_data.reviews_count"),
    "lg077": (S, "순위 상승 제품 — product_metrics.rank_change"),
    "lg078": (S, "신규 진입 제품 — products.first_seen_date"),
    "lg079": (S, "제품 ASIN — raw_data.asin"),
    "lg080": (X, "계절별 순위 패턴. 다년 계절성을 볼 스냅샷이 없음"),
    "lg081": (D, "경쟁 제품 포지셔닝 서술 — 외부 브랜드 가격 중심"),
    "lg082": (S, "Neo Cushion의 face_powder 순위 — raw_data.rank"),
    "lg083": (S, "LANEIGE 최상위 BSR 제품 — raw_data"),
    "lg084": (D, "Cream Skin Toner 경쟁 제품 가격 비교 — 코퍼스 밖 브랜드 중심"),
    "lg085": (X, "제품별 평점 트렌드 — 시계열 추이"),
    "lg086": (S, "Rare Beauty lip_makeup 제품 순위 — raw_data"),
    "lg087": (S, "e.l.f. face_powder 제품 순위 — raw_data"),
    "lg088": (X, "신규 SKU 투입 시 SoS 예측 — 반사실"),
    "lg089": (S, "COSRX Snail Mucin skin_care 순위 — raw_data"),
    "lg090": (D, "TIRTIR 쿠션 성공 요인 — 정성 분석"),
    "lg091": (S, "Lip Sleeping Mask 향별 순위 — raw_data"),
    "lg092": (D, "Water Bank 라인 제품 구성 — 제품 정보"),
    "lg093": (S, "Subscribe & Save 비율 — raw_data.is_subscribe_save"),
    "lg094": (D, "가성비 평가 — 정성 판단"),
    "lg095": (S, "face_powder Top 5 — raw_data"),
    "lg096": (D, "Amazon 번들 구성 — 제품 정보"),
    "lg097": (D, "브랜드의 시장 위치 — 정성"),
    "lg098": (D, "LANEIGE vs COSRX 브랜드 비교 — 정성"),
    "lg099": (D, "K-Beauty 최강 브랜드 — 카테고리 횡단 정성 순위"),
    "lg100": (D, "K-Beauty vs 미국 로컬 경쟁 구도 — 정성"),
    # ---- brand (25) -------------------------------------------------------
    "lg101": (D, "브랜드 아이덴티티"),
    "lg102": (D, "아모레퍼시픽 포트폴리오 내 위치"),
    "lg103": (D, "NYX와의 경쟁 관계 — 정성"),
    "lg104": (D, "Maybelline과의 전략 차이 — 정성"),
    "lg105": (X, "브랜드 충성도(재구매율·구독 비율). 해당 데이터가 DB에 없음"),
    "lg106": (X, "소셜 미디어와 BSR의 연관성 — 상관 주장"),
    "lg107": (D, "BIODANCE와의 브랜드 비교 — 정성"),
    "lg108": (D, "카테고리 확장 전략 — 전략"),
    "lg109": (S, "Aquaphor가 lip_care 1~2위라는 단정 — brand_metrics/raw_data"),
    "lg110": (X, "2018년부터의 성장 히스토리. DB 스냅샷 범위 밖"),
    "lg111": (D, "Beauty of Joseon과의 포지셔닝 비교 — 정성"),
    "lg112": (D, "글로벌 전략과 Amazon 전략의 연계 — 전략"),
    "lg113": (S, "Burt's Bees lip_care SoS 15~20% 단정 — brand_metrics"),
    "lg114": (D, "ChapStick과의 차별점 — 정성"),
    "lg115": (D, "Face Makeup 성공 전략 — 전략"),
    "lg116": (D, "eos의 전략과 관계 — 정성"),
    "lg117": (D, "ANUA 성장 전략 — 정성"),
    "lg118": (D, "아모레퍼시픽 브랜드 시너지 — 전략"),
    "lg119": (X, "Prime Day 성과. 이벤트 효과를 분리할 데이터가 없음"),
    "lg120": (D, "Rare Beauty와의 브랜드 비교 — 정성"),
    "lg121": (D, "브랜드 위협 요소 — 전략 분석"),
    "lg122": (D, "시장 기회 — 전략 분석"),
    "lg123": (D, "파트너십 전략 — 전략"),
    "lg124": (D, "MEDICUBE와의 브랜드 비교 — 정성"),
    "lg125": (D, "장기 경쟁력 유지 방안. 'SoS 5%+'는 관측치가 아니라 목표치"),
    # ---- market (25) ------------------------------------------------------
    "lg126": (X, "2025년 Lip Care 시장 규모·성장률. 매출 데이터 없음"),
    "lg127": (X, "K-Beauty 세그먼트 집중도의 다년 변화"),
    "lg128": (X, "Face Powder 경쟁 구도의 다년 변화"),
    "lg129": (X, "K-Beauty 점유율 추이(2020~2024). 스냅샷 범위 밖"),
    "lg130": (D, "신규 진입이 기존 브랜드에 미치는 영향 — 메커니즘 설명"),
    "lg131": (D, "BSR 알고리즘과 SoS의 관계 — 알고리즘 가이드"),
    "lg132": (X, "Skin Care 프리미엄화 트렌드(2023~2025) — 다년 추이"),
    "lg133": (X, "슬리핑 마스크 세그먼트 성장률 — 세그먼트 매출 없음"),
    "lg134": (S, "Amazon US 뷰티 전체 HHI — market_metrics(beauty)"),
    "lg135": (D, "Lip Makeup 최신 트렌드 — 정성 트렌드 서술"),
    "lg136": (D, "경쟁사 신제품 영향 분석 프레임 — 플레이북"),
    "lg137": (X, "상위 20% 브랜드의 매출 비중. 매출 데이터 없음"),
    "lg138": (D, "클린뷰티 트렌드의 기회·위협 — 정성"),
    "lg139": (X, "쿠션 파운데이션 비중의 다년 변화"),
    "lg140": (X, "Prime Day가 시장 구조에 미치는 영향 — 이벤트 효과"),
    "lg141": (D, "신규 진입 성공 조건 — 플레이북"),
    "lg142": (D, "계절성과 브랜드 전략 — 전략"),
    "lg143": (D, "가격 인하가 HHI에 미치는 영향 — 메커니즘"),
    "lg144": (X, "비건 제품 비중. 비건 여부가 크롤 데이터에 없음"),
    "lg145": (X, "Skin Care vs Makeup 성장률 비교 — 매출 성장률"),
    "lg146": (X, "소셜 트렌드가 시장에 미치는 영향 — 상관 주장"),
    "lg147": (D, "중소 브랜드 성장 가능성 — 정성"),
    "lg194": (S, "'현재 SoS 5.2%(5개 제품)'를 성장 여력의 출발점으로 단정"),
    "lg197": (X, "한·미 브랜드의 미래 전망(2030) — 예측"),
    "lg198": (D, "모니터링 핵심 KPI 3가지 — 지표 선정 가이드"),
    # ---- multi_hop (20) ---------------------------------------------------
    "lg148": (S, "제품→카테고리→HHI. lip_care·face_powder HHI 단정"),
    "lg149": (S, "최강 카테고리의 HHI와 LANEIGE SoS 단정"),
    "lg150": (S, "대표 제품 카테고리의 1위 브랜드 — brand_metrics"),
    "lg151": (S, "COSRX 주력 카테고리의 HHI와 SoS 단정"),
    "lg152": (S, "같은 카테고리 경쟁사 평균 가격 — raw_data.price"),
    "lg153": (S, "face_powder K-Beauty 브랜드 SoS 합계 — brand_metrics"),
    "lg154": (S, "COSRX 최인기 제품의 카테고리 SoS — brand_metrics"),
    "lg155": (S, "'현재 LANEIGE Skin Care SoS ~3~4%' 단정 — brand_metrics"),
    "lg156": (S, "리뷰 수→SoS 경로. '10만+ 리뷰' 단정이 raw_data로 검증 가능"),
    "lg157": (X, "TIRTIR 1위 원인과 HHI 변화 — 시계열 반사실"),
    "lg158": (S, "아모레퍼시픽 자매 브랜드의 Amazon 현황 — brand_metrics"),
    "lg159": (X, "SoS 최고월과 이유 — 월별 계절성 스냅샷 부족"),
    "lg160": (S, "최고가 제품과 카테고리 평균 대비 CPI — raw_data·brand_metrics"),
    "lg161": (S, "1위 브랜드 SoS와 LANEIGE의 격차 — brand_metrics"),
    "lg191": (D, "LANEIGE는 어떤 브랜드인가 — 브랜드 소개"),
    "lg192": (S, "'Lip Care SoS 5.2%'의 이유. 전제 수치가 DB 확인 대상"),
    "lg193": (S, "COSRX의 SoS와 주력 카테고리 HHI 단정"),
    "lg195": (S, "TIRTIR vs LANEIGE face_powder SoS 비교 — brand_metrics"),
    "lg196": (S, "카테고리별 경쟁 강도. HHI 수치를 단정"),
    "lg199": (D, "SoS·HHI·CPI 통합 해석 프레임 — 플레이북"),
    # ---- edge (15) --------------------------------------------------------
    "lg162": (S, "오타 교정 후 현재 순위를 답한다 — raw_data.rank"),
    "lg163": (D, "브랜드 우열 비교 — 정성"),
    "lg164": (S, "영어 질의의 현재 SoS — brand_metrics"),
    "lg165": (D, "범위 밖 질문(날씨) 거절 — 응답 정책"),
    "lg166": (D, "주가 질문 — 범위 설명"),
    "lg167": (D, "성분 안전성 질문 — 응답 정책"),
    "lg168": (D, "모방 요청 거절 — 응답 정책"),
    "lg169": (S, "SoS 100% 반박과 함께 현재 SoS를 제시"),
    "lg170": (S, "'어제와 같은가'에 현재 SoS 수준을 제시"),
    "lg171": (S, "lip_makeup 순위가 낮은 이유 — 실제 순위 단정"),
    "lg172": (D, "구매 링크 요청 거절 — 응답 정책"),
    "lg173": (D, "HHI 음수 가능성 — 정의상 불가"),
    "lg174": (D, "콜라보 정보 없음 — 범위 설명"),
    "lg189": (X, "5년 후 K-Beauty 시장 전망 — 예측"),
    "lg200": (D, "Amazon 전략 성공 요인 종합 — 정성"),
    # ---- time (15) --------------------------------------------------------
    "lg175": (S, "지난 3개월 SoS 추이 — 해당 월 스냅샷이 있으면 DB로 산출"),
    "lg176": (S, "작년 대비 순위 변화 — 해당 시점 스냅샷 필요"),
    "lg177": (X, "월별 HHI 계절 패턴 — 다년 계절성"),
    "lg178": (S, "최근 1년 K-Beauty SoS 성장 — 시작·끝 스냅샷 필요"),
    "lg179": (X, "분기별 HHI 패턴 — 다년 계절성"),
    "lg180": (X, "2018년부터의 순위 성장 히스토리"),
    "lg181": (S, "최근 6개월 CPI 추이 — brand_metrics.cpi"),
    "lg182": (X, "2년 전 대비 HHI 변화 — 스냅샷 범위 밖"),
    "lg183": (X, "TIRTIR face_powder SoS 월별 성장. DB에 해당 데이터 없음"),
    "lg184": (X, "브랜드별 분기 이탈률. churn_rate는 카테고리 단위만 있음"),
    "lg185": (X, "연간 SoS 최고·최저 — 12개월 연속 스냅샷 없음"),
    "lg186": (X, "브랜드별 이탈률 비교 — 브랜드 단위 churn 없음"),
    "lg187": (X, "Prime Day 이후 회복 패턴 — 이벤트 전후 스냅샷 없음"),
    "lg188": (X, "순위 변동성이 높은 시기 — 다년 계절성"),
    "lg190": (X, "연도별 신제품 출시 빈도와 SoS 상관"),
    # ---- ir (12) ----------------------------------------------------------
    "lg201": (D, "AP 1Q25 매출 — docs/ir 원문 인용"),
    "lg202": (D, "AP 1Q25 영업이익 — IR 원문"),
    "lg203": (D, "AP 1Q25 서구권 성장률 — IR 원문"),
    "lg204": (D, "AP 1Q25 중화권 실적 — IR 원문"),
    "lg205": (D, "AP 2Q25 영업이익 — IR 원문"),
    "lg206": (D, "AP 2Q25 지역별 매출 — IR 원문"),
    "lg207": (D, "AP 2Q25 매출총이익률 — IR 원문"),
    "lg208": (D, "2Q25 미주 LANEIGE 실적 — IR 원문"),
    "lg209": (D, "AP 3Q25 영업이익 증가율 — IR 원문"),
    "lg210": (D, "AP 3Q25 순이익 — IR 원문"),
    "lg211": (D, "AP 3Q25 해외 매출 — IR 원문"),
    "lg212": (D, "3Q25 립 슬리핑 마스크 신제품 에디션 — IR 원문"),
}

# 분류상으로는 snapshot이지만, 4단계(refresh_golden_snapshot_values.py)가 as_of 시점의
# DB에 해당 관측이 없음을 확인해 domain_expectation으로 강등한 문항들.
# 여기 적어 두지 않으면 이 스크립트를 다시 돌릴 때 강등이 되돌아가고, 두 스크립트가
# 서로를 덮어쓴다. 분류 판단(CLASSIFICATION)은 그대로 두고 결과만 여기서 보정한다.
DEMOTED_BY_DATA: dict[str, str] = {
    "lg077": "product_metrics에 LANEIGE rank_change가 한 건도 없다 (순위 상승 판정 불가)",
    "lg093": "raw_data.is_subscribe_save가 전부 0이다 (구독 여부 미수집)",
    "lg175": "지난 3개월(2026-06·07) 월 스냅샷 부재",
    "lg176": "작년 대비 비교에 필요한 2025-08~11 스냅샷 부재",
    "lg178": "최근 1년 비교에 필요한 2025-08~11 스냅샷 부재",
    "lg181": "최근 6개월(2026-05~07) 스냅샷 부재 + lip_care CPI가 NULL",
}

# 층이 갈릴 수 있는 문항. 판단 근거를 남겨 재검토가 가능하게 한다.
BORDERLINE: dict[str, str] = {
    "lg054": "SoS-매출 상관. 코퍼스 지표 가이드에 경험칙으로 적혀 있다면 document",
    "lg058": "카테고리 변동성. product_metrics.rank_volatility로 일부는 실측 가능",
    "lg063": "SoS-리뷰 상관은 검증 불가지만 '10만+ 리뷰' 단정은 DB로 검증 가능",
    "lg088": "신규 SKU 예측(반사실)이지만 expected_values의 current_sos는 DB 확인 대상",
    "lg099": "K-Beauty 최강 브랜드. SoS로 뒷받침되지만 카테고리 횡단 정성 순위",
    "lg135": "Lip Makeup 최신 트렌드. 코퍼스 트렌드 문서 유무에 따라 갈림",
    "lg156": "질문은 메커니즘(document)인데 답이 리뷰 수를 단정(snapshot)",
    "lg175": "지난 3개월 추이. 해당 월 스냅샷이 없으면 4단계에서 강등된다",
    "lg192": "질문은 '이유'(document)인데 전제가 현재 SoS 수치(snapshot)",
    "lg194": "질문은 성장 가능성(전략)인데 출발점으로 현재 SoS를 단정",
}


def load_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def apply_classification(rows: list[dict]) -> tuple[list[dict], int]:
    """분류를 적용하고 (갱신된 행, 변경 건수)를 돌려준다. 멱등."""
    changed = 0
    for row in rows:
        source, _ = CLASSIFICATION[row["id"]]
        if row["id"] in DEMOTED_BY_DATA:
            source = X
        meta = row.setdefault("metadata", {})
        as_of = AS_OF if source == "snapshot" else None
        if meta.get("gold_source") != source or meta.get("as_of") != as_of:
            changed += 1
        meta["gold_source"] = source
        if as_of is None:
            meta.pop("as_of", None)
        else:
            meta["as_of"] = as_of
    return rows, changed


def effective_source(item_id: str) -> str:
    """분류 결과에 4단계의 데이터 기반 강등을 반영한 최종 층."""
    return X if item_id in DEMOTED_BY_DATA else CLASSIFICATION[item_id][0]


def print_distribution(rows: list[dict]) -> None:
    by_source = collections.Counter(effective_source(r["id"]) for r in rows)
    print(f"\n총 {len(rows)}문항")
    print("\n| gold_source | 문항 수 | 비율 |")
    print("|---|---|---|")
    for source in (D, S, X):
        n = by_source[source]
        print(f"| {source} | {n} | {n / len(rows):.1%} |")

    print("\n| 도메인 | document | snapshot | domain_expectation |")
    print("|---|---|---|---|")
    grid: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        grid[row["metadata"]["domain"]][effective_source(row["id"])] += 1
    for domain in sorted(grid):
        c = grid[domain]
        print(f"| {domain} | {c[D]} | {c[S]} | {c[X]} |")

    print(f"\nas_of={AS_OF}가 붙는 문항: {by_source[S]}건")
    print(
        f"\n분류상 snapshot이나 DB에 관측이 없어 강등된 문항 {len(DEMOTED_BY_DATA)}건 "
        "(4단계 확인 결과):"
    )
    for item_id, why in DEMOTED_BY_DATA.items():
        print(f"  {item_id} → domain_expectation — {why}")
    print("\n경계 사례 (재검토 대상):")
    for item_id, why in BORDERLINE.items():
        source, reason = CLASSIFICATION[item_id]
        print(f"  {item_id} → {source}\n      분류 근거: {reason}\n      경계인 이유: {why}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="분포만 출력하고 쓰지 않는다")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    missing = {r["id"] for r in rows} - set(CLASSIFICATION)
    extra = set(CLASSIFICATION) - {r["id"] for r in rows}
    if missing or extra:
        print(f"분류표와 데이터셋 불일치 — 누락 {sorted(missing)}, 잉여 {sorted(extra)}")
        return 1

    print_distribution(rows)

    if args.dry_run:
        _, changed = apply_classification([json.loads(json.dumps(r)) for r in rows])
        print(f"\n[dry-run] 변경될 문항: {changed}건 (파일은 그대로)")
        return 0

    rows, changed = apply_classification(rows)
    # 원본은 git이 보관한다 (별도 .bak을 남기지 않는 것이 기존 스크립트 관례)
    with args.dataset.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n{changed}건 갱신 — {args.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
