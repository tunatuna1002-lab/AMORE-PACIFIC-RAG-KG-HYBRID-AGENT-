"""
API Dependencies
================
공통 의존성 모듈 (인증, 레이트리밋, 세션 관리, 헬퍼 함수 등)

dashboard_api.py의 라우트 모듈들이 공유하는 모든 의존성을 제공합니다.
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.rag.retriever import DocumentRetriever
from src.rag.router import QueryType, RAGRouter

logger = logging.getLogger(__name__)

# ============= API Key 인증 =============

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    _env = os.getenv("RAILWAY_ENVIRONMENT", os.getenv("ENV", "development"))
    if _env in ("production", "staging"):
        raise RuntimeError(
            "API_KEY 환경변수가 설정되지 않았습니다. 프로덕션/스테이징 환경에서는 필수입니다."
        )
    logging.warning("API_KEY 환경변수가 설정되지 않았습니다. (개발 환경)")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """API Key 검증 (민감한 엔드포인트용)"""
    import hmac

    if API_KEY is None:
        raise HTTPException(
            status_code=503,
            detail="Server not configured for authenticated access",
        )
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail="API Key가 필요합니다. 헤더에 X-API-Key를 추가하세요.",
        )
    if not hmac.compare_digest(api_key.encode(), API_KEY.encode()):
        raise HTTPException(
            status_code=403,
            detail="유효하지 않은 API Key입니다.",
        )
    return api_key


# ============= Rate Limiter =============

limiter = Limiter(key_func=get_remote_address)


# ============= Session Management =============

conversation_memory: dict[str, list[dict[str, str]]] = defaultdict(list)
session_last_activity: dict[str, datetime] = {}
MAX_MEMORY_TURNS = 10
SESSION_TTL_HOURS = 1
MAX_SESSIONS = 1000


def cleanup_expired_sessions() -> int:
    """만료된 세션 정리"""
    now = datetime.now()
    expired = [
        sid
        for sid, last_time in session_last_activity.items()
        if (now - last_time).total_seconds() > SESSION_TTL_HOURS * 3600
    ]
    for sid in expired:
        if sid in conversation_memory:
            del conversation_memory[sid]
        if sid in session_last_activity:
            del session_last_activity[sid]
    return len(expired)


def get_conversation_history(session_id: str, limit: int = 5) -> str:
    """대화 기록 조회"""
    history = conversation_memory.get(session_id, [])[-limit:]
    if not history:
        return ""

    lines = []
    for turn in history:
        role = "사용자" if turn["role"] == "user" else "AI"
        content = turn["content"][:150] + "..." if len(turn["content"]) > 150 else turn["content"]
        lines.append(f"[{role}]: {content}")

    return "\n".join(lines)


def add_to_memory(session_id: str, role: str, content: str) -> None:
    """대화 메모리에 추가"""
    now = datetime.now()

    if len(session_last_activity) > MAX_SESSIONS or len(session_last_activity) % 100 == 0:
        cleanup_expired_sessions()

    session_last_activity[session_id] = now
    conversation_memory[session_id].append(
        {"role": role, "content": content, "timestamp": now.isoformat()}
    )

    if len(conversation_memory[session_id]) > MAX_MEMORY_TURNS * 2:
        conversation_memory[session_id] = conversation_memory[session_id][-MAX_MEMORY_TURNS * 2 :]


# ============= Audit Trail Logger =============

AUDIT_LOG_DIR = "./logs"


def setup_audit_logger():
    """Audit Trail 로거 설정"""
    Path(AUDIT_LOG_DIR).mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = Path(AUDIT_LOG_DIR) / f"chatbot_audit_{today}.log"

    audit_logger = logging.getLogger("audit_trail")
    audit_logger.setLevel(logging.INFO)
    audit_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    audit_logger.addHandler(file_handler)

    return audit_logger


audit_logger = setup_audit_logger()


def log_chat_interaction(
    session_id: str,
    user_query: str,
    ai_response: str,
    query_type: str,
    confidence: float,
    entities: dict,
    sources: list[str],
    response_time_ms: float,
):
    """챗봇 대화 Audit Trail 기록"""
    audit_entry = {
        "session_id": session_id,
        "timestamp": datetime.now().isoformat(),
        "user_query": user_query,
        "ai_response": ai_response[:500] + "..." if len(ai_response) > 500 else ai_response,
        "query_type": query_type,
        "confidence": round(confidence, 4),
        "entities": entities,
        "sources": sources,
        "response_time_ms": round(response_time_ms, 2),
    }
    audit_logger.info(json.dumps(audit_entry, ensure_ascii=False))


# ============= Data Helpers =============

# Resolve data directory: Railway volume at /data/ vs local ./data/
RESOLVED_DATA_DIR = "/data" if Path("/data").exists() else "./data"
DATA_PATH = f"{RESOLVED_DATA_DIR}/dashboard_data.json"
DOCS_PATH = "./"


def load_dashboard_data() -> dict[str, Any]:
    """대시보드 데이터 로드 (staleness 경고 포함)"""
    import time

    try:
        data_path = Path(DATA_PATH)
        if not data_path.exists():
            logging.warning(
                f"Dashboard data file not found: {DATA_PATH} "
                f"(RESOLVED_DATA_DIR={RESOLVED_DATA_DIR})"
            )
            return {}

        file_age_hours = (time.time() - data_path.stat().st_mtime) / 3600
        if file_age_hours > 24:
            logging.warning(
                f"Dashboard data is stale: {file_age_hours:.1f} hours old. "
                f"Consider running a crawl or calling /api/data/refresh."
            )

        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)

        data.setdefault("metadata", {})
        data["metadata"]["_cache_age_hours"] = round(file_age_hours, 1)
        data["metadata"]["_is_stale"] = file_age_hours > 24

        return data

    except json.JSONDecodeError as e:
        logging.warning(f"Corrupted dashboard data file: {e}")
        return {}


# ============= RAG System =============

rag_router = RAGRouter()
doc_retriever = DocumentRetriever(DOCS_PATH)


def build_data_context(data: dict, query_type: QueryType, entities: dict) -> str:
    """
    데이터 컨텍스트 구성 (Ontology 기반)

    질문 유형과 추출된 엔티티에 따라 필요한 데이터만 선택
    """
    if not data:
        return "현재 데이터가 없습니다."

    context_parts = []

    # 메타데이터 (항상 포함)
    metadata = data.get("metadata", {})
    context_parts.append(f"""[데이터 현황]
- 기준일: {metadata.get("data_date", "N/A")}
- 총 제품 수: {metadata.get("total_products", 0)}개
- LANEIGE 제품 수: {metadata.get("laneige_products", 0)}개""")

    # 질문 유형별 데이터 선택
    brand_kpis = data.get("brand", {}).get("kpis", {})

    # 시장/브랜드 지표 (DEFINITION, INTERPRETATION, ANALYSIS)
    if query_type in [
        QueryType.DEFINITION,
        QueryType.INTERPRETATION,
        QueryType.ANALYSIS,
        QueryType.COMBINATION,
    ]:
        if brand_kpis:
            context_parts.append(f"""
[LANEIGE 브랜드 KPI] (Ontology: BrandMetrics)
- SoS (Share of Shelf): {brand_kpis.get("sos", 0)}% {brand_kpis.get("sos_delta", "")}
- Top 10 제품 수: {brand_kpis.get("top10_count", 0)}개
- 평균 순위: {brand_kpis.get("avg_rank", 0)}위
- HHI (시장 집중도): {brand_kpis.get("hhi", 0)}""")

    # 경쟁사 정보 (ANALYSIS, DATA_QUERY에서 경쟁사 언급 시)
    competitors = data.get("brand", {}).get("competitors", [])
    brands_mentioned = entities.get("brands", [])

    if query_type == QueryType.ANALYSIS or any(
        b for b in brands_mentioned if b.lower() != "laneige"
    ):
        if competitors:
            top_comps = competitors[:5]
            comp_lines = [
                f"  - {c['brand']}: SoS {c['sos']}%, 평균 순위 {c['avg_rank']}위, 제품 {c['product_count']}개"
                for c in top_comps
            ]
            context_parts.append("[경쟁사 현황]\n" + "\n".join(comp_lines))

    # 제품 정보 (DATA_QUERY, 특정 제품 언급 시)
    products = data.get("products", {})
    products_mentioned = entities.get("products", [])

    if query_type == QueryType.DATA_QUERY or products_mentioned:
        if products:
            prod_lines = []
            for _asin, p in list(products.items())[:5]:
                prod_lines.append(f"""  - {p["name"][:40]}
    순위: #{p["rank"]} ({p["rank_delta"]}), 평점: {p["rating"]}, 변동성: {p.get("volatility_status", "N/A")}""")
            context_parts.append(
                "[LANEIGE 제품 현황] (Ontology: ProductMetrics)\n" + "\n".join(prod_lines)
            )

    # 카테고리 정보
    categories = data.get("categories", {})
    categories_mentioned = entities.get("categories", [])

    if categories_mentioned or query_type in [QueryType.ANALYSIS, QueryType.INTERPRETATION]:
        if categories:
            cat_lines = []
            for _cat_id, cat in categories.items():
                cat_lines.append(
                    f"  - {cat['name']}: SoS {cat['sos']}%, 최고 순위 #{cat['best_rank']}, CPI {cat.get('cpi', 100)}"
                )
            context_parts.append(
                "[카테고리 현황] (Ontology: MarketMetrics)\n" + "\n".join(cat_lines)
            )

    # 액션 아이템 (전략 질문)
    if query_type == QueryType.ANALYSIS:
        action_items = data.get("home", {}).get("action_items", [])
        if action_items:
            action_lines = [
                f"  - [{a['priority']}] {a['product_name']}: {a['signal']} → {a['action_tag']}"
                for a in action_items[:4]
            ]
            context_parts.append("[현재 액션 아이템]\n" + "\n".join(action_lines))

    return "\n\n".join(context_parts)


async def get_rag_context(query: str, query_type: QueryType) -> tuple[str, list[str]]:
    """
    RAG 컨텍스트 검색

    Returns:
        (컨텍스트 문자열, 출처 목록)
    """
    # DocumentRetriever 초기화 (처음 호출 시)
    if not doc_retriever._initialized:
        await doc_retriever.initialize()

    # 질문 유형에 맞는 문서 검색
    target_doc = rag_router.get_target_document(query_type)

    # 검색 실행
    results = await doc_retriever.search(query, top_k=3, doc_filter=target_doc)

    if not results:
        return "", []

    # 컨텍스트 구성
    context_parts = []
    sources = []

    for result in results:
        metadata = result.get("metadata", {})
        content = result.get("content", "")
        title = metadata.get("title", "Unknown")
        doc_id = metadata.get("doc_id", "")

        context_parts.append(f"[{title}]\n{content}")

        # 출처 추가
        doc_name_map = {
            "strategic_indicators": "Strategic Indicators Definition",
            "metric_interpretation": "Metric Interpretation Guide",
            "indicator_combination": "Indicator Combination Playbook",
            "home_insight_rules": "Home Page Insight Rules",
        }
        if doc_id in doc_name_map and doc_name_map[doc_id] not in sources:
            sources.append(doc_name_map[doc_id])

    return "\n\n---\n\n".join(context_parts), sources


def generate_dynamic_suggestions(
    query_type: QueryType, entities: dict, response: str, user_query: str = ""
) -> list[str]:
    """동적 후속 질문 제안 (v2 - 개선 버전)"""
    suggestions = []

    # 엔티티 추출
    brands = entities.get("brands", [])
    indicators = entities.get("indicators", [])
    categories = entities.get("categories", [])

    # 1순위: 응답 키워드 기반 제안
    if response:
        keyword_suggestions = _extract_response_keywords(response)
        suggestions.extend(keyword_suggestions)

    # 2순위: 엔티티 기반 제안
    if len(suggestions) < 3:
        entity_suggestions = _generate_entity_suggestions(brands, categories, indicators)
        suggestions.extend(entity_suggestions)

    # 3순위: 쿼리 유형 기반 제안 (폴백)
    if len(suggestions) < 3:
        type_suggestions = _generate_type_suggestions(query_type, brands, indicators)
        suggestions.extend(type_suggestions)

    # 중복 제거 및 상위 3개
    unique = list(dict.fromkeys(suggestions))
    return unique[:3]


def _extract_response_keywords(response: str) -> list[str]:
    """응답에서 후속 질문 관련 키워드 추출"""
    import re

    keywords = []

    patterns = {
        r"순위.{0,10}(하락|급락|떨어)": "순위 하락 원인을 분석해주세요",
        r"순위.{0,10}(상승|급등|올라)": "상승 요인을 상세 분석해주세요",
        r"경쟁사|경쟁 브랜드|competitor": "경쟁사 상세 비교를 해주세요",
        r"가격.{0,10}(인상|인하|변동)": "가격 전략을 분석해주세요",
        r"리뷰|평점|rating": "소비자 피드백을 상세 분석해주세요",
        r"트렌드|유행|trend": "트렌드 상세 분석을 해주세요",
        r"성장.{0,5}(기회|가능|potential)": "성장 전략을 제안해주세요",
        r"위험|리스크|위협|risk": "리스크 대응 전략은?",
        r"SoS|점유율|share": "점유율 개선 전략은?",
        r"Top.{0,3}(10|5)|상위": "Top 10 진입 전략은?",
    }

    for pattern, suggestion in patterns.items():
        if re.search(pattern, response, re.IGNORECASE):
            keywords.append(suggestion)
            if len(keywords) >= 2:
                break

    return keywords


def _generate_entity_suggestions(
    brands: list[str], categories: list[str], indicators: list[str]
) -> list[str]:
    """엔티티 기반 동적 제안 생성"""
    suggestions = []

    if brands:
        brand = brands[0]
        suggestions.append(f"{brand} 경쟁사 비교 분석")
        if len(brands) > 1:
            suggestions.append(f"{brands[0]} vs {brands[1]} 비교")

    if categories:
        cat = categories[0]
        suggestions.append(f"{cat} 시장 트렌드 분석")

    if indicators:
        ind = indicators[0].upper()
        suggestions.append(f"{ind} 개선 전략")

    return suggestions


def _generate_type_suggestions(
    query_type: QueryType, brands: list[str], indicators: list[str]
) -> list[str]:
    """쿼리 유형 기반 폴백 제안"""
    suggestions = []

    if query_type == QueryType.DEFINITION:
        if indicators:
            ind = indicators[0].upper()
            suggestions.append(f"{ind}가 높으면 어떤 의미인가요?")
        suggestions.extend(["관련된 다른 지표는?", "실제 데이터에 적용해주세요"])

    elif query_type == QueryType.INTERPRETATION:
        suggestions.extend(
            ["현재 LANEIGE 수치 분석", "경쟁사와 비교해주세요", "개선 액션 아이템은?"]
        )

    elif query_type == QueryType.DATA_QUERY:
        suggestions.extend(["이 수치가 좋은 건가요?", "최근 7일 추이 분석", "경쟁사 대비 현황"])

    elif query_type == QueryType.ANALYSIS:
        suggestions.extend(["가장 시급한 액션은?", "Top 10 진입 전략", "리스크 요인 분석"])

    elif query_type == QueryType.COMBINATION:
        suggestions.extend(["다른 시나리오 분석", "현재 해당 상황 존재 여부"])

    else:
        suggestions = ["SoS(점유율) 설명해주세요", "LANEIGE 현재 순위는?", "전략적 권고사항"]

    return suggestions


# ============= URL Helper =============


def get_base_url() -> str:
    """배포 환경에 맞는 Base URL 반환"""
    if dashboard_url := os.getenv("DASHBOARD_URL"):
        return dashboard_url.rstrip("/")

    if railway_domain := os.getenv("RAILWAY_PUBLIC_DOMAIN"):
        return f"https://{railway_domain}"

    port = os.getenv("PORT", "8001")
    return f"http://localhost:{port}"


# ============= JWT Helpers =============

from datetime import UTC, timedelta

import jwt

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
EMAIL_VERIFICATION_EXPIRES_MINUTES = 30

_jwt_env = os.getenv("RAILWAY_ENVIRONMENT", os.getenv("ENV", "development"))
if not JWT_SECRET_KEY and _jwt_env in ("production", "staging"):
    raise RuntimeError(
        "JWT_SECRET_KEY 환경변수가 설정되지 않았습니다. 프로덕션/스테이징 환경에서는 필수입니다."
    )

if JWT_SECRET_KEY:
    if len(JWT_SECRET_KEY) < 32:
        logging.warning(
            "JWT_SECRET_KEY가 32자 미만입니다. HS256은 최소 256비트(32바이트) 키를 권장합니다."
        )
    _weak_patterns = [
        r"^changeme",
        r"^secret",
        r"^password",
        r"^(.)\1+$",  # all same character
        r"^0123456789",  # sequential digits
    ]
    import re as _re

    if any(_re.match(p, JWT_SECRET_KEY, _re.IGNORECASE) for p in _weak_patterns):
        logging.warning(
            "JWT_SECRET_KEY가 취약한 패턴을 사용하고 있습니다. 강력한 랜덤 키로 교체하세요."
        )


def create_email_verification_token(
    email: str, expires_minutes: int = EMAIL_VERIFICATION_EXPIRES_MINUTES
) -> str:
    """이메일 인증용 JWT 토큰 생성"""
    if not JWT_SECRET_KEY:
        raise ValueError("JWT_SECRET_KEY 환경변수가 설정되지 않았습니다.")

    payload = {
        "email": email,
        "purpose": "email_verification",
        "exp": datetime.now(UTC) + timedelta(minutes=expires_minutes),
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_jwt_email_token(token: str) -> dict:
    """JWT 이메일 인증 토큰 검증"""
    if not JWT_SECRET_KEY:
        return {"valid": False, "error": "JWT_SECRET_KEY 환경변수가 설정되지 않았습니다."}

    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])

        if payload.get("purpose") != "email_verification":
            return {"valid": False, "error": "유효하지 않은 토큰입니다."}

        return {"valid": True, "email": payload["email"]}

    except jwt.ExpiredSignatureError:
        return {"valid": False, "error": "인증 토큰이 만료되었습니다. 다시 인증해주세요."}
    except jwt.InvalidTokenError:
        return {"valid": False, "error": "유효하지 않은 인증 토큰입니다."}


# ============= State Manager =============

from src.core.state_manager import StateManager, get_state_manager

_state_manager: StateManager | None = None


def get_app_state_manager() -> StateManager:
    """앱 레벨 State Manager 반환"""
    global _state_manager
    if _state_manager is None:
        _state_manager = get_state_manager()
    return _state_manager


# ============= SheetsWriter Singleton =============

from src.tools.storage.sheets_writer import SheetsWriter

_sheets_writer: SheetsWriter | None = None


def get_sheets_writer() -> SheetsWriter:
    """SheetsWriter 싱글톤 인스턴스 반환"""
    global _sheets_writer
    if _sheets_writer is None:
        _sheets_writer = SheetsWriter()
    return _sheets_writer


# ============= Market Intelligence Singleton =============

from src.tools.intelligence.market_intelligence import MarketIntelligenceEngine

_market_intelligence_engine: MarketIntelligenceEngine | None = None


async def get_market_intelligence() -> MarketIntelligenceEngine:
    """Market Intelligence Engine 싱글톤 반환"""
    global _market_intelligence_engine
    if _market_intelligence_engine is None:
        _market_intelligence_engine = MarketIntelligenceEngine()
        await _market_intelligence_engine.initialize()
    return _market_intelligence_engine
