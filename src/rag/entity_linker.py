"""
Entity Linker
==============
텍스트 엔티티를 온톨로지 개념에 연결하는 Entity Linking 모듈

## 핵심 기능
1. **NER 기반 엔티티 추출**
   - 브랜드명 인식 (LANEIGE, COSRX, TIRTIR 등)
   - 제품 카테고리 인식 (Lip Care, Skin Care 등)
   - 성분명 인식 (Peptide, Ceramide 등)
   - 지표명 인식 (SoS, HHI, CPI 등)
   - spaCy 또는 규칙 기반 NER (폴백)

2. **온톨로지 개념 매칭**
   - 추출된 엔티티를 OWL 온톨로지 개념에 매핑
   - 유사도 기반 퍼지 매칭 (브랜드명 변형 처리)
   - 동의어/별칭 처리

3. **신뢰도 점수 계산**
   - Exact match: 1.0
   - Fuzzy match: 0.7-0.9
   - Partial match: 0.5-0.7
   - 컨텍스트 기반 점수 보정

## 사용 예
```python
linker = EntityLinker()
entities = linker.link("LANEIGE Lip Care 경쟁력 분석해줘")
# [
#   LinkedEntity(text="LANEIGE", type="brand", concept_uri="...", confidence=1.0),
#   LinkedEntity(text="Lip Care", type="category", concept_uri="...", confidence=1.0)
# ]
```

## 통합
- EntityExtractor (hybrid_retriever.py)와 호환
- OWLReasoner 통합 지원
- KnowledgeGraph 연동
"""

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)

# spaCy 선택적 import
try:
    import spacy

    SPACY_AVAILABLE = True
    logger.info("spaCy is available")
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed. EntityLinker will use rule-based fallback.")


def product_name_slugs(title: str, brand: str) -> list[str]:
    """제품 타이틀 → 제품 라인명 슬러그 후보 (3단어·2단어 변형)

    예: "LANEIGE Lip Sleeping Mask - Berry" → ["lip_sleeping_mask", "lip_sleeping"]
        "LANEIGE Water Bank Blue Hyaluronic Cream" → ["water_bank_blue", "water_bank"]
    """
    t = title.strip()
    if brand and t.lower().startswith(brand.lower()):
        t = t[len(brand) :]
    for sep in (":", "|", ",", " - ", "–", "("):
        t = t.split(sep)[0]
    words = re.findall(r"[A-Za-z0-9]+", t)
    slugs: list[str] = []
    for n in (3, 2):
        if len(words) >= n:
            slug = "_".join(w.lower() for w in words[:n])
            if slug not in slugs:
                slugs.append(slug)
    return slugs


# =========================================================================
# 온톨로지 등록부 연결 (트랙 O2, 결정 OA-6) — 플래그 `ontology.use_class_reasoning`
# =========================================================================
# 플래그가 꺼져 있으면 아래 코드는 전혀 쓰이지 않는다(출력이 O2 이전과 같다 — 특성화 테스트
# `tests/unit/rag/test_entity_linker_characterization.py`).

CLASS_REASONING_SECTION = "ontology"
CLASS_REASONING_KEY = "use_class_reasoning"

# 등록부에만 있는 별칭 중 일반 단어·짧은 표기라 오탐이 큰 것. 등록부 사전에서 뺀다.
# (기존 사전 config/entities.json·KNOWN_BRANDS에 있던 표기 — elf·boj·eos·로드 등 — 는 그대로 둔다.)
#   려: 한 글자라 "성공하려면"·"고려"에 걸린다 (골든 4문항 실측)
#   ap: "AP 그룹"은 그룹을 가리키는 경우가 많다 → 그룹 구문표에서 처리한다
#   essence: "Snail Mucin 96% Essence" 같은 제품 유형 단어 (골든 1문항)
#   median·matrix·verb·dove: 일반 영어 단어
AMBIGUOUS_REGISTRY_KEYS: frozenset[str] = frozenset(
    {"려", "ap", "essence", "median", "matrix", "verb", "dove"}
)

# 클래스 언급 구문표. 영어 세그먼트 단어는 등록부 세그먼트 라벨(`Luxury`·`Premium`…)에서
# 가져오고, 여기에는 한국어 표현만 적는다. 세그먼트 클래스는 "럭셔리 브랜드"처럼 뒤에
# 브랜드·라인 류 단어가 올 때만 인식한다 ("프리미엄화 트렌드"·"가격 프리미엄" 오탐 방지).
_SEGMENT_SUFFIX = r"\s*(?:브랜드|라인|세그먼트|계열|티어|brands?\b|segments?\b|tiers?\b|lines?\b)"
_SEGMENT_CLASS_TERMS: dict[str, tuple[str, ...]] = {
    "LuxuryBrand": ("럭셔리", "하이엔드", "high-end", "high end"),
    "PremiumBrand": ("프리미엄",),
    "MidTierBrand": ("중가", "미드", "중저가", "mid-tier", "mid tier", "mid-range"),
    "AffordableBrand": ("저가", "중저가", "affordable", "budget"),
    "MassBrand": ("매스", "대중", "mass-market", "mass market"),
}
_KBEAUTY_PATTERN = re.compile(
    r"k[\s\-]?beauty|k[\s\-]?뷰티|케이\s?뷰티|한국\s?(?:브랜드|화장품|코스메틱)"
    r"|korean\s+(?:brands?|beauty|cosmetics)"
)
_SIBLING_PATTERN = re.compile(
    r"같은\s*(?:그룹|계열|회사)|자매\s*브랜드|sister\s+brands?|sibling\s+brands?|same\s+group"
)
# 그룹 언급 뒤에 오면 "그 그룹의 브랜드 집합"(AmorepacificBrand 류)을 가리키는 단어
_GROUP_CLASS_CUE = (
    r"(?:브랜드들|브랜드\s*중|브랜드\s*전체|브랜드\s*목록|브랜드\s*포트폴리오|포트폴리오|소속|산하"
    r"|계열|소유한\s*브랜드|(?:owned\s+)?brands\b|portfolio)"
)
# "AP 그룹"·"AP 계열"·"AP 브랜드" — 등록부에서 AP는 브랜드 `Amore Pacific`의 별칭이지만
# 이 구문에서는 그룹이다.
_AP_GROUP_PATTERN = re.compile(r"(?<![a-z0-9])ap\s*(?:그룹|계열|group|브랜드)")


def class_reasoning_enabled() -> bool:
    """플래그 `ontology.use_class_reasoning` (기본 OFF). 호출마다 읽는다 — 테스트가 env로 바꾼다."""
    try:
        from src.infrastructure.feature_flags import FeatureFlags

        return bool(
            FeatureFlags.get_instance().get_flag(
                CLASS_REASONING_SECTION, CLASS_REASONING_KEY, default=False
            )
        )
    except Exception:
        logger.debug("feature flag read failed; class reasoning OFF", exc_info=True)
        return False


@dataclass(frozen=True)
class _RegistryLexicon:
    """등록부에서 만든 인식 사전 (정규화 키 기준)."""

    brand_keys: dict[str, str]  # normalize_key(표기) → 브랜드 id (가짜·모호 표기 제외)
    group_keys: dict[str, str]  # normalize_key(표기) → 그룹 id
    # 그룹 id → 그 그룹 소속 클래스 (예: amorepacific → AmorepacificBrand)
    group_class: dict[str, str]
    group_class_pattern: re.Pattern[str] | None
    segment_patterns: tuple[tuple[str, re.Pattern[str]], ...]  # (클래스, 패턴)
    kbeauty_class: str | None


_lexicon_cache: tuple[Any, _RegistryLexicon] | None = None


def _registry_lexicon() -> tuple[Any, _RegistryLexicon]:
    """(Ontology, 사전). 같은 Ontology 객체면 다시 만들지 않는다."""
    global _lexicon_cache
    from src.ontology.ontology import get_ontology

    onto = get_ontology()
    if _lexicon_cache is not None and _lexicon_cache[0] is onto:
        return _lexicon_cache
    _lexicon_cache = (onto, _build_registry_lexicon(onto))
    return _lexicon_cache


def _build_registry_lexicon(onto: Any) -> _RegistryLexicon:
    import json

    from src.ontology.ontology import DEFAULT_ONTOLOGY_DIR, normalize_key

    # 로더에는 별칭 목록을 돌려주는 공개 API가 없어 원본의 표기를 읽고, 각 표기가 로더에서
    # 같은 id로 정규화되는지 확인한 것만 쓴다.
    registry = json.loads((DEFAULT_ONTOLOGY_DIR / "brands.json").read_text(encoding="utf-8"))
    brand_keys: dict[str, str] = {}
    for entry in registry.get("brands") or []:
        for text in (entry.get("id"), entry.get("name"), *(entry.get("aliases") or [])):
            if not text:
                continue
            bid = onto.normalize_brand(str(text))
            key = normalize_key(str(text))
            if not bid or not key or onto.is_placeholder(bid):
                continue
            if key in AMBIGUOUS_REGISTRY_KEYS:
                continue
            brand_keys.setdefault(key, bid)

    group_keys: dict[str, str] = {}
    group_texts: dict[str, set[str]] = {}
    for group in registry.get("groups") or []:
        for text in (group.get("id"), group.get("name"), *(group.get("aliases") or [])):
            gid = onto.normalize_group(str(text)) if text else None
            if gid:
                group_keys.setdefault(normalize_key(str(text)), gid)
                group_texts.setdefault(gid, set()).add(str(text).lower())

    # 그룹 → 정의 클래스 (defined_by ownedByGroup = gid)
    classes = set(onto.classes)
    group_class: dict[str, str] = {}
    segment_label_class: dict[str, str] = {}
    kbeauty_class: str | None = None
    for cls in onto.classes:
        spec = onto.class_spec(cls)
        if spec.defined_by is None:
            continue
        predicate, value = spec.defined_by
        if predicate == "ownedByGroup":
            group_class[value] = cls
        elif predicate == "hasSegment":
            label = onto.label_of(value)
            if label:
                segment_label_class[label.lower()] = cls
        elif predicate == "originatesFrom" and value == "south_korea":
            kbeauty_class = cls

    group_class_pattern = None
    alts = sorted(
        {t for texts in group_texts.values() for t in texts} | {"ap"}, key=len, reverse=True
    )
    if alts:
        alt = "|".join(re.escape(a) for a in alts)
        group_class_pattern = re.compile(
            rf"(?<![a-z0-9])(?:{alt})(?![a-z0-9])\s*(?:그룹|group)?\s*(?:이|가|의|에서)?\s*"
            rf"{_GROUP_CLASS_CUE}"
        )

    segment_patterns: list[tuple[str, re.Pattern[str]]] = []
    for cls, terms in _SEGMENT_CLASS_TERMS.items():
        if cls not in classes:
            continue
        words = set(terms)
        words.update(label for label, c in segment_label_class.items() if c == cls)
        alt = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
        segment_patterns.append((cls, re.compile(rf"(?<![a-z0-9])(?:{alt}){_SEGMENT_SUFFIX}")))

    return _RegistryLexicon(
        brand_keys=brand_keys,
        group_keys=group_keys,
        group_class=group_class,
        group_class_pattern=group_class_pattern,
        segment_patterns=tuple(segment_patterns),
        kbeauty_class=kbeauty_class if kbeauty_class in classes else None,
    )


@dataclass
class LinkedEntity:
    """
    링크된 엔티티

    Attributes:
        text: 원본 텍스트
        entity_type: brand, product, category, metric, ingredient, trend
        concept_uri: 온톨로지 개념 URI
        concept_label: 개념 레이블 (사람이 읽을 수 있는 이름)
        confidence: 연결 신뢰도 (0-1)
        context: 추가 컨텍스트
    """

    text: str
    entity_type: str
    concept_uri: str
    concept_label: str
    confidence: float
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "concept_uri": self.concept_uri,
            "concept_label": self.concept_label,
            "confidence": self.confidence,
            "context": self.context,
        }


class EntityLinker:
    """
    텍스트 엔티티를 온톨로지 개념에 연결

    NER → Entity Normalization → Ontology Concept Mapping

    사용 예:
        linker = EntityLinker()
        entities = linker.link("LANEIGE Lip Care 경쟁력 분석")
    """

    # 온톨로지 URI 베이스
    ONTOLOGY_BASE = "http://amorepacific.com/ontology/amore_brand.owl#"

    # 알려진 브랜드 (정규화 포함)
    KNOWN_BRANDS = {
        # 한/영 매핑
        "laneige": "LANEIGE",
        "라네즈": "LANEIGE",
        "라네쥬": "LANEIGE",
        "라네이지": "LANEIGE",
        "라네지": "LANEIGE",
        "cosrx": "COSRX",
        "코스알엑스": "COSRX",
        "코스아르엑스": "COSRX",
        "tirtir": "TIRTIR",
        "티르티르": "TIRTIR",
        "rare beauty": "Rare Beauty",
        "레어뷰티": "Rare Beauty",
        "레어 뷰티": "Rare Beauty",
        "innisfree": "Innisfree",
        "이니스프리": "Innisfree",
        "etude": "ETUDE",
        "에뛰드": "ETUDE",
        "에뛰드하우스": "ETUDE",
        "sulwhasoo": "Sulwhasoo",
        "설화수": "Sulwhasoo",
        "hera": "HERA",
        "헤라": "HERA",
        "missha": "MISSHA",
        "미샤": "MISSHA",
        "skin1004": "SKIN1004",
        "스킨1004": "SKIN1004",
        "anua": "Anua",
        "아누아": "Anua",
        "medicube": "MEDICUBE",
        "메디큐브": "MEDICUBE",
        "biodance": "BIODANCE",
        "바이오던스": "BIODANCE",
        "beauty of joseon": "Beauty of Joseon",
        "조선미녀": "Beauty of Joseon",
        "summer fridays": "Summer Fridays",
        "la roche-posay": "La Roche-Posay",
        "cerave": "CeraVe",
        "neutrogena": "Neutrogena",
        "eos": "eos",
        "이오에스": "eos",
        "e.l.f.": "e.l.f.",
        "nyx": "NYX",
        "maybelline": "Maybelline",
    }

    # 카테고리 매핑
    CATEGORY_MAP = {
        "lip care": ("lip_care", "Lip Care"),
        "립케어": ("lip_care", "Lip Care"),
        "립 케어": ("lip_care", "Lip Care"),
        "lip makeup": ("lip_makeup", "Lip Makeup"),
        "립메이크업": ("lip_makeup", "Lip Makeup"),
        "립 메이크업": ("lip_makeup", "Lip Makeup"),
        "skin care": ("skin_care", "Skin Care"),
        "스킨케어": ("skin_care", "Skin Care"),
        "스킨 케어": ("skin_care", "Skin Care"),
        "face powder": ("face_powder", "Face Powder"),
        "파우더": ("face_powder", "Face Powder"),
        "페이스파우더": ("face_powder", "Face Powder"),
        "beauty": ("beauty", "Beauty & Personal Care"),
        "뷰티": ("beauty", "Beauty & Personal Care"),
    }

    # 지표 매핑
    INDICATOR_MAP = {
        "sos": ("sos", "Share of Shelf"),
        "점유율": ("sos", "Share of Shelf"),
        "share of shelf": ("sos", "Share of Shelf"),
        "hhi": ("hhi", "Herfindahl-Hirschman Index"),
        "시장집중도": ("hhi", "Herfindahl-Hirschman Index"),
        "허핀달": ("hhi", "Herfindahl-Hirschman Index"),
        "cpi": ("cpi", "Category Price Index"),
        "가격지수": ("cpi", "Category Price Index"),
        "churn": ("churn_rate", "Churn Rate"),
        "교체율": ("churn_rate", "Churn Rate"),
        "streak": ("streak_days", "Streak Days"),
        "연속": ("streak_days", "Streak Days"),
        "volatility": ("rank_volatility", "Rank Volatility"),
        "변동성": ("rank_volatility", "Rank Volatility"),
        "shock": ("rank_shock", "Rank Shock"),
        "급변": ("rank_shock", "Rank Shock"),
    }

    # 성분 키워드
    INGREDIENT_MAP = {
        "peptide": ("Peptide", "펩타이드"),
        "펩타이드": ("Peptide", "펩타이드"),
        "ceramide": ("Ceramide", "세라마이드"),
        "세라마이드": ("Ceramide", "세라마이드"),
        "hyaluronic acid": ("HyaluronicAcid", "히알루론산"),
        "히알루론산": ("HyaluronicAcid", "히알루론산"),
        "niacinamide": ("Niacinamide", "나이아신아마이드"),
        "나이아신아마이드": ("Niacinamide", "나이아신아마이드"),
        "retinol": ("Retinol", "레티놀"),
        "레티놀": ("Retinol", "레티놀"),
        "vitamin c": ("VitaminC", "비타민C"),
        "비타민c": ("VitaminC", "비타민C"),
        "centella": ("Centella", "센텔라"),
        "센텔라": ("Centella", "센텔라"),
        "cica": ("Centella", "시카"),
        "시카": ("Centella", "시카"),
        "pdrn": ("PDRN", "PDRN"),
        "글래스스킨": ("GlassSkin", "Glass Skin"),
        "glass skin": ("GlassSkin", "Glass Skin"),
    }

    # 트렌드 키워드
    TREND_KEYWORDS = {
        "모닝쉐드": "MorningShade",
        "morning shade": "MorningShade",
        "글로우": "Glow",
        "glow": "Glow",
        "바이럴": "Viral",
        "viral": "Viral",
        "틱톡": "TikTok",
        "tiktok": "TikTok",
        "인플루언서": "Influencer",
        "influencer": "Influencer",
    }

    # 시간 범위 매핑 (EntityExtractor + RAGRouter 통합)
    TIME_RANGE_MAP: dict[str, str] = {
        "오늘": "today",
        "today": "today",
        "어제": "yesterday",
        "yesterday": "yesterday",
        "이번 주": "week",
        "this week": "week",
        "이번 달": "month",
        "this month": "month",
        "최근 7일": "7days",
        "last 7 days": "7days",
        "last week": "7days",
        "지난주": "7days",
        "최근 30일": "30days",
        "last 30 days": "30days",
        "last month": "30days",
        "3개월": "90days",
        "90 days": "90days",
        "1개월": "30days",
        "1 month": "30days",
    }

    # 감성 키워드 매핑 (EntityExtractor 통합)
    SENTIMENT_MAP: dict[str, str] = {
        "moisturizing": "Hydration",
        "hydrating": "Hydration",
        "보습": "Hydration",
        "수분": "Hydration",
        "촉촉": "Hydration",
        "value for money": "Pricing",
        "가성비": "Pricing",
        "affordable": "Pricing",
        "저렴": "Pricing",
        "easy to use": "Usability",
        "사용감": "Usability",
        "편리": "Usability",
        "효과": "Effectiveness",
        "effective": "Effectiveness",
        "works well": "Effectiveness",
        "scent": "Sensory",
        "향": "Sensory",
        "texture": "Sensory",
        "텍스처": "Sensory",
        "질감": "Sensory",
        "packaging": "Packaging",
        "패키징": "Packaging",
        "포장": "Packaging",
        "gentle": "Skin_Compatibility",
        "순한": "Skin_Compatibility",
        "민감": "Skin_Compatibility",
        "sensitive": "Skin_Compatibility",
        "리뷰": "sentiment_general",
        "review": "sentiment_general",
        "고객 반응": "sentiment_general",
        "customer": "sentiment_general",
        "customer feedback": "sentiment_general",
    }

    # config/entities.json 캐시
    _config_cache: dict | None = None
    _config_loaded_at: float | None = None
    _CONFIG_TTL_SECONDS: int = 300

    def __init__(self, knowledge_graph=None, owl_reasoner=None, use_spacy: bool = True):
        """
        Args:
            knowledge_graph: KnowledgeGraph 인스턴스 (개념 검증용)
            owl_reasoner: OWLReasoner 인스턴스 (온톨로지 쿼리용)
            use_spacy: spaCy NER 사용 여부
        """
        self.kg = knowledge_graph
        self.owl_reasoner = owl_reasoner
        self.use_spacy = use_spacy and SPACY_AVAILABLE

        # spaCy 모델 로드
        self.nlp = None
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}. Using rule-based fallback.")
                self.use_spacy = False

        # 통계
        self._stats = {"total_links": 0, "exact_matches": 0, "fuzzy_matches": 0, "no_matches": 0}

    # =========================================================================
    # 엔티티 링킹 메인 API
    # =========================================================================

    def link(
        self, text: str, entity_types: list[str] | None = None, min_confidence: float = 0.5
    ) -> list[LinkedEntity]:
        """
        텍스트에서 엔티티 추출 및 온톨로지 개념에 링크

        Args:
            text: 입력 텍스트
            entity_types: 추출할 엔티티 유형 필터 (None이면 전체)
            min_confidence: 최소 신뢰도 임계값

        Returns:
            LinkedEntity 리스트
        """
        # 1. NER 기반 엔티티 추출
        if self.use_spacy:
            entities = self._extract_with_spacy(text)
        else:
            entities = self._extract_with_rules(text)

        # 2. 온톨로지 개념 매칭
        linked_entities = []
        for entity_text, entity_type, context in entities:
            # 유형 필터
            if entity_types and entity_type not in entity_types:
                continue

            # 개념 매칭
            concept_uri, concept_label, confidence = self._match_concept(
                entity_text, entity_type, context
            )

            # 신뢰도 임계값
            if confidence < min_confidence:
                continue

            linked = LinkedEntity(
                text=entity_text,
                entity_type=entity_type,
                concept_uri=concept_uri,
                concept_label=concept_label,
                confidence=confidence,
                context=context,
            )
            linked_entities.append(linked)

            # 통계
            self._stats["total_links"] += 1
            if confidence == 1.0:
                self._stats["exact_matches"] += 1
            elif confidence >= 0.7:
                self._stats["fuzzy_matches"] += 1

        if class_reasoning_enabled() and (not entity_types or "brand" in entity_types):
            linked_entities = self._link_registry_brands(text, linked_entities)

        return linked_entities

    def _link_registry_brands(
        self, text: str, linked_entities: list[LinkedEntity]
    ) -> list[LinkedEntity]:
        """플래그 ON: 브랜드 엔티티에 등록부 id를 달고, 가짜 브랜드를 빼고, 기존 단어사전에
        없던 등록부 브랜드를 덧붙인다 (신뢰도 1.0, 위치 정보 없음)."""
        try:
            onto, lexicon = _registry_lexicon()
        except Exception:
            logger.warning("ontology registry unavailable; class reasoning skipped", exc_info=True)
            return linked_entities
        from src.ontology.ontology import normalize_key

        kept: list[LinkedEntity] = []
        seen: set[str] = set()
        for entity in linked_entities:
            if entity.entity_type == "brand":
                bid = onto.normalize_brand(entity.concept_label) or onto.normalize_brand(
                    entity.text
                )
                if bid and onto.is_placeholder(bid):
                    continue
                if bid:
                    entity.context = {**entity.context, "registry_id": bid}
                    seen.add(bid)
            kept.append(entity)

        text_norm = normalize_key(text)
        for key, bid in lexicon.brand_keys.items():
            if bid in seen or not self._mentions(text_norm, key):
                continue
            seen.add(bid)
            name = onto.brand_name(bid) or bid
            kept.append(
                LinkedEntity(
                    text=name,
                    entity_type="brand",
                    concept_uri=f"{self.ONTOLOGY_BASE}Brand/{name.replace(' ', '_')}",
                    concept_label=name,
                    confidence=1.0,
                    context={"matched_key": key, "registry_id": bid, "source": "ontology"},
                )
            )
        return kept

    # =========================================================================
    # Simple dict-format entity extraction (EntityExtractor compat)
    # =========================================================================

    @classmethod
    def _load_entity_config(cls) -> dict[str, Any]:
        """
        config/entities.json에서 엔티티 매핑 로드 (캐싱 적용).

        Returns:
            설정 딕셔너리. 파일이 없으면 빈 딕셔너리.
        """
        import json
        import time
        from pathlib import Path

        now = time.monotonic()
        if (
            cls._config_cache is not None
            and cls._config_loaded_at is not None
            and (now - cls._config_loaded_at) < cls._CONFIG_TTL_SECONDS
        ):
            return cls._config_cache

        config_path = Path("config/entities.json")
        if not config_path.exists():
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config/entities.json"

        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    cls._config_cache = json.load(f)
                    cls._config_loaded_at = now
                    return cls._config_cache
            except Exception as e:
                logger.warning(f"Failed to load entity config: {e}")

        return {}

    def _get_merged_brands(self) -> dict[str, str]:
        """
        KNOWN_BRANDS (class-level) + config/entities.json 브랜드 통합.

        Config brands use lowercase canonical names (e.g. "laneige").
        Class-level KNOWN_BRANDS use proper case (e.g. "LANEIGE").
        Config canonical names take precedence for extract_entities() compat.

        Returns:
            {lowercase_name_or_alias: normalized_name} 매핑
        """
        # Start with config brands (lowercase canonical names)
        config = self._load_entity_config()
        config_norm: dict[str, str] = {}
        for brand_info in config.get("known_brands", []):
            if not isinstance(brand_info, dict):
                continue
            name = brand_info["name"].lower()
            config_norm[name] = name
            for alias in brand_info.get("aliases", []):
                config_norm[alias.lower()] = name

        # Add class-level brands, using config canonical name if available
        merged: dict[str, str] = dict(config_norm)
        for key, proper_name in self.KNOWN_BRANDS.items():
            if key not in merged:
                # Check if proper_name.lower() is a config canonical name
                canonical = proper_name.lower()
                if canonical in config_norm:
                    merged[key] = config_norm[canonical]
                else:
                    merged[key] = canonical

        return merged

    def _get_merged_categories(self) -> dict[str, str]:
        """
        CATEGORY_MAP (class-level) + config/entities.json 카테고리 통합.

        Returns:
            {keyword: category_id} 매핑
        """
        merged: dict[str, str] = {}
        # class-level: (cat_id, cat_label) -> cat_id
        for key, val in self.CATEGORY_MAP.items():
            merged[key] = val[0] if isinstance(val, tuple) else val

        config = self._load_entity_config()
        for key, cat_id in config.get("category_map", {}).items():
            if key.lower() not in merged:
                merged[key.lower()] = cat_id

        return merged

    def _get_merged_indicators(self) -> dict[str, str]:
        """
        INDICATOR_MAP (class-level) + config/entities.json 지표 통합.

        Returns:
            {keyword: indicator_id} 매핑
        """
        merged: dict[str, str] = {}
        for key, val in self.INDICATOR_MAP.items():
            merged[key] = val[0] if isinstance(val, tuple) else val

        config = self._load_entity_config()
        for key, ind_id in config.get("indicator_map", {}).items():
            if key.lower() not in merged:
                merged[key.lower()] = ind_id

        return merged

    def _get_merged_time_ranges(self) -> dict[str, str]:
        """
        TIME_RANGE_MAP (class-level) + config/entities.json 시간범위 통합.

        Returns:
            {keyword: time_id} 매핑
        """
        merged: dict[str, str] = dict(self.TIME_RANGE_MAP)

        config = self._load_entity_config()
        for key, time_id in config.get("time_range_map", {}).items():
            if key.lower() not in merged:
                merged[key.lower()] = time_id

        return merged

    def _get_merged_sentiments(self) -> dict[str, str]:
        """
        SENTIMENT_MAP (class-level) + config/entities.json 감성 통합.

        Returns:
            {keyword: cluster_name} 매핑
        """
        merged: dict[str, str] = dict(self.SENTIMENT_MAP)

        config = self._load_entity_config()
        for key, cluster in config.get("sentiment_map", {}).items():
            if key.lower() not in merged:
                merged[key.lower()] = cluster

        return merged

    def extract_entities(self, query: str, knowledge_graph: Any | None = None) -> dict[str, Any]:
        """
        Simple entity extraction returning dict format.

        Compatible with EntityExtractor.extract() and RAGRouter.extract_entities().

        Args:
            query: 사용자 쿼리
            knowledge_graph: 지식 그래프 (순위 기반 제품 검색용, optional)

        Returns:
            {
                "brands": [...],
                "categories": [...],
                "indicators": [...],
                "time_range": [...],
                "products": [...],
                "sentiments": [...],
                "sentiment_clusters": [...]
            }
        """
        query_lower = query.lower()

        entities: dict[str, Any] = {
            "brands": [],
            "categories": [],
            "indicators": [],
            "time_range": [],
            "products": [],
            "sentiments": [],
            "sentiment_clusters": [],
        }

        # 브랜드 추출 (class-level + config 통합)
        merged_brands = self._get_merged_brands()
        for brand_key, normalized in merged_brands.items():
            if not self._mentions(query_lower, brand_key):
                continue
            if normalized not in entities["brands"]:
                entities["brands"].append(normalized)

        # 카테고리 추출
        merged_cats = self._get_merged_categories()
        for cat_name, cat_id in merged_cats.items():
            if self._mentions(query_lower, cat_name):
                if cat_id not in entities["categories"]:
                    entities["categories"].append(cat_id)

        # 지표 추출
        merged_indicators = self._get_merged_indicators()
        for ind_name, ind_id in merged_indicators.items():
            if self._mentions(query_lower, ind_name):
                if ind_id not in entities["indicators"]:
                    entities["indicators"].append(ind_id)

        # 시간 범위 추출
        merged_time = self._get_merged_time_ranges()
        for time_name, time_id in merged_time.items():
            if time_name in query_lower:
                if time_id not in entities["time_range"]:
                    entities["time_range"].append(time_id)

        # 제품 ASIN 추출 (B0로 시작하는 10자리)
        asin_pattern = r"\bB0[A-Z0-9]{8}\b"
        asins = re.findall(asin_pattern, query)
        if asins:
            entities["products"].extend(asins)

        # 제품명 기반 브랜드 역링크 (지식 그래프 활용)
        if knowledge_graph:
            self._link_products_from_kg(query_lower, entities, knowledge_graph)

        # 순위 기반 제품 추출 (지식 그래프 활용)
        if knowledge_graph:
            rank_patterns = [
                (r"(\d+)위\s*제품", "ko"),
                (r"top\s*(\d+)\s*product", "en"),
                (r"(\d+)위", "ko"),
                (r"rank\s*(\d+)", "en"),
            ]
            for pattern, _lang in rank_patterns:
                matches = re.findall(pattern, query_lower)
                if matches and entities.get("categories"):
                    for rank_str in matches:
                        rank = int(rank_str)
                        for category in entities["categories"]:
                            products = knowledge_graph.query(predicate=None, object_=category)
                            for rel in products:
                                if rel.properties.get("rank") == rank:
                                    asin = rel.subject
                                    if asin not in entities["products"]:
                                        entities["products"].append(asin)
                                    break

        # 감성 키워드 추출
        merged_sentiments = self._get_merged_sentiments()
        for keyword, cluster in merged_sentiments.items():
            if keyword in query_lower:
                if keyword not in entities["sentiments"]:
                    entities["sentiments"].append(keyword)
                if cluster not in entities["sentiment_clusters"]:
                    entities["sentiment_clusters"].append(cluster)

        if class_reasoning_enabled():
            self._apply_ontology(query, entities, merged_brands)

        return entities

    def _apply_ontology(
        self, query: str, entities: dict[str, Any], merged_brands: dict[str, str]
    ) -> None:
        """플래그 ON: 등록부 사전으로 브랜드 보강 + 가짜 브랜드 제거 + 클래스·그룹 언급.

        - ``brands``: 기존 결과를 그대로 두고(순서 유지) 등록부에서 새로 찾은 브랜드를 뒤에
          붙인다. 표기는 기존 사전이 그 브랜드에 쓰던 문자열(예: ``e.l.f.``·``la roche-posay``),
          기존 사전에 없던 브랜드는 등록부 이름 소문자(예: ``it cosmetics``·``charlotte tilbury``)
          — KG·DB 브랜드 표기(소문자)와 같다.
        - 가짜 브랜드(``unknown``·``fresh``·``chi``)는 어느 경로로 들어왔든 뺀다.
        - ``brand_ids``: ``brands`` 중 등록부 브랜드의 id (같은 순서, 등록부 밖은 건너뜀).
        - ``classes``·``groups``·``relations_hint``: 클래스·그룹 언급.
        """
        try:
            onto, lexicon = _registry_lexicon()
        except Exception:
            logger.warning("ontology registry unavailable; class reasoning skipped", exc_info=True)
            return
        from src.ontology.ontology import normalize_key

        query_norm = normalize_key(query)
        query_lower = query.lower()

        # 기존 사전 표기 → 등록부 id (기존 브랜드는 같은 문자열로 내보낸다)
        existing_by_id: dict[str, str] = {}
        for key, value in merged_brands.items():
            bid = onto.normalize_brand(key)
            if bid:
                existing_by_id.setdefault(bid, value)

        brands: list[str] = entities["brands"]
        for key, bid in lexicon.brand_keys.items():
            if not self._mentions(query_norm, key):
                continue
            surface = existing_by_id.get(bid) or (onto.brand_name(bid) or bid).lower()
            if surface not in brands:
                brands.append(surface)

        brands[:] = [b for b in brands if not onto.is_placeholder(str(b))]

        brand_ids: list[str] = []
        for b in brands:
            bid = onto.normalize_brand(str(b))
            if bid and bid not in brand_ids:
                brand_ids.append(bid)
        entities["brand_ids"] = brand_ids

        groups: list[str] = []
        for key, gid in lexicon.group_keys.items():
            if self._mentions(query_norm, key) and gid not in groups:
                groups.append(gid)
        ap_group = onto.normalize_group("amorepacific")
        if ap_group and _AP_GROUP_PATTERN.search(query_lower) and ap_group not in groups:
            groups.append(ap_group)

        classes: list[str] = []
        if lexicon.group_class_pattern is not None and lexicon.group_class_pattern.search(
            query_lower
        ):
            for gid in groups:
                cls = lexicon.group_class.get(gid)
                if cls and cls not in classes:
                    classes.append(cls)
        if lexicon.kbeauty_class and _KBEAUTY_PATTERN.search(query_lower):
            classes.append(lexicon.kbeauty_class)
        for cls, pattern in lexicon.segment_patterns:
            if pattern.search(query_lower) and cls not in classes:
                classes.append(cls)

        entities["classes"] = classes
        entities["groups"] = groups
        entities["relations_hint"] = ["sibling"] if _SIBLING_PATTERN.search(query_lower) else []

    @staticmethod
    def _mentions(query_lower: str, key: str) -> bool:
        """키워드 포함 여부 — 라틴 문자 키는 단어 경계를 요구한다.

        단순 부분 문자열 매칭은 "share of shelf"의 'shelf'에서 브랜드 'elf'를,
        "eos"를 포함하는 임의 단어에서 브랜드 'eos'를 오탐한다. 한글은 교착어라
        조사가 바로 붙으므로(예: "라네즈의") 경계 조건을 걸지 않는다.
        """
        if not key:
            return False
        if re.fullmatch(r"[a-z0-9.\-' ]+", key):
            return re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", query_lower) is not None
        return key in query_lower

    def _get_product_slug_index(self, knowledge_graph: Any) -> dict[str, tuple[str, str]]:
        """KG의 hasProduct 타이틀 → {제품 슬러그: (브랜드, 카테고리)} 인덱스.

        질의가 브랜드명 없이 제품명만 언급하는 경우("Lip Sleeping Mask 순위는?")
        브랜드가 하나도 추출되지 않아 KG 조회가 통째로 비었다. KG가 이미 보유한
        제품 타이틀로 제품명→브랜드를 역링크한다.
        """
        cached = getattr(self, "_product_slug_index", None)
        if cached is not None:
            return cached

        index: dict[str, tuple[str, str]] = {}
        try:
            for rel in knowledge_graph.query():
                predicate = (
                    rel.predicate.value if hasattr(rel.predicate, "value") else str(rel.predicate)
                )
                if predicate != "hasProduct":
                    continue
                title = rel.properties.get("title", "")
                brand = str(rel.subject)
                if not title:
                    continue
                category = str(rel.properties.get("category", ""))
                for slug in product_name_slugs(title, brand):
                    index.setdefault(slug, (brand.lower(), category))
        except Exception:
            logger.debug("product slug index build failed", exc_info=True)

        self._product_slug_index = index
        return index

    def _link_products_from_kg(
        self, query_lower: str, entities: dict, knowledge_graph: Any
    ) -> None:
        """질의에 등장한 KG 제품명을 products/brands 엔티티로 추가."""
        for slug, (brand, _category) in self._get_product_slug_index(knowledge_graph).items():
            phrase = slug.replace("_", " ")
            if len(phrase) < 6 or not self._mentions(query_lower, phrase):
                continue
            if slug not in entities["products"]:
                entities["products"].append(slug)
            if brand not in entities["brands"]:
                entities["brands"].append(brand)

    def extract_concepts(self, query: str) -> list[str]:
        """
        쿼리에서 의미론적 개념(concepts)을 추출합니다.

        indicators(sos, hhi 등)와 query_type(definition, data_query 등)을 결합하여
        gold standard의 concepts 필드와 매칭 가능한 개념 리스트를 반환합니다.

        Args:
            query: 사용자 쿼리

        Returns:
            개념 리스트 (e.g., ["sos", "data_query", "market_share"])
        """
        import json
        from pathlib import Path

        query_lower = query.lower()
        concepts: list[str] = []

        # 1. 지표 개념 (indicators → concepts로 매핑)
        merged_indicators = self._get_merged_indicators()
        for ind_name, ind_id in merged_indicators.items():
            if self._mentions(query_lower, ind_name) and ind_id not in concepts:
                concepts.append(ind_id)

        # 2. concept_taxonomy.json 기반 매칭
        taxonomy_path = Path("config/concept_taxonomy.json")
        if not taxonomy_path.exists():
            project_root = Path(__file__).parent.parent.parent
            taxonomy_path = project_root / "config" / "concept_taxonomy.json"

        if taxonomy_path.exists():
            try:
                with open(taxonomy_path, encoding="utf-8") as f:
                    taxonomy = json.load(f)

                # 메트릭·쿼리유형·분석 개념 매칭
                # 라틴 문자 키워드는 단어 경계를 요구한다 — 'rating'이
                # "ope|rating| profit"에 걸려 review_rating을 오탐하던 사례
                # (2026-08-30 사이클 7 실측). 한글은 조사가 붙으므로 예외.
                for section in ("metric_concepts", "query_type_concepts", "analysis_concepts"):
                    for concept_id, keywords in taxonomy.get(section, {}).items():
                        if concept_id in concepts:
                            continue
                        if any(self._mentions(query_lower, kw) for kw in keywords):
                            concepts.append(concept_id)

            except Exception as e:
                logger.warning(f"Failed to load concept taxonomy: {e}")

        return concepts

    # =========================================================================
    # NER 기반 엔티티 추출
    # =========================================================================

    def _extract_with_spacy(self, text: str) -> list[tuple[str, str, dict[str, Any]]]:
        """
        spaCy NER를 사용한 엔티티 추출

        Returns:
            [(entity_text, entity_type, context), ...]
        """
        entities = []
        doc = self.nlp(text)

        # spaCy 엔티티
        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            if entity_type:
                entities.append(
                    (
                        ent.text,
                        entity_type,
                        {"spacy_label": ent.label_, "start": ent.start_char, "end": ent.end_char},
                    )
                )

        # 규칙 기반 엔티티 추가 (spaCy가 못 잡은 것들)
        rule_entities = self._extract_with_rules(text)
        for ent_text, ent_type, context in rule_entities:
            # 중복 체크 (이미 spaCy가 잡은 것 제외)
            if not any(e[0].lower() == ent_text.lower() for e in entities):
                entities.append((ent_text, ent_type, context))

        return entities

    def _extract_with_rules(self, text: str) -> list[tuple[str, str, dict[str, Any]]]:
        """
        규칙 기반 엔티티 추출 (spaCy 폴백)

        Returns:
            [(entity_text, entity_type, context), ...]
        """
        entities = []
        text_lower = text.lower()

        # 브랜드 추출
        for brand_key in self.KNOWN_BRANDS.keys():
            if brand_key in text_lower:
                # 원본 텍스트에서 위치 찾기 (대소문자 무시)
                pattern = re.compile(re.escape(brand_key), re.IGNORECASE)
                for match in pattern.finditer(text):
                    entities.append(
                        (
                            match.group(),
                            "brand",
                            {"matched_key": brand_key, "start": match.start(), "end": match.end()},
                        )
                    )

        # 카테고리 추출
        for cat_key in self.CATEGORY_MAP.keys():
            if cat_key in text_lower:
                pattern = re.compile(re.escape(cat_key), re.IGNORECASE)
                for match in pattern.finditer(text):
                    entities.append(
                        (
                            match.group(),
                            "category",
                            {"matched_key": cat_key, "start": match.start(), "end": match.end()},
                        )
                    )

        # 지표 추출
        for ind_key in self.INDICATOR_MAP.keys():
            if ind_key in text_lower:
                pattern = re.compile(r"\b" + re.escape(ind_key) + r"\b", re.IGNORECASE)
                for match in pattern.finditer(text):
                    entities.append(
                        (
                            match.group(),
                            "metric",
                            {"matched_key": ind_key, "start": match.start(), "end": match.end()},
                        )
                    )

        # 성분 추출
        for ing_key in self.INGREDIENT_MAP.keys():
            if ing_key in text_lower:
                pattern = re.compile(re.escape(ing_key), re.IGNORECASE)
                for match in pattern.finditer(text):
                    entities.append(
                        (
                            match.group(),
                            "ingredient",
                            {"matched_key": ing_key, "start": match.start(), "end": match.end()},
                        )
                    )

        # 트렌드 키워드 추출
        for trend_key in self.TREND_KEYWORDS.keys():
            if trend_key in text_lower:
                pattern = re.compile(re.escape(trend_key), re.IGNORECASE)
                for match in pattern.finditer(text):
                    entities.append(
                        (
                            match.group(),
                            "trend",
                            {"matched_key": trend_key, "start": match.start(), "end": match.end()},
                        )
                    )

        # ASIN 패턴 추출 (제품)
        asin_pattern = r"\bB0[A-Z0-9]{8}\b"
        for match in re.finditer(asin_pattern, text):
            entities.append(
                (
                    match.group(),
                    "product",
                    {"format": "asin", "start": match.start(), "end": match.end()},
                )
            )

        return entities

    def _map_spacy_label(self, spacy_label: str) -> str | None:
        """
        spaCy 라벨을 도메인 엔티티 유형에 매핑

        Args:
            spacy_label: spaCy NER 라벨 (ORG, PRODUCT, GPE 등)

        Returns:
            엔티티 유형 (brand, product, category 등) 또는 None
        """
        mapping = {
            "ORG": "brand",  # 조직명 → 브랜드
            "PRODUCT": "product",  # 제품명
            "PERCENT": "metric",  # 퍼센트 → 지표
            "MONEY": "metric",  # 금액 → 가격 지표
        }
        return mapping.get(spacy_label)

    # =========================================================================
    # 온톨로지 개념 매칭
    # =========================================================================

    def _match_concept(
        self, entity_text: str, entity_type: str, context: dict[str, Any]
    ) -> tuple[str, str, float]:
        """
        엔티티를 온톨로지 개념에 매칭

        Args:
            entity_text: 엔티티 텍스트
            entity_type: 엔티티 유형
            context: 추출 컨텍스트

        Returns:
            (concept_uri, concept_label, confidence)
        """
        if entity_type == "brand":
            return self._match_brand(entity_text, context)
        elif entity_type == "category":
            return self._match_category(entity_text, context)
        elif entity_type == "metric":
            return self._match_metric(entity_text, context)
        elif entity_type == "ingredient":
            return self._match_ingredient(entity_text, context)
        elif entity_type == "trend":
            return self._match_trend(entity_text, context)
        elif entity_type == "product":
            return self._match_product(entity_text, context)
        else:
            return (f"{self.ONTOLOGY_BASE}Unknown", entity_text, 0.3)

    def _match_brand(self, text: str, context: dict[str, Any]) -> tuple[str, str, float]:
        """브랜드 매칭"""
        text_lower = text.lower()

        # Exact match
        if text_lower in self.KNOWN_BRANDS:
            normalized = self.KNOWN_BRANDS[text_lower]
            uri = f"{self.ONTOLOGY_BASE}Brand/{normalized.replace(' ', '_')}"
            return (uri, normalized, 1.0)

        # Fuzzy match
        best_match, best_score = self._fuzzy_match(text_lower, list(self.KNOWN_BRANDS.keys()))

        if best_score >= 0.8:
            normalized = self.KNOWN_BRANDS[best_match]
            uri = f"{self.ONTOLOGY_BASE}Brand/{normalized.replace(' ', '_')}"
            confidence = 0.7 + (best_score - 0.8) * 0.5  # 0.7 ~ 0.9
            return (uri, normalized, confidence)

        # 매칭 실패 - 원본 텍스트 그대로
        uri = f"{self.ONTOLOGY_BASE}Brand/{text.replace(' ', '_')}"
        return (uri, text, 0.5)

    def _match_category(self, text: str, context: dict[str, Any]) -> tuple[str, str, float]:
        """카테고리 매칭"""
        text_lower = text.lower()

        # Exact match
        if text_lower in self.CATEGORY_MAP:
            cat_id, cat_label = self.CATEGORY_MAP[text_lower]
            uri = f"{self.ONTOLOGY_BASE}Category/{cat_id}"
            return (uri, cat_label, 1.0)

        # Fuzzy match
        best_match, best_score = self._fuzzy_match(text_lower, list(self.CATEGORY_MAP.keys()))

        if best_score >= 0.7:
            cat_id, cat_label = self.CATEGORY_MAP[best_match]
            uri = f"{self.ONTOLOGY_BASE}Category/{cat_id}"
            confidence = 0.7 + (best_score - 0.7) * 0.3  # 0.7 ~ 0.9
            return (uri, cat_label, confidence)

        # 매칭 실패
        uri = f"{self.ONTOLOGY_BASE}Category/{text.replace(' ', '_')}"
        return (uri, text, 0.5)

    def _match_metric(self, text: str, context: dict[str, Any]) -> tuple[str, str, float]:
        """지표 매칭"""
        text_lower = text.lower()

        # Exact match
        if text_lower in self.INDICATOR_MAP:
            ind_id, ind_label = self.INDICATOR_MAP[text_lower]
            uri = f"{self.ONTOLOGY_BASE}Metric/{ind_id}"
            return (uri, ind_label, 1.0)

        # Fuzzy match
        best_match, best_score = self._fuzzy_match(text_lower, list(self.INDICATOR_MAP.keys()))

        if best_score >= 0.7:
            ind_id, ind_label = self.INDICATOR_MAP[best_match]
            uri = f"{self.ONTOLOGY_BASE}Metric/{ind_id}"
            confidence = 0.7 + (best_score - 0.7) * 0.3
            return (uri, ind_label, confidence)

        # 매칭 실패
        uri = f"{self.ONTOLOGY_BASE}Metric/{text.replace(' ', '_')}"
        return (uri, text, 0.5)

    def _match_ingredient(self, text: str, context: dict[str, Any]) -> tuple[str, str, float]:
        """성분 매칭"""
        text_lower = text.lower()

        # Exact match
        if text_lower in self.INGREDIENT_MAP:
            ing_id, ing_label = self.INGREDIENT_MAP[text_lower]
            uri = f"{self.ONTOLOGY_BASE}Ingredient/{ing_id}"
            return (uri, ing_label, 1.0)

        # Fuzzy match
        best_match, best_score = self._fuzzy_match(text_lower, list(self.INGREDIENT_MAP.keys()))

        if best_score >= 0.7:
            ing_id, ing_label = self.INGREDIENT_MAP[best_match]
            uri = f"{self.ONTOLOGY_BASE}Ingredient/{ing_id}"
            confidence = 0.7 + (best_score - 0.7) * 0.3
            return (uri, ing_label, confidence)

        # 매칭 실패
        uri = f"{self.ONTOLOGY_BASE}Ingredient/{text.replace(' ', '_')}"
        return (uri, text, 0.5)

    def _match_trend(self, text: str, context: dict[str, Any]) -> tuple[str, str, float]:
        """트렌드 키워드 매칭"""
        text_lower = text.lower()

        # Exact match
        if text_lower in self.TREND_KEYWORDS:
            trend_id = self.TREND_KEYWORDS[text_lower]
            uri = f"{self.ONTOLOGY_BASE}Trend/{trend_id}"
            return (uri, text, 1.0)

        # Fuzzy match
        best_match, best_score = self._fuzzy_match(text_lower, list(self.TREND_KEYWORDS.keys()))

        if best_score >= 0.7:
            trend_id = self.TREND_KEYWORDS[best_match]
            uri = f"{self.ONTOLOGY_BASE}Trend/{trend_id}"
            confidence = 0.7 + (best_score - 0.7) * 0.3
            return (uri, text, confidence)

        # 매칭 실패
        uri = f"{self.ONTOLOGY_BASE}Trend/{text.replace(' ', '_')}"
        return (uri, text, 0.5)

    def _match_product(self, text: str, context: dict[str, Any]) -> tuple[str, str, float]:
        """제품 매칭 (ASIN)"""
        # ASIN 형식 검증
        if context.get("format") == "asin" and re.match(r"^B0[A-Z0-9]{8}$", text):
            uri = f"{self.ONTOLOGY_BASE}Product/{text}"
            return (uri, text, 1.0)

        # 매칭 실패
        uri = f"{self.ONTOLOGY_BASE}Product/{text.replace(' ', '_')}"
        return (uri, text, 0.5)

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def _fuzzy_match(
        self, query: str, candidates: list[str], min_ratio: float = 0.6
    ) -> tuple[str | None, float]:
        """
        퍼지 문자열 매칭

        Args:
            query: 쿼리 문자열
            candidates: 후보 문자열 리스트
            min_ratio: 최소 유사도

        Returns:
            (best_match, best_score)
        """
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            ratio = SequenceMatcher(None, query, candidate).ratio()
            if ratio > best_score and ratio >= min_ratio:
                best_score = ratio
                best_match = candidate

        return (best_match, best_score)

    def get_ontology_filters(self, entities: list[LinkedEntity]) -> dict[str, Any]:
        """
        연결된 엔티티로부터 ChromaDB 필터 조건 생성 — **검색에 쓰지 말 것** (결함 F9).

        이 조건은 실제 색인에서 항상 0건이다: (1) 키 `brand`·`category`가 색인 메타데이터에
        없고, (2) 값이 canonical id가 아니라 표기 그대로이며(`LANEIGE`·`lip care`),
        (3) OWL 전략의 매칭 함수는 `$or`를 메타데이터 키로 취급했다. 엔티티 신호는 필터가
        아니라 재정렬 보너스로 쓴다 — `src/rag/entity_tags.py` (트랙 4-B, 설계 E10).

        Args:
            entities: LinkedEntity 리스트

        Returns:
            ChromaDB where 조건 딕셔너리
            예: {"$or": [{"brand": "LANEIGE"}, {"category": "lip_care"}]}
        """
        if not entities:
            return {}

        conditions = []

        for entity in entities:
            if entity.entity_type == "brand":
                conditions.append({"brand": entity.concept_label})
            elif entity.entity_type == "category":
                # concept_label에서 ID 추출 (예: "Lip Care" → "lip_care")
                cat_key = entity.context.get(
                    "matched_key", entity.concept_label.lower().replace(" ", "_")
                )
                conditions.append({"category": cat_key})
            elif entity.entity_type == "metric":
                # 지표는 메타데이터 필터로 사용
                metric_key = entity.context.get("matched_key", entity.concept_label.lower())
                conditions.append({"metric_type": metric_key})
            elif entity.entity_type == "ingredient":
                conditions.append({"ingredient": entity.concept_label})
            elif entity.entity_type == "trend":
                conditions.append({"trend": entity.concept_label})
            elif entity.entity_type == "product":
                conditions.append({"asin": entity.text})

        if len(conditions) == 0:
            return {}
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$or": conditions}

    def get_stats(self) -> dict[str, Any]:
        """통계 조회"""
        return self._stats.copy()

    def __repr__(self):
        mode = "spaCy" if self.use_spacy else "rule-based"
        return f"EntityLinker(mode={mode}, stats={self._stats})"


# =========================================================================
# 싱글톤 패턴 (선택)
# =========================================================================

_linker_instance: EntityLinker | None = None


def get_entity_linker(
    knowledge_graph=None, owl_reasoner=None, use_spacy: bool = True
) -> EntityLinker:
    """
    EntityLinker 싱글톤 인스턴스 반환

    Args:
        knowledge_graph: KnowledgeGraph 인스턴스
        owl_reasoner: OWLReasoner 인스턴스
        use_spacy: spaCy 사용 여부

    Returns:
        EntityLinker 인스턴스
    """
    global _linker_instance
    if _linker_instance is None:
        _linker_instance = EntityLinker(
            knowledge_graph=knowledge_graph, owl_reasoner=owl_reasoner, use_spacy=use_spacy
        )
    return _linker_instance
