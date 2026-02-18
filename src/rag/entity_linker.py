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

        return linked_entities

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
            if brand_key in query_lower:
                if normalized not in entities["brands"]:
                    entities["brands"].append(normalized)

        # 카테고리 추출
        merged_cats = self._get_merged_categories()
        for cat_name, cat_id in merged_cats.items():
            if cat_name in query_lower:
                if cat_id not in entities["categories"]:
                    entities["categories"].append(cat_id)

        # 지표 추출
        merged_indicators = self._get_merged_indicators()
        for ind_name, ind_id in merged_indicators.items():
            if ind_name in query_lower:
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

        return entities

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
        연결된 엔티티로부터 ChromaDB 필터 조건 생성

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
