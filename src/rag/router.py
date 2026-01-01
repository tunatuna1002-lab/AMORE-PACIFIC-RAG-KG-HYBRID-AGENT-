"""
RAG Router
질의 의도 파악 및 적절한 문서로 라우팅 + Fallback 처리

질의 유형:
- 지표 정의 질의 → Strategic Indicators Definition
- 지표 해석 질의 → Metric Interpretation Guide
- 지표 조합 질의 → Indicator Combination Playbook
- 인사이트 생성 질의 → Home Page Insight Rules
- 데이터 조회 질의 → Google Sheets 데이터
"""

from typing import Dict, Any, Optional, List, Tuple
from enum import Enum


class QueryType(str, Enum):
    """질의 유형"""
    DEFINITION = "definition"          # 지표 정의
    INTERPRETATION = "interpretation"  # 지표 해석
    COMBINATION = "combination"        # 지표 조합
    INSIGHT_RULE = "insight_rule"      # 인사이트 규칙
    DATA_QUERY = "data_query"          # 데이터 조회
    ANALYSIS = "analysis"              # 분석 요청
    UNKNOWN = "unknown"                # 의도 불명


class RAGRouter:
    """RAG 질의 라우터"""

    # 질의 유형별 키워드
    QUERY_PATTERNS = {
        QueryType.DEFINITION: {
            "keywords": ["뭐야", "무엇", "정의", "산출식", "계산식", "공식", "어떻게 계산"],
            "indicators": ["sos", "hhi", "cpi", "churn", "streak", "volatility", "shock", "rating"]
        },
        QueryType.INTERPRETATION: {
            "keywords": ["해석", "의미", "뜻", "높으면", "낮으면", "어떻게 봐야", "이해"],
            "indicators": ["sos", "hhi", "cpi", "churn", "streak", "volatility", "shock", "rating"]
        },
        QueryType.COMBINATION: {
            "keywords": ["조합", "같이", "동시에", "그리고", "결합", "복합", "시나리오"],
            "patterns": ["높고", "낮고", "상승", "하락", "증가", "감소"]
        },
        QueryType.INSIGHT_RULE: {
            "keywords": ["인사이트", "요약", "문구", "생성", "템플릿", "톤", "표현"],
            "patterns": ["어떻게 쓰", "어떻게 표현", "문장"]
        },
        QueryType.DATA_QUERY: {
            "keywords": ["순위", "랭킹", "현재", "오늘", "어제", "최근", "지금"],
            "entities": ["라네즈", "laneige", "제품", "브랜드", "카테고리"]
        },
        QueryType.ANALYSIS: {
            "keywords": ["분석", "비교", "전략", "제언", "보고서", "리포트", "평가"],
            "patterns": ["vs", "대비", "경쟁사", "트렌드"]
        }
    }

    # 문서 매핑
    DOC_MAPPING = {
        QueryType.DEFINITION: "strategic_indicators",
        QueryType.INTERPRETATION: "metric_interpretation",
        QueryType.COMBINATION: "indicator_combination",
        QueryType.INSIGHT_RULE: "home_insight_rules"
    }

    # Fallback 메시지
    FALLBACK_MESSAGES = {
        QueryType.UNKNOWN: "질문의 의도를 정확히 파악하지 못했습니다. 다음 중 어떤 정보가 필요하신가요?\n\n1. 지표 정의 (예: 'SoS가 뭐야?')\n2. 지표 해석 (예: 'HHI가 높으면 어떤 의미야?')\n3. 지표 조합 해석 (예: 'CPI 높고 평점 낮으면?')\n4. 순위/데이터 조회 (예: '라네즈 현재 순위?')\n5. 분석 요청 (예: '라네즈 3개월 성과 분석해줘')",
        "no_data": "요청하신 데이터를 찾을 수 없습니다. 다음을 확인해주세요:\n- 브랜드/제품명이 정확한가요?\n- 카테고리가 올바른가요?\n- 조회 기간 내 데이터가 있나요?",
        "clarification": "더 정확한 답변을 위해 추가 정보가 필요합니다:\n- 어떤 카테고리인가요? (Lip Care, Skin Care 등)\n- 어떤 기간을 조회할까요?\n- 특정 브랜드나 제품이 있나요?"
    }

    def __init__(self):
        """라우터 초기화"""
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """패턴 전처리"""
        # 지표명 통합 리스트
        self.all_indicators = [
            "sos", "share of shelf",
            "hhi", "herfindahl",
            "cpi", "category price index",
            "churn", "churn rate", "교체율",
            "streak", "streak days", "연속",
            "volatility", "변동성",
            "shock", "rank shock", "급변",
            "rating", "평점", "평가",
            "avg rank", "평균 순위"
        ]

    def classify_query(self, query: str) -> Tuple[QueryType, float]:
        """
        질의 유형 분류 (레거시 호환)

        Args:
            query: 사용자 질의

        Returns:
            (질의 유형, 신뢰도 점수)
        """
        query_type, confidence, _, _ = self.classify_query_detailed(query)
        return query_type, confidence

    def classify_query_detailed(self, query: str) -> Tuple[QueryType, float, float, List[str]]:
        """
        질의 유형 분류 (상세 정보 포함)

        Args:
            query: 사용자 질의

        Returns:
            (질의 유형, 신뢰도 점수, 최고 점수, 매칭된 키워드)
        """
        query_lower = query.lower()
        scores = {qt: 0.0 for qt in QueryType}
        matched_keywords = []

        for query_type, patterns in self.QUERY_PATTERNS.items():
            # 키워드 매칭
            for keyword in patterns.get("keywords", []):
                if keyword in query_lower:
                    scores[query_type] += 2.0
                    matched_keywords.append(keyword)

            # 지표명 매칭
            for indicator in patterns.get("indicators", []):
                if indicator in query_lower:
                    scores[query_type] += 1.5
                    matched_keywords.append(indicator)

            # 패턴 매칭
            for pattern in patterns.get("patterns", []):
                if pattern in query_lower:
                    scores[query_type] += 1.0
                    matched_keywords.append(pattern)

            # 엔티티 매칭
            for entity in patterns.get("entities", []):
                if entity in query_lower:
                    scores[query_type] += 1.5
                    matched_keywords.append(entity)

        # 최고 점수 유형 선택
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]

        # 신뢰도 정규화
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0

        # 최소 점수 미달 시 UNKNOWN
        if max_score < 1.5:
            return QueryType.UNKNOWN, 0.0, max_score, matched_keywords

        return max_type, confidence, max_score, list(set(matched_keywords))

    def get_target_document(self, query_type: QueryType) -> Optional[str]:
        """
        질의 유형에 따른 대상 문서 반환

        Args:
            query_type: 질의 유형

        Returns:
            문서 ID 또는 None
        """
        return self.DOC_MAPPING.get(query_type)

    def route(self, query: str) -> Dict[str, Any]:
        """
        질의 라우팅

        Args:
            query: 사용자 질의

        Returns:
            {
                "query_type": QueryType,
                "confidence": float,
                "max_score": float,           # 절대 점수 (신규)
                "matched_keywords": list,     # 매칭된 키워드 (신규)
                "target_doc": str or None,
                "requires_data": bool,
                "requires_rag": bool,
                "fallback_message": str or None
            }
        """
        query_type, confidence, max_score, matched_keywords = self.classify_query_detailed(query)

        result = {
            "query_type": query_type,
            "confidence": confidence,
            "max_score": max_score,
            "matched_keywords": matched_keywords,
            "target_doc": self.get_target_document(query_type),
            "requires_data": query_type in [QueryType.DATA_QUERY, QueryType.ANALYSIS],
            "requires_rag": query_type in [
                QueryType.DEFINITION,
                QueryType.INTERPRETATION,
                QueryType.COMBINATION,
                QueryType.INSIGHT_RULE,
                QueryType.ANALYSIS
            ],
            "fallback_message": None
        }

        # 낮은 신뢰도 시 Fallback 메시지
        if query_type == QueryType.UNKNOWN or confidence < 0.3:
            result["fallback_message"] = self.FALLBACK_MESSAGES[QueryType.UNKNOWN]

        return result

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """
        질의에서 엔티티 추출

        Args:
            query: 사용자 질의

        Returns:
            {
                "brands": [...],
                "products": [...],
                "categories": [...],
                "indicators": [...],
                "time_range": str or None
            }
        """
        query_lower = query.lower()

        entities = {
            "brands": [],
            "products": [],
            "categories": [],
            "indicators": [],
            "time_range": None
        }

        # 브랜드 추출 (동의어 지원)
        brand_aliases = {
            "laneige": ["laneige", "라네즈", "라네쥬", "라네이지", "라네지"],
            "cosrx": ["cosrx", "코스알엑스", "코스아르엑스"],
            "tirtir": ["tirtir", "티르티르"],
            "rare beauty": ["rare beauty", "레어뷰티", "레어 뷰티"],
            "eos": ["eos", "이오에스"],
            "medicube": ["medicube", "메디큐브"],
            "innisfree": ["innisfree", "이니스프리"],
            "sulwhasoo": ["sulwhasoo", "설화수"],
            "etude": ["etude", "에뛰드", "에뛰드하우스"]
        }

        for canonical_brand, aliases in brand_aliases.items():
            for alias in aliases:
                if alias in query_lower:
                    entities["brands"].append(canonical_brand)
                    break  # 하나의 브랜드는 한 번만 추가

        # 카테고리 추출
        category_map = {
            "lip care": "lip_care",
            "립케어": "lip_care",
            "skin care": "skin_care",
            "스킨케어": "skin_care",
            "beauty": "beauty",
            "뷰티": "beauty",
            "lip makeup": "lip_makeup",
            "face powder": "face_powder"
        }
        for cat_name, cat_id in category_map.items():
            if cat_name in query_lower:
                entities["categories"].append(cat_id)

        # 지표 추출
        for indicator in self.all_indicators:
            if indicator in query_lower:
                entities["indicators"].append(indicator)

        # 시간 범위 추출
        time_patterns = {
            "오늘": "today",
            "어제": "yesterday",
            "이번 주": "week",
            "이번 달": "month",
            "최근 7일": "7days",
            "최근 30일": "30days",
            "3개월": "90days",
            "1개월": "30days"
        }
        for pattern, time_range in time_patterns.items():
            if pattern in query_lower:
                entities["time_range"] = time_range
                break

        return entities

    def needs_clarification(self, route_result: Dict, entities: Dict) -> Optional[str]:
        """
        명확화 필요 여부 확인

        Args:
            route_result: route() 결과
            entities: extract_entities() 결과

        Returns:
            명확화 메시지 또는 None
        """
        query_type = route_result["query_type"]

        # 데이터 조회인데 엔티티 부족
        if query_type == QueryType.DATA_QUERY:
            if not entities["brands"] and not entities["products"]:
                return "어떤 브랜드나 제품의 데이터를 조회할까요?"

        # 분석 요청인데 범위 불명확
        if query_type == QueryType.ANALYSIS:
            if not entities["time_range"]:
                return "어떤 기간의 데이터를 분석할까요? (예: 최근 7일, 3개월)"
            if not entities["brands"] and not entities["categories"]:
                return "어떤 브랜드/카테고리를 분석할까요?"

        return None

    def get_fallback_response(self, reason: str = "unknown") -> str:
        """
        Fallback 응답 반환

        Args:
            reason: Fallback 사유

        Returns:
            Fallback 메시지
        """
        if reason in self.FALLBACK_MESSAGES:
            return self.FALLBACK_MESSAGES[reason]
        return self.FALLBACK_MESSAGES[QueryType.UNKNOWN]
