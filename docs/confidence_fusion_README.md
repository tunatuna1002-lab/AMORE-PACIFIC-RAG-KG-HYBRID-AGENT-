# Confidence Fusion Module

다중 소스 (벡터 검색, 온톨로지 추론, 엔티티 연결)의 신뢰도를 통합하는 Confidence Fusion 모듈

---

## 개요

RAG 시스템에서 여러 소스의 정보를 결합할 때, 각 소스의 신뢰도를 어떻게 통합할지가 중요합니다. Confidence Fusion은 다음 기능을 제공합니다:

- **가중치 기반 융합**: 소스별 중요도에 따른 가중치 적용
- **점수 정규화**: Min-Max, Softmax, Z-Score 정규화 지원
- **다양한 융합 전략**: 가중합, 조화평균, 기하평균, 최대/최소값
- **상충 감지**: 소스 간 점수 불일치 자동 감지
- **설명 생성**: 왜 이 결과가 나왔는지 해석 가능한 설명 제공

---

## 설치 및 임포트

```python
from src.rag.confidence_fusion import (
    ConfidenceFusion,
    SearchResult,
    InferenceResult,
    LinkedEntity,
    create_default_fusion,
    create_conservative_fusion,
    create_optimistic_fusion
)
```

---

## 기본 사용법

### 1. 기본 융합 (Default)

```python
from src.rag.confidence_fusion import create_default_fusion

# Fusion 객체 생성
fusion = create_default_fusion()

# 각 소스 데이터 준비
vector_results = [
    SearchResult(
        content="LANEIGE Lip Sleeping Mask is a bestseller",
        score=0.92,
        metadata={"doc_id": "doc1"}
    )
]

ontology_results = [
    InferenceResult(
        insight="LANEIGE dominates Lip Care category",
        confidence=0.88,
        evidence={"sos": 0.35, "rank": 1}
    )
]

entity_links = [
    LinkedEntity(
        entity_id="brand_laneige",
        entity_name="LANEIGE",
        entity_type="Brand",
        link_confidence=0.95
    )
]

# 융합 실행
result = fusion.fuse(
    vector_results=vector_results,
    ontology_results=ontology_results,
    entity_links=entity_links,
    query="LANEIGE 분석"
)

# 결과 확인
print(f"최종 신뢰도: {result.confidence:.3f}")
print(f"설명: {result.explanation}")

# 소스별 기여도
for source in result.source_scores:
    print(f"{source.source_name}: {source.contribution:.3f}")
```

---

## 주요 클래스

### ConfidenceFusion

메인 융합 클래스

```python
fusion = ConfidenceFusion(
    weights={
        'vector': 0.40,      # 벡터 검색 가중치
        'ontology': 0.35,    # 온톨로지 추론 가중치
        'entity': 0.25       # 엔티티 연결 가중치
    },
    normalization=ScoreNormalizationMethod.MIN_MAX,
    strategy=FusionStrategy.WEIGHTED_SUM,
    min_sources=1,
    conflict_threshold=0.3
)
```

**Parameters:**
- `weights`: 소스별 가중치 (합이 1.0이어야 함)
- `normalization`: 정규화 방법
- `strategy`: 융합 전략
- `min_sources`: 최소 필요 소스 수
- `conflict_threshold`: 상충 감지 임계값

---

## 데이터 클래스

### SearchResult (벡터/키워드 검색 결과)

```python
SearchResult(
    content="문서 내용",
    score=0.85,                    # 0~1 유사도 점수
    metadata={"doc_id": "123"},
    source="vector"                # vector, keyword, hybrid
)
```

### InferenceResult (온톨로지 추론 결과)

```python
InferenceResult(
    insight="추론된 인사이트 문장",
    confidence=0.88,               # 0~1 추론 신뢰도
    evidence={"sos": 0.35},        # 근거 데이터
    rule_name="market_dominance"   # 적용된 규칙명
)
```

### LinkedEntity (엔티티 연결 결과)

```python
LinkedEntity(
    entity_id="brand_laneige",
    entity_name="LANEIGE",
    entity_type="Brand",           # Brand, Product, Category
    link_confidence=0.95,          # 0~1 연결 강도
    context="Query mentioned LANEIGE"
)
```

### FusedResult (최종 융합 결과)

```python
result = FusedResult(
    documents=[...],               # 통합 문서들
    confidence=0.85,               # 최종 신뢰도
    explanation="...",             # 설명
    source_scores=[...],           # 소스별 점수
    fusion_strategy="weighted_sum",
    warnings=[...]                 # 경고 메시지
)
```

---

## 정규화 방법

### MIN_MAX (기본)

점수를 0~1 범위로 정규화

```python
fusion = ConfidenceFusion(
    normalization=ScoreNormalizationMethod.MIN_MAX
)
```

**특징:**
- 점수 범위를 0~1로 스케일링
- 상대적 차이를 극대화
- 모든 점수가 비슷하면 중간값(0.5)에 수렴

### SOFTMAX

확률 분포로 정규화 (합이 1.0)

```python
fusion = ConfidenceFusion(
    normalization=ScoreNormalizationMethod.SOFTMAX
)
```

**특징:**
- 각 소스의 상대적 중요도를 확률로 표현
- 높은 점수가 더 크게 부각됨
- 합이 1.0이 되도록 정규화

### Z_SCORE

표준 점수로 정규화

```python
fusion = ConfidenceFusion(
    normalization=ScoreNormalizationMethod.Z_SCORE
)
```

**특징:**
- 평균과 표준편차 기반 정규화
- 이상치에 강함
- -3~+3 범위를 0~1로 재스케일

### NONE

정규화 없이 원본 점수 사용

```python
fusion = ConfidenceFusion(
    normalization=ScoreNormalizationMethod.NONE
)
```

**사용 시기:** 모든 점수가 이미 0~1 범위이고 직접 비교 가능할 때

---

## 융합 전략

### WEIGHTED_SUM (기본, 추천)

가중치를 적용한 가중합

```python
fusion = ConfidenceFusion(
    strategy=FusionStrategy.WEIGHTED_SUM
)
```

**공식:**
```
confidence = w1 * s1 + w2 * s2 + w3 * s3
```

**특징:**
- 균형잡힌 결과
- 모든 소스가 기여
- 일반적인 상황에 적합

### HARMONIC_MEAN (보수적)

조화평균 사용 - 모든 점수가 높아야 높음

```python
fusion = ConfidenceFusion(
    strategy=FusionStrategy.HARMONIC_MEAN
)
```

**공식:**
```
confidence = 1 / (w1/s1 + w2/s2 + w3/s3)
```

**특징:**
- 매우 보수적
- 낮은 점수 하나가 전체를 끌어내림
- 고신뢰 시나리오에 적합

### GEOMETRIC_MEAN

기하평균 사용

```python
fusion = ConfidenceFusion(
    strategy=FusionStrategy.GEOMETRIC_MEAN
)
```

**공식:**
```
confidence = (s1^w1 * s2^w2 * s3^w3)
```

**특징:**
- Weighted Sum보다 보수적
- 균형잡힌 점수 요구

### MAX_SCORE (낙관적)

가장 높은 점수 사용

```python
fusion = ConfidenceFusion(
    strategy=FusionStrategy.MAX_SCORE
)
```

**특징:**
- 매우 낙관적
- 하나의 소스라도 높으면 높은 신뢰도
- 탐색적 분석에 적합

### MIN_SCORE (매우 보수적)

가장 낮은 점수 사용

```python
fusion = ConfidenceFusion(
    strategy=FusionStrategy.MIN_SCORE
)
```

**특징:**
- 극도로 보수적
- 모든 소스가 높아야만 높음
- 위험 회피 시나리오에 적합

---

## 편의 함수

### create_default_fusion()

기본 설정 (추천)

```python
fusion = create_default_fusion()
# weights: vector=0.4, ontology=0.35, entity=0.25
# normalization: MIN_MAX
# strategy: WEIGHTED_SUM
```

### create_conservative_fusion()

보수적 설정

```python
fusion = create_conservative_fusion()
# strategy: HARMONIC_MEAN
# min_sources: 2
# conflict_threshold: 0.2 (더 엄격)
```

### create_optimistic_fusion()

낙관적 설정

```python
fusion = create_optimistic_fusion()
# strategy: MAX_SCORE
# normalization: SOFTMAX
# conflict_threshold: 0.5 (느슨)
```

---

## 실전 사용 예제

### 예제 1: 챗봇 응답 신뢰도 평가

```python
def evaluate_chatbot_response(query, rag_docs, kg_insights, linked_entities):
    """챗봇 응답의 신뢰도를 평가하여 답변 톤 결정"""

    fusion = create_default_fusion()

    result = fusion.fuse(
        vector_results=rag_docs,
        ontology_results=kg_insights,
        entity_links=linked_entities,
        query=query
    )

    # 신뢰도 기반 응답 톤 선택
    if result.confidence > 0.75:
        tone = "확신 있는 답변"
        prefix = "데이터에 따르면,"
    elif result.confidence > 0.50:
        tone = "중립적 답변"
        prefix = "분석 결과,"
    else:
        tone = "조심스러운 답변"
        prefix = "현재 데이터로는 명확히 말하기 어렵지만,"

    return {
        "confidence": result.confidence,
        "tone": tone,
        "response": f"{prefix} {generate_answer(kg_insights)}",
        "warnings": result.warnings
    }
```

### 예제 2: 소스별 기여도 분석

```python
def analyze_source_contributions(result):
    """각 소스의 기여도를 분석"""

    print("=== 소스별 기여도 ===")
    for source in sorted(result.source_scores, key=lambda s: s.contribution, reverse=True):
        print(f"{source.source_name:10s} "
              f"[{source.confidence_level.upper():6s}] "
              f"기여: {source.contribution:.3f} "
              f"({source.contribution/result.confidence*100:.1f}%)")

    # 주요 기여 소스 식별
    major_contributors = [
        s for s in result.source_scores
        if s.contribution > 0.15
    ]

    print(f"\n주요 근거: {', '.join(s.source_name for s in major_contributors)}")
```

### 예제 3: 다중 쿼리 배치 처리

```python
def batch_evaluate_queries(queries, data_sources):
    """여러 쿼리를 배치로 평가"""

    fusion = create_default_fusion()
    results = []

    for query in queries:
        # 각 쿼리에 대해 소스 데이터 추출
        vector_results = data_sources['vector'][query]
        ontology_results = data_sources['ontology'][query]
        entity_links = data_sources['entities'][query]

        result = fusion.fuse(
            vector_results=vector_results,
            ontology_results=ontology_results,
            entity_links=entity_links,
            query=query
        )

        results.append({
            "query": query,
            "confidence": result.confidence,
            "level": "HIGH" if result.confidence > 0.75 else
                     "MEDIUM" if result.confidence > 0.50 else "LOW"
        })

    return results
```

---

## 신뢰도 해석 가이드

### 신뢰도 수준

| 점수 범위 | 수준 | 의미 | 권장 액션 |
|-----------|------|------|-----------|
| 0.75 ~ 1.0 | HIGH | 높은 신뢰도, 명확한 근거 | 확신 있는 답변 제공 |
| 0.50 ~ 0.75 | MEDIUM | 중간 신뢰도, 일부 근거 | 중립적 답변, 추가 컨텍스트 제공 |
| 0.25 ~ 0.50 | LOW | 낮은 신뢰도, 약한 근거 | 조심스러운 답변, 더 많은 정보 요청 |
| 0.0 ~ 0.25 | VERY_LOW | 매우 낮은 신뢰도 | "정보 부족" 명시, 답변 보류 |

### 소스별 기여도 해석

```python
# 각 소스의 기여도 계산
contribution = normalized_score * weight

# 전체 신뢰도 대비 비율
percentage = (contribution / total_confidence) * 100
```

**예시:**
```
vector:    0.92 * 0.40 = 0.368 (45% 기여)
ontology:  0.88 * 0.35 = 0.308 (38% 기여)
entity:    0.90 * 0.25 = 0.225 (27% 기여)
```

---

## 경고 메시지 처리

### 상충 감지 (Conflict Detection)

소스 간 점수 차이가 임계값을 초과하면 경고 발생

```python
result = fusion.fuse(...)

if result.warnings:
    for warning in result.warnings:
        print(f"⚠️  {warning}")

    # 상충 발생 시 처리
    if "점수 불일치" in result.warnings[0]:
        # 보수적 전략으로 재평가
        conservative = create_conservative_fusion()
        result = conservative.fuse(...)
```

### 예상 경고 유형

1. **점수 불일치**: 소스 간 점수 차이가 큼
2. **소스 부족**: min_sources 미달
3. **가중치 오류**: 가중치 합이 1.0이 아님

---

## 성능 최적화 팁

### 1. 소스 수 제한

불필요한 소스는 제외하여 연산 비용 절감

```python
# 벡터 검색만으로도 충분한 경우
result = fusion.fuse(
    vector_results=vector_results,
    ontology_results=None,
    entity_links=None
)
```

### 2. 정규화 생략

모든 점수가 이미 0~1 범위라면 정규화 생략

```python
fusion = ConfidenceFusion(
    normalization=ScoreNormalizationMethod.NONE
)
```

### 3. 캐싱

동일한 입력에 대해 결과 캐싱

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_fuse(query_hash, vector_hash, ontology_hash):
    # 융합 실행 및 캐싱
    ...
```

---

## 테스트

```bash
# 전체 테스트 실행
python -m pytest tests/test_confidence_fusion.py -v

# 특정 테스트만 실행
python -m pytest tests/test_confidence_fusion.py::test_fuse_all_sources -v

# 데모 실행
PYTHONPATH=. python3 examples/confidence_fusion_demo.py
```

---

## 문제 해결 (Troubleshooting)

### Q: 신뢰도가 예상보다 낮게 나옵니다

**A:** Min-Max 정규화가 점수를 상대화합니다. 원본 점수를 보려면:

```python
# 정규화 없이 실행
fusion = ConfidenceFusion(normalization=ScoreNormalizationMethod.NONE)
```

### Q: 모든 점수가 높은데 신뢰도가 낮습니다

**A:** Min-Max 정규화는 상대 차이를 극대화합니다. 해결책:

```python
# 1. 정규화 비활성화
fusion = ConfidenceFusion(normalization=ScoreNormalizationMethod.NONE)

# 2. 또는 Softmax 사용
fusion = ConfidenceFusion(normalization=ScoreNormalizationMethod.SOFTMAX)
```

### Q: 상충 경고가 자주 발생합니다

**A:** 임계값을 조정하세요:

```python
# 더 느슨한 임계값
fusion = ConfidenceFusion(conflict_threshold=0.5)
```

---

## API 레퍼런스

### ConfidenceFusion.fuse()

```python
def fuse(
    self,
    vector_results: Optional[List[SearchResult]] = None,
    ontology_results: Optional[List[InferenceResult]] = None,
    entity_links: Optional[List[LinkedEntity]] = None,
    query: Optional[str] = None
) -> FusedResult
```

**Returns:** `FusedResult` 객체

**Raises:** `ValueError` - 가중치 검증 실패 시

### FusedResult.to_dict()

```python
def to_dict(self) -> Dict[str, Any]
```

JSON 직렬화 가능한 딕셔너리로 변환

---

## 버전 정보

- **Version**: 1.0.0
- **Author**: AMORE RAG Team
- **Date**: 2026-01-23
- **Python**: 3.11+
- **Dependencies**: numpy

---

## 라이센스

이 모듈은 AMORE RAG-KG Hybrid Agent 프로젝트의 일부입니다.
