# Confidence Fusion Module

다중 소스 (벡터 검색, 온톨로지 추론, 엔티티 연결)의 신뢰도를 통합하는 모듈.

- **파일**: `src/rag/confidence_fusion.py`
- **테스트**: `tests/test_confidence_fusion.py` (19개 통과)
- **데모**: `examples/confidence_fusion_demo.py`
- **버전**: 1.0.0 (2026-01-23)

---

## 아키텍처

```
           User Query
               |
               v
       HybridRetriever
      /       |        \
     v        v         v
  Vector   Ontology   Entity
  Search   Reasoner   Linker
  (0.92)   (0.88)     (0.90)
      \       |        /
       v      v       v
      ConfidenceFusion
      - Normalize scores
      - Apply weights
      - Detect conflicts
      - Generate explanation
               |
               v
         FusedResult
      confidence: 0.85
      level: HIGH
```

### 처리 흐름

1. **Collect**: 각 소스에서 점수 수집 (Vector: Top-K 평균, Ontology: 최대값, Entity: 평균)
2. **Normalize**: MIN_MAX / SOFTMAX / Z_SCORE / NONE 중 선택
3. **Weight**: 소스별 가중치 적용 (vector=0.40, ontology=0.35, entity=0.25)
4. **Fuse**: 전략에 따라 최종 점수 계산
5. **Detect Conflicts**: 소스 간 점수 차이가 임계값 초과 시 경고
6. **Explain**: 해석 가능한 설명 생성

---

## 기본 사용법

```python
from src.rag.confidence_fusion import (
    ConfidenceFusion,
    SearchResult, InferenceResult, LinkedEntity,
    create_default_fusion, create_conservative_fusion, create_optimistic_fusion
)

fusion = create_default_fusion()

result = fusion.fuse(
    vector_results=[SearchResult(content="...", score=0.92, metadata={})],
    ontology_results=[InferenceResult(insight="...", confidence=0.88, evidence={})],
    entity_links=[LinkedEntity(entity_id="brand_laneige", entity_name="LANEIGE",
                               entity_type="Brand", link_confidence=0.95)],
    query="LANEIGE 분석"
)

print(f"신뢰도: {result.confidence:.3f}")  # 0.85
print(f"설명: {result.explanation}")
for source in result.source_scores:
    print(f"  {source.source_name}: {source.contribution:.3f}")
```

### 편의 함수

| 함수 | 전략 | min_sources | conflict_threshold |
|------|------|-------------|-------------------|
| `create_default_fusion()` | WEIGHTED_SUM | 1 | 0.3 |
| `create_conservative_fusion()` | HARMONIC_MEAN | 2 | 0.2 |
| `create_optimistic_fusion()` | MAX_SCORE | 1 | 0.5 |

---

## 데이터 클래스

```python
# 입력
SearchResult(content: str, score: float, metadata: dict, source: str = "vector")
InferenceResult(insight: str, confidence: float, evidence: dict, rule_name: str = None)
LinkedEntity(entity_id: str, entity_name: str, entity_type: str,
             link_confidence: float, context: str = None)

# 출력
FusedResult(documents: list, confidence: float, explanation: str,
            source_scores: list[SourceScore], fusion_strategy: str, warnings: list)
```

---

## 융합 전략 비교

점수 예시: Vector=0.9, Ontology=0.7, Entity=0.8 / 가중치: 0.40, 0.35, 0.25

| 전략 | 공식 | 결과 | 특징 |
|------|------|------|------|
| **WEIGHTED_SUM** (기본) | w1*s1 + w2*s2 + w3*s3 | 0.805 | 균형, 모든 소스 반영 |
| HARMONIC_MEAN | 1/(w1/s1 + w2/s2 + w3/s3) | 0.795 | 보수적, 낮은 점수가 끌어내림 |
| GEOMETRIC_MEAN | s1^w1 * s2^w2 * s3^w3 | 0.799 | 약간 보수적 |
| MAX_SCORE | max(scores) | 0.900 | 낙관적 |
| MIN_SCORE | min(scores) | 0.700 | 극도로 보수적 |

순서: MIN < HARMONIC < GEOMETRIC < WEIGHTED < MAX

---

## 정규화 방법

| 방법 | 설명 | 사용 시기 |
|------|------|----------|
| **MIN_MAX** (기본) | 0~1 범위로 스케일링 | 일반적 |
| SOFTMAX | 확률 분포 (합=1.0) | 상대적 중요도 비교 |
| Z_SCORE | 표준점수 기반 | 이상치에 강함 |
| NONE | 원본 점수 유지 | 이미 0~1 범위일 때 |

> **주의**: MIN_MAX는 모든 점수가 비슷하면 상대 차이를 극대화합니다.
> 모든 점수가 높은데 신뢰도가 낮게 나오면 `NONE` 또는 `SOFTMAX`를 사용하세요.

---

## 신뢰도 수준

| 점수 | 수준 | 의미 | 챗봇 동작 |
|------|------|------|----------|
| 0.75~1.0 | HIGH | 명확한 근거 | "데이터에 따르면," |
| 0.50~0.75 | MEDIUM | 일부 근거 | "분석 결과," |
| 0.25~0.50 | LOW | 약한 근거 | "현재 데이터로는 명확하지 않지만," |
| 0.0~0.25 | VERY_LOW | 정보 부족 | 답변 보류 |

---

## HybridRetriever 통합

```python
class HybridRetriever:
    def __init__(self, ...):
        self.confidence_fusion = create_default_fusion()

    def retrieve(self, query, top_k=5, return_confidence=True):
        vector_docs = self.document_retriever.search(query, top_k=top_k)
        kg_insights = self.ontology_reasoner.infer(query)
        linked_entities = self._link_entities(query)

        if return_confidence:
            fused = self.confidence_fusion.fuse(
                vector_results=[SearchResult(content=d["content"], score=d["score"], metadata=d.get("metadata", {})) for d in vector_docs],
                ontology_results=[InferenceResult(insight=i["insight"], confidence=i["confidence"], evidence=i.get("evidence", {})) for i in kg_insights],
                entity_links=[LinkedEntity(**e) for e in linked_entities],
                query=query
            )
            return {
                "documents": vector_docs[:top_k],
                "kg_insights": kg_insights,
                "confidence": fused.confidence,
                "confidence_level": fused.explanation,
                "source_breakdown": {s.source_name: s.contribution for s in fused.source_scores},
                "warnings": fused.warnings,
            }
        return {"documents": vector_docs[:top_k], "kg_insights": kg_insights}
```

---

## 상충 감지

소스 간 점수 차이가 `conflict_threshold`(기본 0.3)를 초과하면 경고.

```python
result = fusion.fuse(...)
if result.warnings:
    # 보수적 전략으로 재평가
    conservative = create_conservative_fusion()
    result = conservative.fuse(...)
```

---

## 성능 및 제한

- **연산**: <1ms (numpy 기반)
- **의존성**: `numpy>=1.24.0`
- **제한**: MIN_MAX는 점수 범위에 민감 / 가중치는 수동 조정 (학습 기반 아님)

### 향후 개선

- 사용자 피드백 기반 가중치 자동 조정
- 쿼리 타입별 가중치 전략
- 시간적 감쇠 (오래된 정보의 신뢰도 감소)
- 앙상블 전략 (보팅 메커니즘)

---

## 테스트

```bash
python3 -m pytest tests/test_confidence_fusion.py -v      # 유닛 테스트
PYTHONPATH=. python3 examples/confidence_fusion_demo.py    # 데모
```
