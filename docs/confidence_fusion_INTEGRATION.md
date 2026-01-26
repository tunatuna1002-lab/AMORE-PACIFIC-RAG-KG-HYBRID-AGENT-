# Confidence Fusion 통합 가이드

기존 `HybridRetriever`에 Confidence Fusion을 통합하는 단계별 가이드

---

## 통합 목표

기존 `src/rag/hybrid_retriever.py`의 `HybridRetriever` 클래스에 신뢰도 융합 기능을 추가하여:

1. 다중 소스의 신뢰도를 통합
2. 검색 결과에 신뢰도 점수 추가
3. 소스별 기여도 분석 제공
4. 상충 감지 및 경고

---

## Step 1: Import 추가

`src/rag/hybrid_retriever.py` 파일 상단에 추가:

```python
from src.rag.confidence_fusion import (
    ConfidenceFusion,
    SearchResult,
    InferenceResult,
    LinkedEntity,
    create_default_fusion
)
```

---

## Step 2: HybridRetriever 클래스 수정

### 2.1 초기화에 Fusion 추가

```python
class HybridRetriever:
    def __init__(self, ...):
        # 기존 초기화 코드
        self.knowledge_graph = KnowledgeGraph()
        self.ontology_reasoner = OntologyReasoner(self.knowledge_graph)
        self.document_retriever = DocumentRetriever()

        # NEW: Confidence Fusion 추가
        self.confidence_fusion = create_default_fusion()
```

### 2.2 검색 메서드 수정

기존 `retrieve()` 메서드를 확장:

```python
def retrieve(
    self,
    query: str,
    top_k: int = 5,
    return_confidence: bool = True  # NEW: 신뢰도 반환 옵션
) -> Dict[str, Any]:
    """
    하이브리드 검색 수행

    Args:
        query: 검색 쿼리
        top_k: 반환할 문서 수
        return_confidence: 신뢰도 정보 포함 여부

    Returns:
        검색 결과 (신뢰도 포함)
    """
    # 1. 기존 벡터 검색
    vector_docs = self.document_retriever.search(query, top_k=top_k)

    # 2. 기존 온톨로지 추론
    kg_insights = self.ontology_reasoner.infer(query)

    # 3. 기존 엔티티 연결
    linked_entities = self._link_entities(query)

    # 4. NEW: Confidence Fusion 적용
    if return_confidence:
        fused_result = self._compute_confidence(
            query=query,
            vector_docs=vector_docs,
            kg_insights=kg_insights,
            linked_entities=linked_entities
        )

        return {
            "query": query,
            "documents": vector_docs[:top_k],
            "kg_insights": kg_insights,
            "entities": linked_entities,
            # NEW: 신뢰도 정보
            "confidence": fused_result.confidence,
            "confidence_level": self._get_confidence_level(fused_result.confidence),
            "explanation": fused_result.explanation,
            "source_breakdown": {
                s.source_name: {
                    "score": s.raw_score,
                    "contribution": s.contribution,
                    "level": s.confidence_level
                }
                for s in fused_result.source_scores
            },
            "warnings": fused_result.warnings
        }
    else:
        # 기존 동작 유지 (하위 호환성)
        return {
            "query": query,
            "documents": vector_docs[:top_k],
            "kg_insights": kg_insights,
            "entities": linked_entities
        }
```

### 2.3 신뢰도 계산 메서드 추가

```python
def _compute_confidence(
    self,
    query: str,
    vector_docs: List[Dict],
    kg_insights: List[Dict],
    linked_entities: List[Dict]
) -> "FusedResult":
    """
    다중 소스 신뢰도 계산

    Args:
        query: 검색 쿼리
        vector_docs: 벡터 검색 결과
        kg_insights: 온톨로지 추론 결과
        linked_entities: 엔티티 연결 결과

    Returns:
        FusedResult 객체
    """
    # 벡터 검색 결과를 SearchResult로 변환
    vector_results = [
        SearchResult(
            content=doc.get("content", ""),
            score=doc.get("score", 0.0),
            metadata=doc.get("metadata", {}),
            source="vector"
        )
        for doc in vector_docs
    ]

    # 온톨로지 추론 결과를 InferenceResult로 변환
    ontology_results = [
        InferenceResult(
            insight=insight.get("insight", ""),
            confidence=insight.get("confidence", 0.0),
            evidence=insight.get("evidence", {}),
            rule_name=insight.get("rule_name")
        )
        for insight in kg_insights
    ]

    # 엔티티 연결 결과를 LinkedEntity로 변환
    entity_links = [
        LinkedEntity(
            entity_id=entity.get("entity_id", ""),
            entity_name=entity.get("entity_name", ""),
            entity_type=entity.get("entity_type", ""),
            link_confidence=entity.get("link_confidence", 0.0),
            context=entity.get("context")
        )
        for entity in linked_entities
    ]

    # Confidence Fusion 실행
    return self.confidence_fusion.fuse(
        vector_results=vector_results,
        ontology_results=ontology_results,
        entity_links=entity_links,
        query=query
    )

def _get_confidence_level(self, confidence: float) -> str:
    """신뢰도 수준 레이블"""
    if confidence >= 0.75:
        return "HIGH"
    elif confidence >= 0.50:
        return "MEDIUM"
    elif confidence >= 0.25:
        return "LOW"
    else:
        return "VERY_LOW"
```

---

## Step 3: 챗봇 에이전트 통합

`src/agents/hybrid_chatbot_agent.py` 수정:

### 3.1 신뢰도 기반 응답 생성

```python
class HybridChatbotAgent:
    def chat(self, user_query: str) -> str:
        """챗봇 응답 생성"""

        # 하이브리드 검색 (신뢰도 포함)
        search_result = self.retriever.retrieve(
            query=user_query,
            top_k=5,
            return_confidence=True  # 신뢰도 활성화
        )

        confidence = search_result["confidence"]
        confidence_level = search_result["confidence_level"]

        # NEW: 신뢰도 기반 답변 톤 결정
        if confidence_level == "HIGH":
            tone = "확신 있는 답변"
            prefix = "데이터에 따르면,"
        elif confidence_level == "MEDIUM":
            tone = "중립적 답변"
            prefix = "분석 결과,"
        elif confidence_level == "LOW":
            tone = "조심스러운 답변"
            prefix = "현재 데이터로는 명확하지 않지만,"
        else:
            # VERY_LOW - 답변 보류
            return "죄송합니다. 해당 질문에 대한 충분한 정보가 없습니다. 다른 방식으로 질문해 주시겠어요?"

        # 컨텍스트 구성
        context = self._build_context(search_result)

        # LLM 프롬프트에 신뢰도 정보 포함
        prompt = f"""
질문: {user_query}

신뢰도: {confidence:.2f} ({confidence_level})
답변 톤: {tone}

근거 데이터:
{context}

위 정보를 바탕으로 {prefix} 답변해주세요.
"""

        # LLM 호출
        response = self.llm.generate(prompt)

        # NEW: 경고가 있으면 추가 안내
        if search_result.get("warnings"):
            response += "\n\n💡 참고: 일부 정보원 간에 불일치가 있어 추가 확인이 필요할 수 있습니다."

        return response
```

---

## Step 4: 대시보드 표시 (선택)

`dashboard/amore_unified_dashboard_v4.html` 또는 FastAPI 응답에 신뢰도 추가:

### 4.1 API 응답 수정

`dashboard_api.py`:

```python
@app.post("/api/v2/chat")
async def chat_v2(request: ChatRequest):
    """채팅 엔드포인트 (신뢰도 포함)"""

    result = chatbot_agent.chat(request.message)

    # NEW: 신뢰도 정보 포함
    return {
        "answer": result["answer"],
        "confidence": result.get("confidence", 0.0),
        "confidence_level": result.get("confidence_level", "UNKNOWN"),
        "source_breakdown": result.get("source_breakdown", {}),
        "warnings": result.get("warnings", [])
    }
```

### 4.2 프론트엔드 표시

```javascript
// 신뢰도 배지 표시
function displayConfidenceBadge(confidence, level) {
    const colors = {
        "HIGH": "success",
        "MEDIUM": "warning",
        "LOW": "danger",
        "VERY_LOW": "secondary"
    };

    return `
        <span class="badge badge-${colors[level]}">
            신뢰도: ${(confidence * 100).toFixed(0)}% (${level})
        </span>
    `;
}

// 소스별 기여도 차트
function displaySourceBreakdown(breakdown) {
    const canvas = document.getElementById('sourceChart');
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: Object.keys(breakdown),
            datasets: [{
                label: '기여도',
                data: Object.values(breakdown).map(s => s.contribution)
            }]
        }
    });
}
```

---

## Step 5: 로깅 추가 (선택)

`src/monitoring/logger.py`에 신뢰도 로깅:

```python
def log_chat_with_confidence(query, result):
    """채팅 이벤트 로깅 (신뢰도 포함)"""
    logger.info(
        "chat_response",
        query=query,
        confidence=result.get("confidence", 0.0),
        confidence_level=result.get("confidence_level", "UNKNOWN"),
        source_breakdown=result.get("source_breakdown", {}),
        warnings=result.get("warnings", [])
    )
```

---

## Step 6: 테스트

### 6.1 유닛 테스트

`tests/test_hybrid_retriever.py`:

```python
def test_retrieve_with_confidence():
    """신뢰도 포함 검색 테스트"""
    retriever = HybridRetriever()

    result = retriever.retrieve(
        query="LANEIGE Lip Sleeping Mask",
        return_confidence=True
    )

    # 신뢰도 필드 존재 확인
    assert "confidence" in result
    assert "confidence_level" in result
    assert "source_breakdown" in result

    # 신뢰도 범위 확인
    assert 0.0 <= result["confidence"] <= 1.0

    # 소스별 기여도 확인
    assert "vector" in result["source_breakdown"]
```

### 6.2 통합 테스트

```bash
# 챗봇 테스트
curl -X POST http://localhost:8001/api/v2/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "LANEIGE 분석"}' \
  | jq '.confidence'
```

---

## Step 7: 배포

### 7.1 의존성 확인

`requirements.txt`에 numpy 확인:

```txt
numpy>=1.24.0
```

### 7.2 환경 변수 (선택)

`.env`:

```bash
# Confidence Fusion 설정
CONFIDENCE_MIN_SOURCES=1
CONFIDENCE_THRESHOLD=0.3
CONFIDENCE_STRATEGY=weighted_sum  # weighted_sum, harmonic_mean, max_score
```

### 7.3 배포 체크리스트

- [ ] `src/rag/confidence_fusion.py` 파일 존재 확인
- [ ] `numpy` 설치 확인
- [ ] 기존 테스트 통과 확인
- [ ] 새 테스트 작성 및 통과
- [ ] API 응답 스키마 업데이트 (문서화)
- [ ] 프론트엔드 신뢰도 표시 (선택)
- [ ] 로깅 설정 (선택)

---

## 사용 예제

### 예제 1: 기본 사용

```python
retriever = HybridRetriever()

# 신뢰도 포함 검색
result = retriever.retrieve(
    query="LANEIGE Lip Sleeping Mask 분석",
    return_confidence=True
)

print(f"신뢰도: {result['confidence']:.3f}")
print(f"수준: {result['confidence_level']}")

# 신뢰도 기반 처리
if result['confidence_level'] == 'HIGH':
    print("확신 있는 답변 제공")
elif result['confidence_level'] == 'MEDIUM':
    print("중립적 답변 제공")
else:
    print("조심스러운 답변 또는 답변 보류")
```

### 예제 2: 소스별 기여도 분석

```python
result = retriever.retrieve(query, return_confidence=True)

print("\n소스별 기여도:")
for source, scores in result['source_breakdown'].items():
    print(f"  {source}: {scores['contribution']:.3f} ({scores['level']})")
```

### 예제 3: 경고 처리

```python
result = retriever.retrieve(query, return_confidence=True)

if result['warnings']:
    print("\n⚠️  경고:")
    for warning in result['warnings']:
        print(f"  • {warning}")

    # 보수적 전략으로 재평가
    retriever.confidence_fusion = create_conservative_fusion()
    result = retriever.retrieve(query, return_confidence=True)
```

---

## 하위 호환성 유지

기존 코드가 깨지지 않도록:

```python
# 기존 방식 (신뢰도 없음) - 여전히 동작
result = retriever.retrieve(
    query="LANEIGE",
    return_confidence=False  # 또는 생략
)
# result는 기존 형식 그대로

# 새 방식 (신뢰도 포함)
result = retriever.retrieve(
    query="LANEIGE",
    return_confidence=True
)
# result에 confidence, source_breakdown 등 추가
```

---

## 문제 해결

### Q: 신뢰도가 항상 낮게 나옵니다

**A:** Min-Max 정규화 때문일 수 있습니다. 정규화 없이 시도:

```python
from src.rag.confidence_fusion import ConfidenceFusion, ScoreNormalizationMethod

self.confidence_fusion = ConfidenceFusion(
    normalization=ScoreNormalizationMethod.NONE
)
```

### Q: 특정 소스의 가중치를 높이고 싶습니다

**A:** 커스텀 가중치 설정:

```python
self.confidence_fusion = ConfidenceFusion(
    weights={
        'vector': 0.50,      # 벡터 검색 강조
        'ontology': 0.30,
        'entity': 0.20
    }
)
```

### Q: 매우 보수적인 신뢰도를 원합니다

**A:** Harmonic Mean 전략 사용:

```python
from src.rag.confidence_fusion import create_conservative_fusion

self.confidence_fusion = create_conservative_fusion()
```

---

## 성능 고려사항

- **연산 비용**: 매우 낮음 (numpy 연산, <1ms)
- **메모리**: 무시할 수준 (인메모리 처리)
- **확장성**: 소스 수에 선형 비례 (O(n))

---

## 다음 단계

1. ✅ 기본 통합 완료
2. ⬜ A/B 테스트로 최적 가중치 탐색
3. ⬜ 사용자 피드백 수집
4. ⬜ 쿼리 타입별 전략 최적화
5. ⬜ 대시보드에 신뢰도 시각화

---

**통합 가이드 완료 ✅**
