# eval/ - 평가 프레임워크

## 개요

RAG-KG 하이브리드 시스템의 성능을 정량적으로 평가하는 5단계 평가 파이프라인입니다.
쿼리 품질부터 최종 답변까지 전체 파이프라인을 체계적으로 측정합니다.

## 평가 파이프라인

```
Query → L1 → L2 → L3 → L4 → L5 → Final Score
         ↓    ↓    ↓    ↓    ↓
      Query Ret  KG  Onto Ans
      Quality    ...  ...  Quality
```

## 디렉토리 구조

| 디렉토리 | 파일 수 | 설명 |
|----------|---------|------|
| `metrics/` | 8개 | L1-L5 평가 메트릭 |
| `judge/` | 4개 | 판단 엔진 (LLM, NLI) |
| `validators/` | 1개 | Ontology 검증기 |
| 루트 | 8개 | 평가 실행, 리포트, CLI |

## 5단계 평가 레벨

### L1: Query Quality (쿼리 품질)

| 메트릭 | 설명 | 범위 |
|--------|------|------|
| Clarity | 쿼리 명확성 | 0-1 |
| Complexity | 쿼리 복잡도 | 1-5 |
| Intent Recognition | 의도 파악 정확도 | 0-1 |

**파일**: `metrics/l1_query.py`

```python
# 사용 예시
from eval.metrics.l1_query import QueryQualityMetric

metric = QueryQualityMetric()
score = metric.evaluate(query="LANEIGE 1위 제품은?")
# {'clarity': 0.95, 'complexity': 2, 'intent': 0.9}
```

### L2: Retrieval Quality (검색 품질)

| 메트릭 | 설명 | 범위 |
|--------|------|------|
| Precision | 검색된 문서 중 관련 문서 비율 | 0-1 |
| Recall | 관련 문서 중 검색된 문서 비율 | 0-1 |
| F1 Score | Precision과 Recall의 조화 평균 | 0-1 |
| MRR | Mean Reciprocal Rank | 0-1 |

**파일**: `metrics/l2_retrieval.py`

```python
from eval.metrics.l2_retrieval import RetrievalMetric

metric = RetrievalMetric()
score = metric.evaluate(
    retrieved_docs=[doc1, doc2, doc3],
    relevant_docs=[doc1, doc4]
)
# {'precision': 0.33, 'recall': 0.5, 'f1': 0.4}
```

### L3: Knowledge Graph Quality (KG 품질)

| 메트릭 | 설명 | 범위 |
|--------|------|------|
| Triple Accuracy | 추출된 트리플 정확도 | 0-1 |
| Coverage | KG 커버리지 | 0-1 |
| Freshness | 데이터 신선도 | 0-1 |

**파일**: `metrics/l3_kg.py`

```python
from eval.metrics.l3_kg import KGMetric

metric = KGMetric()
score = metric.evaluate(
    kg=knowledge_graph,
    ground_truth=expected_triples
)
# {'accuracy': 0.85, 'coverage': 0.7, 'freshness': 0.9}
```

### L4: Ontology Reasoning Quality (추론 품질)

| 메트릭 | 설명 | 범위 |
|--------|------|------|
| Inference Correctness | 추론 정확도 | 0-1 |
| Consistency | 논리적 일관성 | 0-1 |
| Completeness | 추론 완성도 | 0-1 |

**파일**: `metrics/l4_ontology.py`

```python
from eval.metrics.l4_ontology import OntologyMetric

metric = OntologyMetric()
score = metric.evaluate(
    reasoner=reasoner,
    test_cases=ontology_test_cases
)
# {'correctness': 0.8, 'consistency': 0.95}
```

### L5: Answer Quality (답변 품질)

| 메트릭 | 설명 | 범위 |
|--------|------|------|
| Relevance | 질문과의 관련성 | 0-1 |
| Correctness | 사실적 정확성 | 0-1 |
| Completeness | 답변 완성도 | 0-1 |
| Fluency | 자연스러움 | 0-1 |

**파일**: `metrics/l5_answer.py`

```python
from eval.metrics.l5_answer import AnswerQualityMetric

metric = AnswerQualityMetric()
score = metric.evaluate(
    answer="LANEIGE Lip Sleeping Mask는 1위입니다.",
    expected="LANEIGE Lip Sleeping Mask가 1위입니다.",
    context=retrieved_docs
)
# {'relevance': 0.95, 'correctness': 1.0, 'completeness': 0.9}
```

## 추가 메트릭

### Semantic Similarity

**파일**: `metrics/semantic.py`

```python
from eval.metrics.semantic import SemanticSimilarity

metric = SemanticSimilarity()
score = metric.calculate(
    text1="LANEIGE는 1위입니다",
    text2="LANEIGE가 1위입니다"
)
# 0.98 (매우 유사)
```

### Aggregator

**파일**: `metrics/aggregator.py`

L1-L5 점수를 가중 평균으로 통합합니다.

```python
from eval.metrics.aggregator import MetricAggregator

aggregator = MetricAggregator(weights={
    'l1': 0.1,
    'l2': 0.2,
    'l3': 0.2,
    'l4': 0.2,
    'l5': 0.3
})

final_score = aggregator.aggregate({
    'l1': 0.9,
    'l2': 0.8,
    'l3': 0.85,
    'l4': 0.75,
    'l5': 0.9
})
# 0.84
```

## Judge (판단 엔진)

### LLM-as-Judge

**파일**: `judge/llm.py`

GPT-4를 판단자로 사용합니다.

```python
from eval.judge.llm import LLMJudge

judge = LLMJudge(model="gpt-4.1-mini")
verdict = await judge.judge(
    question="LANEIGE 1위 제품은?",
    answer="Lip Sleeping Mask입니다",
    criteria="correctness"
)
# {'score': 0.9, 'reasoning': '정확하지만 브랜드명 누락'}
```

### NLI Judge

**파일**: `judge/nli.py`

Natural Language Inference 모델을 사용합니다.

```python
from eval.judge.nli import NLIJudge

judge = NLIJudge()
result = judge.judge(
    premise="LANEIGE는 1위입니다",
    hypothesis="LANEIGE가 1위입니다"
)
# 'entailment' (함의)
```

### Stub Judge (테스트용)

**파일**: `judge/stub.py`

Mock 판단자 (테스트용).

## Validators

### Ontology Validator

**파일**: `validators/ontology_validator.py`

Ontology 무결성 검증:

```python
from eval.validators.ontology_validator import OntologyValidator

validator = OntologyValidator()
issues = validator.validate(ontology)
# [{'type': 'inconsistency', 'message': 'Cycle detected'}]
```

## 핵심 파일

### loader.py

골든셋 데이터 로더.

```python
from eval.loader import GoldenSetLoader

loader = GoldenSetLoader("tests/golden/")
queries = loader.load_queries()
# [{'query': '...', 'expected': '...', 'category': 'factual'}]
```

### runner.py

평가 실행기.

```python
from eval.runner import EvaluationRunner

runner = EvaluationRunner(
    metrics=['l1', 'l2', 'l3', 'l4', 'l5'],
    golden_set='tests/golden/'
)
results = await runner.run()
# {'overall_score': 0.84, 'l1': 0.9, ...}
```

### schemas.py

평가 데이터 스키마.

```python
from eval.schemas import EvaluationResult, MetricScore

result = EvaluationResult(
    query_id="Q001",
    scores={
        'l1': MetricScore(value=0.9, metadata={}),
        'l2': MetricScore(value=0.8, metadata={})
    }
)
```

### report.py

리포트 생성기.

```python
from eval.report import ReportGenerator

generator = ReportGenerator()
report = generator.generate(results, format='html')
# HTML 리포트 생성
```

### regression.py

회귀 테스트.

```python
from eval.regression import RegressionTester

tester = RegressionTester()
diff = tester.compare(
    baseline='results/baseline.json',
    current='results/current.json'
)
# {'degraded': ['l2'], 'improved': ['l5']}
```

### cost_tracker.py

LLM API 비용 추적.

```python
from eval.cost_tracker import CostTracker

tracker = CostTracker()
tracker.record(model='gpt-4.1-mini', tokens=1500)
report = tracker.summary()
# {'total_cost': 0.045, 'total_tokens': 30000}
```

### cli.py

CLI 인터페이스.

```bash
# 전체 평가 실행
python3 -m eval.cli run --golden tests/golden/

# 회귀 테스트
python3 -m eval.cli regression --baseline results/baseline.json

# 비용 리포트
python3 -m eval.cli cost-report
```

## 실행 방법

### 1. 골든셋 평가

```bash
# 전체 30개 쿼리 평가
python3 scripts/evaluate_golden.py --verbose

# 특정 카테고리만
python3 scripts/evaluate_golden.py --category factual

# HTML 리포트 생성
python3 scripts/evaluate_golden.py --format html --output report.html
```

### 2. 회귀 테스트

```bash
# 베이스라인 저장
python3 -m eval.runner --save-baseline results/baseline.json

# 회귀 비교
python3 -m eval.regression --baseline results/baseline.json
```

### 3. 비용 추적

```bash
# 비용 추적 활성화
TRACK_COST=true python3 scripts/evaluate_golden.py

# 리포트 확인
python3 -m eval.cost_tracker --report
```

## 골든셋 구조

```
tests/golden/
├── queries.json              # 30개 쿼리
├── expected_answers.json     # 정답 세트
└── metadata.json             # 난이도, 카테고리
```

### queries.json 예시

```json
[
  {
    "id": "Q001",
    "query": "LANEIGE 1위 제품은?",
    "category": "factual",
    "difficulty": "easy"
  },
  {
    "id": "Q002",
    "query": "경쟁사 대비 LANEIGE 시장 점유율은?",
    "category": "analytical",
    "difficulty": "medium"
  }
]
```

## 평가 지표 요약

| Level | 핵심 메트릭 | 목표 값 |
|-------|------------|---------|
| L1 | Clarity | ≥ 0.9 |
| L2 | F1 Score | ≥ 0.7 |
| L3 | Triple Accuracy | ≥ 0.8 |
| L4 | Consistency | ≥ 0.9 |
| L5 | Correctness | ≥ 0.85 |
| **Overall** | **Weighted Avg** | **≥ 0.8** |

## 주의사항

1. **LLM 비용**: LLM-as-Judge는 비용이 높음 (NLI 우선 사용)
2. **골든셋 품질**: 정답 세트의 품질이 평가 신뢰도를 결정
3. **캐싱**: 반복 평가 시 LLM 응답 캐싱 권장
4. **회귀 감지**: 변경 전후 반드시 회귀 테스트
5. **통계적 유의성**: 최소 30개 이상 샘플로 평가

## 개선 방향

- [ ] RAGAS 메트릭 통합 (Context Relevance, Faithfulness)
- [ ] 자동 골든셋 생성 (LLM 기반)
- [ ] A/B 테스트 프레임워크
- [ ] 실시간 모니터링 대시보드
- [ ] 다국어 평가 지원

## 참고

- `tests/golden/` - 골든셋 데이터
- `scripts/evaluate_golden.py` - 평가 실행 스크립트
- `CLAUDE.md` - 평가 목표 및 기준
- [RAGAS 문서](https://docs.ragas.io/)
