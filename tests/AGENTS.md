# tests/ - 테스트 구조

## 개요

프로젝트의 모든 테스트 코드를 포함합니다.
단위 테스트, 평가 테스트, 통합 테스트로 구성됩니다.

## 테스트 통계

- **전체 테스트**: 841개 collected
- **커버리지**: 10.11% (목표: 60%)
- **테스트 프레임워크**: pytest, pytest-cov
- **환경 분리**: `.env.test` 사용

## 디렉토리 구조

| 디렉토리 | 테스트 수 | 설명 |
|----------|-----------|------|
| `unit/` | 대부분 | 단위 테스트 (레이어별) |
| `eval/` | ~50개 | 평가 프레임워크 테스트 |
| `integration/` | ~10개 | 통합 테스트 (파이프라인) |
| `golden/` | 30개 | 골든셋 테스트 케이스 |

## unit/ - 단위 테스트

### 레이어별 구조

| 디렉토리 | 파일 수 | 주요 테스트 대상 |
|----------|---------|-----------------|
| `domain/` | 3개 | Entities, Interfaces |
| `agents/` | 5개 | Chatbot, Insight, Crawler |
| `core/` | 3개 | Brain, ReAct Agent |
| `ontology/` | 4개 | KG, Reasoner, Rules |
| `rag/` | 7개 | Retriever, Router, Fusion |
| `tools/` | 8개 | Scraper, Metrics, Collectors |
| `api/` | 2개 | FastAPI endpoints |
| `prompts/` | 1개 | Prompt 템플릿 |
| `application/` | 5개 | Workflows (예정) |
| `infrastructure/` | 2개 | DB, Storage (예정) |
| `memory/` | 1개 | History |
| `monitoring/` | 1개 | 모니터링 |

### 주요 테스트 파일

```
unit/
├── domain/
│   ├── test_entities.py          # Product, Category, Brand
│   ├── test_interfaces.py        # Protocol 검증
│   └── test_value_objects.py     # VO 불변성
├── agents/
│   ├── test_hybrid_chatbot_agent.py
│   ├── test_hybrid_insight_agent.py
│   ├── test_crawler_agent.py
│   ├── test_alert_agent.py
│   └── test_workflow_agent.py
├── core/
│   ├── test_brain.py             # UnifiedBrain
│   ├── test_react_agent.py      # ReAct Self-Reflection
│   └── test_decision_maker.py
├── ontology/
│   ├── test_knowledge_graph.py   # Triple Store
│   ├── test_reasoner.py          # 추론 엔진
│   ├── test_kg_query.py
│   └── test_kg_updater.py
├── rag/
│   ├── test_retriever.py         # 문서 검색
│   ├── test_router.py            # 쿼리 라우팅
│   ├── test_confidence_fusion.py # 신뢰도 통합
│   ├── test_context_builder.py
│   ├── test_query_rewriter.py
│   └── test_templates.py
└── tools/
    ├── test_amazon_scraper.py
    ├── test_metric_calculator.py
    ├── test_tiktok_collector.py
    ├── test_instagram_collector.py
    └── test_youtube_collector.py
```

## eval/ - 평가 프레임워크

### 5단계 평가 파이프라인

| Level | 평가 대상 | 메트릭 |
|-------|----------|--------|
| **L1** | Query Quality | Clarity, Complexity, Intent |
| **L2** | Retrieval | Precision, Recall, F1 |
| **L3** | KG | Triple Accuracy, Coverage |
| **L4** | Ontology | Reasoning, Consistency |
| **L5** | Answer | Relevance, Correctness, Completeness |

### 파일 구조

```
eval/
├── metrics/
│   ├── l1_query.py               # Query 품질
│   ├── l2_retrieval.py           # Retrieval 성능
│   ├── l3_kg.py                  # KG 정확도
│   ├── l4_ontology.py            # Ontology 추론
│   ├── l5_answer.py              # 답변 품질
│   ├── semantic.py               # 의미적 유사도
│   ├── aggregator.py             # 종합 점수
│   └── base.py                   # Base Metric
├── judge/
│   ├── llm.py                    # LLM-as-Judge
│   ├── nli.py                    # NLI 모델
│   ├── stub.py                   # Mock Judge
│   └── interface.py              # Judge Protocol
├── validators/
│   └── ontology_validator.py    # Ontology 검증
├── loader.py                     # Golden Set 로더
├── runner.py                     # 평가 실행기
├── schemas.py                    # 평가 스키마
├── report.py                     # 리포트 생성
├── regression.py                 # 회귀 테스트
├── cost_tracker.py               # 비용 추적
└── cli.py                        # CLI 인터페이스
```

### 평가 실행

```bash
# 전체 평가 (골든셋 30개)
python3 scripts/evaluate_golden.py --verbose

# 회귀 테스트 (이전 결과와 비교)
python3 -m eval.regression --baseline results/baseline.json

# 비용 추적
python3 -m eval.cost_tracker
```

## integration/ - 통합 테스트

엔드투엔드 워크플로우 테스트:

```python
# integration/test_pipeline.py
async def test_full_pipeline():
    # 크롤링 → DB 저장 → KG 업데이트 → 인사이트 생성
    products = await crawler.crawl("beauty")
    await storage.save(products)
    await kg.update(products)
    insight = await insight_agent.generate()
    assert insight.confidence > 0.7
```

## golden/ - 골든셋

### 구조

```
golden/
├── queries.json                  # 30개 쿼리
├── expected_answers.json         # 정답 세트
└── metadata.json                 # 난이도, 카테고리
```

### 골든셋 카테고리

| 카테고리 | 쿼리 수 | 예시 |
|----------|---------|------|
| Factual | 10개 | "LANEIGE 1위 제품은?" |
| Analytical | 10개 | "경쟁사 대비 시장 점유율은?" |
| Comparative | 5개 | "Summer Fridays와 비교해줘" |
| Temporal | 5개 | "지난달 대비 순위 변동은?" |

## conftest.py - 테스트 설정

### Fixture 목록

```python
@pytest.fixture
def mock_llm_client():
    """LLM 클라이언트 Mock"""
    pass

@pytest.fixture
def mock_storage():
    """Storage Mock"""
    pass

@pytest.fixture
def test_db():
    """테스트용 SQLite DB"""
    pass

@pytest.fixture
def mock_kg():
    """Knowledge Graph Mock"""
    pass
```

### 환경 분리

```python
# conftest.py
import os
os.environ["ENV_FILE"] = ".env.test"
```

```bash
# .env.test
OPENAI_API_KEY=sk-test-...
DB_PATH=data/test.db
KG_PATH=data/test_kg.json
```

## 테스트 실행

### 전체 테스트

```bash
python3 -m pytest tests/ -v
```

### 레이어별 테스트

```bash
# Domain Layer
python3 -m pytest tests/unit/domain/ -v

# Application Layer
python3 -m pytest tests/unit/application/ -v

# RAG System
python3 -m pytest tests/unit/rag/ -v
```

### 커버리지 측정

```bash
# HTML 리포트 생성
python3 -m pytest tests/ -v --cov=src --cov-report=html

# 터미널 출력
python3 -m pytest tests/ -v --cov=src --cov-report=term-missing

# 리포트 열기
open coverage_html/index.html
```

### 특정 테스트만 실행

```bash
# 파일 단위
python3 -m pytest tests/unit/core/test_react_agent.py -v

# 함수 단위
python3 -m pytest tests/unit/core/test_react_agent.py::test_react_loop -v

# 마커 단위
python3 -m pytest tests/ -m slow -v
```

## 테스트 작성 규칙

### 1. AAA 패턴

```python
def test_calculate_sos():
    # Arrange
    products = [Product(...), Product(...)]

    # Act
    sos = calculate_sos(products, "LANEIGE")

    # Assert
    assert 0 <= sos <= 100
```

### 2. Mock 최소화

```python
# ❌ Bad - 과도한 Mock
@patch("src.agents.chatbot.LLMClient")
@patch("src.rag.retriever.ChromaDB")
@patch("src.ontology.kg.KnowledgeGraph")
def test_chatbot(...):
    pass

# ✅ Good - 실제 객체 사용
def test_chatbot(test_db, mock_llm_client):
    retriever = Retriever(test_db)  # 실제 객체
    chatbot = Chatbot(retriever, mock_llm_client)
```

### 3. Fixture 재사용

```python
@pytest.fixture
def sample_products():
    return [
        Product(asin="B001", brand="LANEIGE", rank=1),
        Product(asin="B002", brand="Summer Fridays", rank=2)
    ]

def test_sos(sample_products):
    sos = calculate_sos(sample_products, "LANEIGE")
    assert sos == 50.0
```

## 주의사항

1. **환경 분리**: `.env.test` 사용 (프로덕션 DB 보호)
2. **독립성**: 테스트 간 의존성 없음
3. **속도**: 단위 테스트 < 1초, 통합 테스트 < 10초
4. **정리**: teardown에서 테스트 데이터 삭제
5. **Mock 남용 금지**: 필요한 경우만 사용

## 커버리지 목표

- **현재**: 10.11%
- **목표**: 60% (프로젝트 전체)
- **우선순위**: domain > application > rag > agents

## 참고

- `pytest.ini` - pytest 설정
- `pyproject.toml` - 커버리지 설정
- `scripts/evaluate_golden.py` - 골든셋 평가
- `CLAUDE.md` - TDD 워크플로우
