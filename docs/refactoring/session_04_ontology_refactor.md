# Session 4: Ontology 모듈 리팩토링 (중복 통합)

> ⏱ 예상 시간: 40~50분 | 위험도: 🟡 중간 | 선행 조건: Session 2 완료

---

## 프롬프트 (아래를 복사해서 새 Claude Code 세션에 붙여넣기)

```
너는 20년 베테랑 Python 개발자이자 Knowledge Graph/Ontology 전문가야. AMORE RAG-KG Hybrid Agent의 ontology 모듈을 리팩토링하는 세션이야.

## 이번 세션 목표
`src/ontology/`의 중복 파일을 통합하고, God Object를 분할해서 유지보수 가능한 구조로 만들어.

## 컨텍스트
- 프로젝트: `/Users/leedongwon/Desktop/AMORE-RAG-ONTOLOGY-HYBRID AGENT/`
- 전체 마스터 플랜: `docs/refactoring/00_MASTER_PLAN.md` 참조
- Python 3.13.7 (`python3` 사용)
- 이 모듈은 `src/domain` 만 의존함 (Clean Architecture 준수 중)

## 현재 구조 & 문제점
```
src/ontology/
├── knowledge_graph.py    # 1,842줄 — God Object!
├── business_rules.py     # 1,720줄 — God Object!
├── owl_reasoner.py       # 985줄
├── reasoner.py           # 980줄 — owl_reasoner와 중복?
├── relations.py
└── schema.py
```

총 5,861줄. 문제:
1. `knowledge_graph.py` (1842줄) — 너무 많은 책임
2. `reasoner.py` vs `owl_reasoner.py` — 중복 가능성
3. `business_rules.py` (1720줄) — 너무 많은 규칙이 한 파일에

## 수행할 작업 (TDD 방식)

### 1. 중복 분석: reasoner.py vs owl_reasoner.py
- 두 파일을 읽고 비교해줘
- 어디서 import되는지 추적 (`grep -r "from src.ontology.reasoner" src/`, `grep -r "from src.ontology.owl_reasoner" src/`)
- 하나로 통합할 수 있으면 통합. 역할이 다르면 이름을 명확하게 분리.

### 2. knowledge_graph.py 분할
1842줄을 다음과 같이 분할해줘:
- `knowledge_graph.py` — 핵심 Triple Store (CRUD, 로드/저장)
- `kg_query.py` (NEW) — KG 쿼리/검색 로직
- `kg_updater.py` (NEW) — KG 업데이트/동기화 로직
각 파일이 500줄 이하가 되도록.

### 3. business_rules.py 분할
1720줄을 카테고리별로 분할:
- `business_rules.py` — 규칙 엔진 코어 (BaseRule, RuleEngine)
- `rules/market_rules.py` (NEW) — 시장 관련 규칙
- `rules/brand_rules.py` (NEW) — 브랜드 관련 규칙
- `rules/alert_rules.py` (NEW) — 알림 조건 규칙

### 4. 테스트 보강
- 기존 `tests/unit/ontology/` 테스트가 통과하는지 확인
- 분할된 모듈에 대한 테스트 추가
- KG CRUD, 추론, 규칙 평가 각각 테스트

### 5. Import 경로 업데이트
- 분할 후 다른 모듈에서 import하는 경로를 업데이트
- `__init__.py`에서 re-export해서 기존 import 경로 호환성 유지:
  ```python
  # src/ontology/__init__.py
  from .knowledge_graph import KnowledgeGraph
  from .reasoner import OntologyReasoner  # 통합된 이름
  from .business_rules import RuleEngine
  ```

### 6. 검증
- `python3 -m pytest tests/unit/ontology/ -v` — ontology 테스트 통과
- `python3 -m pytest tests/ -v --tb=short` — 전체 테스트 통과
- `python3 -c "from src.ontology.knowledge_graph import KnowledgeGraph; print('OK')"` — import 확인

## 주의사항
- 기능 변경 없이 구조만 변경 (동작 동일)
- `__init__.py` re-export로 기존 import 호환성 유지 필수
- Context7 MCP로 owlready2 문서 참조 가능
```

---

## 체크리스트

- [ ] reasoner.py vs owl_reasoner.py 분석 및 통합
- [ ] knowledge_graph.py 분할 (500줄 이하)
- [ ] business_rules.py 분할
- [ ] `__init__.py` re-export 설정
- [ ] 기존 테스트 통과
- [ ] 새 테스트 추가
- [ ] 전체 테스트 통과
