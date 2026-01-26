# Entity Linker

텍스트 엔티티를 온톨로지 개념(OWL URI)에 자동 연결하는 NER 기반 Entity Linking 모듈입니다.

## 목차

- [개요](#개요)
- [핵심 기능](#핵심-기능)
- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [상세 가이드](#상세-가이드)
- [통합 예제](#통합-예제)
- [API 레퍼런스](#api-레퍼런스)

---

## 개요

Entity Linker는 사용자 쿼리에서 다음을 수행합니다:

1. **엔티티 추출** (NER): 텍스트에서 브랜드, 카테고리, 지표 등을 인식
2. **개념 매칭**: 추출된 엔티티를 온톨로지 URI에 매핑
3. **신뢰도 계산**: 매칭 품질을 0-1 점수로 평가

### 지원 엔티티 유형

- **Brand**: LANEIGE, COSRX, Beauty of Joseon 등
- **Category**: Lip Care, Skin Care, Face Powder 등
- **Metric**: SoS, HHI, CPI, Churn Rate 등
- **Ingredient**: Peptide, Ceramide, Niacinamide 등
- **Trend**: Glass Skin, Morning Shade, Viral 등
- **Product**: ASIN 형식 (B0BSHRYY1S 등)

---

## 핵심 기능

### 1. NER 기반 엔티티 추출

- **spaCy NER** (선택): en_core_web_sm 모델 사용
- **규칙 기반 폴백**: spaCy 미설치 시 자동 폴백
- **다국어 지원**: 한글/영문 동시 인식

### 2. 온톨로지 개념 매칭

- **정확 매칭** (confidence: 1.0): "LANEIGE" → LANEIGE
- **퍼지 매칭** (confidence: 0.7-0.9): "라네즈" → LANEIGE
- **동의어 처리**: "점유율" → Share of Shelf
- **OWL URI 생성**: `http://amorepacific.com/ontology/amore_brand.owl#Brand/LANEIGE`

### 3. 신뢰도 점수

| 매칭 유형 | 신뢰도 | 예시 |
|----------|--------|------|
| Exact match | 1.0 | "LANEIGE" → LANEIGE |
| High fuzzy | 0.8-0.9 | "라네즈" → LANEIGE |
| Mid fuzzy | 0.7-0.8 | "Lanege" → LANEIGE (오타) |
| Partial | 0.5-0.7 | 정규화 실패 |

---

## 설치

### 필수 패키지

```bash
pip install -r requirements.txt
```

### 선택적 패키지 (spaCy)

```bash
pip install spacy>=3.7.0
python -m spacy download en_core_web_sm
```

> **Note**: spaCy 없이도 규칙 기반 모드로 동작합니다.

---

## 빠른 시작

### 기본 사용법

```python
from src.rag.entity_linker import EntityLinker

# 인스턴스 생성
linker = EntityLinker()

# 엔티티 링킹
query = "LANEIGE Lip Care 경쟁력 분석해줘"
entities = linker.link(query)

# 결과 출력
for ent in entities:
    print(f"[{ent.entity_type}] {ent.text}")
    print(f"  → {ent.concept_label} (confidence: {ent.confidence:.2f})")
    print(f"  → URI: {ent.concept_uri}")
```

**출력:**

```
[brand] LANEIGE
  → LANEIGE (confidence: 1.00)
  → URI: http://amorepacific.com/ontology/amore_brand.owl#Brand/LANEIGE

[category] Lip Care
  → Lip Care (confidence: 1.00)
  → URI: http://amorepacific.com/ontology/amore_brand.owl#Category/lip_care
```

---

## 상세 가이드

### 엔티티 유형 필터링

특정 유형의 엔티티만 추출:

```python
# 브랜드만 추출
brands = linker.link(query, entity_types=["brand"])

# 지표만 추출
metrics = linker.link(query, entity_types=["metric"])

# 여러 유형 지정
entities = linker.link(query, entity_types=["brand", "category"])
```

### 신뢰도 임계값 설정

낮은 신뢰도 엔티티 제외:

```python
# 높은 신뢰도만 (>= 0.9)
high_conf = linker.link(query, min_confidence=0.9)

# 중간 이상 (>= 0.7)
mid_conf = linker.link(query, min_confidence=0.7)

# 모든 엔티티 (>= 0.5, 기본값)
all_ents = linker.link(query, min_confidence=0.5)
```

### 다국어 쿼리

한/영 혼합 쿼리 자동 처리:

```python
queries = [
    "라네즈 립케어 제품 분석",      # 한글
    "LANEIGE Lip Care analysis",   # 영문
    "COSRX vs 라네즈 비교",        # 혼합
]

for query in queries:
    entities = linker.link(query)
    # 정규화된 영문 레이블 반환
```

### LinkedEntity 데이터 접근

```python
entities = linker.link("LANEIGE SoS 분석")

for ent in entities:
    # 원본 텍스트
    print(ent.text)  # "LANEIGE", "SoS"

    # 엔티티 유형
    print(ent.entity_type)  # "brand", "metric"

    # 온톨로지 URI
    print(ent.concept_uri)

    # 정규화된 레이블
    print(ent.concept_label)  # "LANEIGE", "Share of Shelf"

    # 신뢰도
    print(ent.confidence)  # 0.0 ~ 1.0

    # 추가 컨텍스트
    print(ent.context)  # {"matched_key": "laneige", "start": 0, "end": 7}
```

### 직렬화 (JSON)

API 응답이나 로깅에 사용:

```python
import json

entities = linker.link(query)

# 딕셔너리 변환
entity_dicts = [ent.to_dict() for ent in entities]

# JSON 직렬화
json_str = json.dumps(entity_dicts, indent=2, ensure_ascii=False)
print(json_str)
```

---

## 통합 예제

### HybridRetriever와 통합

```python
from src.rag.entity_linker import EntityLinker
from src.rag.hybrid_retriever import HybridRetriever

linker = EntityLinker()
retriever = HybridRetriever()

await retriever.initialize()

# 1. Entity Linking
query = "LANEIGE Lip Care SoS 분석"
entities = linker.link(query)

# 2. Convert to retriever format
entity_dict = {
    "brands": [e.concept_label for e in entities if e.entity_type == "brand"],
    "categories": [
        e.context.get("matched_key", e.text.lower())
        for e in entities if e.entity_type == "category"
    ],
    "indicators": [
        e.context.get("matched_key", e.text.lower())
        for e in entities if e.entity_type == "metric"
    ]
}

# 3. Hybrid Retrieval
context = await retriever.retrieve(
    query=query,
    current_metrics={"summary": {"laneige_sos_by_category": {"lip_care": 0.12}}}
)

print(f"Ontology facts: {len(context.ontology_facts)}")
print(f"Inferences: {len(context.inferences)}")
```

### KnowledgeGraph 연동

```python
from src.rag.entity_linker import EntityLinker
from src.ontology.knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph()
linker = EntityLinker(knowledge_graph=kg)

# 링킹 후 KG 쿼리
query = "LANEIGE 제품"
entities = linker.link(query)

for ent in entities:
    if ent.entity_type == "brand":
        # KG에서 브랜드 제품 조회
        products = kg.get_brand_products(ent.concept_label)
        print(f"{ent.concept_label}: {len(products)} products")
```

### OWLReasoner 통합

```python
from src.rag.entity_linker import EntityLinker
from src.ontology.owl_reasoner import OWLReasoner

owl = OWLReasoner()
linker = EntityLinker(owl_reasoner=owl)

# URI를 사용한 SPARQL 쿼리 구성
entities = linker.link("LANEIGE vs COSRX 비교")

brand_uris = [e.concept_uri for e in entities if e.entity_type == "brand"]

if len(brand_uris) >= 2:
    # SPARQL-like 쿼리
    print(f"SELECT ?relation WHERE {{")
    print(f"  <{brand_uris[0]}> ?relation <{brand_uris[1]}> .")
    print(f"}}")
```

---

## API 레퍼런스

### EntityLinker

#### `__init__(knowledge_graph=None, owl_reasoner=None, use_spacy=True)`

Entity Linker 인스턴스 생성

**Parameters:**
- `knowledge_graph` (KnowledgeGraph, optional): 지식 그래프 인스턴스
- `owl_reasoner` (OWLReasoner, optional): OWL 추론기 인스턴스
- `use_spacy` (bool, default=True): spaCy NER 사용 여부

#### `link(text, entity_types=None, min_confidence=0.5) -> List[LinkedEntity]`

텍스트에서 엔티티 추출 및 온톨로지 링킹

**Parameters:**
- `text` (str): 입력 텍스트
- `entity_types` (List[str], optional): 추출할 엔티티 유형 필터
- `min_confidence` (float, default=0.5): 최소 신뢰도 임계값

**Returns:**
- `List[LinkedEntity]`: 링크된 엔티티 리스트

**Example:**
```python
entities = linker.link(
    text="LANEIGE SoS 분석",
    entity_types=["brand", "metric"],
    min_confidence=0.7
)
```

#### `get_stats() -> Dict[str, Any]`

Entity Linker 통계 조회

**Returns:**
- `Dict[str, Any]`: 통계 정보
  - `total_links`: 전체 링크 수
  - `exact_matches`: 정확 매칭 수
  - `fuzzy_matches`: 퍼지 매칭 수
  - `no_matches`: 매칭 실패 수

### LinkedEntity

#### `to_dict() -> Dict[str, Any]`

엔티티를 딕셔너리로 변환 (JSON 직렬화용)

**Returns:**
- `Dict[str, Any]`: 엔티티 정보 딕셔너리

**Example:**
```python
import json

entity = entities[0]
json_str = json.dumps(entity.to_dict(), ensure_ascii=False)
```

---

## 인식 가능한 엔티티 전체 목록

### 브랜드 (Brand)

**한/영 매핑 지원:**
- LANEIGE / 라네즈
- COSRX / 코스알엑스
- TIRTIR / 티르티르
- Rare Beauty / 레어뷰티
- Innisfree / 이니스프리
- ETUDE / 에뛰드
- Sulwhasoo / 설화수
- HERA / 헤라
- MISSHA / 미샤
- SKIN1004 / 스킨1004
- Anua / 아누아
- MEDICUBE / 메디큐브
- BIODANCE / 바이오던스
- Beauty of Joseon / 조선미녀

**글로벌 브랜드:**
- Summer Fridays, La Roche-Posay, CeraVe, Neutrogena
- e.l.f., NYX, Maybelline, L'Oreal
- The Ordinary, Paula's Choice, Drunk Elephant
- Fenty Beauty, Huda Beauty, Charlotte Tilbury

### 카테고리 (Category)

- Lip Care / 립케어
- Lip Makeup / 립메이크업
- Skin Care / 스킨케어
- Face Powder / 파우더
- Beauty / 뷰티

### 지표 (Metric)

- SoS / 점유율 → Share of Shelf
- HHI / 시장집중도 / 허핀달 → Herfindahl-Hirschman Index
- CPI / 가격지수 → Category Price Index
- Churn / 교체율 → Churn Rate
- Streak / 연속 → Streak Days
- Volatility / 변동성 → Rank Volatility
- Shock / 급변 → Rank Shock

### 성분 (Ingredient)

- Peptide / 펩타이드
- Ceramide / 세라마이드
- Hyaluronic Acid / 히알루론산
- Niacinamide / 나이아신아마이드
- Retinol / 레티놀
- Vitamin C / 비타민C
- Centella / Cica / 센텔라 / 시카
- PDRN
- Glass Skin / 글래스스킨

### 트렌드 (Trend)

- Morning Shade / 모닝쉐드
- Glow / 글로우
- Viral / 바이럴
- TikTok / 틱톡
- Influencer / 인플루언서

### 제품 (Product)

- ASIN 형식: `B0[A-Z0-9]{8}` (예: B0BSHRYY1S)

---

## 문제 해결

### spaCy 모델 로드 실패

**증상:**
```
Failed to load spaCy model: [E050] Can't find model 'en_core_web_sm'
```

**해결:**
```bash
python -m spacy download en_core_web_sm
```

또는 규칙 기반 모드 사용:
```python
linker = EntityLinker(use_spacy=False)
```

### 인식되지 않는 브랜드

새 브랜드 추가하려면 `src/rag/entity_linker.py`의 `KNOWN_BRANDS` 딕셔너리 수정:

```python
KNOWN_BRANDS = {
    # 기존 브랜드...
    "newbrand": "NewBrand",
    "뉴브랜드": "NewBrand",
}
```

### 낮은 신뢰도 문제

퍼지 매칭 임계값 조정 (`_fuzzy_match` 함수):

```python
# 현재: min_ratio=0.6
best_match, score = self._fuzzy_match(query, candidates, min_ratio=0.6)

# 더 엄격하게: min_ratio=0.8
best_match, score = self._fuzzy_match(query, candidates, min_ratio=0.8)
```

---

## 추가 자료

- **테스트 스크립트**: `test_entity_linker.py`
- **통합 예제**: `examples/entity_linker_integration.py`
- **프로젝트 문서**: `CLAUDE.md` (Entity Linker 섹션)

---

## 라이선스

이 프로젝트의 일부로 AMORE Pacific 내부용입니다.
