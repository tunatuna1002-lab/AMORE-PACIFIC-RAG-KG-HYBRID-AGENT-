# RAG + Ontology + Knowledge Graph 하이브리드 시스템 심층 분석

> **작성 목적**: AMORE RAG-KG Hybrid Agent의 RAG, 온톨로지, 지식 그래프 아키텍처를 설명용으로 상세 분석
> **작성일**: 2026-02-19
> **상태**: 완료 (Phase 1-5)
> **코드 기준**: main 브랜치 (commit c55eb91)

---

## 목차

1. [핵심 비유: 인간의 사고방식과 AI 에이전트의 1:1 매핑](#1-핵심-비유-인간의-사고방식과-ai-에이전트의-11-매핑)
2. [RAG 시스템 심층 분석](#2-rag-시스템-심층-분석)
3. [Knowledge Graph 심층 분석](#3-knowledge-graph-심층-분석)
4. [Ontology & 추론 시스템 심층 분석](#4-ontology--추론-시스템-심층-분석)
5. [하이브리드 통합: HybridRetriever](#5-하이브리드-통합-hybridretriever)
6. [전체 쿼리 흐름 (End-to-End)](#6-전체-쿼리-흐름-end-to-end)
7. [할루시네이션 방지 메커니즘](#7-할루시네이션-방지-메커니즘)
8. [비즈니스 도메인 적용 사례](#8-비즈니스-도메인-적용-사례)
9. [아키텍처 요약 다이어그램](#9-아키텍처-요약-다이어그램)

---

## 1. 핵심 비유: 인간의 사고방식과 AI 에이전트의 1:1 매핑

### 1.1 인간은 어떻게 질문에 대답하는가?

사람이 **"LANEIGE의 Lip Care 시장 경쟁력은 어때?"** 라는 질문을 받았다고 하자.

사람의 머릿속에서는 다음이 **동시에** 일어난다:

```
1. 책장에서 자료 꺼내기 (RAG)
   → "최근에 읽었던 시장 보고서가 있었는데..."
   → 관련 문서를 찾아서 읽는 행위

2. 머릿속 개념 지도 활성화 (Knowledge Graph)
   → "LANEIGE는 아모레퍼시픽 브랜드이고..."
   → "Lip Care는 Skin Care 하위 카테고리이고..."
   → "경쟁사는 COSRX, TIRTIR 같은 다른 회사 브랜드들이지..."
   → 이미 알고 있는 관계/사실들이 자동으로 떠오름

3. 세계관/규칙 적용 (Ontology + Reasoning)
   → "점유율 30% 이상이면 시장 지배적이라고 볼 수 있고..."
   → "같은 그룹 브랜드끼리는 경쟁 관계가 아니야"
   → "HHI가 높으면 시장이 집중돼 있다는 뜻이야"
   → 이 세계의 규칙/분류 체계로 판단하는 행위
```

이 세 가지가 합쳐져서 **정확하고 맥락 있는 답변**이 나온다.

### 1.2 이 시스템에서의 1:1 매핑

| 인간의 인지 과정 | 시스템 컴포넌트 | 핵심 역할 |
|-----------------|----------------|-----------|
| **책장에서 자료 찾기** | **RAG** (`src/rag/`) | 문서 검색 + 관련 텍스트 추출 |
| **머릿속 관계 지도** | **Knowledge Graph** (`src/ontology/knowledge_graph.py`) | 엔티티 간 관계 저장 + 조회 |
| **세계관/분류 체계/규칙** | **Ontology** (`src/ontology/owl_reasoner.py`, `reasoner.py`) | 개념 정의 + 규칙 기반 추론 |
| **세 가지를 종합해 판단** | **HybridRetriever** (`src/rag/hybrid_retriever.py`) | 세 소스를 가중 병합하여 최종 컨텍스트 생성 |

### 1.3 왜 이 세 가지가 모두 필요한가?

**RAG만 있을 때의 문제:**
```
질문: "LANEIGE의 경쟁사는?"

RAG만의 답변: "LANEIGE Lip Sleeping Mask는 Lip Care 카테고리에서
              LANEIGE Water Sleeping Mask와 경쟁하고 있습니다."
              ← 같은 브랜드 제품끼리 "경쟁"이라고 말하는 할루시네이션!
```

**KG + Ontology가 있을 때:**
```
KG 사실:  (LANEIGE, OWNED_BY_GROUP, AMOREPACIFIC)
KG 사실:  (LANEIGE Lip Sleeping Mask, OWNED_BY, LANEIGE)
KG 사실:  (LANEIGE Water Sleeping Mask, OWNED_BY, LANEIGE)
Ontology 규칙: 같은 브랜드의 제품은 경쟁 관계가 아님
Ontology 규칙: 경쟁사 = 다른 기업 소유 브랜드

올바른 답변: "LANEIGE의 Lip Care 카테고리 경쟁사는
             COSRX, TIRTIR, Neutrogena 등입니다.
             LANEIGE의 다른 제품과는 같은 브랜드이므로 경쟁 관계가 아닙니다."
```

KG는 **"무엇이 무엇인지"** 사실을 알고, Ontology는 **"그 사실로부터 무엇을 추론해야 하는지"** 규칙을 안다. RAG는 **"관련 문서에 뭐라고 써있는지"** 텍스트를 가져온다. 이 세 가지가 합쳐져야 할루시네이션 없는 답변이 가능하다.

---

## 2. RAG 시스템 심층 분석

### 2.1 RAG란? (이 시스템에서의 의미)

RAG(Retrieval Augmented Generation)는 LLM이 답변을 생성할 때, **외부 문서를 검색해서 참조하도록** 하는 기법이다. 인간 비유로 **"책장에서 관련 자료를 찾아 읽는 행위"** 에 해당한다.

이 시스템의 RAG는 단순한 벡터 검색이 아니라, **14개 전문 마크다운 문서**를 관리하는 체계적인 문서 라이브러리이다.

### 2.2 문서 라이브러리 (DocumentRetriever)

> 파일: `src/rag/retriever.py` (1,383줄)

#### 문서 구조 — 5가지 유형

```
[Type A] 지표 가이드 (Metric Guides)
  ├── A1. 지표 해석 가이드        — SoS, HHI, CPI 정의와 해석법
  ├── A2. 시장 분석 플레이북      — 분석 절차, 의사결정 프레임워크
  └── A3. 종합 분석 가이드         — 복합 지표 통합 분석법

[Type B] 시장 정보 (Market Intelligence)
  ├── B1. 경쟁 환경 분석          — 브랜드별 포지셔닝
  ├── B2. 카테고리 구조           — 뷰티 카테고리 계층 구조
  └── B3. 트렌드 리포트           — 시장 동향

[Type C] 대응 가이드 (Response Guides)
  ├── C1. 순위 변동 대응          — 급등/급락 대응 전략
  ├── C2. 가격 전략               — CPI 기반 가격 분석
  └── C3. 프로모션 영향           — 프로모 효과 분석

[Type D] IR 보고서 (Quarterly Reports)
  ├── D1. 아모레퍼시픽 IR         — 실적, 매출, 전략
  └── D2. 경쟁사 IR               — LG생활건강, 에스티로더 등

[Type E] 고급 분석 (Advanced)
  ├── E1. 시계열 분석             — 장기 트렌드 패턴
  └── E2. 카테고리 교차 분석      — 크로스 카테고리 인사이트
```

이 문서들은 **시스템이 참조하는 전문 지식 문서**이다. 인간 비유로는 **분석가의 책상 위에 놓인 14권의 전문서적**에 해당한다.

#### 이중 검색 (Dense + Sparse)

RAG 검색은 두 가지 방식을 **동시에** 사용한다:

```
┌─────────────────────────────────────────────────────┐
│                사용자 쿼리                             │
│        "LANEIGE 점유율 트렌드 분석해줘"                  │
└──────────────────────┬──────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
   [Dense Search]            [Sparse Search]
   ChromaDB + 임베딩          BM25 키워드 매칭
   (의미적 유사도)             (정확한 용어 매칭)
           │                       │
           │  "점유율"과 의미적으로    │  "LANEIGE", "점유율"
           │  유사한 문서 상위 N개    │  정확히 포함된 문서
           │                       │
           └───────────┬───────────┘
                       ▼
              [RRF 융합 (Reciprocal Rank Fusion)]
              두 결과의 순위를 결합하여 최종 순위 결정
              score = Σ 1/(k + rank_i)  (k=60)
                       │
                       ▼
              최종 문서 청크 Top-K 반환
└─────────────────────────────────────────────────────┘
```

**왜 두 가지를 쓰는가?**

- **Dense Search (벡터)**: "시장 경쟁력" → "market competitiveness", "brand positioning" 같은 **의미적으로 유사한** 표현도 찾아줌. 하지만 정확한 숫자나 고유명사에 약함.
- **Sparse Search (BM25)**: "LANEIGE", "SoS", "37.2%" 같은 **정확한 키워드**를 잘 찾음. 하지만 동의어나 의미 확장에 약함.
- **RRF 융합**: 두 검색의 장점을 결합. 의미적으로도 관련 있고 + 키워드도 매칭되는 문서가 최상위에 올라옴.

#### Self-RAG 게이트

모든 질문에 문서 검색이 필요한 것은 아니다. 시스템은 **검색 필요 여부를 먼저 판단**한다.

```python
# src/rag/retriever.py의 _needs_retrieval() 로직
def _needs_retrieval(query: str) -> bool:
    """검색 없이 답변 가능한 쿼리인지 판단"""
    # 인사/잡담 → 검색 불필요
    if is_greeting(query):        # "안녕", "고마워" 등
        return False
    # 시스템 명령 → 검색 불필요
    if is_command(query):          # "도움말", "상태 확인" 등
        return False
    # 그 외 → 검색 필요
    return True
```

이것은 인간이 "안녕하세요"라는 인사에 굳이 자료를 찾지 않는 것과 같다. **불필요한 검색을 건너뛰어** 응답 속도를 높이고 노이즈를 줄인다.

#### 문서 청킹 (Chunker)

긴 문서를 검색하기 위해 적절한 크기로 나눈다:

```
원본 문서 (수천 자) → 청킹 → 검색 가능한 조각들 (500~800자)
```

- **테이블 인식 청킹**: 마크다운 표를 감지하여 표가 잘리지 않도록 처리
- **문서 유형별 청크 크기**: 지표 가이드(A)는 800자, 대응 가이드(C)는 500자 등 유형별 최적화
- **오버랩**: 청크 간 100자 정도 겹침을 두어 문맥 단절 방지

#### 임베딩 캐시

같은 문서를 반복 임베딩하면 낭비이므로 캐시를 사용한다:

```
임베딩 요청 → 캐시 확인 → Hit? → 캐시 반환 (0ms)
                         → Miss? → OpenAI API 호출 → 캐시 저장 → 반환
```

- **InMemory 캐시**: 메모리에 저장 (빠르지만 재시작 시 소멸)
- **SQLite 캐시**: 디스크에 저장 (재시작 후에도 유지)
- **TTL**: 기본 300초(5분) 캐시 유효 기간

### 2.3 인텐트 기반 검색 전략 (RetrievalStrategy)

> 파일: `src/rag/retrieval_strategy.py` (712줄)

질문의 **의도(Intent)** 에 따라 검색 가중치를 다르게 적용한다. 이것은 인간이 질문 유형에 따라 **"이건 자료를 봐야 해"** vs **"이건 내가 아는 관계로 추론하면 돼"** 를 자동으로 판단하는 것과 같다.

```
12가지 인텐트 → 검색 가중치 매핑:

┌─────────────────┬───────────────────────────────────────┐
│  질문 인텐트      │  가중치 (KG : RAG : Inference)         │
├─────────────────┼───────────────────────────────────────┤
│  DIAGNOSIS       │  0.50 : 0.30 : 0.20  ← KG 중심       │
│  (진단/원인분석)    │  "왜 순위가 떨어졌어?"                    │
├─────────────────┼───────────────────────────────────────┤
│  GENERAL         │  0.20 : 0.50 : 0.30  ← RAG 중심      │
│  (일반 질문)       │  "SoS가 뭐야?"                         │
├─────────────────┼───────────────────────────────────────┤
│  ANALYSIS        │  0.30 : 0.30 : 0.40  ← 추론 중심      │
│  (시장 분석)       │  "시장 구조 분석해줘"                     │
├─────────────────┼───────────────────────────────────────┤
│  TREND           │  0.35 : 0.35 : 0.30  ← 균형          │
│  (트렌드)         │  "최근 트렌드는?"                        │
├─────────────────┼───────────────────────────────────────┤
│  COMPETITION     │  0.45 : 0.25 : 0.30  ← KG 중심       │
│  (경쟁 분석)       │  "COSRX랑 비교해줘"                     │
├─────────────────┼───────────────────────────────────────┤
│  RANKING         │  0.40 : 0.35 : 0.25  ← KG 중심       │
│  (순위 조회)       │  "LANEIGE 순위 알려줘"                  │
├─────────────────┼───────────────────────────────────────┤
│  PRICING         │  0.35 : 0.35 : 0.30  ← 균형          │
│  (가격 분석)       │  "가격 경쟁력은?"                        │
├─────────────────┼───────────────────────────────────────┤
│  SENTIMENT       │  0.30 : 0.40 : 0.30  ← RAG 중심      │
│  (감성 분석)       │  "리뷰 감성 분석해줘"                    │
├─────────────────┼───────────────────────────────────────┤
│  RECOMMENDATION  │  0.30 : 0.30 : 0.40  ← 추론 중심     │
│  (추천/제안)       │  "어떤 전략이 좋을까?"                   │
└─────────────────┴───────────────────────────────────────┘
```

**핵심 통찰**: 경쟁사 관련 질문(COMPETITION, DIAGNOSIS)은 KG 가중치가 높다. 왜? **"누가 경쟁사인가"는 문서가 아니라 관계(KG)에서 가장 정확하게 답할 수 있기 때문이다.** 반면 일반 지식 질문(GENERAL)은 문서(RAG)에서 답을 찾아야 한다.

### 2.4 신뢰도 융합 (ConfidenceFusion)

> 파일: `src/rag/confidence_fusion.py` (649줄)

여러 소스에서 온 정보의 **신뢰도를 통합 평가**한다. 인간이 여러 출처의 정보를 종합할 때 "이건 확실하고, 이건 좀 불확실해" 라고 판단하는 것과 같다.

```
3가지 소스의 신뢰도 가중치:

  벡터 검색 (RAG)      : 0.40  ← 문서에서 찾은 정보
  온톨로지 추론         : 0.35  ← 규칙/지식으로 추론한 정보
  엔티티 매칭           : 0.25  ← 개체명 인식의 정확도

최종 신뢰도 = 0.40×(RAG점수) + 0.35×(Ontology점수) + 0.25×(Entity점수)
```

**충돌 감지 (Conflict Detection)**: 소스 간 점수 차이가 0.3 이상이면 경고를 발생시킨다.

```
예: 벡터 검색 점수 = 0.9, 온톨로지 점수 = 0.3
    → 차이 = 0.6 > 임계값 0.3
    → 경고: "소스 간 충돌 감지 — 문서는 관련 있다고 하지만 온톨로지는 아니라고 함"
```

이것은 "책에는 이렇게 써있는데, 내가 아는 규칙으로는 말이 안 되는데?" 라는 인간의 비판적 사고와 같다.

#### 6가지 융합 전략

| 전략 | 수식 | 용도 |
|------|------|------|
| WEIGHTED_SUM | Σ(wi × si) | 기본 융합 (가장 일반적) |
| HARMONIC_MEAN | n / Σ(1/si) | 모든 소스가 고르게 높아야 높은 점수 |
| GEOMETRIC_MEAN | ∏(si)^(1/n) | 하나라도 0이면 전체 0 |
| MAX_SCORE | max(si) | 가장 자신 있는 소스만 반영 |
| MIN_SCORE | min(si) | 가장 보수적인 판단 |
| RRF | Σ 1/(k+ri) | 순위 기반 융합 (문서 병합용) |

### 2.5 Cross-Encoder 재순위화 (Reranker)

> 파일: `src/rag/reranker.py`

2단계 검색 파이프라인에서 **정밀 재순위화**를 담당한다:

```
Stage 1 (Recall — 빠르게):
  Bi-Encoder(임베딩)로 후보 Top-100 추출

Stage 2 (Precision — 정확하게):
  Cross-Encoder로 쿼리-문서 쌍을 직접 비교하여 Top-5 재정렬
```

- Bi-Encoder: 쿼리와 문서를 **독립적으로** 임베딩 후 코사인 유사도 비교 (빠르지만 덜 정확)
- Cross-Encoder: 쿼리+문서를 **함께** 모델에 넣어 관련도 직접 예측 (느리지만 정확)

지원 모델: ms-marco-MiniLM-L-6-v2 (기본), bge-reranker-base/large, 또는 OpenAI API

### 2.6 엔티티 링킹 (EntityLinker)

> 파일: `src/rag/entity_linker.py`

자연어 텍스트에서 **엔티티를 추출하고 온톨로지 개념에 연결**한다:

```
입력: "LANEIGE Lip Care 경쟁력 분석해줘"

엔티티 추출 결과:
  ├── "LANEIGE"  → type: brand,    concept: amore:brand/LANEIGE,   confidence: 1.0
  ├── "Lip Care" → type: category, concept: amore:category/lip_care, confidence: 1.0
  └── "경쟁력"    → type: metric,   concept: amore:metric/competitiveness, confidence: 0.8
```

- **NER 엔진**: spaCy (설치 시) 또는 규칙 기반 폴백 (정규식 + 사전 매칭)
- **퍼지 매칭**: "라네즈" → "LANEIGE" (유사도 0.7~0.9), "립케어" → "Lip Care"
- **동의어 처리**: "점유율" = "SoS" = "Share of Shelf"

---

## 3. Knowledge Graph 심층 분석

### 3.1 KG란? (이 시스템에서의 의미)

Knowledge Graph는 **세상의 사실들을 (주체, 관계, 객체) 트리플로 저장**하는 구조이다. 인간 비유로 **"머릿속의 개념 관계 지도"** 에 해당한다.

사람이 "LANEIGE는 아모레퍼시픽 브랜드야"를 알고 있는 것처럼, KG는 이것을 구조화된 형태로 저장한다:

```
(LANEIGE, OWNED_BY_GROUP, AMOREPACIFIC)    ← "LANEIGE는 아모레퍼시픽 소속"
(LANEIGE, HAS_PRODUCT, Lip Sleeping Mask)  ← "LANEIGE는 이 제품을 갖고 있다"
(Lip Sleeping Mask, BELONGS_TO_CATEGORY, lip_care) ← "이 제품은 Lip Care 카테고리"
(LANEIGE, COMPETES_WITH, COSRX)            ← "LANEIGE는 COSRX와 경쟁"
(LANEIGE, SOS_SCORE, 37.2)                 ← "LANEIGE의 점유율은 37.2%"
```

### 3.2 Triple Store 구현

> 파일: `src/ontology/knowledge_graph.py` (551줄)

#### 핵심 구조

```python
class KnowledgeGraph(KGQueryMixin, KGUpdaterMixin, KGIRIMixin):
    """
    인메모리 트리플 스토어 + JSON 영속화

    구성:
    - triples: List[Relation]          ← 모든 트리플 저장
    - subject_index: Dict[str, List]   ← 주체별 인덱스
    - object_index: Dict[str, List]    ← 객체별 인덱스
    - predicate_index: Dict[RelationType, List] ← 관계유형별 인덱스
    - entity_metadata: Dict[str, Dict] ← 엔티티 부가 정보
    """
```

3중 인덱스를 사용하여 **어떤 방향으로든 빠르게 탐색**할 수 있다:

```
"LANEIGE의 모든 관계를 알려줘" → subject_index["LANEIGE"] → 즉시 반환
"Lip Care에 속하는 모든 것"    → object_index["lip_care"] → 즉시 반환
"모든 경쟁 관계를 알려줘"       → predicate_index[COMPETES_WITH] → 즉시 반환
```

#### 트리플 쿼리 (SPARQL-like)

> 파일: `src/ontology/kg_query.py`

```python
# 사용 예 (SPARQL 스타일 패턴 매칭)
kg.query(subject="LANEIGE", predicate=RelationType.HAS_PRODUCT)
# → [Relation(LANEIGE, HAS_PRODUCT, Lip Sleeping Mask, confidence=0.95),
#    Relation(LANEIGE, HAS_PRODUCT, Water Sleeping Mask, confidence=0.95)]

kg.query(predicate=RelationType.BELONGS_TO_CATEGORY, object_="lip_care")
# → [Relation(Lip Sleeping Mask, BELONGS_TO_CATEGORY, lip_care, ...),
#    Relation(Chapstick Original, BELONGS_TO_CATEGORY, lip_care, ...)]
```

**그래프 탐색 (BFS)**: 한 엔티티에서 출발하여 N-hop 관계를 탐색할 수 있다:

```
LANEIGE에서 2-hop 탐색:
  1-hop: LANEIGE → [제품들, 카테고리, 경쟁사, 아모레퍼시픽]
  2-hop: 각 1-hop 결과 → [제품의 순위, 카테고리의 하위 카테고리, 경쟁사의 제품들, ...]
```

#### JSON 영속화와 안전성

```
메모리                    디스크
┌──────────────┐   save()   ┌─────────────────────────┐
│ List[Relation]│ ────────→ │ data/knowledge_graph.json│
│ (인메모리)     │   atomic   │ (928KB, ~5000 트리플)     │
│              │   write    │                         │
└──────────────┘ ←──────── └─────────────────────────┘
                  load()
```

- **원자적 쓰기 (Atomic Write)**: 임시 파일에 먼저 쓰고 → rename으로 교체 (중간에 죽어도 데이터 손상 없음)
- **Thread-safe**: `threading.Lock`으로 동시 접근 보호
- **싱글톤**: `get_knowledge_graph()`로 전체 앱에서 하나의 KG 인스턴스만 사용
- **스마트 퇴거**: 최대 50,000 트리플 제한. 초과 시 중요도 낮은 트리플부터 삭제 (보호 관계: OWNED_BY, PARENT_CATEGORY 등은 절대 삭제 안 됨)

### 3.3 관계 유형 체계 (RelationType)

> 파일: `src/domain/entities/relations.py` (697줄)

이 시스템의 KG가 표현할 수 있는 **30개 이상의 관계 유형**이 7개 범주로 분류되어 있다:

```
[1. 엔티티 관계 — "무엇이 무엇인가"]
  HAS_PRODUCT          "LANEIGE는 Lip Sleeping Mask를 가짐"
  BELONGS_TO_CATEGORY  "Lip Sleeping Mask는 Lip Care에 속함"
  HAS_SUBCATEGORY      "Skin Care는 Lip Care를 하위에 가짐"
  PARENT_CATEGORY      "Lip Care의 상위는 Skin Care"
  HAS_INGREDIENT       "제품에 이 성분이 포함"

[2. 기업 소유 관계 — "누가 누구 소속인가" ★ 할루시네이션 방지 핵심]
  OWNED_BY             "제품이 브랜드에 소속"
  OWNED_BY_GROUP       "브랜드가 그룹(기업)에 소속"  ★
  OWNS_BRAND           "그룹이 브랜드를 소유"        ★
  SIBLING_BRAND        "같은 그룹 내 형제 브랜드"     ★

[3. 순위/랭킹 관계]
  RANKED_IN            "제품이 카테고리에서 순위 보유"
  RANK_CHANGE          "순위 변동 발생"
  RANK_TREND           "순위 추세 (상승/하락/유지)"

[4. 경쟁 관계 — "누가 누구와 경쟁하는가"]
  COMPETES_WITH        "브랜드 간 경쟁" (대칭적)
  MARKET_LEADER        "시장 리더"
  MARKET_CHALLENGER    "시장 도전자"
  MARKET_FOLLOWER      "시장 추종자"

[5. 지표 관계]
  SOS_SCORE            "점유율 수치"
  HHI_SCORE            "시장 집중도 수치"
  CPI_SCORE            "가격 경쟁력 수치"
  PRICE_RANGE          "가격대"

[6. 시간/트렌드 관계]
  TREND_UP / TREND_DOWN / TREND_STABLE  "추세"
  SEASONAL_PEAK        "시즌 피크"
  PERIOD_COMPARISON    "기간 비교"

[7. 감성 관계]
  SENTIMENT_POSITIVE / NEGATIVE / NEUTRAL  "감성"
  REVIEW_TOPIC         "리뷰 주제"
```

#### 핵심: 기업 소유 관계가 왜 중요한가

```
사실:
  (LANEIGE, OWNED_BY_GROUP, AMOREPACIFIC)
  (이니스프리, OWNED_BY_GROUP, AMOREPACIFIC)
  (LANEIGE, SIBLING_BRAND, 이니스프리)

추론 규칙:
  IF brand_A.group == brand_B.group
  THEN brand_A와 brand_B는 경쟁 관계가 아님

결과:
  "LANEIGE의 경쟁사를 알려줘" 질문에
  → 이니스프리는 제외됨 (같은 아모레퍼시픽 그룹)
  → COSRX, TIRTIR, Neutrogena 등만 반환
```

이것이 바로 **KG가 할루시네이션을 방지하는 핵심 메커니즘**이다.

### 3.4 IRI 스킴 (엔티티 식별)

모든 KG 엔티티에 고유한 식별자(IRI)를 부여한다:

```
amore:brand/LANEIGE              ← 브랜드
amore:brand/COSRX                ← 브랜드
amore:product/B09SBP8FJS         ← 제품 (ASIN)
amore:category/lip_care          ← 카테고리
amore:metric/sos                 ← 지표
amore:group/AMOREPACIFIC         ← 기업 그룹
```

IRI 스킴은 RDF/OWL 표준을 따르며, `rdflib.Namespace`를 사용한다:
```python
AMORE = Namespace("http://amore.ontology/")
```

---

## 4. Ontology & 추론 시스템 심층 분석

### 4.1 Ontology란? (이 시스템에서의 의미)

Ontology는 **"세계의 개념 체계와 규칙"** 을 정의한다. KG가 **사실(fact)** 을 저장한다면, Ontology는 **"그 사실들을 어떻게 해석하고, 무엇을 추론해야 하는지"** 의 규칙을 정의한다.

인간 비유:
- KG = "LANEIGE 점유율은 37.2%이다" (사실)
- Ontology = "점유율 30% 이상이면 시장 지배적이라고 판단한다" (규칙)
- 추론 결과 = "따라서 LANEIGE는 시장 지배적 브랜드이다" (새로운 지식)

### 4.2 이 시스템의 3중 추론 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    UnifiedReasoner (통합 추론 엔진)            │
│                                                             │
│    ┌─────────────────┐    ┌──────────────────────────┐      │
│    │  OWL Reasoner    │    │  Business Rules Engine   │      │
│    │  (형식 논리 추론)  │    │  (비즈니스 규칙 추론)       │      │
│    │                 │    │                          │      │
│    │  owlready2 +    │    │  Forward Chaining +      │      │
│    │  Pellet/HermiT  │    │  Python 조건-결론 규칙     │      │
│    │                 │    │                          │      │
│    │  가중치: 0.6     │    │  가중치: 0.4              │      │
│    └────────┬────────┘    └─────────────┬────────────┘      │
│             │                           │                   │
│             └──────────┬────────────────┘                   │
│                        ▼                                    │
│              [결과 융합 (Weighted Merge)]                     │
│              OWL 추론 × 0.6 + 규칙 추론 × 0.4                │
│                        │                                    │
│                        ▼                                    │
│              최종 추론 결과 (InferenceResult[])                │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 OWL Reasoner (형식 논리 추론)

> 파일: `src/ontology/owl_reasoner.py`

OWL (Web Ontology Language)은 시맨틱 웹의 표준 온톨로지 언어이다. 이 시스템은 **OWL2**를 사용하여 형식 논리 기반 추론을 수행한다.

#### OWL 클래스 계층 (T-Box: 개념 정의)

```
Thing
├── Brand
│   ├── DominantBrand   (SoS ≥ 30%)
│   ├── StrongBrand     (SoS 15~30%)
│   └── NicheBrand      (SoS < 15%)
├── Product
├── Category
└── Trend
    ├── UpwardTrend
    ├── DownwardTrend
    └── StableTrend
```

이것은 인간이 "점유율 30% 이상이면 지배적 브랜드로 분류하자"는 **분류 기준**을 세우는 것과 같다.

#### OWL 속성 (Properties)

**Object Properties (엔티티 간 관계):**
```
hasBrand:             Product → Brand
hasProduct:           Brand → Product
belongsToCategory:    Product → Category
competsWith:          Brand → Brand  (대칭적: A가 B와 경쟁 ↔ B가 A와 경쟁)
```

**Data Properties (엔티티의 속성값):**
```
shareOfShelf:    Brand → float  (0-100)
averageRank:     Brand → float
productCount:    Brand → int
rank:            Product → int
price:           Product → float
rating:          Product → float
```

#### 자동 분류 추론

```
OWL 추론 예:
  사실: LANEIGE.shareOfShelf = 37.2
  규칙: DominantBrand ≡ Brand AND shareOfShelf ≥ 30
  추론: LANEIGE는 DominantBrand로 자동 분류됨

  사실: TIRTIR.shareOfShelf = 12.5
  규칙: NicheBrand ≡ Brand AND shareOfShelf < 15
  추론: TIRTIR는 NicheBrand로 자동 분류됨
```

Pellet 또는 HermiT 추론기가 이 자동 분류를 수행한다. owlready2 라이브러리가 없으면 (선택적 의존성) graceful fallback으로 규칙 기반 추론만 사용한다.

### 4.4 Business Rules Engine (비즈니스 규칙 추론)

> 파일: `src/ontology/reasoner.py` (932줄)

OWL이 **형식 논리**라면, Business Rules Engine은 **도메인 전문가의 분석 규칙**을 코드로 표현한 것이다.

#### Forward Chaining (전방 추론) 패턴

```python
# 규칙 구조
InferenceRule(
    name="시장 지배력 진단",
    conditions=[
        RuleCondition("sos_above", lambda ctx: ctx.sos > 30),
        RuleCondition("rank_top3", lambda ctx: ctx.avg_rank <= 3),
    ],
    conclusion="시장 지배적 포지션. 현 전략 유지 및 방어 체계 강화 권장.",
    insight_type=InsightType.MARKET_POSITION,
    confidence=0.85,
)
```

이것은 인간 분석가의 판단 규칙을 코드화한 것이다:
```
IF 점유율 > 30% AND 평균 순위 ≤ 3위
THEN "시장 지배적 포지션이다. 방어 전략을 강화하라."
(신뢰도: 85%)
```

#### 8가지 규칙 카테고리

> 파일: `src/ontology/rules/` 디렉토리

```
[1] Market Position Rules (시장 포지션)
    → SoS 기반 지배력/도전자/니치 판단
    → 예: "SoS 30% 이상 + 순위 3위 이내 → 시장 지배적"

[2] Competitive Threat Rules (경쟁 위협)
    → 경쟁사 성장률, 순위 역전 감지
    → 예: "경쟁사 SoS 5%↑ + 우리 SoS 3%↓ → 위협 경고"

[3] Growth Opportunity Rules (성장 기회)
    → 미개척 영역, 성장 잠재력 분석
    → 예: "카테고리 HHI < 0.15 + 우리 SoS < 10% → 진입 기회"

[4] Risk Alert Rules (위험 경고)
    → 급격한 순위 하락, 가격 전쟁 감지
    → 예: "순위 10단계 이상 급락 → 즉시 대응 필요"

[5] Price Position Rules (가격 포지션)
    → CPI 기반 가격 경쟁력 판단
    → 예: "CPI > 1.2 → 프리미엄 포지셔닝"

[6] IR Cross-Analysis Rules (IR 교차 분석)
    → 분기 실적과 시장 데이터 교차 검증
    → 예: "IR 매출 성장 + SoS 하락 → 단가 상승 but 점유율 하락"

[7] Sentiment Rules (감성 분석)
    → 리뷰/SNS 감성과 시장 지표 교차
    → 예: "부정 감성 30%↑ + 순위 하락 → 품질 이슈 경고"

[8] Rank-Discount Causality (순위-할인 인과)
    → 가격 할인과 순위 변동의 인과관계
    → 예: "할인율 20%↑ → 순위 5단계↑ → 할인 효과 확인"
```

#### 쿼리 인텐트 → 규칙 필터링

모든 규칙을 매번 실행하지 않는다. **질문의 의도에 맞는 규칙만 선택적으로 실행**한다:

```
질문: "LANEIGE 가격 경쟁력 분석해줘"
  → 인텐트: PRICING
  → 활성화 규칙: Price Position Rules + Market Position Rules
  → 비활성화: Sentiment, IR, Growth 등 (불필요)

질문: "경쟁사 위협 분석해줘"
  → 인텐트: COMPETITION
  → 활성화 규칙: Competitive Threat + Market Position + Risk Alert
  → 비활성화: Price, Sentiment, IR 등
```

이것은 인간이 "가격 물어봤으니 가격 관련 지식만 떠올리면 되겠네"라고 판단하는 것과 같다.

#### 추론 결과와 설명 가능성

모든 추론은 **왜 그런 결론을 내렸는지 설명**을 포함한다:

```python
InferenceResult(
    rule_name="시장 지배력 진단",
    insight_type=InsightType.MARKET_POSITION,
    insight="LANEIGE는 Lip Care 시장에서 지배적 포지션을 유지하고 있습니다.",
    confidence=0.85,
    evidence=[
        "SoS: 37.2% (임계값 30% 초과)",
        "평균 순위: 2.3위 (Top 3 이내)",
        "제품 수: 5개 (카테고리 최다)"
    ],
    recommendation="현 포지션 방어를 위한 신제품 라인업 강화 및 프로모션 전략 수립"
)
```

### 4.5 UnifiedReasoner (통합 추론 엔진)

> 파일: `src/ontology/unified_reasoner.py`

OWL Reasoner와 Business Rules Engine의 결과를 **가중 융합**한다:

```
Step 1: OWL 추론 실행
  → OWL 결과: "LANEIGE = DominantBrand (SoS ≥ 30%)"  [confidence: 0.90]

Step 2: Business Rules 추론 실행
  → 규칙 결과: "시장 지배적 포지션, 방어 전략 권장"     [confidence: 0.85]

Step 3: 가중 융합
  최종 = OWL결과 × 0.6 + Rules결과 × 0.4
       = 0.90 × 0.6 + 0.85 × 0.4
       = 0.54 + 0.34 = 0.88

Step 4: 중복 제거 & 정렬
  → 유사한 추론 결과 병합
  → 신뢰도 순 정렬
```

**왜 두 가지를 쓰는가?**
- **OWL**: 형식 논리로 **정확한 분류**와 **일관성 검증** (예: "이 브랜드가 동시에 DominantBrand이면서 NicheBrand일 수는 없다")
- **Rules**: 도메인 전문가의 **실용적 인사이트** 생성 (예: "점유율이 떨어지고 있으면 어떤 전략을 써야 한다")
- 둘의 **상호 보완**: OWL은 "무엇인지" 분류하고, Rules는 "그래서 어떻게 해야 하는지" 제안한다

---

## 5. 하이브리드 통합: HybridRetriever

> 파일: `src/rag/hybrid_retriever.py` (1,583줄) — **이 시스템의 심장부**

### 5.1 역할

HybridRetriever는 RAG, KG, Ontology **세 가지를 하나로 통합하는 오케스트레이터**이다. 인간 비유로, "자료 찾기 + 머릿속 지도 + 세계관 규칙"을 **동시에 활성화하고 종합 판단을 내리는 뇌의 통합 기능**에 해당한다.

### 5.2 컴포넌트 구성

```python
class HybridRetriever:
    def __init__(self):
        self.kg = KnowledgeGraph          # 지식 그래프 (머릿속 관계 지도)
        self.reasoner = OntologyReasoner  # 규칙 기반 추론 (세계관)
        self.retriever = DocumentRetriever # 문서 검색 (책장)
        self.entity_extractor = EntityExtractor  # 엔티티 추출
        self.relevance_grader = RelevanceGrader  # 관련성 평가
        self.query_enhancer = QueryEnhancer      # 쿼리 강화
        self.confidence_fusion = ConfidenceFusion # 신뢰도 융합
```

### 5.3 retrieve() 메인 플로우

```
사용자 질문: "LANEIGE Lip Care 경쟁사 대비 점유율 추이는?"

════════════════════════════════════════════════════════════
Step 0: Self-RAG Gate (검색 필요 여부 판단)
════════════════════════════════════════════════════════════
  → 이 질문은 데이터 분석이 필요 → 검색 진행

════════════════════════════════════════════════════════════
Step 1: 인텐트 분류 (Intent Classification)
════════════════════════════════════════════════════════════
  "경쟁사 대비 점유율 추이"
  → 1차: COMPETITION (경쟁사 키워드)
  → 2차: TREND (추이 키워드)
  → 최종: COMPETITION (우선순위 적용)

  → 가중치 결정: KG=0.45, RAG=0.25, Inference=0.30

════════════════════════════════════════════════════════════
Step 2: 엔티티 추출 (Entity Extraction)
════════════════════════════════════════════════════════════
  입력: "LANEIGE Lip Care 경쟁사 대비 점유율 추이는?"
  추출:
    brands:     ["LANEIGE"]
    categories: ["lip_care"]
    indicators: ["sos"]        (점유율 = SoS)
    sentiments: []
    products:   []

════════════════════════════════════════════════════════════
Step 3: 쿼리 강화 (Query Enhancement)
════════════════════════════════════════════════════════════
  원본: "LANEIGE Lip Care 경쟁사 대비 점유율 추이는?"
  강화: "LANEIGE brand Lip Care category competitor SoS
         share of shelf trend analysis"
  → 영문 키워드 추가로 RAG 검색 품질 향상

════════════════════════════════════════════════════════════
Step 4: Knowledge Graph 조회 (KG Query) ★
════════════════════════════════════════════════════════════
  4-a. 브랜드 정보:
    query(subject="LANEIGE") →
      (LANEIGE, SOS_SCORE, 37.2)
      (LANEIGE, OWNED_BY_GROUP, AMOREPACIFIC)
      (LANEIGE, HAS_PRODUCT, Lip Sleeping Mask)

  4-b. 경쟁사 탐색:
    query(subject="LANEIGE", predicate=COMPETES_WITH) →
      (LANEIGE, COMPETES_WITH, COSRX)
      (LANEIGE, COMPETES_WITH, TIRTIR)
      (LANEIGE, COMPETES_WITH, Neutrogena)

  4-c. 경쟁사 네트워크 (2-hop):
    각 경쟁사의 SoS, 순위, 제품 수 등 조회
      COSRX → SoS: 15.3%, avg_rank: 8.2
      TIRTIR → SoS: 12.1%, avg_rank: 5.7
      Neutrogena → SoS: 8.5%, avg_rank: 12.3

  4-d. 카테고리 계층:
    lip_care → parent: skin_care → parent: beauty

  4-e. 트렌드 정보:
    (LANEIGE, TREND_UP, lip_care, period="2026-01~02")

  4-f. 감성 데이터 (있으면):
    (LANEIGE, SENTIMENT_POSITIVE, Hydration, score=0.82)

════════════════════════════════════════════════════════════
Step 5: 추론 컨텍스트 구축 (Inference)
════════════════════════════════════════════════════════════
  KG 사실들을 OntologyReasoner에 전달:

  활성화 규칙: Market Position + Competitive Threat

  추론 결과:
  1. "LANEIGE는 Lip Care 시장 지배적 브랜드 (SoS 37.2%, Top 3)"
     [confidence: 0.88]
  2. "TIRTIR의 빠른 성장세(SoS 12.1%↑) 주의 필요"
     [confidence: 0.75]
  3. "COSRX 안정적 2위, 직접 위협은 낮지만 모니터링 필요"
     [confidence: 0.70]

════════════════════════════════════════════════════════════
Step 6: 문서 검색 (RAG Search)
════════════════════════════════════════════════════════════
  강화된 쿼리로 Dense + Sparse 검색:

  검색 결과 (RRF 융합 후 Top-5):
  1. [A1] 지표 해석 가이드 - "SoS 산출 방법 및 해석..."  [score: 0.87]
  2. [B1] 경쟁 환경 분석 - "Lip Care 카테고리 경쟁 구도..."  [score: 0.82]
  3. [A2] 시장 분석 플레이북 - "점유율 트렌드 분석 절차..."  [score: 0.78]
  4. [C1] 순위 변동 대응 - "경쟁사 순위 변동 시..."  [score: 0.71]
  5. [D1] 아모레퍼시픽 IR - "Lip Care 매출 성장..."  [score: 0.65]

════════════════════════════════════════════════════════════
Step 7: 관련성 평가 (Relevance Grading)
════════════════════════════════════════════════════════════
  각 RAG 결과의 쿼리 관련성 평가:
  → 5개 중 4개 통과, 1개(D1 IR 문서) 제외 (관련도 부족)

════════════════════════════════════════════════════════════
Step 8: 가중 병합 (Weighted Merge) ★
════════════════════════════════════════════════════════════
  인텐트 가중치 적용 (COMPETITION):
    KG 사실들      × 0.45 → 상위 배치
    RAG 문서 청크   × 0.25 → 중위 배치
    추론 인사이트    × 0.30 → 중상위 배치

  최대 컨텍스트 항목 수 제한:
    KG: 최대 15개, RAG: 최대 10개, Inference: 최대 8개

  선선도 보정 (Freshness):
    최근 데이터(24시간 이내)에 1.2x 가중치 부여

════════════════════════════════════════════════════════════
Step 9: 컨텍스트 조합 (Context Combination)
════════════════════════════════════════════════════════════
  최종 HybridContext 생성:
  {
    query: "LANEIGE Lip Care 경쟁사 대비 점유율 추이는?",
    entities: {brands: [LANEIGE], categories: [lip_care], ...},
    ontology_facts: [
      "LANEIGE는 AMOREPACIFIC 소속",
      "LANEIGE SoS: 37.2%",
      "경쟁사: COSRX(15.3%), TIRTIR(12.1%), Neutrogena(8.5%)",
      ...
    ],
    inferences: [
      "시장 지배적 브랜드 (SoS 37.2%)",
      "TIRTIR 성장세 주의 필요",
      ...
    ],
    rag_chunks: [
      "SoS 산출 방법: 카테고리 내 브랜드 제품 수 / 전체 ...",
      "Lip Care 경쟁 구도: 상위 5개 브랜드가 ...",
      ...
    ],
    combined_context: "위의 모든 정보가 하나의 텍스트로 합쳐진 것",
    metadata: {
      confidence: 0.84,
      intent: "COMPETITION",
      sources_used: ["kg", "rag", "inference"],
      conflict_warnings: [],
      ...
    }
  }

════════════════════════════════════════════════════════════
Step 10: 신뢰도 산출 (Confidence Fusion)
════════════════════════════════════════════════════════════
  vector_score: 0.82 × 0.40 = 0.328
  ontology_score: 0.88 × 0.35 = 0.308
  entity_score: 1.0 × 0.25 = 0.250

  최종 신뢰도: 0.886
  충돌 경고: 없음

════════════════════════════════════════════════════════════
                    → LLM에 전달 →
════════════════════════════════════════════════════════════
  combined_context가 LLM 프롬프트의 {{context}} 부분에 삽입됨
  → LLM은 이 컨텍스트를 기반으로 답변 생성
  → KG 사실이 포함되어 있으므로 "같은 브랜드 경쟁" 할루시네이션 방지
```

### 5.4 두 가지 검색 경로 (Dual Path)

HybridRetriever는 **두 가지 전략 경로**를 지원한다:

```
[경로 1: OWL Strategy (신규)]
  OWL 온톨로지가 활성화된 경우 사용
  → EntityLinker로 OWL 개념 매핑
  → OWLReasoner로 형식 추론
  → Cross-Encoder 재순위화
  → ConfidenceFusion으로 다중 소스 융합

[경로 2: Legacy Strategy (기본)]
  OWL 없이도 작동하는 기본 경로
  → EntityExtractor로 키워드 추출
  → KG + OntologyReasoner(규칙 기반)
  → RRF로 문서 융합
  → 가중 병합

retrieve_unified() 메서드가 자동 선택:
  if owl_strategy 설정됨 → 경로 1
  else → 경로 2
```

---

## 6. 전체 쿼리 흐름 (End-to-End)

사용자의 질문이 시스템을 통과하는 전체 여정:

```
사용자                    시스템
  │
  │  "LANEIGE 경쟁사 분석해줘"
  │
  ▼
[FastAPI Endpoint: /api/v3/chat]
  │
  ▼
[QueryRouter]─────────────────────────────────────────────
  │ 1. 쿼리 카테고리 분류 (COMPETITIVE)
  │ 2. 복합 쿼리면 서브쿼리로 분해
  │    (단일 쿼리면 그대로 통과)
  │
  ▼
[HybridChatbotAgent]─────────────────────────────────────
  │ 1. 대화 이력 확인 (ConversationMemory)
  │ 2. 세션 컨텍스트 로드
  │
  ▼
[HybridRetriever.retrieve()]──────────────────────────────
  │
  │  ┌─────────────┬─────────────────┬──────────────────┐
  │  ▼             ▼                 ▼                  │
  │ [KG Query]   [RAG Search]    [Ontology Reasoning]   │
  │  브랜드 사실    문서 검색         규칙 기반 추론        │
  │  경쟁사 관계    Dense+Sparse     Forward Chaining     │
  │  카테고리 계층   RRF 융합         + OWL 추론           │
  │  │             │                 │                  │
  │  └─────────────┴────────┬────────┘                  │
  │                         ▼                           │
  │              [Weighted Merge]                        │
  │              인텐트 기반 가중 병합                      │
  │                         │                           │
  │                         ▼                           │
  │              [Confidence Fusion]                     │
  │              신뢰도 산출 + 충돌 감지                    │
  │                         │                           │
  │                         ▼                           │
  │              HybridContext (통합 컨텍스트)              │
  │                                                     │
  ▼
[LLM (GPT-4.1-mini via LiteLLM)]────────────────────────
  │ 시스템 프롬프트 + HybridContext + 사용자 질문
  │ → 답변 생성
  │
  ▼
[HallucinationDetector]──────────────────────────────────
  │ 답변이 컨텍스트와 일치하는지 검증
  │ 근거 없는 주장 감지
  │
  ▼
[SourceProvider]──────────────────────────────────────────
  │ 출처 정보 추출 및 포매팅
  │
  ▼
[SuggestionEngine]───────────────────────────────────────
  │ 후속 질문 3개 자동 생성
  │
  ▼
사용자에게 응답:
  "LANEIGE는 Lip Care 시장에서 37.2%의 점유율로 1위를 유지하고 있습니다.
   주요 경쟁사로는 COSRX(15.3%), TIRTIR(12.1%), Neutrogena(8.5%)가 있으며,
   특히 TIRTIR의 최근 성장세가 주목됩니다.

   [출처: KG 데이터, 시장 분석 플레이북]
   [신뢰도: 88.6%]

   후속 질문 제안:
   1. TIRTIR의 성장 요인은 무엇인가요?
   2. LANEIGE의 가격 경쟁력은 어떤가요?
   3. 최근 순위 변동 추이를 보여주세요."
```

---

## 7. 할루시네이션 방지 메커니즘

이 시스템이 할루시네이션을 방지하는 **7가지 레이어**:

### Layer 1: KG 사실 기반 제약 (Knowledge Grounding)
```
메커니즘: KG에 저장된 사실만 "확실한 정보"로 LLM에 제공
효과: "LANEIGE SoS = 37.2%"라는 KG 사실이 있으므로
      LLM이 "약 40%"라고 반올림하거나 임의의 숫자를 만들 수 없음
코드: hybrid_retriever.py의 _query_knowledge_graph()
```

### Layer 2: 기업 소유 관계 (Corporate Ownership)
```
메커니즘: OWNED_BY_GROUP, SIBLING_BRAND 관계로
         같은 그룹 브랜드를 경쟁사에서 자동 제외
효과: "LANEIGE vs 이니스프리 경쟁" 같은 오류 원천 차단
코드: domain/entities/relations.py의 RelationType
```

### Layer 3: Ontology 규칙 기반 일관성
```
메커니즘: OWL 추론으로 논리적 모순 감지
         (예: DominantBrand이면서 NicheBrand일 수 없음)
효과: 자기 모순적인 답변 방지
코드: owl_reasoner.py의 OWL 클래스 계층
```

### Layer 4: 인텐트 기반 검색 가중치
```
메커니즘: 질문 유형에 따라 가장 신뢰할 수 있는 소스에
         높은 가중치 부여
효과: 경쟁사 질문에 KG 가중치↑ → 문서의 모호한 정보보다
      KG의 정확한 관계가 우선 반영
코드: retrieval_strategy.py의 IntentRetrievalConfig
```

### Layer 5: 신뢰도 융합 + 충돌 감지
```
메커니즘: 소스 간 점수 차이 > 0.3이면 경고
         "문서는 관련 있다고 하는데 KG는 아니라고 함"
효과: 모순되는 정보를 그대로 LLM에 넘기지 않고 필터링
코드: confidence_fusion.py의 conflict detection
```

### Layer 6: HallucinationDetector (사후 검증)
```
메커니즘: LLM이 생성한 답변을 컨텍스트와 비교하여
         근거 없는 주장을 감지
효과: LLM이 컨텍스트에 없는 정보를 "지어낸" 경우 감지
코드: core/hallucination_detector.py
```

### Layer 7: 출처 추적 (Source Tracking)
```
메커니즘: 모든 정보에 출처를 명시
         "KG 데이터", "시장 분석 플레이북 p.3" 등
효과: 사용자가 답변의 근거를 직접 확인 가능
코드: agents/source_provider.py
```

---

## 8. 비즈니스 도메인 적용 사례

### 사례 1: "LANEIGE 경쟁사는?"

```
[Without KG+Ontology — 순수 RAG만]
  RAG가 문서에서 "LANEIGE"가 나오는 문서를 검색
  → 여러 LANEIGE 제품이 함께 언급된 문서를 찾음
  → LLM: "LANEIGE Lip Sleeping Mask의 경쟁사는 LANEIGE Water Sleeping Mask입니다"
  → ❌ 같은 브랜드 제품을 경쟁사로 답변 (할루시네이션)

[With KG+Ontology — 하이브리드]
  KG: (LANEIGE, OWNED_BY_GROUP, AMOREPACIFIC)
  KG: (LANEIGE, COMPETES_WITH, COSRX)
  KG: (LANEIGE, COMPETES_WITH, TIRTIR)
  Ontology: 같은 OWNED_BY_GROUP → 경쟁 관계 아님
  → LLM: "LANEIGE의 주요 경쟁사는 COSRX, TIRTIR, Neutrogena입니다"
  → ✅ 정확한 답변
```

### 사례 2: "시장 점유율이 얼마야?"

```
[Without KG — 순수 RAG]
  RAG가 문서에서 과거의 점유율 수치를 찾아올 수 있음
  → "37.2%" (하지만 이것이 현재인지 과거인지 불명확)

[With KG — 하이브리드]
  KG: (LANEIGE, SOS_SCORE, 37.2, timestamp="2026-02-18")
  → 날짜가 명시된 최신 데이터를 KG에서 직접 가져옴
  → Freshness 가중치로 최신 데이터 우선
  → "2026년 2월 18일 기준 LANEIGE의 Lip Care SoS는 37.2%입니다"
```

### 사례 3: "Lip Care와 Lip Makeup 차이가 뭐야?"

```
KG 카테고리 계층:
  Beauty & Personal Care
  ├── Skin Care
  │   └── Lip Care         ← LANEIGE Lip Sleeping Mask
  └── Makeup
      └── Lip Makeup       ← 립스틱, 립글로스

Ontology 규칙: PARENT_CATEGORY가 다르면 다른 시장 세그먼트
추론: "Lip Care(보습)와 Lip Makeup(색조)은 다른 카테고리입니다"

→ 카테고리 혼동 할루시네이션 방지
```

### 사례 4: "시장이 과점 상태야?"

```
KG: (lip_care, HHI_SCORE, 0.22)
Ontology 규칙:
  IF HHI > 0.25 THEN "고도 집중"
  IF 0.15 < HHI ≤ 0.25 THEN "중간 집중"
  IF HHI ≤ 0.15 THEN "경쟁적"

추론: HHI 0.22 → "중간 집중 시장"
→ "Lip Care 시장은 HHI 0.22로 중간 집중도를 보이며,
   상위 3개 브랜드가 시장의 약 65%를 점유하고 있습니다."
```

---

## 9. 아키텍처 요약 다이어그램

### 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        사용자 질문                                │
│                "LANEIGE 경쟁력 분석해줘"                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    QueryRouter                                 │
│             인텐트 분류 + 복합쿼리 분해                            │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌═══════════════════════════════════════════════════════════════┐
║                  HybridRetriever (심장부)                      ║
║                                                               ║
║   ┌─────────────┐  ┌─────────────────┐  ┌────────────────┐   ║
║   │  Knowledge   │  │   Ontology +    │  │      RAG       │   ║
║   │   Graph      │  │   Reasoning     │  │   (Documents)  │   ║
║   │             │  │                 │  │                │   ║
║   │ ┌─────────┐ │  │ ┌─────────────┐ │  │ ┌────────────┐ │   ║
║   │ │ Triples │ │  │ │ OWL Reasoner│ │  │ │  ChromaDB  │ │   ║
║   │ │ (S,P,O) │ │  │ │ (형식 논리)  │ │  │ │  (벡터 DB) │ │   ║
║   │ └─────────┘ │  │ └─────────────┘ │  │ └────────────┘ │   ║
║   │ ┌─────────┐ │  │ ┌─────────────┐ │  │ ┌────────────┐ │   ║
║   │ │ Indexes │ │  │ │ Business    │ │  │ │   BM25     │ │   ║
║   │ │ (3-way) │ │  │ │ Rules (8종) │ │  │ │ (키워드)   │ │   ║
║   │ └─────────┘ │  │ └─────────────┘ │  │ └────────────┘ │   ║
║   │ ┌─────────┐ │  │ ┌─────────────┐ │  │ ┌────────────┐ │   ║
║   │ │  JSON   │ │  │ │ Unified     │ │  │ │    RRF     │ │   ║
║   │ │  File   │ │  │ │ Reasoner    │ │  │ │   융합     │ │   ║
║   │ └─────────┘ │  │ └─────────────┘ │  │ └────────────┘ │   ║
║   └──────┬──────┘  └────────┬────────┘  └───────┬────────┘   ║
║          │                  │                    │             ║
║          └──────────────────┼────────────────────┘             ║
║                             ▼                                  ║
║              ┌──────────────────────────┐                      ║
║              │    Weighted Merge        │                      ║
║              │  인텐트 기반 가중 병합     │                      ║
║              └─────────────┬────────────┘                      ║
║                            ▼                                   ║
║              ┌──────────────────────────┐                      ║
║              │   Confidence Fusion      │                      ║
║              │  신뢰도 산출 + 충돌 감지   │                      ║
║              └─────────────┬────────────┘                      ║
║                            ▼                                   ║
║                    HybridContext                                ║
║         (KG사실 + 추론결과 + RAG문서 = 통합 컨텍스트)              ║
╚═══════════════════════════╤═══════════════════════════════════╝
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                 LLM (GPT-4.1-mini)                             │
│          시스템프롬프트 + HybridContext + 질문 → 답변 생성         │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│  HallucinationDetector → SourceProvider → SuggestionEngine    │
│          사후검증              출처첨부          후속질문 생성      │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        사용자 응답                                │
│   "LANEIGE는 SoS 37.2%로 시장 1위... 경쟁사: COSRX, TIRTIR..."  │
│   [신뢰도: 88.6%] [출처: KG, 시장분석 플레이북]                    │
└─────────────────────────────────────────────────────────────────┘
```

### 인간-에이전트 1:1 매핑 요약

```
┌──────────────────────────────────────────────────────────────┐
│                    인간의 사고 과정                             │
│                                                              │
│   ┌──────────┐   ┌──────────────┐   ┌──────────────────┐    │
│   │ 자료 찾기  │   │ 관계 지도     │   │ 세계관/판단 규칙   │    │
│   │ (책, 보고서)│   │ (경험, 상식)  │   │ (분류, 규칙, 논리) │    │
│   └─────┬────┘   └──────┬───────┘   └────────┬─────────┘    │
│         │               │                     │              │
│         └───────────────┼─────────────────────┘              │
│                         ▼                                    │
│               종합 판단 & 답변 생성                              │
└──────────────────────────────────────────────────────────────┘
                         ↕  1:1 매핑
┌──────────────────────────────────────────────────────────────┐
│                    AI 에이전트 시스템                            │
│                                                              │
│   ┌──────────┐   ┌──────────────┐   ┌──────────────────┐    │
│   │  RAG     │   │ Knowledge    │   │  Ontology +      │    │
│   │ 문서 검색  │   │ Graph        │   │  Reasoning       │    │
│   │ 14개 문서  │   │ 트리플 스토어  │   │  OWL + 규칙 8종   │    │
│   └─────┬────┘   └──────┬───────┘   └────────┬─────────┘    │
│         │               │                     │              │
│         └───────────────┼─────────────────────┘              │
│                         ▼                                    │
│              HybridRetriever → LLM → 답변                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 부록: 주요 파일 참조 테이블

| 컴포넌트 | 파일 | 줄 수 | 핵심 역할 |
|---------|------|-------|----------|
| HybridRetriever | `src/rag/hybrid_retriever.py` | 1,583 | RAG+KG+Ontology 통합 오케스트레이터 |
| DocumentRetriever | `src/rag/retriever.py` | 1,383 | 14개 문서 라이브러리 + Dense/Sparse 검색 |
| RetrievalStrategy | `src/rag/retrieval_strategy.py` | 712 | 인텐트 기반 검색 전략 패턴 |
| ConfidenceFusion | `src/rag/confidence_fusion.py` | 649 | 다중 소스 신뢰도 융합 |
| EntityLinker | `src/rag/entity_linker.py` | ~400 | NER + 온톨로지 개념 매핑 |
| CrossEncoderReranker | `src/rag/reranker.py` | ~300 | 2-stage 재순위화 |
| KnowledgeGraph | `src/ontology/knowledge_graph.py` | 551 | 인메모리 Triple Store |
| KGQueryMixin | `src/ontology/kg_query.py` | ~400 | SPARQL-like 쿼리 + BFS 탐색 |
| OntologyReasoner | `src/ontology/reasoner.py` | 932 | Forward Chaining 규칙 엔진 |
| OWLReasoner | `src/ontology/owl_reasoner.py` | ~600 | OWL2 형식 논리 추론 |
| UnifiedReasoner | `src/ontology/unified_reasoner.py` | ~300 | OWL + Rules 가중 융합 |
| RelationType | `src/domain/entities/relations.py` | 697 | 30+ 관계 유형 정의 |
| QueryRouter | `src/core/query_router.py` | ~200 | 쿼리 분류 + 라우팅 |
| Business Rules | `src/ontology/rules/` (6파일) | ~1,500 | 8가지 도메인 규칙 세트 |

---

> **문서 끝**
> 작성: Claude Code (AI Agent Engineer 관점 분석)
> 코드 기반: AMORE RAG-KG Hybrid Agent main 브랜치
