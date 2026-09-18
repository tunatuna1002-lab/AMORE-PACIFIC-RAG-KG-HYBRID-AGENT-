# 온톨로지 검토 보고서 (2026-09-18)

> 질문: "온톨로지는 역할을 하고 있나? 안 쓴다면 왜인가?"
> 방법: 읽기 전용 코드 조사 3건(OWL 코드·이력 / 질의 경로의 온톨로지 지식 / 평가로 잴 수 있는 방법) + 리드 확인. 운영 `data/`는 JSON을 직접 읽기만 했다(`KnowledgeGraph()`는 로드 중 파일을 다시 쓰므로 쓰지 않음).
> 기준 커밋: `a7ac285` (브랜치 `feat/evidence-react-ontology-2026-09`, PR #12). 모든 내용은 공모전 이후의 사후 점검이다.
> 표시: **[확인]** = 리드가 코드·데이터로 직접 확인, **[조사]** = 조사 에이전트 측정값(작동 계획서 O0에서 재측정한다).

---

## 1. 결론 요약

이 프로젝트의 "온톨로지"는 세 층으로 나뉘고, 층마다 답이 다르다.

| 층 | 무엇 | 실질적 역할 | 한 줄 근거 |
|---|---|---|---|
| ① 규칙 추론 | `src/ontology/reasoner.py` + 규칙 37개, 입력은 증거 카드(`rule_contracts.py`) | **있음 (측정 확인)** | 규칙 on/off 비교에서 규칙 정답 일치율 0.469 → 0.779(범위 비겹침), 발화 문항 0 → 57% |
| ② 지식 그래프 | `data/knowledge_graph.json` 트리플 3,480개 | **있음, 효과는 부분 입증** | 관계 카드로 프롬프트에 실린다. 다만 KG를 끄면 근거성만 떨어지고 정답성 지표 차이는 없었다(`docs/experiments/kg_ablation_2026-09.md`) |
| ③ OWL 온톨로지 | `src/ontology/owl_reasoner.py`(1,247줄), `ontology_knowledge_graph.py`, `cosmetics_ontology.owl` | **없음** | 서비스 경로에서 생성하는 곳 0곳 **[확인]**. 이 머신에 Java가 없어 추론기(Pellet/HermiT)는 어느 경로로도 돌지 않는다 **[조사]** |

**더 큰 문제는 ③이 아니라 "클래스·계층 지식이 질의 경로에서 전혀 쓰이지 않는다"는 점이다.** 그룹 소속, 세그먼트, 원산지(K-Beauty), 카테고리 포함 관계 같은 지식은 설정 파일과 KG에 이미 있는데, 질의 경로는 전부 문자열 일치만 하고 이 지식을 **걸러서 버린다**. 다단계·관계 골든 54문항 중 그룹 소속 18문항, 세그먼트 5문항, 원산지 5문항이 이 지식을 필요로 한다 **[조사]**.

---

## 2. OWL은 왜 안 쓰이는가

### 2.1 코드 사실

1. **생성 호출처 0** **[확인]**: `OWLReasoner`는 `OntologyKnowledgeGraph` 안에서만 import되고, `OntologyKnowledgeGraph`는 `src/ontology/__init__.py`가 export만 한다. 쓰는 곳은 테스트와 `scripts/migrate_kg_to_ontology.py`뿐이다. `git log -S "OntologyKnowledgeGraph("` 결과, 서비스 코드에 연결된 적이 한 번도 없다 **[조사]**.
2. **`entity_linker.py:323`의 `self.owl_reasoner`는 저장만 되고 읽히지 않는다** **[확인]**. 생성처는 모두 `owl_reasoner` 없이 만든다.
3. **플래그 이름이 동작을 속인다** **[확인]**:
   - `reasoner.use_owl_reasoner`는 OWL과 무관하다. `use_unified_reasoner`와 함께 **Python 규칙 추론 전체의 on/off 스위치**다(`hybrid_retriever.py:577`, `tool_registry.py:328`).
   - `ontology.use_ontology_kg`는 `OntologyKnowledgeGraph`가 아니라 **일반 KG 조회의 on/off**다(`hybrid_retriever.py:549`, `tool_registry.py:324,409`).
4. **Java 없음** **[조사]**: owlready2 0.50은 import되지만 추론기는 Java(JRE)가 필요하다. 로컬에 없고, 배포 이미지(`python:3.11-slim`)에도 없다.
5. **OWL 파일 두 벌이 서로 맞지 않는다** **[조사]**:
   - `cosmetics_ontology.owl`에는 클래스 23개, SWRL 규칙 3개가 있지만 로드하는 코드가 없다.
   - `owl_reasoner.py`는 이 파일을 쓰지 않고 코드에서 클래스를 따로 만든다.
   - 네임스페이스는 `cosmetics#`, `amore_brand.owl#`, `amore_brand.owl#Brand/X` 세 가지다.
   - 속성 이름도 `competsWith`(오타), `competitorOf`로 갈린다.
6. **카테고리 계층은 OWL이 아니다** **[확인]**: 계층은 `config/category_hierarchy.json`을 `kg_updater.load_category_hierarchy()`가 읽어 메모리 KG에 넣는다. 그러므로 CLAUDE.md·AGENTS.md·포트폴리오 문서의 "OWL은 카테고리 계층 어휘로만 사용"은 **사실과 다르다**(작동 계획서 O6에서 정정).

### 2.1-추가. Java 설치 후 재확인 (2026-09-18, 사용자가 Java 설치) **[확인]**

- 설치된 Java: `1.8.0_503`(Java 8). `.venv`의 Python은 **3.14.3**이다(CLAUDE.md의 3.13.7과 다름).
- **HermiT: 동작한다.** 임시 예제 온톨로지(`AmorepacificBrand ≡ Brand ⊓ ownedBy value amorepacific`)에서 laneige·cosrx를 자동 분류하고 elf는 뺐다.
- **Pellet: 실패한다.** owlready2 0.50에 든 Apache Jena가 Java 25용(class file 69)으로 컴파일돼 있어 Java 8에서 `UnsupportedClassVersionError`가 난다. Pellet을 쓰려면 JDK 25 이상이 필요하다.
- **`src/ontology/cosmetics_ontology.owl`은 읽기부터 실패한다.** 81행 `Strong competitor (15% < SoS ≤ 30%)`의 `<`가 XML에서 이스케이프(`&lt;`)되지 않아 RDF/XML 파싱 오류가 난다. 이 파일은 도입(`06f5ea4`) 이후 어떤 도구로도 로드할 수 없는 상태였다.
- 배포 이미지(`python:3.11-slim`)에는 여전히 Java가 없다. 로컬에서 추론이 된다고 서비스 경로에서 쓸 수 있는 것은 아니다.

### 2.2 결정 이력 **[조사]**

| 날짜 | 커밋·결정 | 내용 |
|---|---|---|
| 2026-01-23 | `06f5ea4` | `.owl` 파일·`owl_reasoner` 도입 |
| 02-09 | `7ca17ce` | `OntologyKnowledgeGraph` 추가(서비스 연결 없음) |
| 02-15 | `f049cb8` | 리팩터 후 생성자 인자 불일치로 TypeError, 예외를 삼켜 OWL 검색 전략이 항상 None |
| 02-18 | `ac332bb` `92da983` | 개수 제약·분리 공리·일관성 검사 추가(여전히 미연결) |
| 09-17 | `26cd8e6`, 결정 D1 | OWL 전략 배선 수리, 검증 전이라 기본 OFF |
| 09-17 | `kg_ablation_2026-09.md` §6 | OWL 전략을 켜면 `$or` 필터 미처리 등으로 엔티티가 연결된 124/130문항에서 문서 0건, 종합 −0.187 → OFF 유지 |
| 09-18 | `eb5dff2`, 결정 S4-1 | OWL 검색 전략·플래그 삭제. 근거: 설계 E3 "OWL은 검색 전략이 아니라 어휘·일관성 검사 역할" |

**정리**: OWL은 처음부터 서비스에 연결된 적이 없고, 유일하게 연결을 시도한 "검색 전략"은 나빠서 삭제됐다. 설계 E3가 남긴 역할(어휘·일관성 검사)은 **맡을 코드가 한 번도 만들어지지 않았다.** 이것이 "안 쓰는 이유"의 전부다.

### 2.3 OWL 모듈의 잠복 결함 **[조사]**

| # | 결함 | 위치 |
|---|---|---|
| 1 | `sync_owl_inferences`가 항상 0건. `fact["property"]`를 읽는데 실제 키는 `"relation"`이고, 값도 오타(`competsWith`) | `ontology_knowledge_graph.py:288`, `owl_reasoner.py:624` |
| 2 | OKG `check_consistency`가 OWL 불일치를 보고하지 못함(`ConsistencyReport` 객체가 항상 truthy) | `ontology_knowledge_graph.py:357-359` |
| 3 | Java가 없으면 일관성 검사가 경고만 남기고 통과로 보고. 관련 테스트 174개 통과는 실제 추론 없이 얻은 결과 | `owl_reasoner.py:1024` |
| 4 | `fallback_reasoner`는 저장만 되고 쓰이지 않음 | `owl_reasoner.py:125-133` |
| 5 | 인스턴스를 만들 때마다 개수 제약이 중복 누적 | `owl_reasoner.py:230` |
| 6 | `add_product`의 `if price:`가 0.0을 버림 | `owl_reasoner.py:411` |
| 7 | `owlready2>=0.45`가 서비스에서 쓰이지 않는 의존성 | `requirements.txt:76` |

---

## 3. 질의 경로에서 온톨로지 지식이 버려지는 곳

### 3.1 지식은 있는데 쓰이지 않는다

| 개념 | 정의된 곳 | 질의 경로에서 | 문제 |
|---|---|---|---|
| 브랜드 어휘 | `config/entities.json`(27), `entity_linker.KNOWN_BRANDS`(약 30), 스크레이퍼 코드 여러 곳, `brands.json` | 사전 부분 문자열 일치 | 5곳 이상에 흩어져 서로 다르다. KG 브랜드 83개 중 약 65개, 아모레퍼시픽 브랜드 31개 중 약 25개를 인식하지 못한다 **[조사]**(예: IT Cosmetics, Almay, Charlotte Tilbury, IOPE, primera) |
| 그룹 소속 | `brands.json`, KG `ownedByGroup`·`ownsBrand`·`siblingBrand` | 그룹을 브랜드 집합으로 전개하지 않음 | "아모레퍼시픽 브랜드 중…" 질문이 `amorepacific` 한 브랜드로만 조회된다 |
| 세그먼트·티어 | `brands.json`(AP는 `segment`, 경쟁사는 `tier`), KG `hasSegment` 31 | **버려짐** | 술어 허용 목록(`priority_preds`, `hybrid_retriever.py:1045`)에 없다 **[확인]**. 엔티티 메타데이터는 "날짜 없음" 이유로 제외된다(`EXCLUDED_METADATA`, `evidence_adapters.py:56,735`) **[확인]** |
| 원산지(K-Beauty) | `brands.json` korean_brands 16, KG `originatesFrom` 31(AP만) | **버려짐** | 경쟁 K-Beauty 브랜드(anua, medicube, biodance 등)는 원산지 정보가 없다 |
| 카테고리 계층 | `config/category_hierarchy.json` | 조상·자식 카드로 출력만 함 | 규칙 카드 매칭, 브랜드 조회, 문서 태그가 모두 카테고리 **정확 일치**라 포함 관계로 넓히지 않는다. 디스크 KG에는 계층 트리플이 0건 |
| 관계 술어 | `src/domain/entities/relations.py`(약 35개) | enricher가 합쳐 버림 | `hasSoS`·`hasHHI`·가격 포지션이 `hasPosition` 하나로 뭉치고, `rankedIn`은 `belongsToCategory`로 바뀐다. `ownedBy`와 `ownedByGroup`이 함께 존재한다 |
| 엔티티 타입 | 없음 | — | KG에 타입 트리플 0건, `entity_metadata` 0개 **[확인]**. 브랜드·제품·카테고리를 구분할 수 없고, `unknown`·`fresh`·`chi` 같은 가짜 브랜드가 섞인다 |

### 3.2 KG 데이터 품질 **[조사]**

- **대소문자 이원화**: `LANEIGE`/`laneige`, `COSRX`/`cosrx`, `e.l.f.`/`elf`. 시드 로더는 원 표기, enricher는 소문자를 쓴다. 질의 경로가 대소문자 4가지 변형을 모두 조회해 가리고 있지만, `make ON`, `Mise-en-scène` 같은 표기에서는 실패한다.
- **경쟁 관계 오염**: `competesWith` 616건은 "같은 Top 100에 함께 나옴"일 뿐이다. 비대칭 374쌍이고, `unknown`(SoS 30%), `fresh`(eos 등의 제목에 "Fresh"), `chi`(KimChiChic)가 섞여 있다.
- **수치 엣지에 날짜 없음**: 수치 엣지 1,864건 모두 `valid_from`이 없다. SoS는 덮어써져 이력이 없고, HHI는 값이 곧 목적어라 카테고리당 9~15개가 쌓인다.

### 3.3 규칙 37개 중 발화 가능한 규칙은 13개 **[조사]**

- **발화 불가 24개**: 이력(기간 비교) 필요 11, 감성 데이터 없음 8, IR 구조화 추출기 없음 5.
- **클래스 지식이 있으면 넓어지는 규칙**:
  - `is_target`이 `laneige`로 고정된 규칙 4개. "AP 그룹 브랜드", "K-Beauty 브랜드" 클래스로 일반화할 수 있다.
  - 가격 규칙 2개에 세그먼트·티어를 넣을 수 있다.
  - 소유 검증 규칙에 원산지·세그먼트·인수 정보를 넣을 수 있다. 지금은 KG에 있는데도 "입력 없음"으로 표시된다(`rule_contracts.py:722-725`).

### 3.4 골든 문항이 요구하는 온톨로지 추론 **[조사]**

multihop 27 + relation 27 = 54문항 기준(한 문항이 여러 태그를 가질 수 있음).

| 필요한 추론 | multihop | relation | 예 |
|---|---|---|---|
| 그룹 소속·전개 | 6 | 12 | "COSRX와 같은 그룹 브랜드 중 Face Powder Top 100에 제품이 있는 브랜드 수" |
| 세그먼트 | 0 | 5 | "LANEIGE와 같은 세그먼트 브랜드" |
| 원산지·K-Beauty | 3 | 2 | "K-Beauty 브랜드 중 가장 강한" |
| 제품·브랜드 → 카테고리 | 19 | 4 | "LANEIGE 제품이 속한 카테고리의 HHI" |
| 카테고리 포함 관계 | 2 | 0 | lg155(skin_care ⊃ lip_care) |
| 경쟁사 집합 | 5 | 6 | |
| 온톨로지 불필요 | 4 | 0 | |

rl015·rl016("TIRTIR·Beauty of Joseon은 자매 브랜드가 아님")은 **부정 판정**이 필요하다. 브랜드 등록부를 닫힌 세계로 보는 규칙이 있어야 답할 수 있다.

---

## 4. 평가로 효과를 잴 수 있는가 **[조사]**

| 지표 | 현재 상태 | 온톨로지 효과를 재려면 |
|---|---|---|
| L4 `constraint_violation_rate` | **233문항 모두 0.0 고정**. 규칙 결과가 `trace.rule_evaluation`에만 있고 이 지표는 비어 있는 `inferences`를 읽는다 | 규칙 발화 결과를 입력으로 연결 |
| L4 `type_consistency_rate` | **모두 1.0 고정**. `expected_types`를 넘기지 않는다 | 골드 엔티티 타입을 넘김 |
| L3 edge recall | 골드 엣지가 없는 103/233문항은 1.0으로 기본 처리돼 평균이 부풀려진다 | 골드 엣지가 있는 문항만, **술어별로** 집계 |
| 골드 ↔ 방출 술어 | `ownedByGroup` 0/29(런타임은 `ownedBy`로 방출. 이름만 맞춰도 14/29), `hasSegment` 0/14, `originatesFrom`·`siblingBrand`·`acquiredIn`은 방출 0 | 술어 정규화 + 버려지던 술어 노출 |
| 규칙 정답 일치율 | rule 시험지 32문항에만 골드가 있다. s5 기준 0.781(25/32) | 그대로 사용 |

무비용(LLM 호출 없음)으로 먼저 볼 수 있는 것도 있다: 브랜드 인식 범위, 골드 엣지 도달률, KG 정합성 위반 수, 카테고리 포함 확장의 적중률.

---

## 5. 판단

1. **OWL 추론기(owlready2 + Java)를 살리는 것은 권하지 않는다.**
   - 배포 이미지에 Java가 없다.
   - 현재 OWL이 하는 일(SoS 구간 분류)은 Python 규칙이 이미 한다.
   - 한 번 연결했던 경로(검색 전략)는 측정에서 나빴다.
2. **대신 "온톨로지"를 질의 경로가 실제로 읽는 단일 원본 스키마로 만든다.** 이것이 설계 E3("어휘·일관성 검사")가 원래 뜻한 역할이다.
   - **클래스와 계층**: Brand / CorporateGroup / Category / Product, 세그먼트, 원산지.
   - **술어 정의**: 도메인·범위, 역관계, 대칭.
   - **브랜드 등록부**: 이름, 별칭, 그룹, 세그먼트·티어, 원산지.

   이 원본에서 다음을 만든다:
   - 엔티티 인식 사전
   - KG 시드와 쓰기 검증
   - 질의 확장: 그룹 전개, 클래스 소속, 카테고리 포함
   - 규칙 입력
3. 효과는 먼저 무비용 지표로 확인하고, 그다음 54문항·42문항 측정으로 판정한다. 켜기 기준은 지난 작업과 같은 노이즈 규칙을 쓴다.
4. 과장 금지: 이 작업 전까지는 "OWL 온톨로지 추론"이라는 표현을 쓰지 않는다. 지금 정확한 표현은 "규칙 기반 추론(37개 규칙, 13개 발화 가능) + 지식 그래프 관계 조회"다.

실행 계획은 [`docs/plans/ontology-activation-plan-2026-09-18.md`](../plans/ontology-activation-plan-2026-09-18.md)에 있다.
