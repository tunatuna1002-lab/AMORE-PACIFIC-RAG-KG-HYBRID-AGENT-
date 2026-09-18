## §O2 — 인식·정규화 (연결기 등록부 사전) [2026-09-18]

> `docs/experiments/ontology_activation_2026-09.md`가 이 브랜치 기준(0b56e4b)에 없어 따로 적었다. 리드가 그 문서의 §O2로 옮긴다.

- 계획서 §6 O2, 결정 OA-6(새 인식은 `ontology.use_class_reasoning` 뒤). 플래그 읽기는 `FeatureFlags.get_flag("ontology", "use_class_reasoning", default=False)`를 **호출마다** 한다(env `FF_ONTOLOGY_USE_CLASS_REASONING`).
- API 비용 $0, LLM 호출 0건. 데이터는 `eval_output/evidence-2026-09/eval_data_snapshot_base`의 **사본**(amore_data.db sha1 fc4271c8…, knowledge_graph.json sha1 770ed1d0…, 실행 전후 동일, KG `auto_save=False`).

### 바뀐 것

| 대상 | 플래그 OFF | 플래그 ON |
|---|---|---|
| `EntityLinker.extract_entities` | O2 이전과 바이트 단위로 같음 (특성화 테스트 87개 질의 + KG 경로 3개) | 등록부 이름·별칭으로 브랜드 보강, 가짜 브랜드 제거, `brand_ids`·`classes`·`groups`·`relations_hint` 키 추가 |
| `EntityLinker.link` | 같음 | 브랜드 엔티티에 `context.registry_id`, 등록부 브랜드 추가, 가짜 브랜드 제거 |
| `resolve_entity` 도구 | 같음 (7개 표기 스냅샷) | 카드 metadata에 `registry_id`·`classes`(폐포 후)·`group`, 그룹·클래스 언급도 카드, `entities.groups/classes/brand_ids` |
| `entity_tags` | 변경 없음 | 코드 변경 없음. 색인 시 ON이면 새 브랜드 태그가 붙을 뿐 태그 문자열 형식(`|laneige|`)은 같다 |

- 브랜드 문자열: 기존 사전이 알던 브랜드는 기존 문자열 그대로(`e.l.f.`·`la roche-posay`·`l'oreal`), 새 브랜드는 등록부 이름 소문자(`it cosmetics`·`charlotte tilbury` — KG·DB 표기와 같음).
- 등록부 사전에서 뺀 모호 표기: `려`(RYO 한글, "성공하려면"에 걸림 — 골든 4문항), `ap`(구문표에서 그룹으로 처리), `essence`·`median`·`matrix`·`verb`·`dove`(일반 단어).

### 측정 1 — 골든 L1 엔티티 연결 F1 (오프라인, typed multihop 27 + relation 27 + rule 42 = 96문항)

러너와 같은 식(`L1QueryMetrics(use_fuzzy=True)._compute_entity_link_f1`: 브랜드·카테고리·지표·제품 합집합 vs `gold.kg_entities`).

| 조건 | 파일 | OFF | ON | 차이 |
|---|---|---|---|---|
| KG 사본 사용(파이프라인과 같음) | multihop | 0.5211 | 0.5211 | +0.0000 |
| | relation | 0.6860 | 0.7412 | +0.0552 |
| | rule | 0.6951 | 0.7560 | +0.0609 |
| | **전체** | **0.6436** | **0.6858** | **+0.0422** |
| KG 없음 | 전체 | 0.6631 | 0.7089 | +0.0458 |
| exact(퍼지 없음), KG 사용 | 전체 | 0.6337 | 0.6633 | +0.0297 |

- multihop이 그대로인 이유: 이 파일의 미스는 브랜드가 아니라 제품·지표·그룹 표기다(골드가 `amorepacific` 등을 요구하지만 질문에 브랜드가 없음).
- 골드 `kg_entities`는 등록부 id 표기(`it_cosmetics`·`loreal`)라 exact F1에서는 문자열(`it cosmetics`·`l'oreal`)이 일부만 맞는다. `brand_ids`를 채점에 쓰면 더 오른다(평가 쪽 결정, O2 범위 밖).

### 측정 2 — 새로 인식한 브랜드와 오탐 (골든 전체 273개 질문, KG 없음)

- 브랜드 목록이 바뀐 질문 17개, **전부 추가, 제거 0**.
- 새로 인식: Almay(3), COVERGIRL(2), IT Cosmetics, Jouer, Charlotte Tilbury, AESTURA(2), TATA HARPER, Aquaphor(2), Burt's Bees(2), ChapStick, L'Oreal(질문 원문 기준 — KG 사용 시 OFF에서도 제품 역링크로 잡힘).
- IOPE·primera·한율 등은 골든 질문에 나오지 않는다(단위 테스트로 인식 확인).
- **새 오탐 0건**: 17건 모두 질문에 실제로 쓰인 브랜드다. 2건(`Lip Care에서 Aquaphor 제품은 몇 위`, `LANEIGE와 Burt's Bees 비교`)은 골드 `kg_entities`에 해당 브랜드가 빠져 있어 채점상 FP로 잡히지만 골드 누락이다.
- 남은 기존 오탐(ON에서도 유지 — "기존 인식은 잃지 않는다" 원칙): `config/entities.json`의 `rhode` 별칭 `로드`가 "로드맵"에 걸린다. 설정 파일은 O2 소유가 아니라 고치지 않았다.
- 클래스·그룹 언급(ON): `classes` 26문항, `groups` 28문항, `relations_hint=sibling` 7문항. 예: mh001 "아모레퍼시픽 그룹 소속 브랜드 중…" → `AmorepacificBrand`·`amorepacific`, lg099 "K-Beauty 브랜드 중…" → `KBeautyBrand`, rl013~016 "같은 그룹에 속한 자매 브랜드인가요?" → `sibling`. 세그먼트 클래스는 "럭셔리 브랜드"처럼 뒤에 브랜드·라인 류 단어가 있을 때만 인식한다("Skin Care 카테고리 프리미엄화 트렌드"는 인식 안 함).

### 측정 3 — 오프라인 규칙 일치율 (생성 rule 32문항, 방법은 `eval_output/evidence-2026-09/notes/3b_rule_offline_match.md`)

| 경로 | 이전 기록(3-B) | 이번 OFF | 이번 ON |
|---|---|---|---|
| A 파이프라인(실제 `retrieve`) | 25/32 | 25/32 (재현) | **29/32** |
| B 골드 엔티티 | 29/32 | 29/32 | 29/32 |

- ON에서 rg021(IT Cosmetics)·rg022(Jouer)·rg023(Almay)·rg026(Charlotte Tilbury)이 발화로 돌아섰다. 공허 일치였던 rg007·rg020·rg024·rg025(브랜드 미인식)도 이제 실제 조건 평가로 판정된다. 공허 일치 6개 → 2개(rg004 카테고리 전용, rg032 TIRTIR 소유 카드 없음 — 3-B의 B 경로와 같음).
- 남은 3개(rg001~003)는 골드가 조건 단위라 규칙 발화로 표현될 수 없는 문항이다(3-B 해석과 같음). 파이프라인이 골드 엔티티 상한에 도달했다.
- rg001에서 `nivea`가 추가로 추출되는 현상은 그대로다 — 사전이 아니라 KG 제품 슬러그 역링크 경로다(O2 사전 변경과 무관).

### 판단 기록

- 등록부 별칭 목록을 돌려주는 공개 API가 로더에 없어, 연결기가 `config/ontology/brands.json`의 표기를 읽고 각 표기가 `normalize_brand`로 같은 id가 되는 것만 쓴다(로더는 수정하지 않음).
- `아모레퍼시픽`: 기존 사전이 브랜드 `amorepacific`로 내보내던 것은 유지하고(ON에서도), 그룹 `groups=["amorepacific"]`를 함께 낸다. `brand_ids`에는 넣지 않는다(등록부에서 그룹). `Amore Pacific`(공백)은 등록부대로 브랜드 라인 `amore_pacific`.
- `AmorepacificBrand` 클래스는 그룹 언급 뒤에 "브랜드들/브랜드 중/소속/산하/계열/포트폴리오/소유한 브랜드/brands"가 올 때만 낸다. "아모레퍼시픽 그룹 브랜드 COSRX의 원산지"처럼 특정 브랜드를 한정하는 표현은 그룹만 낸다.
- "중저가"는 `MidTierBrand`·`AffordableBrand` 둘 다로 둔다.
- ON의 `brand_ids` 키는 레거시 신뢰도 점수(`query_graph.py` 엔티티 개수, 상한 3)에 더해진다. 판정은 O7 측정에서 본다.
- 범위 밖으로 남김: `amazon_scraper.py`의 브랜드 오귀속(가짜 브랜드 `unknown`·`fresh`·`chi`의 출처)은 고치지 않았다 — 조회 쪽에서 `is_placeholder`로만 거른다.
