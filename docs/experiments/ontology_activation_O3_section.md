## §O3 — 질의 경로 온톨로지 추론 (플래그 `ontology.use_class_reasoning`, 기본 OFF) [2026-09 사후]

> 이 파일은 `docs/experiments/ontology_activation_2026-09.md`(O0가 만드는 문서)에 붙일 절이다.
> O3 작업 기준 커밋 `0b56e4b`에는 그 문서가 없어 따로 적었다. 리드가 병합 때 옮긴다.
> 모든 수치는 LLM 호출 없이(API $0) 잰 것이다.

### 무엇을 바꿨나

| 항목 | 내용 | 위치 |
|---|---|---|
| 플래그 | `ontology.use_class_reasoning`(bool, 기본 false, ENV `FF_ONTOLOGY_USE_CLASS_REASONING`), `kg.write_validation`(`off`/`warn`/`enforce`, 기본 `warn`, 잘못된 값 → `warn`, ENV `FF_KG_WRITE_VALIDATION`) | `src/infrastructure/feature_flags.py`, `config/feature_flags.json` |
| 질의 해석 | 엔티티 → 등록부 id·그룹·클래스·힌트 → 전개 후보 → 순위·상한(12) | `src/rag/ontology_context.py`(신규) |
| 그룹·클래스 전개 | 그룹만 언급(소속 브랜드가 함께 언급되지 않음) → 소속 브랜드, 클래스(`LuxuryBrand` 등) → 인스턴스, 자매·세그먼트 힌트는 기준 브랜드 1개일 때만. 가짜 브랜드 제외. 수치 조회 브랜드 상한 3 → 12(전개가 있을 때만). 넘으면 `expansionTruncated` 카드(뺀 브랜드 목록). 순서: 크롤 DB 등장 → KG 크롤 관계 등장 → 이름 | `ontology_context.py`, `hybrid_retriever._plan_ontology`, `metric_facts.collect(max_brands=)` |
| 정적 사실 카드(OE5) | 질의 브랜드의 `ownedByGroup`·`hasSegment`·`originatesFrom`·`acquiredIn`, 전개 브랜드는 전개 근거 술어 1장. 출처 `ontology:registry`, `as_of` = 온톨로지 `as_of`(2026-09-18). KG 사실 5개 상한(`_weighted_merge`)과 관계 카드 20장 상한 밖에서 최대 24장까지 프롬프트에 싣는다(기존 관계 카드를 밀어내지 않음) | `evidence_adapters._RelationBuilder.ontology_card`, `hybrid_retriever._with_ontology_cards` |
| 닫힌 세계(OE4) | 그룹 소속·자매를 묻는 질의(그룹 언급·그룹 클래스·자매 힌트)에서 등록부 브랜드끼리만 `notOwnedByGroup`·`notSiblingBrand`. 등록부 밖 브랜드는 `groupMembershipUnknown`("모름", "아님"으로 쓰지 않음) | `ontology_context.static_edges` |
| 읽을 때 정식화 | `ownedBy`→`ownedByGroup`(별칭), `hasPosition`→`hasSoS`·`hasHHI`·`hasPricePosition`(`original_predicate`로 분리). 등록부 브랜드의 KG 정적 트리플은 등록부 카드가 대신한다(중복 방지). 레거시 표기와 O5 정식 표기를 모두 받는다 | `hybrid_retriever._query_knowledge_graph`, `EvidenceAdapter(ontology=…)` |
| `EXCLUDED_METADATA` | ON에서는 날짜 없는 **수치** 메타데이터만 뺀다. 정적 키(`segment`·`group`·`origin`·`acquired`)는 관계 카드 | `evidence_adapters._kg_brand_info` |
| `priority_preds` | ON에서 `ownedByGroup`·`ownsBrand`·`siblingBrand`·`hasSegment`·`originatesFrom`·`acquiredIn` 포함 | `hybrid_retriever._PRIORITY_PREDS_CANONICAL` |
| 카테고리 포함(OE3) | 질의 카테고리의 하위 카테고리를 KG 조회(`category_brands`), DB 조회(시장 지표가 있는 하위 카테고리만, 최대 4개), 규칙 조합(수치 카드가 있는 것만), 문서 태그 가산점 대상에 더한다. **수치 카드는 자기 카테고리 그대로이고 합산·환산 코드는 없다** — `test_inclusion_widens_scope_but_never_rescales_numbers`가 모든 SoS 카드 = DB(브랜드, 자기 카테고리)/100, skin_care의 LANEIGE는 부재 카드 그대로, HHI는 카테고리별 원값임을 확인 | `hybrid_retriever`, `metric_facts._scope_categories` |
| 카드 수 추적 | `metadata["ontology"]`: 온톨로지 카드 수(전체·프롬프트·술어별), 전개·제외 브랜드, 조회 범위 카테고리, 전체·프롬프트 카드 수. OFF면 키 없음 | `hybrid_retriever._ontology_trace` |

**OFF 불변**: 변경 전 코드(`0b56e4b`)로 5질의(그룹·세그먼트·원산지·포함·부정)의 `HybridContext` 전체를 스냅샷으로 고정하고(`tests/unit/rag/fixtures/o3_flag_off_snapshot.json`), OFF(ENV 없음·`false`)에서 완전 일치를 확인한다(`test_class_reasoning_off_characterization.py`, 10건).

### 무비용 골드 엣지 도달률 (OFF → ON)

- 방법: `scripts/o3_gold_edge_reachability.py`. typed `multihop.jsonl`+`relation.jsonl` 54문항, **골드 `kg_entities`를 엔티티로 입력**(연결기 O2 미사용), `eval_output/evidence-2026-09/eval_data_snapshot_tagged`의 KG·DB를 임시 디렉터리에 복사해 사용(원본 불변, KG 사본도 바이트 동일 확인), 문서 검색은 빈 결과, `as_of` 2026-08-31.
- 도달 = 골드 엣지가 방출 카드(관계 카드 전체) 또는 평가 러너가 읽는 `metric_edges`·`ontology_static` 원문에 있음. 비교는 평가 L3와 같이 엣지 전체 소문자.
- "별칭 동일시"는 골드·방출 술어를 온톨로지 정식 이름으로 맞춰 비교한 참고값이다(`ownedBy` ≡ `ownedByGroup`).

골드 엔티티만 (힌트 없음):

| 술어 | 골드 | OFF | ON | OFF (별칭 동일시) | ON (별칭 동일시) |
|---|---:|---:|---:|---:|---:|
| ownedByGroup | 27 | 0 | 23 | 23 | 23 |
| rankedIn | 15 | 2 | 2 | 2 | 2 |
| hasSegment | 14 | 0 | 14 | 0 | 14 |
| belongsToCategory | 12 | 0 | 0 | 0 | 0 |
| competesWith | 10 | 4 | 4 | 4 | 4 |
| hasProduct | 8 | 0 | 0 | 0 | 0 |
| ownedBy | 6 | 6 | 0 | 6 | 6 |
| hasSoS | 3 | 0 | 0 | 0 | 0 |
| originatesFrom | 2 | 0 | 1 | 0 | 1 |
| siblingBrand | 2 | 0 | 2 | 0 | 2 |
| acquiredIn | 1 | 0 | 1 | 0 | 1 |
| **전체** | 100 | 12 | 47 | 35 | 53 |

골드 엔티티 + 질문 표현으로 만든 `relations_hint`("같은 그룹"·"자매" → sibling, "같은 세그먼트" → segment; O2 흉내):
`ownedByGroup` 0 → **27/27**, 전체 12 → **51** (별칭 동일시 35 → 57). 나머지 술어는 위와 같다.

평균 카드 수(54문항, 골드 엔티티 입력): 전체 evidence OFF 103.4 → ON 114.5(힌트 116.1), **프롬프트 카드 OFF 53.9 → ON 59.4**(힌트 60.0), 그중 온톨로지 카드 5.1(5.6), 프롬프트 최대 79장.

해석과 한계:
- `ownedByGroup`·`hasSegment`·`siblingBrand`·`acquiredIn`은 ON에서 도달 가능해졌다. OFF는 같은 사실을 `ownedBy`로만 내서 엄격 비교 0이었다.
- **ON은 골드의 레거시 술어 `ownedBy` 6건(lg102·lg158)을 엄격 비교에서 잃는다**(카드가 정식 이름 `ownedByGroup`만 쓰므로). L3 채점이 온톨로지 `canonical_predicate`로 양쪽을 맞추면 잃지 않는다(별칭 동일시 열 6/6). O0/리드 판단 필요.
- `originatesFrom` 1/2: rl011 골드가 `cosrx -originatesFrom-> korea`인데 등록부 국가 id는 `south_korea`. 골드 어휘 불일치라 고치지 않았다(골드 수정 금지).
- `hasSegment` 14/14는 골드 엔티티에 답 브랜드가 들어 있어서 쉬운 값이다(rl018·rl019). 실제 연결기에서는 세그먼트 힌트(O2)가 있어야 전개된다.
- `rankedIn`·`belongsToCategory`·`hasProduct`·`hasSoS`는 O3 범위 밖(제품 슬러그·수치 엣지·`_weighted_merge` 5개 상한)이라 OFF와 같다.
- 평가 러너 `_extract_l3_trace`는 `metric_edges`·`competitors`·`category_hierarchy`만 읽는다. ON의 정적 사실은 `ontology_static` 사실(같은 `data.edges` 모양)에 있으므로 **러너가 이 유형도 읽어야 O7 L3에 잡힌다**(한 줄 변경, O0 소유 파일).

### 현재 연결기(O2 이전) + ON 스모크 관찰 (LLM 없음)

| 문항 | 연결기 엔티티 | 온톨로지 카드(프롬프트) | 관찰 |
|---|---|---|---|
| mh001 그룹 | brands `amorepacific`, face_powder | 13 | 소속 12개 전개(DB 등장 순: etude·innisfree·laneige·cosrx·hera…), 19개 제외 카드 |
| rl007 세그먼트 | cosrx, amorepacific | 4 | `COSRX hasSegment K-Beauty` 포함, 그룹은 맥락이라 전개 안 함 |
| rl010 원산지 | `amorepacific`만(TATA HARPER 미인식) | 13 | 연결기가 브랜드를 못 잡아 그룹 전개로 빠짐 — tata_harper는 이름 순위에서 제외됨. O2 병합 후 재확인 필요 |
| lg155 포함 | laneige, cosrx, skin_care | 7 | 조회 범위에 lip_care 포함, skin_care의 LANEIGE는 부재 그대로 |
| rl015 부정 | laneige, tirtir | 5 | 자매 힌트가 없어 부정 카드 없음 — O2의 `relations_hint`가 있어야 `notOwnedByGroup`·`notSiblingBrand`가 나온다(단위 테스트로 확인) |

### 남긴 것 / 다른 트랙에 넘길 것

- O0: L3 추출에 `ontology_static` 추가, 술어 별칭 정식화 채점.
- O4: `rule_contracts.PARENT_GROUP` 바인딩이 `ownedBy`만 본다. ON에서는 카드 술어가 `ownedByGroup`이므로 `extra_predicates=("ownedByGroup",)`가 필요하다(ON에서 소유 검증 규칙 입력이 비는 회귀).
- O2: `relations_hint`에 `sibling`/`segment` 값을 내 주면 쌍 판정·자매 전개·세그먼트 전개가 켜진다(받는 값: `sibling`·`siblings`·`same_group`·`group`·`siblingbrand`, `segment`·`same_segment`·`hassegment`).
- 그룹 전개 + 카테고리 없는 질의는 12개 브랜드 × 자동 카테고리 3개를 조회해 전체 카드가 크게 는다(rl010 관찰 218장, 프롬프트는 60장으로 상한). O7에서 비용·지연과 함께 본다.
