## §O4 — 규칙 입력 연결 (플래그 `ontology.use_class_reasoning`, 기본 OFF) [2026-09 사후]

> 리드가 `docs/experiments/ontology_activation_2026-09.md`에 옮길 절이다.
> 작업 기준 커밋 `e0ffac7`. 비용 $0이고 LLM 호출은 0건이다(litellm 호출을 예외로 막고 호출 수를 셌다).
> 데이터는 `eval_output/evidence-2026-09/eval_data_snapshot_tagged`의 DB·KG를 임시 디렉터리에 복사해 썼다.
> sha1은 amore_data.db fc4271c8…, knowledge_graph.json 770ed1d0…이다. 원본과 사본 모두 실행 전후에 같았고, KG는 `auto_save=False`로 열었다.

### 무엇을 바꿨나 (`src/ontology/rule_contracts.py`)

| 입력 | 이전 | 이후 |
|---|---|---|
| `parent_group` (소유 검증 필수 입력) | 카드 술어 `ownedBy`만 읽었다 | `ownedBy`와 `ownedByGroup`을 모두 읽는다. 바인딩에 `ontology_aliases=True`를 두고, `get_ontology().canonical_predicate`가 같은 정식 이름을 내는 술어는 같은 술어로 본다 |
| `country_of_origin` | gap("표시용 — 전용 카드 변환 규칙 없음") | 등록부 카드 `originatesFrom`에서 표시 이름을 읽는다(`South Korea`) |
| `segment` | gap | 등록부 카드 `hasSegment`에서 표시 이름을 읽는다(`K-Beauty`, `Premium`…) |
| `acquired` | gap | 등록부 카드 `acquiredIn`에서 읽는다(`2024`) |

- 새 바인딩 옵션은 두 가지다.
  - `EvidenceBinding.sources`: 카드 출처를 제한한다.
  - `Reduce.OBJECT_LABEL`: 목적어의 등록부 표시 이름을 쓰고, 표시 이름이 없으면 목적어를 그대로 쓴다.
- 원산지·세그먼트·인수 입력은 **출처 `ontology:registry` 카드만** 읽는다(`REGISTRY_SOURCE`).
  - KG에 있는 AP 브랜드 30개의 원산지 "Korea"는 `kg_updater`가 넣은 기본값이다. 원본에서 온 진술이 아니다(결정 OA-5).
  - 등록부 카드는 플래그가 ON일 때만 생긴다. 그래서 OFF에서는 이 세 입력이 계속 결측이다.
  - 입력이 결측이면 규칙 결론은 기존 기본값 `'Korea'`를 쓴다. 이 기본값은 규칙 파일에 있고 바꾸지 않았다.
- 세 입력은 모두 선택 입력이다. 소유 검증 규칙은 결론 문장과 metadata에서만 이 값을 쓰므로 **발화 여부는 바뀌지 않는다**. 발화 조건은 여전히 `parent_group == amorepacific` 하나다.

### OFF 불변 증거

1. **특성화 스냅샷.** `tests/unit/ontology/test_rule_contracts_ontology.py`의 스냅샷 파일은 `fixtures/o4_rule_off_snapshot.json`이다.
   - 변경 전 코드(`e0ffac7`)에서 만든 뒤 따로 커밋했다(`af2bf5b`).
   - 입력은 OFF 어댑터 카드다. OFF 경로가 받을 수 있는 정적 술어 표기를 모두 넣었다.
     - `metric_edges`: `ownedByGroup`·`ownedBy`·`hasSegment`·`originatesFrom`·`acquiredIn`
     - `brand_info`: 정적 메타 키
   - 이 입력으로 두 조합 세트의 전체 `RuleRun`이 같은지 비교한다. 비교 대상은 요약, 판정마다의 사유·입력·근거 카드, 결론 문장·metadata·evidence다.
   - OFF 어댑터는 `ownedByGroup`을 `ownedBy`로 바꾼다. 그래서 OFF 카드에는 `ownedByGroup`도, 출처 `ontology:registry`도 없다. 같은 파일의 테스트가 이것을 확인한다.
2. **오프라인 32문항 전 판정 비교.** 변경 전후의 OFF 실행에서 A·B 두 경로 모두, 문항마다 모든 조합 × 모든 규칙의 판정이 **완전히 같았다**. 비교 대상은 발화 여부, 사유 라벨, 입력 이름, 결론 문장과 metadata다.

### 오프라인 규칙 일치율 (생성 rule 32문항)

- **방법**: `eval_output/evidence-2026-09/notes/3b_rule_offline_match.md`와 같은 일치식을 쓴다. 일치 조건은 `bool(발화 ∩ rule_ids) == expected fires`이다.
- **A 경로**: 실제 `HybridRetriever.retrieve(질문)`을 돌린다. 연결기도 실제 것을 쓰고, 문서 검색기만 빈 가짜로 바꾼다.
- **B 경로**: 같은 `retrieve`를 부르되, 연결기 출력을 골드 `rule_context`의 brand·category로 바꿔 넣는다.
  - 기존 `scripts/ontology_offline_checks.py` §4의 B는 `_query_knowledge_graph`를 온톨로지 계획 없이 부른다. 그래서 ON 경로(정식 어댑터·등록부 카드)를 재지 못해 방법을 이렇게 바꿨다.
  - OFF 값은 3-B·O2 기록과 같다(25/29).

| 코드 | 플래그 | A 연결기 | B 골드 엔티티 |
|---|---|---:|---:|
| O4 이전 (`e0ffac7`) | OFF | 25/32 | 29/32 |
| O4 이전 (`e0ffac7`) | ON | **27/32** | **27/32** |
| O4 이후 | OFF | 25/32 (판정 전부 동일) | 29/32 |
| O4 이후 | ON | **29/32** | **29/32** |

- **O4 이전 ON의 회귀**: rg030(COSRX)과 rg031(innisfree)이 `missing_input:parent_group`으로 불일치했다.
  - 원인은 O3가 넘긴 문제다. ON 카드 술어는 `ownedByGroup`인데 계약은 `ownedBy`만 읽었다.
  - O4 이후 두 문항 모두 발화해 O2 보고값 29/32로 돌아왔다. 골드 엔티티 상한과 같다.
- **남은 불일치 3개(rg001~003)**: 골드가 규칙이 아니라 조건 단위로 적혀 있어 규칙 발화로 표현할 수 없는 문항이다(3-B 해석과 같다).
- **공허 일치**: ON에서는 rg004(카테고리 전용)와 rg032(TIRTIR) 두 개이고, O2 기록과 같다.
  - rg032: TIRTIR는 등록부에 그룹이 없어 `parent_group`이 여전히 결측이다.
  - 이제 원산지(South Korea)와 세그먼트(Mid) 입력은 채워진다. 하지만 선택 입력이라 발화하지 않는다. 골드도 "발화 안 함"이다.

### 새로 발화 가능해진 규칙

| 비교 | 경로 | 새로 판정 가능(결측 → 평가) | 새로 발화 | 발화를 잃은 규칙 |
|---|---|---|---|---|
| ON, O4 이전 → O4 이후 | A | `brand_ownership_verification` (14문항) | `brand_ownership_verification` | 없음 |
| ON, O4 이전 → O4 이후 | B | `brand_ownership_verification` (14문항) | `brand_ownership_verification` | 없음 |
| O4 이후, OFF → ON | B | 없음 | 없음 | 없음 |
| O4 이후, OFF → ON | A | 12개 규칙 (rg007·rg020~026) | `fragmented_market_competition`, `premium_price_position`, `price_quality_mismatch`, `value_position` | 없음 |

- 14문항은 rg009~019, rg029, rg030, rg031이다. 브랜드가 LANEIGE·COSRX·innisfree인 문항에서 소유 검증 판정이 결측에서 발화로 바뀌었다.
- 마지막 행(A 경로 OFF → ON)은 **O2 연결기 효과**다. IT Cosmetics·Jouer·Almay·Charlotte Tilbury·COVERGIRL을 인식하게 된 결과이고, O4 코드가 만든 차이가 아니다.
- 32문항 전체 판정 기준으로 발화한 규칙의 종류와 판정 수는 아래와 같다.

| 조건 | A 규칙 종류 | A 발화 판정 | B 규칙 종류 | B 발화 판정 |
|---|---:|---:|---:|---:|
| O4 이전 ON | 8종 | 66 | 8종 | 57 |
| O4 이후 ON | 9종 | 80 | 9종 | 71 |
| O4 이후 OFF | 8종 | 69 | 9종 | 71 |

- 원산지·세그먼트 입력은 ON 소유 검증 판정 중 A·B 각각 18건에서 채워졌다. 인수 연도는 1건(COSRX 2024)에서 채워졌다.
- 예: rg030의 결론은 "cosrx는 아모레퍼시픽 그룹 소속 브랜드입니다. 원산지: South Korea. 인수 연도: 2024"이다. 이전에는 원산지가 기본값 "Korea"였고 인수 연도는 비어 있었다.

### 가격 규칙과 세그먼트·티어 — 연결하지 않음

`value_position`과 `premium_price_position`은 조건과 결론 모두 `cpi`·`rating_gap`·`brand`·`asin`만 읽는다. 세그먼트나 티어를 쓰는 로직이 없으므로 입력을 추가하지 않았다. 입력만 추가하면 계약에 뜻이 없는 입력이 생긴다.

"럭셔리 브랜드면 CPI 기준을 다르게" 같은 새 의미를 규칙에 넣으려면 rule 골드를 다시 정의해야 한다. 이는 O4 범위 밖이다. 테스트 `TestPriceRulesDoNotReadSegment`가 이 판단을 고정한다.

### `is_target` 일반화 — 하지 않음

- **검토한 안**: `is_target`을 "질의 브랜드면 참"으로 바꾼다. 조합의 브랜드는 모두 질의 브랜드다.
- **확인 방법**: 같은 오프라인 스크립트를 `--generalize-target`으로 돌렸다. `_derive`만 바꾸고 나머지 경로는 그대로 두었다.
- **결과**: rule 골드 32문항 중 **15문항의 발화 규칙 집합이 바뀌었다**. ON·OFF 모두, A·B 모두에서 그렇다.
  - rg001, rg005~008, rg020, rg022~028, rg030, rg031이 해당한다.
  - 예: rg006·rg008 medicube에 `top3_achievement`가 발화했다. rg028 L'Oreal에는 `strong_avg_rank`·`category_entry_opportunity`·`strong_rating_position`·`top3_achievement`가 발화했다. rg030 COSRX에는 `category_entry_opportunity`가 발화했다.
- 일치 수는 A OFF가 25에서 26으로, A ON이 29에서 30으로 올랐다. 늘어난 1은 rg001이고, 올바른 이유로 맞은 것이 아니다.
  - 연결기가 오인식한 nivea가 조합에 들어갔다.
  - 그 nivea에서 `category_entry_opportunity`가 발화해 골드 규칙과 겹쳤다.
  - 골드 rg001은 브랜드가 없는 카테고리 문항이다.
- **결론**: 골드 영향이 0이 아니다. rule 골드는 LANEIGE를 기준으로 정의되어 있으므로 일반화하지 않는다. 이 근거는 `IS_TARGET.note`에 적었다. 규칙 동작은 바꾸지 않았다.

### 판단 기록

- **`ontology_aliases`는 문자열 비교를 먼저 한다.** 정식화는 이름이 다를 때만 부른다.
  - 온톨로지 로드에 실패하면 별칭 확장 없이 기존 이름만 쓴다(`_canonical_predicate`가 None을 돌려준다). 규칙 판정이 온톨로지 원본의 형식 오류에 막히지 않게 하려는 것이다.
  - 이 확장은 `PARENT_GROUP`과 등록부 프로필 입력에만 켰다. 경쟁사 등 다른 바인딩은 바꾸지 않았다.
- **ON에서 등록부 밖 브랜드의 KG `ownedBy`**: ON 어댑터가 이것을 `ownedByGroup`으로 바꾸므로 `parent_group`으로는 여전히 읽힌다. 출처 제한은 원산지·세그먼트·인수 입력에만 두었다.
- **표시 이름을 입력으로 쓴 이유**: 규칙 결론 문장이 이 값을 그대로 출력한다. 등록부 id(`south_korea`·`k_beauty`)가 아니라 라벨이 기존 기본값 "Korea"와 같은 층위다.
  - 그룹(`parent_group`)은 조건 비교에 쓰이므로 기존대로 id를 쓴다.
- **소유 파일**: `src/ontology/rules/*.py`는 바꾸지 않았다. 입력 선언은 모두 `rule_contracts.py`에 있다.

### 테스트

- `tests/unit/ontology/test_rule_contracts_ontology.py`에 12개를 새로 넣었다.
  - OFF 특성화 2개
  - ON `ownedByGroup`·등록부 그룹 2개
  - 프로필 입력·결론·결측·KG 출처 제외·TIRTIR 미발화 5개
  - 가격 규칙 2개
  - 출처 상수 일치 1개
- `tests/unit/ontology tests/unit/rag tests/unit/core`: **3406 passed**, 실패 0.

### 재현

오프라인 스크립트는 워크트리 scratchpad에서 돌렸다(`o4_rule_match.py`, 저장소에 커밋하지 않음). `scripts/`는 이 트랙의 소유가 아니다. 필요하면 리드가 `scripts/ontology_offline_checks.py` §4의 B 경로를 "연결기 출력 대체 + `retrieve`" 방식으로 바꾸면 ON을 잴 수 있다.

- **실행 흐름**: 플래그 OFF·ON마다 A·B를 돌린다. `hybrid_retriever.evaluate_rules_on_cards`를 감싸 `RuleRun`을 수집한다.
- **B 경로**: 연결기 출력을 `{brands: [rc.brand], categories: [rc.category]}`로 바꾸고, 그 밖의 목록 키는 비운다.
- **일반화 검토**: `--generalize-target`으로 `rule_contracts._derive`의 `is_target`을 "브랜드가 있으면 참"으로 바꾼다.
