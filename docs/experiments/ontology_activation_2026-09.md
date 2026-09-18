# 온톨로지 작동 실험 기록 (2026-09) [2026-09 사후]

> 계획: [`docs/plans/ontology-activation-plan-2026-09-18.md`](../plans/ontology-activation-plan-2026-09-18.md) (O0~O7).
> 근거: [`docs/analysis/ontology-review-2026-09-18.md`](../analysis/ontology-review-2026-09-18.md) (이하 검토 보고서).
> 이 문서는 단계별로 절을 이어 붙인다. 모든 내용은 공모전 이후의 사후 보완이다.

---

## §O0 — 측정 수리와 무비용 기준선 (API $0)

- 기준 커밋 `d12668d`, 작업 커밋 `05ce6e9`·`64e0cae`·`6906326`·`e488a9e` (브랜치 `feat/ontology-activation-2026-09` 위 worktree).
- LLM·임베딩 호출 0건. 재채점은 저장된 리포트의 trace만, 무비용 점검은 KG JSON 사본과 스냅샷 **사본**만 읽었다.
- 산출물: `eval_output/ontology-2026-09/rescore/`(리포트별 JSON, `summary.json`, `summary.md`),
  `eval_output/ontology-2026-09/offline_checks/`(`offline_checks.json`, `offline_checks.md`).

### O0-A. 지표 수리

**레거시 필드와 게이트는 그대로 두고 새 필드를 더했다.** `constraint_violation_rate`·`type_consistency_rate`는
`overall_score`와 통과 게이트(`constraint_violation_max 0.05`, `type_consistency_min 0.90`)에 들어간다. 계산식을
바꾸면 재사용 기준선과 새 실행의 통과율이 서로 다른 잣대가 된다. 그래서 과거 값(0.0/1.0 고정)은 비교용으로 남기고,
새 지표는 별도 필드로 두었다. 게이트에 새 지표를 쓸지는 O7에서 정한다.

| 새 필드 | 정의 | 판정 없음 |
|---|---|---|
| L3 `kg_edge_recall_gold_only` | 골드 엣지가 1개 이상인 문항의 엣지 recall. 기존 `kg_edge_recall`과 같은 문자열 일치(소문자·공백 정리) | 골드 엣지 없는 문항은 None(기존은 1.0) |
| L3 `edge_recall_by_predicate` | 골드 술어(골드 표기 그대로)별 {matched, total}. **별칭을 맞춰 주지 않는다** — 런타임이 `ownedBy`로 내서 `ownedByGroup`을 놓치는 것도 지금 시스템의 결과로 센다 | — |
| L4 `rule_constraint_violation_rate` | `trace.rule_evaluation.fired`에 든 규칙의 추론(`l4_ontology.inferences`) 중 제약을 하나라도 어긴 비율 | 검사한 추론 0이면 None |
| L4 `typed_consistency_rate` | 트레이스의 타입 주장(방출 엣지의 주어·목적어, ontology_facts가 함의하는 노드 타입) 중 기대 타입과 맞는 비율 | 검사 0건이면 None |

**규칙 제약 위반의 정의** (`eval/validators/ontology_validator.py` `check_rule_inferences`). 추론 하나가 아래 중 하나라도
해당하면 위반 1건이다.

- `schema`: 필수 필드 누락, confidence가 [0,1] 밖, 알 수 없는 insight_type (기존 검사와 같음)
- `subject_type`: `context_snapshot.brand`가 비었거나 등록부상 브랜드가 아님(가짜 브랜드·그룹·카테고리)
- `category_type`: `context_snapshot.category`가 비었거나 카테고리가 아님
- `related_entity_invalid`: `related_entities`에 빈 문자열이나 가짜 브랜드(unknown·fresh·chi)
- `value_range`: SoS·HHI가 [0,1] 밖, rating_gap이 [-5,5] 밖, 순위가 [1,100] 밖, 가격·CPI ≤ 0, 숫자 자리에 숫자가 아님
- `missing_as_of`: 수치 입력으로 발화했는데 `as_of`가 없음
- `ownership_mismatch`: `parent_group`이 등록부의 소속 그룹과 다름(등록부가 모르면 판정하지 않음)

**기대 타입의 출처.** 골든 233문항에는 엔티티 타입이 **하나도 없다**. 그래서 우선순위를 두었다:
골드 명시 타입(`GoldEvidence.kg_entity_types`, 새 필드이며 현재 비어 있음) → 골드 엣지의 술어 시그니처
(`cosrx -ownedByGroup-> amorepacific`이면 cosrx=brand) → O1 온톨로지 로더(`src.ontology.ontology.get_ontology()`가
`entity_type()`을 제공할 때만. 지금은 없음) → config 등록부(`config/brands.json`·`config/category_hierarchy.json`, 읽기 전용)
→ ASIN 모양. 문항별로 `type_source`와 `types_registry_derived`(현재 전 문항 True)를 기록한다. 어느 출처도 모르는
엔티티(nivea·carmex 등 등록부 밖 브랜드, 제품 슬러그)는 `type_untyped`로 세고 검사하지 않는다.

술어 시그니처는 계획서 §6 O1의 정의를 따랐다(`ownedBy`는 `ownedByGroup`의 별칭, `rankedIn` 도메인 Brand·Product,
`belongsToCategory` 도메인 Product, `hasPosition`·`hasHHI`·`acquiredIn`의 범위는 리터럴이라 검사하지 않음).
기존 `ALLOWED_RELATIONS`는 `(product, ownedBy, brand)`처럼 런타임 사용법과 맞지 않아 새 검사에 쓰지 않았다.

### O0-A. 재사용 기준선 재채점

`scripts/rescore_l3_l4.py`로 저장 리포트를 다시 채점했다. 리포트에는 골드가 저장되지 않아 `eval/data/golden/typed/combined_v1.jsonl`에서
문항 ID로 가져왔고, **기존 `kg_edge_recall`을 다시 계산해 저장값과 전 문항 일치**함을 확인했다(불일치 0건 — 골드가 그때와 같다).
인프라 실패 문항(`trace.error`)은 리포트 집계처럼 뺐다.

| 구성 | 커밋 | runs | 문항 | L3 recall(기존) | L3 recall(골드 엣지 문항) | 골드 엣지 문항 | L3 micro |
|---|---|---|---|---|---|---|---|
| s6a (multihop+relation 54) | `5e4f603` | 3 | 53~54 | 0.236 (0.235~0.239) | **0.140** (0.139~0.142) | 47~48 | 0.161 (0.160~0.163) |
| s5 (233) | `ba714eb` | 1 | 232 | 0.592 | **0.266** | 129 | 0.243 |
| s3 규칙 on (233) | `1e42d78` | 3 | 231~232 | 0.594 (0.592~0.597) | **0.271** (0.270~0.272) | 128~129 | 0.248 |
| s3 규칙 on, rule 42문항만 | `1e42d78` | 3 | 41~42 | 0.880 (0.878~0.881) | 0.286 | 7 | 0.286 |
| s3-roff 규칙 off (rule 42) | `1e42d78` | 3 | 42 | 0.881 | 0.286 | 7 | 0.286 |

- 기존 recall은 골드 엣지 없는 문항(233문항 중 103, rule 42문항 중 35)이 1.0으로 들어가 부풀려져 있었다.
  골드 엣지가 있는 문항만 보면 233문항 기준 **0.27**, 54문항 기준 **0.14**다.
- rule 42문항은 골드 엣지가 7문항에만 있어 L3로는 규칙 on/off를 가를 수 없다(on·off 모두 0.286).

| 구성 | L4 위반(기존) | **L4 규칙 위반율** (문항 평균) | 위반 추론 / 검사 추론 | 규칙 검사 문항 | L4 타입 일관성(기존) | **L4 타입 일관성** (문항 평균) | 타입 검사 문항 |
|---|---|---|---|---|---|---|---|
| s6a | 0.000 | 0.093 (0.093~0.095) | 43 / 300~303 (0.142~0.143) | 42~43 | 1.000 | 0.969 (0.969~0.970) | 49~50 |
| s5 | 0.000 | 0.113 | 160 / 1134 (0.141) | 133 | 1.000 | 0.955 | 192 |
| s3 (233) | 0.000 | 0.113 | 160 / 1,127~1,132 (0.141~0.142) | 132~133 | 1.000 | 0.956 (0.955~0.956) | 191~192 |
| s3, rule 42문항 | 0.000 | 0.118 (0.116~0.122) | 16 / 114~118 | 23~24 | 1.000 | 0.927 | 35~36 |
| s3-roff | 0.000 | — (추론 0건) | — | 0 | 1.000 | 0.927 | 36 |

- **규칙 위반은 전부 한 종류다: `related_entity_invalid`.** `premium_defense_success`·`price_quality_mismatch`·
  `value_position` 추론의 `related_entities`에 빈 문자열이 들어간다(s5 기준 위반 160건 중 157건 = 77+77+3).
  나머지 3건은 related_entities에 가짜 브랜드 `fresh`가 든 경우다. 가격 규칙이 제품 ID 자리를 빈 값으로 채우는
  결함으로 보이며 원인은 조사하지 않았다(규칙 코드는 `src/`).
  나머지 제약(값 범위·날짜·소유 관계·주어 타입)은 이 기준선에서 위반 0건이다.
- **타입 위반은 전부 가짜 브랜드다.** `category_brands`의 top_brands에 `unknown`이 브랜드로 실린 경우가 대부분이고
  (s5 141건), `competitors`·`competesWith`에 가짜 브랜드가 끼는 경우가 6건이다. 타입을 정할 수 없어 건너뛴 끝점이
  검사 건수의 약 30%(s5: 검사 5,133 / 미검사 1,579)라, 등록부가 넓어지면(O1) 이 값은 바뀐다.
- s3-roff와 s3(rule 42문항)의 타입 일관성이 같다(0.927). 규칙 on/off는 KG 조회 경로를 바꾸지 않기 때문이다.

**술어별 recall** (골드 표기 기준, 별칭 정규화 없음. [ ] = 골드 엣지 수)

| 구성 | ownedByGroup | ownedBy | hasSegment | originatesFrom | siblingBrand | acquiredIn | belongsToCategory | hasProduct | rankedIn | competesWith | hasSoS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| s6a | 0.000 [27] | 0.333 [6] | 0.000 [14] | 0.000 [2] | 0.000 [2] | 0.000 [1] | 0.333 [12] | 0.375 [8] | 0.350 (0.333~0.385) [13~15] | 0.000 [10] | 0.667 [3] |
| s5 | 0.000 [29] | 0.455 [11] | 0.000 [14] | 0.000 [2] | 0.000 [2] | 0.000 [1] | 0.375 [16] | 0.413 [46] | 0.175 [40] | 0.032 [31] | 0.765 [17] |
| s3 (233) | 0.000 [29] | 0.455 [11] | 0.000 [14] | 0.000 [2] | 0.000 [2] | 0.000 [1] | 0.375 [16] | 0.426 [47] | 0.184 [38] | 0.031 [32] | 0.781 (0.765~0.812) [16~17] |

- 검토 보고서의 `ownedByGroup` 0/29, `hasSegment` 0/14를 재확인했다. `originatesFrom`·`siblingBrand`·`acquiredIn`도 0이다.
- 골드 엣지 수가 run마다 조금 다른 것은 인프라 실패 문항이 run마다 달라서다.

**trace에 없던 필드.** 재채점에 필요한 필드는 모두 있었다. 예외: `s3-roff-run{1,2,3}`에는 `trace.rule_evaluation`이 없다
(규칙 off 구성이라 규칙 판정을 하지 않음). 추론도 0건이라 규칙 위반율은 "판정 없음"으로 처리했다. 대체 계산은 하지 않았다.

### O0-B. 무비용 온톨로지 점검 기준값

`scripts/ontology_offline_checks.py`, 입력 KG = `eval_output/evidence-2026-09/eval_data_snapshot_tagged/knowledge_graph.json`
(sha1 `770ed1d0…`, 실행 전후 동일, 트리플 3,500, entity_metadata 0), 골든 = typed `combined_v1.jsonl` 233문항.
같은 입력으로 두 번 실행한 JSON이 바이트 단위로 같다(결정적). O1 로더(`src/ontology/ontology.py`)는 아직 없어
등록부 기준 인식 범위 절은 건너뛰었다.

**1. 브랜드 인식 범위** (`EntityLinker(use_spacy=False)`, 런타임과 같은 사전 경로)

| 대상 | 인식 | 전체 | 비율 |
|---|---|---|---|
| KG 브랜드 노드(가짜 3개 제외, 표기 하나라도 인식되면 인식) | 19 | 106 | **17.9%** |
| 골든 문항의 골드 브랜드 중 질문 원문에 영문 표기로 나온 것(런타임처럼 KG 제품명 역링크 포함) | 147 | 160 | **91.9%** |

- KG 브랜드 노드는 술어 위치(hasProduct 주어, ownedByGroup 주어, competesWith 양끝 등)로 골랐다. KG에 타입 트리플이 0건이라서다.
  검토 보고서의 "약 83개 중 약 65개 미인식"과 모집단 정의가 달라(시드 온톨로지의 아모레퍼시픽 브랜드 31개 포함) 수가 다르다.
- 미인식 예: almay, it_cosmetics, jouer, charlotte_tilbury, covergirl, aquaphor, nivea, iope, primera, mise_en_scène, make_on 등 87개.
- 질문에 나왔는데 연결 못 한 골드 브랜드: almay×3, aestura×2, covergirl×2, aquaphor·chapstick·charlotte_tilbury·it_cosmetics·jouer·tata_harper×1.
- 골드 브랜드 132건은 질문에 영문 표기가 없어(기본 대상 LANEIGE 48, 한글 별칭 등) 이 비율에서 뺐다 — 대부분은 골드가 질문에 없는 브랜드를
  암묵적으로 넣은 것이라 연결기 문제가 아니다. 다만 한글 별칭(라네즈 등)은 "질문에 나옴"으로 세지 않았으므로 이 132건에는
  한글로 언급된 브랜드도 섞여 있다(한글 별칭 인식률은 이 표에 없다).

**2. 골드 엣지 도달률** (골드 233문항의 엣지 212개)

검색기가 이름으로 방출하는 술어: `priority_preds` = {competesWith, hasHHI, hasPosition, hasSoS, ownedBy, rankedIn}
(`src/rag/hybrid_retriever.py`, ast로 읽음) + 제품 슬러그 엣지(hasProduct·belongsToCategory) + competitors fact의 competesWith
+ 카테고리 계층의 hasSubcategory. 검색기는 KG의 `ownedByGroup`을 `ownedBy`로 바꿔 방출한다. 증거 카드에서는
`KG_NUMERIC_PREDICATES` = {hasHHI, hasPosition, hasRank, hasSoS}가 날짜 없는 수치로 빠진다(`src/rag/evidence_adapters.py`).

| 술어 | 골드 | KG에 같은 이름으로 있음 | 별칭 포함 KG에 있음 | 런타임이 그 이름으로 방출 | **지금 도달 가능** | 이름만 정규화하면 |
|---|---|---|---|---|---|---|
| ownedByGroup | 29 | 29 | 29 | 0 | **0** | 29 |
| ownedBy | 11 | 0 | 11 | 11 | 11 | 11 |
| hasSegment | 14 | 14 | 14 | 0 | **0** | 0 |
| originatesFrom | 2 | 2 | 2 | 0 | **0** | 0 |
| siblingBrand | 2 | 2 | 2 | 0 | **0** | 0 |
| acquiredIn | 1 | 1 | 1 | 0 | **0** | 0 |
| hasSoS | 17 | 17 | 17 | 17 | 17 | 17 |
| hasProduct | 47 | 38 | 38 | 47 | 38 | 38 |
| rankedIn | 40 | 22 | 22 | 40 | 22 | 22 |
| belongsToCategory | 16 | 8 | 8 | 16 | 8 | 8 |
| competesWith | 32 | 7 | 7 | 32 | 7 | 7 |
| hasHHI | 1 | 0 | 0 | 1 | 0 | 0 |

- "도달 가능"은 상한이다(KG에 사실이 있고 런타임이 같은 이름을 낼 수 있음). 실제 recall(위 재채점)은 방출 상한 12개·질의
  엔티티 인식에 더 깎인다. 예: hasSoS 상한 17/17, 실측 0.77.
- `ownedByGroup` 29건은 KG에 모두 있는데 이름 변환 때문에 0이다. `hasSegment`·`originatesFrom`·`siblingBrand`·`acquiredIn`은
  KG에 있는데 `priority_preds`에 없어 버려진다(검토 보고서 §3.1 확인).
- `competesWith` 골드 32건 중 KG에 있는 쌍은 7건뿐이다. 골드 경쟁 관계는 문서 기반이고 KG의 competesWith는 "같은 Top 100에 함께 나옴"이다.

**3. KG 정합성**

| 항목 | 값 |
|---|---|
| 대소문자·표기 중복 노드 키 | 16 (예: LANEIGE/laneige, COSRX/cosrx, HERA/Hera/hera, La Roche-Posay/la roche-posay) |
| 가짜 브랜드 관련 트리플 | 420 (unknown 272, fresh 92, chi 56) |
| competesWith 비대칭 쌍 | 371 / 625 |
| siblingBrand 비대칭 쌍 | 0 / 874 |
| 수치 속성이 있는 엣지 중 `valid_from` 없음 | 1,884 / 1,884 (belongsToCategory 710·hasProduct 632의 rank 포함, hasSoS 169·rankedIn 169·hasPosition 124·hasHHI 66·competesWith 14) |
| `hasPosition`의 원 술어 | hasSoS 169 · hasHHI 66 · hasPosition 124 (세 의미가 한 술어로 합쳐짐) |
| 도메인·범위 위반 (config 등록부 + ASIN 기준) | 420 — 전부 가짜 브랜드가 브랜드 자리에 온 것 (hasProduct 주어 227, competesWith 156, hasSoS·rankedIn 주어 각 15, hasPosition 7). 타입 모름 끝점 1,262 |

- 검토 보고서의 "수치 엣지 1,864건"과 20건 차이가 나는 것은 정의 차이다. 여기서는 속성에 숫자가 하나라도 있으면(순위 포함) 센다.

**4. 규칙 일치율 오프라인** (`--rule-match`, `notes/3b_rule_offline_match.md`와 같은 방법, DB·KG는 임시 사본, LLM 호출 0)

| 경로 | 일치 |
|---|---|
| A: 실제 `HybridRetriever.retrieve` (연결기 엔티티) | **25/32** |
| B: 골드 `rule_context` 엔티티 | **29/32** |

지난 기록(연결기 25/32 vs 골드 29/32)을 태그 스냅샷에서 그대로 재현했다. 불일치 문항도 같다: rg001~003(골드가 조건 단위라
규칙 발화로 표현 불가, A·B 모두 불일치), rg021·rg022·rg023·rg026(IT Cosmetics·Jouer·Almay·Charlotte Tilbury 미인식, B에서는 일치).

### O0 판단 (과장 금지)

- 지금 수치로 "온톨로지 효과"를 말할 수 있는 지표는 **골드 엣지 문항 L3 recall(0.14 / 0.27)과 술어별 recall**이다.
  L4 두 새 지표는 온톨로지 추론이 아니라 **검색·규칙 출력의 데이터 결함**(빈 related entity, 가짜 브랜드)을 잰다.
- 가장 큰 무비용 개선 여지는 이름 정규화(ownedByGroup 29건)와 버려지는 술어 노출(hasSegment 14 등)이다. 이는 O2·O3의 범위다.
- 규칙 on/off(rule 42문항)는 L3·타입 일관성으로 구분되지 않는다. 규칙 효과는 계속 규칙 일치율(32문항)로 본다.
- 판단 한 줄 기록: 레거시 L4 필드와 게이트를 바꾸지 않은 이유, 등록부에서 카테고리 ID를 세그먼트보다 우선한 이유
  ('makeup'), 가짜 브랜드를 등록부보다 우선한 이유('fresh')는 커밋 메시지와 `ontology_validator.py` 주석에 있다.

### 재현

```bash
# 워크트리 루트에서 (ROOT = 메인 체크아웃)
.venv/bin/python scripts/rescore_l3_l4.py --base $ROOT/eval_output/evidence-2026-09 \
  --out $ROOT/eval_output/ontology-2026-09/rescore \
  --subset s3-rule42:s3=eval/data/golden/typed/rule.jsonl \
  s6a-run1 s6a-run2 s6a-run3 s5-run1 s3-run1 s3-run2 s3-run3 s3-roff-run1 s3-roff-run2 s3-roff-run3
.venv/bin/python scripts/ontology_offline_checks.py \
  --kg $ROOT/eval_output/evidence-2026-09/eval_data_snapshot_tagged/knowledge_graph.json \
  --snapshot $ROOT/eval_output/evidence-2026-09/eval_data_snapshot_tagged --rule-match --registry \
  --out $ROOT/eval_output/ontology-2026-09/offline_checks/offline_checks.json
```
