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
# §O5 — KG 쓰기 정합성 (트랙 O5, 결정 OA-7) [2026-09 사후]

> `docs/experiments/ontology_activation_2026-09.md`가 이 트랙의 기준 커밋(`0b56e4b`)에 없어서 별도
> 파일로 둔다. 리드가 병합할 때 본 문서의 "§O5" 절로 옮긴다.

## 1. 한 일

| 항목 | 내용 |
|---|---|
| 쓰기 검증 | `src/ontology/kg_write_validation.py`(규칙, 순수 함수) + `KnowledgeGraph.add_relation`(적용). 플래그 `kg.write_validation` = `off`/`warn`/`enforce`, 기본 **`warn`**, ENV `FF_KG_WRITE_VALIDATION`. `FeatureFlags.kg_write_validation_mode()`가 생기면 그 값을 우선 쓴다 |
| warn | 위반을 사유별로 세고, 사유마다 첫 예시 3개만 WARNING 로그. enricher는 저장 후 요약 1줄(INFO). **저장 내용은 off와 같다**(특성 테스트: enricher·updater·카테고리 계층 로드 결과를 off/warn으로 저장해 비교 — 벽시계 값 `created_at`·`saved_at`만 빼고 동일) |
| enforce | 술어 정식화(`hasPosition`→`hasSoS`/`hasHHI`/`hasPricePosition`, `belongsToCategory`(rankedIn)→`rankedIn`, `ownedBy`→`ownedByGroup`, 옛 이름은 `original_predicate`), 브랜드 정식 표기, 가짜 브랜드 차단, 수치 `as_of` 필수, 도메인·범위·리터럴 위반 차단, `entity_metadata`에 `type`·`ontology_types` 기록 |
| KG 로드 | `_load()` 중에는 검증하지 않는다 — 파일 내용을 그대로 읽는다 |
| enricher | `enrich_and_store(crawl_data, as_of=None)`. enforce일 때만 수치 트리플에 크롤 날짜를 붙인다(`as_of` 인자 → `crawl_data["as_of"/"snapshot_date"]` → 제품들의 `snapshot_date`가 하나일 때). warn/off에서는 붙이지 않는다 |
| 도메인 | `RelationType`에 `HAS_SOS`·`HAS_HHI`·`HAS_PRICE_POSITION` 추가(분리 술어가 저장된 KG를 `Relation.from_dict`가 읽을 수 있게). 추가만 했으므로 기존 파일 읽기는 그대로 |
| 마이그레이션 | `scripts/migrate_kg_ontology.py` — 기본 dry-run, `--apply`는 `--out`·변경 기록만 쓴다. `--out`이 `--in`과 같거나 저장소(워크트리 포함) `data/`·`/data` 아래면 거부. 결정적(같은 입력 → 같은 바이트, 재실행으로 확인) |
| 평가 스냅샷 | `eval_output/evidence-2026-09/eval_data_snapshot_ontology` = 태그 스냅샷 사본 + 마이그레이션한 KG |

### 정식 브랜드 KG 문자열: 등록부 표시 이름의 소문자

- 예: `LANEIGE`→`laneige`, `elf`→`e.l.f.`, `Beauty of Joseon`→`beauty of joseon`.
- 근거: 질의 경로 `hybrid_retriever._query_knowledge_graph`는 엔티티 링커가 낸 이름(`LANEIGE`·`e.l.f.`·`La Roche-Posay`처럼 등록부식 표기)의 **원형·lower·upper·title** 4변형을 조회한다. 소문자 표시 이름은 항상 `lower` 변형과 같다.
- 등록부 id(`elf`·`beauty_of_joseon`)는 링커 출력의 어느 변형과도 같지 않아 쓰지 않았다.
- enricher가 이미 소문자로 쓰고 있어서, 기존 트리플 대부분은 그대로 남는다.
- 브랜드를 바꾸는 자리는 술어 정의상 Brand 자리(도메인·범위)뿐이다. 등록부 밖 브랜드는 그대로 둔다.
- 그룹 `AMOREPACIFIC`과 카테고리 id는 바꾸지 않는다.

### 판단한 것

1. **카테고리 계층 별칭은 저장 이름을 유지한다.** `parentCategory`·`hasSubcategory`는 정식 이름(`subCategoryOf`·`hasSubCategory`)으로 바꾸지 않는다. KG 계층 API(`get_category_hierarchy`)가 enum 값으로 조회하기 때문이다. 정식화는 읽기(O3)에서 한다. 위반으로도 세지 않는다.
2. **온톨로지 밖 술어는 막지 않는다.** `hasAlert`·`hasSentiment`·`hasAISummary`는 enforce에서도 그대로 저장한다. 막으면 감성·알림 데이터가 사라진다. `outside_ontology`로 따로 센다.
3. **나눌 수 없는 `hasPosition`은 enforce에서 막는다.** `DOMINATES_CATEGORY`가 해당한다. 같은 정보(`share`)가 `hasSoS`(`sos_pct`)에 있다. 옛 KG에서는 같은 (s,p,o)로 합쳐져서 따로 남은 트리플이 0건이다.
4. **`competesWith`는 대칭으로 만든다.** 스키마가 대칭으로 선언했으므로, 빠진 역방향 추가는 새 사실을 지어내는 것이 아니다. 추가한 트리플에는 `inferred_by: "symmetric:competesWith"`를 표시했다. 공동 출현 기반이라 품질이 낮다는 문제(검토 보고서 §3.2)는 이 트랙에서 고치지 않았다.
5. **`created_at`으로 `as_of`를 채우지 않는다.**
   - `add_relation`은 같은 (s,p,o)가 다시 오면 속성만 덮어쓰고 `created_at`은 처음 값을 남긴다.
   - 그래서 SoS 값은 최신 크롤 값인데 `created_at`은 2026-08-30으로 남아 있다.
   - 속성의 날짜 키(`as_of`·`snapshot_date`·`collected_at`)나 `valid_from`만 쓴다. 이 스냅샷에는 둘 다 없어서 **모두 날짜 없음으로 남겼다**.
6. **가짜 브랜드가 원인인 도메인·범위 위반은 한 번만 센다.** `PlaceholderBrand`는 `Brand`와 서로소라서, 같은 트리플이 도메인/범위 위반으로도 한 번 더 잡힌다. 이런 경우는 `placeholder_brand`로만 센다.

## 2. 평가 스냅샷 KG 마이그레이션 결과

- 입력: `eval_data_snapshot_tagged/knowledge_graph.json`의 사본. sha256 `876ea40fb4f9c395e34476f7309c7340b9ebfe436f851aaebc0f54b8ebe2fcaa`, 3,500 트리플, `entity_metadata` 0.
- 출력: `eval_data_snapshot_ontology/knowledge_graph.json`. sha256 `1600836e081bb075d515f82cff9f8ed767bfe71af8fcc3ec6b08e11c8d5c5a6e`, 3,338 트리플, `entity_metadata` 732.
- 변경 기록: `eval_data_snapshot_ontology/knowledge_graph.json.changes.json` (2,948건).
- 태그 스냅샷 KG의 해시는 작업 후에도 `876ea40f…`로 **바뀌지 않았다**. 스냅샷 디렉터리의 나머지 파일은 `diff -rq`로 동일함을 확인했다.
- 출력 KG는 `KnowledgeGraph(persist_path=<사본>, auto_save=False)`로 3,338 트리플을 모두 읽는다. 읽은 뒤에도 파일은 바뀌지 않았다.

### 사유별 변경 건수

| 구분 | 사유 | 건수 |
|---|---|---:|
| 삭제 | `placeholder_brand` (`unknown`·`fresh`·`chi`·`Fresh`) | 415 |
| 삭제 | `merged_duplicate` (정규화 후 같은 (s,p,o)) | 29 |
| 삭제 | `range_violation` (`hasSegment` 목적어 `Body`×3·`Makeup`이 카테고리 이름과 겹침) | 4 |
| 삭제 | `literal_violation` (`TATA HARPER acquiredIn "True"`) | 1 |
| 수정 | `brand_canonicalized` | 1,708 |
| 수정 | `predicate_canonicalized` | 484 |
| 추가 | `symmetric_closure` (역방향 `competesWith`) | 287 |
| 메타데이터 | `type_record_added` | 732 |
| 정보 | `undated_numeric` (정규화 단계, 병합 전) | 337 |

- 삭제한 `hasSegment` 4건은 등록부에 정식 값이 있다: espoir→`makeup_line`, Illiyoon·Happy Bath·SKIN U→`body`. O3의 정적 사실 카드가 등록부에서 이 값을 싣는다.
- `acquiredIn "True"`는 원인이 따로 있다. `kg_updater.load_brand_ownership`가 `acquired: true`(bool)를 `isinstance(acquired, int)`로 통과시켜 `"True"`를 쓴다. 등록부에는 TATA HARPER의 인수 연도가 없다. 이 버그는 off/warn 저장 내용을 바꾸므로 고치지 않고 FUTURE_WORK 후보로 남긴다. enforce 모드는 이 트리플을 막는다.

### 정합성 위반 수 (LLM 호출 없음)

| 항목 | 이전 | 이후 |
|---|---:|---:|
| `non_canonical_brand` | 1,029 | 0 |
| `non_canonical_predicate` | 528 | 0 |
| `placeholder_brand` | 415 | 0 |
| `range_violation` (가짜 브랜드 원인 제외) | 4 | 0 |
| `domain_violation` (가짜 브랜드 원인 제외) | 0 | 0 |
| `literal_violation` | 1 | 0 |
| `unresolvable_legacy_predicate` | 0 | 0 |
| `missing_as_of` (날짜 없는 수치 엣지) | 359 | **333** |
| 대소문자 중복 브랜드 | 16 | 0 |
| 대칭 술어인데 역방향 없음 | 388 | 0 |
| 타입 기록 없는 개체 | 756 | 0 |
| **합계** | **3,496** | **333** |

**남은 333건 = 날짜 없는 수치 엣지**: `hasSoS` 151, `hasPricePosition` 116, `hasHHI` 66.

- 사유: 원본 KG에 관측 날짜가 없다(§1 판단 5). 날짜를 지어내지 않기로 해서 남겼다.
- 해소하려면 enforce 모드에서 크롤 날짜와 함께 다시 쓰거나, 그날의 DB 스냅샷에서 다시 계산해야 한다.
- `hasHHI`는 값이 목적어라 카테고리마다 여러 값이 쌓여 있다(66건). 날짜가 없어서 어느 값이 최신인지도 가를 수 없다.
- 증거 계층은 날짜 없는 수치 엣지를 이미 증거에서 뺀다(`evidence_adapters.KG_NUMERIC_PREDICATES`). 단, 이 집합에 **`hasPricePosition`은 없다** — 아래 §4 참고.

## 3. 운영 KG 적용 명령 (소유자 승인 후, 이 트랙은 실행하지 않음)

전제 조건 두 가지가 먼저 필요하다.

- 이 트랙 커밋(`RelationType` 추가 포함)이 운영 코드에 병합·배포돼 있어야 한다. 없으면 `hasSoS` 등을 읽지 못해 KG 로드가 실패한다.
- 22:00 KST 크롤 전후는 피한다.

```bash
cd "/Users/leedongwon/Desktop/AMORE-RAG-ONTOLOGY-HYBRID AGENT"
TS=$(date +%Y%m%d-%H%M%S)
# 1) 백업 (kg_backup 롤링과 별도로 보관)
mkdir -p data/backups/kg
cp -p data/knowledge_graph.json "data/backups/kg/knowledge_graph.pre-ontology-$TS.json"
# 2) 사본으로 마이그레이션 (스크립트는 data/ 아래로 쓰기를 거부한다)
cp -p data/knowledge_graph.json "/tmp/kg_in_$TS.json"
.venv/bin/python scripts/migrate_kg_ontology.py --in "/tmp/kg_in_$TS.json" --out "/tmp/kg_ontology_$TS.json"          # dry-run 검토
.venv/bin/python scripts/migrate_kg_ontology.py --in "/tmp/kg_in_$TS.json" --out "/tmp/kg_ontology_$TS.json" --apply  # 변경 기록: /tmp/kg_ontology_$TS.json.changes.json
# 3) 그사이 크롤이 KG를 바꾸지 않았는지 확인한 뒤 교체
shasum -a 256 "/tmp/kg_in_$TS.json" data/knowledge_graph.json   # 두 해시가 같아야 한다
cp "/tmp/kg_ontology_$TS.json" data/knowledge_graph.json
# 되돌리기: cp -p "data/backups/kg/knowledge_graph.pre-ontology-$TS.json" data/knowledge_graph.json
```

Railway(`/data/knowledge_graph.json`)도 같은 순서다. `/data` 아래 쓰기는 스크립트가 거부하므로 사본을 만든 뒤 `cp`로 교체한다.

예상 변경 건수는 태그 스냅샷 기준 위 표와 비슷하다(삭제 449, 수정 2,192, 추가 287, 메타데이터 732). 실제 건수는 dry-run 출력으로 확인한다.

## 4. 남은 일·다른 트랙에 알릴 것

1. **운영에 적용한다면 `enforce`를 같이 켜야 한다.** 기본 `warn`인 매일 크롤은 예전 형태를 계속 쓴다: `hasPosition`, updater의 대문자 브랜드, 가짜 브랜드. 그러면 KG가 다시 섞인다.
2. **`enforce`를 켜기 전에 exporter가 크롤 날짜를 넘겨야 한다.** `dashboard_exporter.py`(약 1542행)의 `enrich_and_store(...)`에 `as_of=latest_date`를 추가한다. `scripts/enrich_kg_from_crawl.py`도 같다. 넘기지 않으면 enforce에서 SoS·HHI·가격 포지션 엣지가 전부 `blocked:missing_as_of`로 막힌다. 이 파일들은 O5 소유가 아니라 고치지 않았다.
3. **O3·리드 확인이 필요하다.**
   - `hybrid_retriever`의 `priority_preds`에 `hasPricePosition`이 없다. 온톨로지 스냅샷에서는 가격 포지션 엣지가 노출되지 않는다(예전에는 `hasPosition`으로 노출됐다).
   - `evidence_adapters.KG_NUMERIC_PREDICATES`에 `hasPricePosition`을 넣어야, 날짜 없는 가격 포지션이 증거 카드가 되지 않는다.
4. **타입 기록이 검색 결과를 바꿀 수 있다.** `entity_metadata`에 브랜드 항목이 생기면, `_query_knowledge_graph`가 `brand_info` 사실을 싣기 시작할 수 있다. `get_entity_metadata`는 정확히 일치하는 키로만 찾으므로 소문자로 질의할 때만 해당한다. O7에서 온톨로지 스냅샷 효과를 읽을 때 고려한다.
5. `kg_updater`의 `acquired: true` → `"True"` 버그(§2).
