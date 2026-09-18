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
## §O6 — OWL 모듈 정리·플래그 이름·문서 (2026-09-18, API $0)

기준 커밋 `e0ffac7`. 결정 OA-1(JSON 원본)에 따라 OE8의 JSON 쪽(OWL 모듈 삭제)을 따른다.

### O6-1. 호출처 확인표 (삭제 전)

조사: `grep -rn` (src·scripts·tests·eval·examples·config·docs 코드 블록, `* 2.*` 제외) + `git log -S "OntologyKnowledgeGraph(" -- src`(도입 커밋 `7ca17ce` 1건뿐 — 서비스 코드가 생성한 적 없음).

| 삭제 대상 심볼 | 호출처 | 종류 | 처리 |
|---|---|---|---|
| `OWLReasoner`·`ConsistencyReport`·`get_owl_reasoner`·`OWLREADY2_AVAILABLE` (`src/ontology/owl_reasoner.py`) | `src/ontology/ontology_knowledge_graph.py` | 같이 삭제되는 모듈 | 삭제 |
| 〃 | `tests/unit/ontology/test_owl_reasoner.py`, `test_owl_consistency.py` | OWL 전용 테스트 | 삭제 |
| 〃 | `tests/integration/test_sprint9_integration.py` `TestOWLConsistencyIntegration`(3개) | OWL 전용 테스트 클래스 | 그 클래스만 삭제 |
| 〃 | `tests/unit/rag/test_owl_strategy_removed.py::test_owl_reasoner_module_is_kept_as_vocabulary` | "어휘 용도로 남는다" 고정 테스트 | "모듈이 삭제됐다"로 반전 |
| `owl_reasoner` 인자·속성 (`src/rag/entity_linker.py` `EntityLinker.__init__`, `get_entity_linker`) | 저장만 하고 읽지 않음. 생성처(src·tests·examples) 모두 인자 없이 호출 | 죽은 인자 | 제거 |
| `OntologyKnowledgeGraph`·`OWL_CLASS_MAPPING` (`src/ontology/ontology_knowledge_graph.py`) | `src/ontology/__init__.py` export만 | export | export 제거 |
| 〃 | `tests/unit/ontology/test_ontology_knowledge_graph.py`, `test_ontology_kg.py` | OKG 전용 테스트 | 삭제 |
| 〃 | `scripts/migrate_kg_to_ontology.py` | 개발 스크립트(OKG 전용) | 삭제 |
| `cosmetics_ontology.owl` | 로드하는 코드 없음(`owl_reasoner.py`도 이 파일을 쓰지 않음, 81행 `<` 때문에 파싱 불가) | 정적 파일 | 삭제 |
| 문서 코드 예시 | `src/rag/README_ENTITY_LINKER.md`(OWLReasoner 통합 예시), `examples/entity_linker_integration.py` docstring | 문서 | 정정 |

**서비스 호출처: 0건.** `src/core/brain.py`·`container.py`에도 없다. `tests/unit/ontology/test_iri_migration.py`는 `KnowledgeGraph`의 IRI 기능 테스트라 OWL 전용이 아니므로 **남긴다.**

owlready2 import 위치(삭제 후): `scripts/export_ontology_owl.py`, `scripts/check_ontology_owl.py`(개발 전용, 유지)와 그 테스트(`importorskip`). 런타임(`src/`) import 0건 → `requirements.txt`에서 `requirements-dev.txt`로 옮긴다. Dockerfile은 `requirements.txt`만 설치한다.

이름만 OWL인 플래그(모듈과 무관, 삭제하지 않고 이름만 바로잡음 — O6-3):

| 호출처 | 옛 이름 | 실제 뜻 |
|---|---|---|
| `hybrid_retriever.py` 규칙 판정 분기, `tool_registry.py` `apply_rules` 가용성 | `reasoner.use_owl_reasoner` (+ `use_unified_reasoner`, OR) | 규칙 추론 on/off |
| `hybrid_retriever.py` KG 조회, `tool_registry.py` `kg_neighbors` | `ontology.use_ontology_kg` | 일반 KG 조회 on/off |
| `eval/ablation.py`, 실험 스크립트 | `FF_REASONER_USE_OWL_REASONER`, `FF_ONTOLOGY_USE_ONTOLOGY_KG` | 위와 같음 — 별칭으로 계속 동작해야 함 |

### O6-2. 삭제한 코드 (`7e50cf1`)

| 파일 | 줄 |
|---|---|
| `src/ontology/owl_reasoner.py` | 1,247 |
| `src/ontology/ontology_knowledge_graph.py` | 398 |
| `src/ontology/cosmetics_ontology.owl` | 770 |
| `scripts/migrate_kg_to_ontology.py` | 229 |
| `tests/unit/ontology/test_owl_reasoner.py`·`test_owl_consistency.py`·`test_ontology_knowledge_graph.py`·`test_ontology_kg.py` | 648 + 228 + 771 + 47 |
| `tests/integration/test_sprint9_integration.py` `TestOWLConsistencyIntegration` | 약 60 |

합계: 소스·스크립트·OWL 2,644줄, 테스트 약 1,750줄(커밋 기준 −4,423 / +33). `entity_linker`의 죽은 `owl_reasoner` 인자 제거. owlready2는 `requirements-dev.txt`로 이동(`src/` import 0건, Dockerfile은 `requirements.txt`만 설치). `scripts/export_ontology_owl.py`·`check_ontology_owl.py`는 유지.

### O6-3. 플래그 이름 (`7f2a3f9`)

| 새 이름 (ENV) | 옛 이름 (ENV) | 기본 |
|---|---|---|
| `reasoner.enabled` (`FF_REASONER_ENABLED`) | `reasoner.use_owl_reasoner` (`FF_REASONER_USE_OWL_REASONER`) | True |
| `kg.enabled` (`FF_KG_ENABLED`) | `ontology.use_ontology_kg` (`FF_ONTOLOGY_USE_ONTOLOGY_KG`) | True |

- 해석 순서: ENV 새 > ENV 옛 > JSON 새 > JSON 옛 > 기본. ENV가 JSON보다 앞선다는 기존 규칙을 지켜, 평가의 규칙 off 명령(`FF_REASONER_USE_OWL_REASONER=false FF_REASONER_USE_UNIFIED_REASONER=false`)이 `config/feature_flags.json`의 새 키(`reasoner.enabled: true`)보다 우선한다.
- 옛 이름이 실제로 값을 정하면 프로세스당 1회 `WARNING` 로그("deprecated; use … instead").
- 규칙 판정 조건은 `use_unified_reasoner() or reasoner_enabled()`로 이전과 같은 뜻. `use_owl_reasoner()`·`use_ontology_kg()` 메서드는 새 메서드로 위임하는 deprecated 별칭으로 남겼다.
- `config/feature_flags.json`은 새 이름으로 바꿨다(`reasoner.enabled`, `kg.enabled`). 테스트: `tests/unit/infrastructure/test_feature_flags_aliases.py`(두 플래그 × 옛/새 × JSON/ENV 조합, 경고 1회, 규칙 off 명령, 저장소 설정 키).

### O6-4. 배포 쪽 온톨로지 상태 (OE10, `471f1ea`)

- `GET /api/v4/brain/status` 응답에 `ontology` 추가: `{version, as_of, class_count, brand_count, use_class_reasoning, kg_write_validation}`. `get_ontology()` 캐시를 쓰고, 로드 실패는 상태 조회를 깨뜨리지 않고 `error` 필드로 싣는다(Brain 초기화 실패 응답에도 포함).
- `scripts/start.py`가 uvicorn 전에 `get_ontology()`를 한 번 불러, 원본 형식 오류면 로그를 남기고 `exit 1`. Dockerfile·Railway 설정은 바꾸지 않았다(선택 과제인 Docker 다단계 Pellet 검증은 하지 않음, FUTURE_WORK 9.10).

### O6-5. 문서 정정 (`35ba26f`, `0e2079f`)

"OWL은 카테고리 계층 어휘로만 사용"(검토 보고서 §2.1-6)을 `[2026-09 사후]` 표시로 정정: `CLAUDE.md`(기술 스택·트리·모듈 표·엔드포인트), 루트 `AGENTS.md`, `src/ontology/AGENTS.md`, `README.md`, `src/rag/retrieval_strategy.py` 모듈 주석, `docs/portfolio/amore_architecture_evidence.md`(원문 보존, 정정 덧붙임), `src/rag/README_ENTITY_LINKER.md`·`examples/entity_linker_integration.py`(삭제 모듈 참조). FUTURE_WORK 9.10 신설.

### O6 게이트

`.venv/bin/python -m pytest tests/unit -q --no-cov` — 5,601 통과, 7 skip (알려진 다른 트랙 실패 `test_ontology_offline_checks.py::test_real_retriever_filters_are_found`는 제외하고 실행). `tests/integration/test_sprint9_integration.py` 15 통과.
# §O0-추가 (2026-09-18, O0 이어가기)

> `docs/experiments/ontology_activation_2026-09.md`의 §O0(O0-A·O0-B) 뒤에 이어 붙일 절 초안이다.
> 리드가 검토 후 그 문서로 옮긴다. 계획: [`docs/plans/ontology-activation-plan-2026-09-18.md`](../plans/ontology-activation-plan-2026-09-18.md).
> API 비용 $0 (LLM·임베딩 호출 0건). worktree 기준 `git log -1`을 `e0ffac7`로 맞추고 시작했다.

## 무엇을 왜 고쳤나

O3(`f4e995d` 플래그 `ontology.use_class_reasoning`)가 `hybrid_retriever.py`의 `priority_preds`를
모듈 상수 하나에서 `_PRIORITY_PREDS_CANONICAL if canonical else _PRIORITY_PREDS`(지역 변수, 조건식)로
바꾸면서 O0-B가 ast로 읽던 `retriever_predicate_filters`가 빈 집합을 돌려주게 됐다
(`tests/unit/scripts/test_ontology_offline_checks.py::test_real_retriever_filters_are_found` 실패).
또한 O3가 정적 정의 술어(그룹·세그먼트·원산지·인수·자매)를 `metric_edges`가 아니라 새 fact
타입 `ontology_static`(`src/rag/ontology_context.py:static_fact`)으로 내면서, 평가 L3 트레이스
추출(`eval/runner.py:_extract_l3_trace`)이 `metric_edges`만 보고 있어 그 카드들이 통째로
안 잡히고 있었다.

## 1. `scripts/ontology_offline_checks.py` 감지 수리 (플래그 OFF/ON 분리)

`retriever_predicate_filters`가 이제 `_PRIORITY_PREDS`·`_PRIORITY_PREDS_CANONICAL`(둘 다 모듈
상수라 ast로 잡힌다)과 `ontology_context.py`의 `STATIC_BRAND_PREDICATES`를 따로 읽어 `off`/`on`
키로 나눠 낸다. 최상위 `priority_preds`·`runtime_emitted_predicates`는 하위 호환을 위해 OFF 값을
그대로 유지한다. `run_checks`도 `gold_edge_reachability`(OFF)와 `gold_edge_reachability_on`(ON)을
각각 계산하고, `render_markdown`이 "2a. 플래그 OFF"·"2b. 플래그 ON" 두 절로 나눠 출력한다.

**부수로 잡은 버그**: `gold_edge_reachability` 안에 "검색기가 KG의 ownedByGroup을 ownedBy로 바꿔
방출한다"는 가정이 하드코딩돼 있었다. OFF에서는 맞는 가정이지만(OFF는 실제로 `ownedBy`만 낸다),
그대로 두면 ON(정식 이름 `ownedByGroup`을 그대로 낸다)에서 `ownedByGroup`이 emitted_names에서
통째로 사라져 "런타임 이름 방출"이 항상 0으로 나왔다. 호출자가 넘기는 `runtime_emitted_predicates`가
이미 모드별 실제 방출 이름이므로 이 하드코딩을 지우고 그대로 썼다 — 이제 모드별 값이 정확하다.

**재현**(태그 스냅샷 KG, 골든 233문항):

```bash
.venv/bin/python scripts/ontology_offline_checks.py \
  --kg $ROOT/eval_output/evidence-2026-09/eval_data_snapshot_tagged/knowledge_graph.json
```

| 술어 | 골드 | OFF 지금 도달 가능 | ON 지금 도달 가능 |
|---|---|---|---|
| ownedByGroup | 29 | 0 | **29** |
| hasSegment | 14 | 0 | **14** |
| originatesFrom | 2 | 0 | **2** |
| siblingBrand | 2 | 0 | **2** |
| acquiredIn | 1 | 0 | **1** |
| ownedBy(별칭 표기) | 11 | 11 | 0 (ON은 이 이름을 안 씀) |
| hasSoS·rankedIn·competesWith·hasProduct·belongsToCategory | — | 변화 없음 | 변화 없음 |

정적 정의 5개 술어(ownedByGroup·hasSegment·originatesFrom·siblingBrand·acquiredIn)는 KG에 사실이
있는 만큼(같은 이름 기준) ON에서 전부 "지금 도달 가능"으로 바뀐다 — `ontology_static` 카드가
KG 조회 경로 밖에서 이 술어들을 정식 이름 그대로 내기 때문이다(O3 결과, `docs/experiments/ontology_activation_2026-09.md` §O3 절 참고). `ownedBy`(레거시 별칭 표기)는 ON에서 아예 안 쓰이므로
0으로 바뀌는 게 맞다 — 골드가 `ownedBy`로 쓰였으면 오히려 ON에서 놓친다(별칭 문제, 아래 §3).

## 2. `eval/runner.py` L3 추출에 `ontology_static` 포함

`_extract_l3_trace`의 `elif fact_type == "metric_edges":` 분기를
`elif fact_type in ("metric_edges", "ontology_static"):`로 넓혔다. `ontology_static` 카드 중
`object`가 없는 것(`expansionTruncated` — 엣지가 아니라 "몇 개 잘랐다"는 집계 카드)은 건너뛴다.
플래그 OFF에서는 `ontology_static` fact 자체가 생기지 않으므로(O3, `hybrid_retriever.py`) 이
변경은 OFF 트레이스에 전혀 영향이 없다 — `tests/eval/test_runner_trace.py`의
`test_off_flag_trace_unaffected_no_ontology_static_fact`로 고정했다. `eval/brain_adapter.py`는
`ontology_facts`를 그대로 전달만 하므로 고칠 곳이 없었다(확인만 함).

## 3. 정식 술어(canonical) per-predicate recall 추가

`eval/metrics/l3_kg.py`에 `canonicalize_edge()`를 추가했다: 술어는
`src.ontology.ontology.get_ontology().canonical_predicate`(모르면 원래 이름), 끝점은
`normalize_brand` 우선 → 안 되면 `normalize_group` → 둘 다 안 되면 원문(소문자, 즉 항등)으로
맞춘다. **나라는 등록부에 정규화 함수가 없어 의도적으로 별칭 처리하지 않는다** —
`korea`와 `south_korea`는 canonical에서도 계속 다른 값이다. 온톨로지 로더를 못 불러오는 극단적
환경에서는 항등 함수로 대체해(`_load_ontology_normalizers`) canonical 값이 raw와 같아지도록
방어했다.

기존 raw 필드(`edge_recall_by_predicate` 등)는 그대로 두고, `L3Metrics`에
`*_canonical` 4필드(`kg_edge_recall_gold_only_canonical`·`gold_edge_count_canonical`·
`gold_edge_matched_canonical`·`edge_recall_by_predicate_canonical`)를 추가했다(additive).
`aggregate_l3_extended`와 `eval/report.py`의 `by_layer`(`l3_kg_edge_recall_gold_only_canonical`)에도
같은 원칙으로 더했다. `scripts/rescore_l3_l4.py`에 `KEY_PREDICATES_CANONICAL`(raw 목록에서
별칭으로 흡수되는 `ownedBy`를 뺀 것)과 canonical 표를 추가하고, 메인 표에 canonical
골드-엣지-문항 recall·micro 두 열을 더했다.

### 재채점 (재사용 기준선, 새 파일만 — 기존 `rescore/*.json`은 그대로 둠)

```bash
.venv/bin/python scripts/rescore_l3_l4.py \
  --base $ROOT/eval_output/evidence-2026-09 \
  --out $ROOT/eval_output/ontology-2026-09/rescore/canonical \
  --subset s3-rule42:s3=eval/data/golden/typed/rule.jsonl \
  s6a-run1 s6a-run2 s6a-run3 s5-run1 s3-run1 s3-run2 s3-run3 s3-roff-run1 s3-roff-run2 s3-roff-run3
```

**canonical per-predicate recall — mean (min~max) [골드 엣지 수]** (raw는 §O0-B 표 참고, 전부 태그
스냅샷·OFF 코드 경로 기준. rule on/off·s6a/s5 구성은 §O0-A와 동일)

| 구성 | ownedByGroup | hasSegment | originatesFrom | siblingBrand | acquiredIn | belongsToCategory |
|---|---|---|---|---|---|---|
| s6a (multihop+relation 54, 3회) | 0.424 [33] | 0.000 [14] | 0.000 [2] | 0.000 [2] | 0.000 [1] | 0.333 [12] |
| s5 (233, 1회) | 0.475 [40] | 0.000 [14] | 0.000 [2] | 0.000 [2] | 0.000 [1] | 0.375 [16] |
| s3 규칙 on (233, 3회) | 0.475 (0.475~0.475) [40] | 0.000 [14] | 0.000 [2] | 0.000 [2] | 0.000 [1] | 0.375 [16] |
| s3-roff / s3-rule42 (rule 42, 3회) | 1.000 [2] | — [0] | — [0] | — [0] | — [0] | — [0] |

- **`ownedByGroup`만 바뀐다.** raw 0.000 → canonical 0.424~0.475(s6a·s5·s3), 골드 엣지 수도
  29(raw, `ownedByGroup` 표기만) → 40(canonical, 골드의 `ownedBy`(11) + `ownedByGroup`(29)을 합침)로
  는다. 이 시점 KG·검색기는 아직 **레거시(플래그 OFF)** 경로만 담겨 있다 — OFF는 `ownedBy`로
  방출하므로, 골드가 `ownedByGroup`으로 쓴 문항 중 방출이 `ownedBy`로 표기가 달라 raw가
  놓치던 것을 canonical이 잡는다. rule 42문항(s3-roff)은 골드 2건이 전부 canonical에서
  일치(1.000)로 바뀐다.
- **`hasSegment`·`originatesFrom`·`siblingBrand`는 안 바뀐다(0.000 그대로).** 이 술어들은
  OFF `priority_preds`에 아예 없어(§O0-B 표) 이름이 뭐든 KG 조회 경로에서 통째로 버려진다 —
  별칭·표기 정규화로 고칠 수 있는 문제가 아니라 **누락**이라 canonical도 0이다. (ON에서는
  §1의 도달률처럼 완전히 살아난다 — 그 효과는 실제 답변 recall로는 O7 플래그 on/off 측정에서
  잰다. 이 rescore는 아직 저장된 리포트의 trace만 다시 채점한 것이라 코드 경로가 안 바뀐다.)
- `belongsToCategory`는 등록부에 정규화 대상(브랜드·그룹)이 아니라 canonical == raw다(0.375·0.333).
- `s3`는 3회 모두 0.475로 완전히 같다 — 이 술어는 실행 간 흔들림이 없었다.

### 가져야 할 결론

- **O0-B 무비용 도달률(§1)은 이름 정규화(canonical_predicate)만으로 `ownedByGroup`이 이미
  29/29까지 도달 가능함을 보였는데(플래그 OFF에서도!), 재사용 기준선 L3 recall 재채점(§3)은
  실제로 그 절반 정도(0.42~0.48)만 잡는다.** 나머지는 정규화로 못 채우는 구조적 한계다 —
  질의 브랜드 인식(O2 범위)·상한 12개 잘림·평가 골드가 `ownedBy`/`ownedByGroup` 어느 쪽으로도
  안 쓰인 문항 등. §1의 "도달 가능(상한)"과 §3의 "실측 canonical recall" 차이를 O2·O3·O7에서
  구분해 봐야 한다.
- `hasSegment`·`originatesFrom`·`siblingBrand`·`acquiredIn`은 canonical로 못 고친다 — O3의
  플래그 ON(정적 사실 카드)이 유일한 해법이고, §1에서 확인했듯 이미 도달률은 100%로 올라간다.

## Owned 파일 변경 요약

- `scripts/ontology_offline_checks.py`: OFF/ON 분리 감지 + `ownedByGroup` rename 버그 수정.
- `eval/runner.py`: `_extract_l3_trace`가 `ontology_static` 카드도 담음.
- `eval/schemas.py` / `eval/metrics/l3_kg.py` / `eval/report.py`: canonical per-predicate 필드
  additive 추가.
- `scripts/rescore_l3_l4.py`: canonical 집계·표 추가, `eval_output/ontology-2026-09/rescore/canonical/`에
  새로 재채점.
- `scripts/typed_eval_summary.py`: `by_layer`에는 있었지만 `TRACKED`에 빠져 있던
  `l3_kg_edge_recall_gold_only`·`l3_kg_edge_recall_gold_only_canonical`·
  `l4_rule_constraint_violation_rate`·`l4_typed_consistency_rate`를 추가(구형 리포트는 "—").
  `eval_output/evidence-2026-09/notes/s6_compare.py`는 `T.TRACKED`를 그대로 참조해서(제외 목록에
  안 걸림) 따로 고치지 않아도 새 필드를 그대로 물려받는다(파일 자체는 `eval_output/` 아래라
  수정 금지 대상이기도 하다).
- 테스트: `tests/unit/scripts/test_ontology_offline_checks.py`,
  `tests/unit/scripts/test_rescore_l3_l4.py`, `tests/eval/test_runner_trace.py`,
  `tests/eval/test_metrics_l3_gold_only.py`에 새 동작 고정.
- `tests/eval tests/unit/scripts -q` 562 passed, 3 skipped(기존 스킵, 무관), 0 failed.

---

## §O7 — 측정과 판정 (2026-09-18, API 약 $7.54) [2026-09 사후]

> 결정: OA-10~OA-13 (`docs/plans/ontology-activation-decisions-2026-09.md`).
> 분석 원자료(API $0, 저장된 리포트만 읽음): `eval_output/ontology-2026-09/analysis/o7_summary.md`(요약)·`o7_tables.md`(전체 표)·`o7_data.json`·`o7_analyze.py`·`o7_l4_detail.py`.
> 새 리포트와 재사용 리포트(s6a·s5·s3)의 L3·L4 새 필드는 모두 `scripts/rescore_l3_l4.rescore_report`로 같은 코드로 다시 채점했다. o7 리포트의 저장값과 재채점값은 모든 run에서 일치했다.

### O7-1. 구성과 실행

모든 실행은 커밋 `433f233`에서 `eval_output/ontology-2026-09/run_eval_onto.sh`로 돌렸다. 실행마다 별도 worktree를 만들고 동결 스냅샷의 **사본**을 `data/`로 복사한다(결정 S2-1). 공통 옵션은 `--target v4 --data-as-of 2026-08-31 --judge llm --judge-model gpt-4.1-mini --semantic-similarity --concurrency 4`이다. 플래그는 `FF_ONTOLOGY_USE_CLASS_REASONING`만 바꾸고, 다른 `FF_*` 변수는 실행 전에 지웠다.

| 구성 | 스냅샷 (KG sha256 앞 8자) | 시험지 | 반복 | 실행 이름 |
|---|---|---|---|---|
| (a) 플래그 OFF | 태그 (`876ea40f`) | 54문항(`s6_mh_rel54.jsonl`, multihop 27 + relation 27) | 3회 | `o7a-run1~3` |
| (b) 플래그 ON | 태그 (`876ea40f`) | 54문항 | 3회 | `o7b-run1~3` |
| (c) 플래그 ON | 온톨로지 (§O5 마이그레이션, `1600836e`) | 54문항 | 3회 | `o7c-run1~3` |
| rule OFF / ON | 태그 | `typed/rule.jsonl` 42문항 | 각 2회 | `o7r-off-run1~2`, `o7r-on-run1~2` |
| 회귀 확인 (b 구성) | 태그 | `typed/combined_v1.jsonl` 233문항 | 1회 | `o7full-b-run1` |
| 회귀 신호 확인 (추가) | 태그 | `typed/numeric.jsonl` 31문항 | OFF·ON 각 2회 | `o7n-off-run1~2`, `o7n-on-run1~2` |

- DB는 모든 실행에서 같다(sha256 앞 8자 `7a0dce00`). 실행 전후 KG 해시도 같았다(`runs.log`).
- 인프라 실패(judge 시간 초과)로 채점에서 빠진 문항: `o7a-run2` lg155, `o7a-run3` lg161, `o7full-b-run1` mh004, `o7n-on-run2` lg089. 이 run의 통과 수는 빠진 문항을 뺀 기준이다.
- 비교 기준 "기준6"은 재사용 s6a 3회와 o7a 3회를 합친 것이다. s6a-run2도 lg155가 인프라 실패로 빠져 있다.
- 판정 규칙: |평균 차| ≥ max(노이즈 기준, 두 구성의 run 간 폭)이고 범위가 겹치지 않으면 有다. 노이즈 기준은 종합·검색 0.01, 근거성·관련성 0.03, 수치 0.05, L4 0.05, 통과 5건이다. **최종 판정은 (a) 대비와 기준6 대비가 모두 有이고 방향이 같을 때만 有**다.
- 표기: 평균 [최소, 최대].

### O7-2. 비용

장부는 `eval_output/risk-remediation-2026-09-17/cost_ledger.md`의 "온톨로지 작동 작업 O0~O7" 절이다. 비용 = 리포트 비용 × 1.1.

| 항목 | 리포트 비용 | 장부 비용 |
|---|---:|---:|
| 스모크 5문항 2회 (O7 전) | 0.0610 | ≈ 0.067 |
| O7 측정 14회: (a) 1.024, (b) 1.095, (c) 1.095, rule OFF×2 0.544, rule ON×2 0.528, 233×1 1.698 | 5.984 | ≈ 6.58 |
| numeric 31문항 OFF×2·ON×2 (회귀 신호 확인) | 0.875 | ≈ 0.96 |
| **누적 (상한 $10, 경보선 $8)** | | **≈ 7.61** |

### O7-3. 54문항 결과 (multihop 27 + relation 27, 각 3회)

| 지표 | (a) OFF | (b) ON 태그 | (c) ON 온톨로지 KG | 기준6 | b−a | b 최종 | c 최종 | c−b |
|---|---|---|---|---|---|---|---|---|
| 종합 점수 | 0.692 [0.692, 0.693] | 0.745 [0.742, 0.748] | 0.744 [0.742, 0.748] | 0.691 [0.685, 0.693] | +0.053 | 有↑ | 有↑ | −0.001 無 |
| 통과 수 | 2.7 [2, 3] | 5.3 [5, 6] | 5.0 [4, 6] | 2.8 [2, 4] | +2.7 | 無 | 無 | −0.3 無 |
| L2 개념 Recall | 0.872 [0.867, 0.874] | 0.855 [0.849, 0.867] | 0.858 [0.849, 0.867] | 0.861 | −0.017 | 無 | 無 | 無 |
| L3 Recall(전체, 기존 지표) | 0.234 [0.230, 0.239] | 0.610 | 0.610 | 0.235 | +0.375 | 有↑ | 有↑ | 0 無 |
| L3 Recall(골드 엣지 문항) | 0.137 [0.131, 0.142] | 0.561 | 0.561 | 0.139 | +0.423 | 有↑ | 有↑ | 0 無 |
| L3 Recall(골드 엣지 문항, canonical) | 0.301 [0.296, 0.307] | 0.582 | 0.582 | 0.302 | +0.281 | 有↑ | 有↑ | 0 無 |
| L4 규칙 제약 위반율(새) | 0.093 [0.091, 0.095] | 0.072 | 0.072 | 0.093 | −0.021 | 無 | 無 | 0 無 |
| L4 타입 일관성(새) | 0.969 [0.968, 0.970] | 0.976 | 0.996 | 0.969 | +0.007 | 無 | 無 | +0.019 無 |
| L5 근거성 | 0.867 [0.844, 0.889] | 0.968 [0.958, 0.978] | 0.953 [0.941, 0.959] | 0.867 | +0.101 | 有↑ | 有↑ | −0.015 無 |
| L5 관련성 | 0.912 [0.895, 0.926] | 0.960 [0.951, 0.967] | 0.966 [0.958, 0.971] | 0.909 | +0.048 | 有↑ | 有↑ | +0.006 無 |
| L5 토큰 F1 | 0.199 [0.194, 0.201] | 0.225 [0.223, 0.227] | 0.227 [0.223, 0.232] | 0.195 | +0.027 | 有↑ | 有↑ | +0.002 無 |
| L5 수치 정확도 | 0.601 [0.560, 0.629] | 0.733 [0.675, 0.781] | 0.755 [0.714, 0.819] | 0.600 | +0.132 | 有↑ | 有↑ | +0.021 無 |
| 규칙 발화 문항 비율 | 0.794 | 0.852 | 0.852 | 0.794 | +0.058 | 有↑ | 有↑ | 0 無 |
| 지연(초/문항) | 6.60 [6.28, 7.15] | 6.91 [6.55, 7.10] | 6.73 [6.47, 7.09] | 6.36 | +0.31 | 無 | 無 | −0.18 無 |
| 파이프라인 비용($/문항, L5) | 0.0023 | 0.0025 | 0.0025 | 0.0023 | +0.0002 | 증가(+9%) | 증가(+9%) | 0 無 |
| L5 프롬프트 토큰/문항 | 4,572 [4,532, 4,635] | 4,988 [4,880, 5,128] | 5,085 [5,071, 5,096] | 4,574 | +417 | 증가(+9%) | 증가(+11%) | +97 無 |

- (b)(c)의 L3·L4는 run 간 폭이 0이다. 검색·KG 경로는 결정적이고, 흔들리는 것은 답변 생성과 judge뿐이다.
- 54문항 L4 값은 O7-6의 결함 수리(`3421ef8`) **이전** 코드로 잰 것이다.

**술어별 canonical recall (골드 엣지 문항, 54문항, 모든 구성에서 run 간 폭 0)**

| 술어 [골드 엣지 수] | (a) | (b) | (c) | b·c 최종 |
|---|---|---|---|---|
| ownedByGroup [33] | 0.424 | 0.909 | 0.909 | 有↑ |
| hasSegment [14] | 0.000 | 0.429 | 0.429 | 有↑ |
| originatesFrom [2] | 0.000 | 0.500 | 0.500 | 有↑ |
| siblingBrand [2] | 0.000 | 1.000 | 1.000 | 有↑ |
| acquiredIn [1] | 0.000 | 1.000 | 1.000 | 有↑ |
| belongsToCategory [12] | 0.333 | 0.333 | 0.333 | 無 |
| competesWith [10] | 0.000 | 0.000 | 0.000 | 無 |
| hasSoS [3] | 0.667 | 0.667 | 0.667 | 無 |

`originatesFrom`이 1/2에서 멈춘 것은 rl011 골드가 `korea`이고 등록부 국가 id가 `south_korea`이기 때문이다(§O3 기록과 같다. 골드는 고치지 않았다).

**유형별 (54 안)**

| 유형 | 지표 | (a) | (b) | b 최종 |
|---|---|---|---|---|
| relation 27 | 종합 | 0.685 | 0.770 | 有↑ |
| relation 27 | 토큰 F1 | 0.145 | 0.205 | 有↑ |
| relation 27 | 통과 수 | 0 | 0 | 無 |
| multihop 27 | 종합 | 0.700 | 0.721 | 有↑ |
| multihop 27 | 토큰 F1 | 0.254 | 0.246 | 無 |
| multihop 27 | 관련성 | 0.980 [0.967, 0.987] | 0.952 [0.935, 0.965] | 無 (−0.027, 폭 이내) |
| multihop 27 | 근거성 | 0.904 | 0.946 | 有↑ |

### O7-4. 온톨로지 문항 부분집합 (54문항 안)

문항 단위 태그가 시험지에 없어서 **골드 `kg_edges`의 술어로 분류**했다.

- group: `ownedByGroup`·`ownedBy`·`ownsBrand`·`siblingBrand`
- segment: `hasSegment`
- origin: `originatesFrom`
- negative: rl015·rl016

부분집합이 작아 통과 수 노이즈는 1건으로 두었다. 원래 기준(5건)을 쓰면 통과 수는 모두 無다.

| 부분집합 (n) | 지표 | (a) | (b) | (c) | b 최종 | c 최종 |
|---|---|---|---|---|---|---|
| 합집합 (25) | 종합 | 0.697 [0.696, 0.698] | 0.806 [0.805, 0.808] | 0.806 [0.805, 0.808] | 有↑ | 有↑ |
| 합집합 (25) | 통과 수 | 0 | 2 | 2 | 有↑ (mh001·mh002, 3회 모두) | 有↑ |
| 합집합 (25) | 토큰 F1 | 0.201 | 0.258 | 0.263 | 有↑ | 有↑ |
| group (19) | 종합 / 토큰 F1 | 0.715 / 0.247 | 0.812 / 0.274 | 0.813 / 0.280 | 有↑ / 有↑ | 有↑ / 有↑ |
| segment (6) | 종합 / 토큰 F1 | 0.640 / 0.057 | 0.785 / 0.207 | 0.786 / 0.207 | 有↑ / 有↑ | 有↑ / 有↑ |
| origin (2) | 종합 / 토큰 F1 | 0.642 / 0.045 | 0.795 / 0.127 | 0.776 / 0.139 | 有↑ / 有↑ | 有↑ / 有↑ |
| negative (2) | 종합 / 토큰 F1 | 0.744 / 0.286 | 0.849 / 0.383 | 0.854 / 0.406 | 有↑ / 有↑ | 有↑ / 有↑ |
| segment·origin·negative | 통과 수 | 0 | 0 | 0 | 無 | 無 |
| **other (29)** | 종합 | 0.689 [0.687, 0.691] | 0.693 [0.688, 0.697] | 0.691 [0.686, 0.698] | 無 | 無 |
| other (29) | 통과 수 | 2.7 | 3.3 | 3.0 | 無 | 無 |
| other (29) | 토큰 F1 | 0.196 | 0.198 | 0.197 | 無 | 無 |

- 모든 온톨로지 부분집합에서 종합·토큰 F1이 有↑다.
- 통과 수가 오른 것은 합집합·group의 2문항(mh001·mh002)뿐이다. relation 문항은 모든 구성에서 통과 0이라 통과 수가 거의 움직이지 않는다.
- 나머지 29문항("other")에는 회귀가 없다.

### O7-5. rule 42문항 (OFF·ON 각 2회)

| 지표 | OFF | ON | s3 rule 42 (재사용 3회) | ON−OFF | ON−s3 |
|---|---|---|---|---|---|
| 규칙 정답 일치율 | 0.781 [0.781, 0.781] | 0.906 [0.906, 0.906] | 0.779 [0.774, 0.781] | +0.125 有 | +0.127 有 |
| 통과 수 | 22.5 [22, 23] | 30.5 [30, 31] | 23.7 [23, 24] | +8.0 有 | +6.8 有 |
| 종합 | 0.794 [0.791, 0.797] | 0.811 [0.810, 0.812] | 0.800 [0.797, 0.803] | +0.017 有 | +0.011 有 |
| 규칙 발화 문항 비율 | 0.571 | 0.762 | 0.568 | +0.190 有 | +0.194 有 |
| L5 수치 정확도 | 0.774 | 1.000 | 0.777 | +0.226 有 | +0.223 有 |
| L5 토큰 F1 | 0.266 | 0.280 | 0.276 | +0.015 有 | +0.005 無 |
| L5 근거성 / 관련성 | 0.934 / 0.974 | 0.950 / 0.998 | 0.966 / 1.000 | 無 / 無 | 無 / 無 |
| L4 규칙 제약 위반율(새) | 0.116 | 0.228 | 0.118 | +0.112 (악화) | +0.110 (악화) |

- 일치율이 바뀐 문항은 rg021·rg022·rg023·rg026 네 개뿐이다(OFF 2회 불일치 → ON 2회 일치). 반대 방향은 0문항이다. 이 4문항은 O2 연결기가 새로 인식한 브랜드(IT Cosmetics·Jouer·Almay·Charlotte Tilbury) 문항이다(§O2 측정 3).
- 문항별 통과 횟수(OFF/ON): 0/0 = 10, 0/2 = 8, 1/2 = 3, 2/1 = 3, 2/2 = 18.
- `s3-roff`(규칙 추론 자체를 끈 구성, 발화 0)는 온톨로지 OFF 기준이 아니라서 판정에 쓰지 않았다. 참고로 종합 점수는 s3-roff 0.822가 ON 0.811보다 높다(`o7_tables.md` §3).

### O7-6. L4 규칙 위반율 상승과 수리 (결정 OA-13)

- **원인**: ON에서 새로 발화한 가격 규칙(rg020~rg026의 `value_position`·`price_quality_mismatch` 등)의 `related_entities`에 빈 문자열 `''`이 들어가 `related_entity_invalid`로 잡혔다. `reasoner.py` `InferenceRule.apply()`가 결측 입력의 빈 문자열을 그대로 옮긴 것이다.
  - 위반 건수: OFF 16건/추론 118 → ON 21건/130. `ownership_mismatch`·`value_range` 위반은 0이었다.
  - 새로 발화한 문항이 분모에 들어가면서 문항 평균 비율이 두 배가 됐다.
- **수리**: `3421ef8`에서 `related_entities`의 빈 문자열과 가짜 브랜드를 뺀다.
- **확인**: 오프라인 재채점만 했다. `o7r-on-run1` 트레이스를 재채점하면 위반율이 0.228 → 0.0이 되고, 발화 규칙과 결론 130건은 수리 전과 같다.
- 수리 후 코드로 LLM 평가를 다시 돌리지는 않았다.

### O7-7. 233문항 회귀 확인 (1회) + numeric 확인 측정

`o7full-b-run1`(ON, 232문항 채점)을 재사용 `s5-run1`(1회)과 비교했다. 참고 노이즈는 max(기준, s3-run1~3 폭)이다. 1회 측정이라 확정 판정이 아니다. s5는 커밋 `ba714eb`라 온톨로지 밖의 차이(O2 연결기 등)도 섞여 있다.

| 유형 | 종합 | 통과 | 근거성 | 관련성 | 토큰 F1 | 수치 정확도 |
|---|---|---|---|---|---|---|
| numeric | 0.707→0.699 | 6→7 | 0.954→0.935 | 0.947→0.937 | 0.207→**0.193** (−0.014, 폭 0.010) | 0.672→0.683 |
| relation | 0.685→0.772 ↑ | 0→0 | 0.839→0.975 ↑ | 0.852→0.963 ↑ | 0.156→0.206 ↑ | 0.750→0.750 |
| rule | 0.793→0.808 ↑ | 23→31 ↑ | 0.940→0.959 | 0.962→0.986 | 0.265→0.281 ↑ | 0.774→1.000 ↑ |
| multihop | 0.706→0.711 | 4→6 | 0.900→0.946 ↑ | 0.981→**0.946** (−0.035, 폭 0.006) | 0.257→0.255 | 0.548→0.737 ↑ |
| other(유형 외) | 0.602→0.615 ↑ | 21→19 | 0.830→0.898 ↑ | 0.823→0.903 ↑ | 0.105→0.123 ↑ | 0.139→**0.056** (−0.083, 폭 0.056) |
| 전체 232 | 0.672→0.690 ↑ | 54→63 ↑ | 0.875→0.928 ↑ | 0.886→0.934 ↑ | 0.171→0.185 ↑ | 0.586→0.684 ↑ |

노이즈를 넘은 하락 3건(굵게)은 이렇게 처리했다(결정 OA-10).

1. **numeric 토큰 F1**: numeric 31문항을 OFF·ON 각 2회 다시 쟀다(`o7n-*`, 커밋 `433f233`, 태그 스냅샷). 하락은 재현되지 않았다.

   | 지표 | OFF run1 / run2 | ON run1 / run2 | 평균 OFF → ON |
   |---|---|---|---|
   | 종합 | 0.698 / 0.700 | 0.710 / 0.713 | 0.699 → 0.712 |
   | 통과 수 | 6 / 6 | 7 / 8 (run2는 30문항, lg089 제외) | 6 → 7.5 |
   | L5 토큰 F1 | 0.202 / 0.190 | 0.206 / 0.206 | 0.196 → 0.206 |
   | L5 수치 정확도 | 0.683 / 0.717 | 0.667 / 0.672 | 0.700 → 0.670 (−0.030, 노이즈 0.05 이내 → 無) |
   | L5 근거성 | 0.935 / 0.930 | 0.962 / 0.987 | 0.933 → 0.975 |
   | L5 관련성 | 0.927 / 0.958 | 0.950 / 0.932 | 0.943 → 0.941 |

   값은 각 리포트의 `aggregates`에서 읽었다.
2. **multihop 관련성**: 54문항 3회 측정에서 −0.027이고 run 간 폭 이내라 無였다(O7-3 유형별 표).
3. **유형 외 문항 수치 정확도 −0.083**: 수치 채점 문항이 소수라 판정을 보류하고 기록만 한다.

233문항에서 지연은 6.41 → 7.21초(+0.8초), 프롬프트 토큰은 문항당 +295(+6.5%)였다.

### O7-8. 카드 수 (결정 OA-12)

`trace.evidence`는 프롬프트에 실린 카드이고, 출처 `ontology:*`가 온톨로지 카드다.

| 구성 | 프롬프트 카드 평균 | 최대 | 70장 초과 문항 (run별) | 온톨로지 카드 평균 | 전체 evidence 평균 |
|---|---|---|---|---|---|
| (a) OFF 54 | 58.1 | 100 | 26 / 25 / 25 | 0 | 111.5 |
| (b) ON 54 | 69.9 | 91 | 32 / 32 / 32 | 6.8 (54문항 중 52문항에 실림) | 171.4 |
| (c) ON 54 | 70.5 | 105 | 33 / 33 / 33 | 6.8 | 171.6 |
| rule OFF / ON | 56.7 / 62.3 | 73 / 76 | 17 / 20 | 0 / 2.5 | 81.3 / 102.4 |
| 233: s5 / full-b | 55.8 / 64.7 | 95 / 91 | 102 / 131 | 0 / 4.0 | 105.8 / 154.8 |

- ON은 프롬프트 카드를 평균 약 +12장(54문항), +9장(233문항) 늘렸다.
- **54문항 평균이 5단계에서 지적한 한계선 "질의마다 약 70장"에 닿았다.** segment 부분집합은 34.5 → 72.2장이다.
- 기록하고 수용했다(OA-12). 카드 상한 조정은 FUTURE_WORK 9.10에 남겼다.

### O7-9. 문항별 일관 하락: lg158 한 문항

ON 3회가 모두 OFF 3회의 최솟값보다 낮은 문항은 lg158뿐이다((b) −0.133, (c) −0.108). 반대로 ON 3회가 모두 OFF 최댓값보다 높은 문항은 (b) 26개, (c) 24개다. 대부분 rl001~rl019·mh001~mh007(온톨로지 부분집합)이고, 목록은 `o7_tables.md` §6에 있다.

- 질문: "LANEIGE 모회사와 해당 기업의 다른 브랜드 Amazon 현황".
- 원인 1 — **골드가 레거시 술어를 씀**: 골드 엣지가 `ownedBy`(laneige·innisfree·sulwhasoo)다. OFF는 `laneige -ownedBy-> amorepacific`을 내서 raw L3 1/3이고, ON은 `ownedByGroup`으로 내서 raw L3 0/3이다. 종합 점수는 raw L3를 쓴다.
- 원인 2 — **소속 브랜드 전개가 일어나지 않음**: 질문에 LANEIGE가 함께 나와 "그룹만 언급" 조건(§O3)을 채우지 못한다. 그래서 답변이 "다른 브랜드 현황은 데이터에 없음"이라고 한다. run1은 SoS 수치도 빼서 수치 정확도가 0이 됐다.
- 골드는 고치지 않았다(계획서 §8). FUTURE_WORK 9.10에 기록했다.

### O7-10. (c) KG 마이그레이션 스냅샷 효과 (결정 OA-11)

(c)와 (b)는 카드 수(+0.7장) 말고는 모든 지표가 無였다. 질의 경로가 KG를 읽을 때 정식화(§O3)를 하므로, 저장 형식을 정리한 추가 효과가 이번 측정에서는 보이지 않았다.

**운영 KG 적용은 권하지 않는다(효과 없음).** 적용 명령은 §O5 "3. 운영 KG 적용 명령"에 남기고 소유자 판단에 맡긴다.

### O7-11. 판정 (결정 OA-10)

계획서 O7 기준은 "온톨로지 문항 부분집합 개선 + 다른 유형 회귀 없음 → 기본 ON"이다.

- 온톨로지 부분집합: 종합·토큰 F1·canonical L3·근거성·관련성이 (a)·기준6 대비 모두 有↑.
- 다른 유형: 54문항 "other" 29문항에 회귀 없음. rule 42문항은 일치율·통과 수 有↑. 233문항 1회의 하락 3건은 O7-7처럼 처리했다.
- **`ontology.use_class_reasoning` 기본 ON.** `config/feature_flags.json`만 `true`로 바꾸고, 코드 기본값(키가 없을 때)은 False로 둔다.

### O7-12. 최종 전체 테스트와 `data/` 해시

- 커밋 `433f233`에서 전체 테스트를 1회 돌렸다: **6,361 통과, 7 skip, 0 실패** (`eval_output/ontology-2026-09/final_full_test.log`).
  - 이 실행은 `related_entities` 수리(`3421ef8`)와 기본값 전환 **이전** 코드다. 그 뒤 재실행 결과는 리드가 따로 기록한다.
- 실행 전후 `data/` sha256 비교(`data_sha_before.txt`·`data_sha_after.txt`·`data_sha_diff.txt`): 바뀐 파일은 `data/chroma/chroma.sqlite3` 하나다(`b8dbd05b…` → `d9beabd6…`).
  - Chroma 컬렉션 크기는 변하지 않았다(임베딩 440개, 리드 확인). 알려진 현상 S3-2와 맞는다.
  - 다만 실행 전 사본을 떠 두지 않았다. 그래서 **테이블 단위 전후 비교(`acquire_write`만 달라졌는지)는 확인하지 못했다.**

---

## §요약 — 기준선 → 최종 [2026-09 사후]

54문항(multihop 27 + relation 27), 태그 스냅샷, 커밋 `433f233` 기준이다. 기준선은 (a) 플래그 OFF 3회, 최종은 (b) 플래그 ON 3회다. 규칙 지표만 rule 42문항 각 2회다. 판정은 (a) 대비와 기준6 대비가 모두 有일 때만 有다.

| 지표 | 기준선 (a) OFF | 최종 (b) ON | 차이 | 판정 |
|---|---|---|---|---|
| 종합 점수 | 0.692 [0.692, 0.693] | 0.745 [0.742, 0.748] | +0.053 | 有↑ |
| 통과 수 | 2.7 [2, 3] | 5.3 [5, 6] | +2.7 | 無 (노이즈 5건 미만) |
| L3 골드 엣지 recall (canonical) | 0.301 [0.296, 0.307] | 0.582 | +0.281 | 有↑ |
| L3 골드 엣지 recall (raw) | 0.137 [0.131, 0.142] | 0.561 | +0.423 | 有↑ |
| L5 근거성 | 0.867 [0.844, 0.889] | 0.968 [0.958, 0.978] | +0.101 | 有↑ |
| L5 관련성 | 0.912 [0.895, 0.926] | 0.960 [0.951, 0.967] | +0.048 | 有↑ |
| L5 토큰 F1 | 0.199 [0.194, 0.201] | 0.225 [0.223, 0.227] | +0.027 | 有↑ |
| L5 수치 정확도 | 0.601 [0.560, 0.629] | 0.733 [0.675, 0.781] | +0.132 | 有↑ |
| 온톨로지 부분집합 25문항 종합 | 0.697 [0.696, 0.698] | 0.806 [0.805, 0.808] | +0.109 | 有↑ |
| 나머지 29문항 종합 | 0.689 [0.687, 0.691] | 0.693 [0.688, 0.697] | +0.004 | 無 (회귀 없음) |
| 규칙 정답 일치율 (rule 42) | 0.781 [0.781, 0.781] | 0.906 [0.906, 0.906] | +0.125 | 有↑ (ON−OFF, ON−s3 모두) |
| L4 규칙 제약 위반율 (rule 42) | 0.116 | 0.228 → 수리 후 재채점 0.0 (`3421ef8`, run1 트레이스) | — | 결함 수리 (OA-13) |
| 프롬프트 카드/문항 | 58.1 | 69.9 | +11.8 | 증가 (한계선 ~70장에 닿음) |
| 파이프라인 비용 ($/문항) | 0.0023 | 0.0025 | +9% | 증가 |
| 지연 (초/문항) | 6.60 | 6.91 | +0.31 | 無 |

**플래그 판정**

| 플래그 | 결정 | 근거 |
|---|---|---|
| `ontology.use_class_reasoning` | **기본 ON** (`config/feature_flags.json`만, 코드 기본값 False) | OA-10 |
| `kg.write_validation` | 기본 `warn` 유지 (로그만, 저장 내용 불변) | OA-7. `enforce` 선행 조건은 FUTURE_WORK 9.10 |
| `reasoner.enabled`·`kg.enabled` | 이름만 바로잡음. 기본 True, 옛 이름은 별칭 | §O6-3 |

**삭제한 코드** (§O6-2, 커밋 `7e50cf1`)

| 파일 | 줄 |
|---|---:|
| `src/ontology/owl_reasoner.py` | 1,247 |
| `src/ontology/ontology_knowledge_graph.py` | 398 |
| `src/ontology/cosmetics_ontology.owl` | 770 |
| `scripts/migrate_kg_to_ontology.py` | 229 |
| OWL 전용 테스트 4파일 + `TestOWLConsistencyIntegration` | 약 1,750 |

- 소스·스크립트·OWL 합계는 2,644줄이고, 커밋 기준으로 −4,423 / +33줄이다.
- 서비스 호출처는 0건이었다(§O6-1 호출처 확인표).
- owlready2는 `requirements-dev.txt`로 옮겼다.

**운영 KG 적용**: 명령과 예상 변경 건수는 §O5 "3. 운영 KG 적용 명령"에 있다. OA-11에 따라 **권하지 않는다**(이번 측정에서 효과 없음). 적용한다면 `kg.write_validation=enforce`를 같이 켜야 하고, 그 전에 exporter가 `as_of`를 넘겨야 한다(§O5 4절).

**남긴 항목** (자세한 내용은 `docs/dev/FUTURE_WORK.md` 9.10)

- 프롬프트 카드 수가 한계선 ~70장에 닿음(OA-12). 카드 상한·선별 조정이 필요하다.
- 골드 어휘 불일치(기록만 함, 골드 수정은 사용자 결정): lg102·lg158의 레거시 `ownedBy`, rl011의 `korea`(등록부는 `south_korea`).
- lg158: 그룹과 소속 브랜드를 함께 언급하면 소속 브랜드 전개가 일어나지 않는다.
- `kg.write_validation=enforce` 선행 조건(OA-8): exporter의 `as_of` 전달, 날짜 없는 수치 엣지 333건.
- CI가 `requirements-dev.txt`를 설치하지 않아 Pellet 교차 검증 테스트가 skip된다.
- `is_target` 일반화는 하지 않았다(§O4, rule 골드 15문항의 발화 집합이 바뀜).
- `3421ef8` 이후 코드로 LLM 재측정은 하지 않았다(L4는 오프라인 재채점만). 유형 외 문항 수치 정확도 −0.083(233문항, 1회)은 판정 보류다.
