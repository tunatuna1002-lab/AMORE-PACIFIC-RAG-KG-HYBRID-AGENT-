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
