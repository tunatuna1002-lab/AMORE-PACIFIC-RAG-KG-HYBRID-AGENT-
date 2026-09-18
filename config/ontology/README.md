# config/ontology — 온톨로지 단일 원본 [2026-09 사후]

결정 OA-1(JSON 원본), OA-2(카테고리는 `config/category_hierarchy.json` 참조), OA-3(지원 공리 형식 제한), OA-4(SWRL 규칙은 옮기지 않음).
계획서: `docs/plans/ontology-activation-plan-2026-09-18.md` 트랙 O1.

| 파일 | 내용 |
|---|---|
| `schema.json` | 클래스(`subClassOf`, 정의 클래스 `defined_by`), 서로소 집합, 술어(도메인·범위·역·대칭·전이·별칭·`static`·수치 `as_of` 필수), KG 옛 술어 분리 표(`kg_legacy_predicates`), 세그먼트·국가·지표 개체 |
| `brands.json` | 브랜드 등록부(119개)와 기업 그룹(1개). `_meta`에 통계와 작성 원칙 |
| (참조) `../category_hierarchy.json` | 카테고리 개체와 `subCategoryOf`(`parent_id`에서 생성) |

## 사용

```python
from src.ontology.ontology import get_ontology

onto = get_ontology()                      # 한 번 로드, 스레드 안전 캐시
onto.normalize_brand("e.l.f.")             # "elf"
onto.instances_of("AmorepacificBrand")     # 31개 (정렬)
onto.closed_world_member("TIRTIR", "amorepacific")   # False (OE4)
onto.category_descendants("skin_care")     # 조회 범위 확장 전용 (OE3)
```

런타임에는 Java·추론기를 부르지 않는다(OE2). OWL 의미론과 같은지는 개발 시점에 확인한다.

```bash
.venv/bin/python scripts/export_ontology_owl.py --out /tmp/amore_ontology.owl   # Protégé용
.venv/bin/python scripts/check_ontology_owl.py   # Pellet: 모순 0 + 폐포 교차 검증. Java < 25면 종료 코드 3(SKIPPED)
```

## 등록부 통계 (2026-09-18)

| 항목 | 값 |
|---|---|
| 항목 수 | 119 (실제 브랜드 116 + 가짜 3: `unknown`, `fresh`, `chi`) |
| 그룹 있음 | 31 (모두 `amorepacific`) |
| 세그먼트 있음 | 51 |
| 원산지 있음 | 28 |
| 인수 연도 있음 | 1 (COSRX 2024. TATA HARPER는 원본에 연도 없이 `acquired: true`) |
| KG 스냅샷 브랜드 문자열 | 원문 125개(대소문자 무시 109개, enricher 계열 83개) — 제외 0개 |

## 원칙

- **원본에 적힌 사실만** 싣는다. 모르는 그룹·세그먼트·원산지는 `null`로 둔다. 추정으로 채우지 않는다.
  - KG 스냅샷의 AP 브랜드 30개 `originatesFrom Korea`는 쓰지 않았다. `kg_updater.load_brand_ownership`의 기본값(`country="Korea"`)이 만든 값이기 때문이다. 그래서 IOPE·primera 등은 원산지 `null`이고 `KBeautyBrand`에 들지 않는다.
- 세그먼트: AP 브랜드의 `segment`(Premium/Luxury/…)와 경쟁사의 `tier`(premium/mid/affordable/mass)를 한 어휘로 합쳤다. 대응표는 `schema.json`의 `individuals.Segment[*].source_values`.
  - `config/brands.json`의 `segments` 블록은 브랜드 항목과 어긋난다(HERA·ETUDE +Makeup, HOLITUAL Health↔Wellness, OSULLOC Lifestyle↔Tea). 브랜드 항목을 따랐다.
  - 세그먼트 `Makeup`의 id는 `makeup_line`이다. `makeup`이 카테고리 id라서 겹치지 않게 했다(개체 id는 종류와 상관없이 유일해야 한다).
- `아모레퍼시픽`은 그룹 별칭으로만 둔다. `config/brands.json`·`entities.json`에서 `Amore Pacific` 브랜드와 그룹 양쪽에 적혀 있다. 브랜드 `Amore Pacific`(id `amore_pacific`)과 그룹 `AMOREPACIFIC`(id `amorepacific`)은 공백 유무로 구분된다.
- 가짜 항목은 KG 산출물이다(검토 보고서 §3.2). `normalize_brand`는 id를 돌려주지만 `is_placeholder()`가 참이고, 클래스는 `Brand`가 아니라 `PlaceholderBrand`다.
- `hasPosition`은 한 별칭이 아니다. `original_predicate`로 `hasSoS`·`hasHHI`·`hasPricePosition`으로 나눠야 한다(`resolve_kg_predicate`). `belongsToCategory` 중 `original_predicate=rankedIn`인 것은 `rankedIn`이다.
- `ownedBy`는 `ownedByGroup`의 별칭이다(런타임 카드와 `rule_contracts`의 사용법). `relations.py` 주석은 Product→Brand라고 적고 있으나 KG에서 그 뜻으로 쓰이지 않는다.

## 고칠 때

- 브랜드 추가·수정은 `brands.json`을 직접 고친다. `id` 순 정렬, `sources`는 비우지 않는다. `_meta` 수치도 맞춘다(테스트가 확인한다).
- 스키마에 새 공리 형식을 넣으면 로더가 `UnsupportedAxiomError`로 거부한다. 폐포 계산과 Pellet 교차 검증을 함께 늘려야 한다.
- 확인: `.venv/bin/python -m pytest tests/unit/ontology/test_ontology_core.py tests/unit/ontology/test_ontology_owl_check.py -q`
