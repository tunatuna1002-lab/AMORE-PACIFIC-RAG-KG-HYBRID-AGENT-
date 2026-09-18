# 온톨로지 작동 계획서 — 실행 지시서 (O0~O7)

> 목적: 온톨로지가 **질의 경로에서 실제로 답을 바꾸는 역할**을 하게 만들고, 그 효과를 측정으로 판정한다.
> 근거: [`docs/analysis/ontology-review-2026-09-18.md`](../analysis/ontology-review-2026-09-18.md) (이하 **검토 보고서**). 먼저 읽을 것.
> 작성: 2026-09-18. 브랜치 `feat/ontology-activation-2026-09`(기준 `a7ac285`, PR #12 브랜치에서 분기). 모든 작업은 공모전 이후의 사후 보완이며 문서에 **[2026-09 사후]**로 표시한다.

---

## 0. 원칙과 권한

**첫째 원칙은 정확하게, 둘째 원칙은 빠르게.** 둘이 부딪히면 정확함을 택한다.

- **주장은 확인하고 쓴다.** 모든 주장은 코드·테스트·측정으로 확인한다. 추측은 추측이라고 적는다.
- **mock으로 버그를 가리지 않는다.** 이 저장소는 `1cd4307`에서 "테스트가 버그를 정상으로 박제"한 전례가 있다. 결과를 유리하게 다듬지 않는다(면접 근거 자료다).
- **"온톨로지 추론" 표현은 아껴 쓴다.** 효과가 측정으로 확인된 범위만 이 이름으로 부른다.

**병렬 오케스트레이션 허용.** 서브에이전트(`Agent`, 코드 수정은 `isolation: "worktree"`)와 백그라운드 실행을 쓴다. 리드는 설계 결정·작업 분배·diff 리뷰·병합·게이트 판정만 한다.

**유료 API 상한 $10**(경보선 80% = $8). 장부는 `eval_output/risk-remediation-2026-09-17/cost_ledger.md`에 새 절로 이어 적는다. 지난 작업 누적 ≈ $26.1은 이 상한에 포함하지 않는다. 비용 = 리포트 비용 × 1.1.

**멈추고 질문할 곳은 세 가지뿐이다.**
1. OE1(원본 형식)의 최종 확인 — 작업 시작 시 1회.
2. 운영 `data/` 수정이 필요할 때. O5의 KG 정리를 운영 KG에 적용하는 것도 여기에 들어간다.
3. 경보선에 닿아 측정을 줄여야 할 때. 줄이는 순서는 §3 끝에 정해 두었으니 적용 후 보고만 해도 된다.

그 밖의 모호함은 스스로 판단하고 근거를 커밋 메시지나 결정표에 한 줄로 남긴다.

---

## 1. 먼저 읽을 것

1. 검토 보고서 전체 (짧다).
2. `docs/plans/evidence-react-ontology-decisions-2026-09.md`: 결정 S2-0(SoS 스케일), S2-1(평가 스냅샷), S3-1, S4-1(OWL 전략 삭제), S4-3(태그 스냅샷).
3. `docs/experiments/evidence_pipeline_2026-09.md`: 3단계(규칙 on/off)와 6단계(측정 방법·노이즈 판정).
4. 코드 (필요한 부분만):
   - `src/rag/hybrid_retriever.py` 958~1246(`_query_knowledge_graph`), 1248~(`_evaluate_rules`)
   - `src/rag/entity_linker.py` 128~171, 542~704
   - `src/rag/evidence_adapters.py` 40~80, 640~760
   - `src/ontology/rule_contracts.py` 700~944
   - `src/core/tool_registry.py` 400~560
   - `config/brands.json`, `config/category_hierarchy.json`, `config/entities.json`

---

## 2. 현재 상태 (재조사하지 말 것 — 검토 보고서 §1~§4 요약)

- **OWL**:
  - 서비스 경로 생성 0곳. Java가 없어 추론기가 돌지 않는다.
  - OWL 파일 두 벌의 네임스페이스·속성 이름이 불일치한다.
  - `sync_owl_inferences`는 항상 0건이고, OKG 일관성 검사는 불일치를 보고하지 못한다.
- **플래그 이름이 동작과 다르다**:
  - `reasoner.use_owl_reasoner`는 사실 규칙 추론 전체의 스위치다.
  - `ontology.use_ontology_kg`는 사실 일반 KG 조회의 스위치다.
- **질의 경로는 문자열 일치뿐이다.**
  - `hasSegment`·`originatesFrom`·`siblingBrand`·`ownsBrand`·`acquiredIn`은 `priority_preds`(`hybrid_retriever.py:1045`)와 `EXCLUDED_METADATA`(`evidence_adapters.py:735`)에서 버려진다.
  - 그룹 전개, 클래스 소속, 카테고리 포함 확장이 없다.
- **브랜드 어휘가 5곳 이상에 흩어져 있다.** KG 브랜드 약 83개 중 약 65개를 인식하지 못한다.
- **KG 데이터 문제**: 대소문자 이원화, 가짜 브랜드(`unknown`·`fresh`·`chi`), 합쳐진 술어(`hasPosition`), 날짜 없는 수치.
- **평가 지표 문제**:
  - L4 두 지표는 0.0/1.0으로 고정돼 의미가 없다.
  - L3 recall은 골드 엣지 없는 문항이 1.0으로 들어가 부풀려진다.
  - 골드 술어 `ownedByGroup` 0/29, `hasSegment` 0/14.
- **측정 인프라**:
  - `eval_output/evidence-2026-09/run_eval3.sh`(커밋별 worktree + 태그 스냅샷)
  - `notes/s6_compare.py`(3회 비교·노이즈 판정)
  - `scripts/typed_eval_summary.py`
- **재사용할 기준선 리포트**(파이프라인 코드가 현재와 동일):
  - `s6a-run{1,2,3}`: 54문항 × 3, `5e4f603`
  - `s5-run1`: 233문항 × 1, `ba714eb`
  - `s3-run{1,2,3}`, `s3-roff-run{1,2,3}`: rule 42문항
- 전체 테스트 `9d2e565` 기준 6,013 passed / 0 failed.

---

## 3. 확정 설계 (OE1~OE9)

**OE1. 온톨로지 단일 원본 — 권장: JSON 스키마 파일 + Python 로더 (시작 시 사용자에게 1회 확인).**

`config/ontology/` 아래에 원본을 둔다.

| 파일 | 내용 |
|---|---|
| `schema.json` | 클래스와 상하위 관계. 술어별 도메인·범위·역관계·대칭·정식 이름과 별칭 |
| `brands.json` | 브랜드 등록부: 정식 이름, 별칭, 그룹, 세그먼트·티어, 원산지, 인수 연도, 가짜 여부 |
| `categories` | 기존 `config/category_hierarchy.json`을 참조 |

로더 `src/ontology/ontology.py`가 로드 시 한 번 폐포를 계산한다: 상위 클래스의 전이, 역관계, 대칭, 그룹 소속.

- **이 방식을 권하는 이유**:
  - 배포 이미지에 Java가 없다.
  - 저장소의 설정이 이미 JSON이다.
  - 사람이 고치기 쉽다.
  - 평가·테스트에서 결정적이다.
- **대안 — OWL 파일을 원본으로 쓰는 경우**:
  - owlready2로 로드하되, 추론은 Java 없이 Python 폐포로 한다.
  - 표준 형식이라는 장점이 있다.
  - 편집이 어렵고, 파일 두 벌을 먼저 통합해야 한다.
- **어느 쪽이든 공통 조건**:
  - 런타임에서 Java나 외부 추론기를 부르지 않는다(OE2).
  - `scripts/export_ontology_owl.py`로 OWL 내보내기를 제공해 표준 도구(Protégé)로 볼 수 있게 한다. 내보내기는 개발 전용이다.

**OE2. 런타임 추론은 Python 폐포 계산만 한다.** owlready2 추론기와 Java는 쓰지 않는다. 폐포 결과는 결정적이어야 하고, 단위 테스트로 고정한다.

**OE2-추가 (2026-09-18, Java 8 설치 후).** 로컬에서 HermiT가 동작한다(Pellet은 JDK 25 이상 필요, 검토 보고서 §2.1-추가). 그래서 OWL 추론기에 **개발·검증 단계의 역할**을 준다. 서비스 런타임은 여전히 Python 폐포만 쓴다.
- O1에 `scripts/check_ontology_owl.py`를 추가한다. 원본 스키마·등록부를 OWL로 내보낸 뒤 HermiT로 **① 일관성(모순 클래스 0) ② 클래스 분류 결과가 Python 폐포와 같은지**를 확인한다. 이 교차 검증이 "Python 폐포가 OWL 의미론과 맞다"는 근거가 된다.
- 테스트에서는 Java가 없으면 `skip`한다(조용히 통과시키지 않는다 — 검토 보고서 §2.3 결함 3 재발 방지). 건너뛴 사실을 테스트 출력에 남긴다.
- OE1에서 OWL 원본을 택하면, 기존 `cosmetics_ontology.owl`의 파싱 오류(81행 `<`)부터 고쳐야 로드된다.
- **JDK 25 설치 후 갱신**: 로컬에 Pellet(JDK 25+)과 HermiT가 모두 동작한다. 검증 스크립트는 **Pellet을 기본**으로 쓴다(SWRL 숫자 비교까지 검증 가능). HermiT는 SWRL 내장 연산이 없는 순수 분류 교차 검사용 보조다. 규칙 37개의 숫자 임계를 SWRL로 옮길지는 O4에서 판단하되, 서비스 런타임은 계속 Python 규칙을 쓴다.

**OE3. 카테고리 포함 관계는 조회 범위를 넓히는 데만 쓴다.**
- 허용: "스킨케어" 질의가 `lip_care` 카드·문서까지 가져오는 것.
- 금지: SoS·순위·HHI를 상위 카테고리로 합산·환산하는 것. `category_hierarchy.json`의 `category_rank_context` 규칙이 이를 금지하고, lg155 정답도 이를 전제한다.

이 제약은 테스트로 고정한다.

**OE4. 닫힌 세계는 등록부 안에서만 적용한다.**
- 등록부에 있고 그룹이 명시되지 않은 브랜드는 "그 그룹 소속이 아님"으로 판정할 수 있다(rl015·rl016).
- 등록부 밖의 브랜드는 "알 수 없음"이다.
- 이 판정도 카드로 싣고, 출처는 `ontology:registry`로 적는다.

**OE5. 정적 정의 사실은 날짜 없이도 카드로 싣는다.**
- 대상: 그룹, 세그먼트, 원산지, 인수 연도, 자매 브랜드.
- 수치 엣지를 날짜 없다고 빼는 규칙(`EXCLUDED_METADATA`)은 **수치에만** 적용한다.
- 카드 `as_of`는 등록부 버전 날짜로 둔다.

**OE6. 운영 `data/`는 건드리지 않는다.**
- KG 정리(대소문자 통일, 술어 정식화, 가짜 브랜드 제거, 타입 트리플 추가)는 **마이그레이션 스크립트 + 평가 스냅샷 사본**에서만 한다.
- 운영 KG에 적용하려면 사용자 승인이 필요하다.
- 조회 시 정규화(읽을 때 정식화)는 운영 데이터 변경 없이 적용할 수 있다.

**OE7. 새 동작은 플래그 뒤에 두고 측정 후 켠다.**
- 신규 플래그 `ontology.use_class_reasoning`(기본 OFF)에 그룹 전개·클래스 소속·포함 확장·정적 사실 카드를 묶는다.
- 기존 플래그는 이름을 바로잡는다.

| 새 이름 | 옛 이름 | 옛 이름 처리 |
|---|---|---|
| `reasoner.enabled` | `reasoner.use_owl_reasoner` | 하위 호환 별칭으로 남기고 경고 로그를 낸다 |
| `kg.enabled` | `ontology.use_ontology_kg` | 같음 |

**OE8. OWL 모듈은 O6에서 정리한다.**
- OE1에서 JSON을 택하면: `owl_reasoner.py`·`ontology_knowledge_graph.py`·`cosmetics_ontology.owl`·`scripts/migrate_kg_to_ontology.py`와 관련 테스트를 삭제한다. 대신 OWL 내보내기 스크립트를 둔다. owlready2는 `requirements-dev` 성격으로 옮긴다.
- OWL을 택하면: 두 파일을 하나로 통합하고, 검토 보고서 §2.3의 결함 1·2·5를 수정한다.
- 어느 쪽이든 호출처 확인표를 먼저 만든다.

**OE9. 지표를 먼저 고친다.** L4 두 지표와 L3의 골드 엣지 있는 문항만 대상으로 한 술어별 recall을 O0에서 고친다. 그 전에는 온톨로지 효과를 주장하지 않는다.

**경보선 도달 시 측정 축소 순서**(고정):
1. O7의 233문항 회귀 측정을 1회에서 생략하고, 54문항 결과로 대신한다.
2. rule 42문항 on/off를 3회에서 2회로 줄인다.
3. 54문항 on/off를 3회에서 2회로 줄인다.

on/off 비교는 각 최소 2회를 지킨다.

---

## 4. 병렬 실행 규칙

- 같은 파일을 두 트랙이 동시에 고치지 않는다. 트랙마다 "소유 파일"을 지정한다(§6).
- O1(원본 + 로더)이 병합돼야 O2·O3·O5를 시작한다. O0은 O1과 병렬로 돌릴 수 있다.
- O2·O3·O5는 소유 파일이 겹치지 않아 병렬로 돌릴 수 있다. O4는 O3 병합 뒤에 한다.
- 측정은 동시 3개까지, 시작 간격 45초. 22:00 전후에는 소유자의 크롤 LaunchAgent가 `data/`를 갱신하므로 전체 테스트·측정을 피한다.

---

## 5. 작업 규칙 (앞 세션에서 배운 것 — 반드시)

1. **서브에이전트에게 "작은 단위로 자주 커밋"을 지시한다.** 세션이 끊기면 미커밋 작업물은 사라진다.
2. **worktree 기준 커밋을 확인한다.** 지시 첫 단계에 "`git log -1 --format=%h`가 지정 해시인지 확인, 아니면 `git reset --hard <해시>`"를 넣는다. worktree가 `origin/main`(`a46ce76`)에서 생성되는 사고가 반복됐다. 병합 전에 `git merge-base --is-ancestor`로 확인한다.
3. **병합은 리드가 `git cherry-pick`으로 하나씩 한다.** RED를 건너뛴 트랙은 새 테스트를 변경 전 코드에 돌려 실패를 확인한다.
4. **트랙이 다른 트랙의 기대값을 깨면**, 테스트를 고치기 전에 "왜 값이 바뀌었는지"를 수치로 설명하게 한다.
5. **`KnowledgeGraph()` 기본 생성은 로드 중 KG 파일을 다시 쓴다.** 스크립트·테스트는 `persist_path`=임시, `auto_save=False`로 한다. KG 조사는 JSON을 직접 읽는다.
6. **서브에이전트 셸 가드가 `eval` 문자열을 막을 수 있다.** Read/Edit 도구로 우회하게 한다.
7. **`* 2.py` 등 iCloud 충돌 사본은 수정·삭제 금지.** 소스 전수 검사 테스트에서 제외한다. `.git/refs` 안에도 사본이 생겨 `git fetch`를 깨뜨린 적이 있다. 그때는 저장소 밖으로 옮겨 보관한다.
8. **pre-commit ruff(v0.8.0)와 PATH ruff(0.14.x)가 포맷을 핑퐁한다.** 훅이 고친 파일은 다시 스테이징하고, 무관한 재포맷은 커밋하지 않는다. `mixed-line-ending --fix=lf` 때문에 CRLF 파일은 건드리면 전체가 LF로 바뀐다(실질 변경만 리뷰할 것).
9. **셸의 `python3`가 3.14를 가리킬 수 있다.** 테스트·평가는 `.venv/bin/python`으로 한다.
10. **측정은 세션이 끊겨도 살아남게 분리 실행한다**(macOS에는 `setsid`가 없음):
    ```
    nohup python3 -c "
    import os,subprocess,sys
    if os.fork(): sys.exit(0)
    os.setsid()
    subprocess.run(['zsh','<run_eval3.sh 절대경로>','<이름>','<데이터셋 절대경로>','<커밋>', 'FF_...=...'], stdout=open('<로그>','w'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    " >/dev/null 2>&1 &
    ```
11. **측정 전에는 스모크 5문항으로 새 경로가 실제로 도는지 확인한다.** 6단계에서 스모크가 평가 어댑터 결함과 경로 문제를 먼저 잡았다.
12. **A/A 흔들림이 있다.** 3회 측정에서 토큰 F1은 같은 코드끼리도 +0.024까지 흔들렸다. 판정은 (a) 대비와 "기준 6회"(재사용 기준선 + 새 OFF 실행) 대비를 함께 보고, 결론이 갈리면 "차이 없음"으로 읽는다.

---

## 6. 단계별 작업

### O0 — 측정 수리와 무비용 기준선 (API $0, O1과 병렬)

- **O0-A 지표 수리** (소유: `eval/metrics/l3_kg.py`, `eval/metrics/l4_ontology.py`, `eval/validators/ontology_validator.py`, `eval/runner.py`의 해당 부분, 관련 테스트)
  - L4 `constraint_violation_rate`가 `trace.rule_evaluation`의 발화 결과를 입력으로 받게 한다.
  - `type_consistency_rate`에 골드 엔티티 타입을 넘긴다. 골드에 타입이 없으면 등록부 기준 타입으로 하되, 그 사실을 리포트에 표시한다.
  - L3에 "골드 엣지 있는 문항만" 평균과 **술어별 recall**을 추가한다. 기존 필드는 유지한다(과거 비교용).
  - 재사용 기준선 리포트를 새 지표로 **재채점**한다(LLM 호출 없음, trace에서 계산). 재채점 스크립트는 `scripts/rescore_l3_l4.py`.
- **O0-B 무비용 온톨로지 점검 스크립트** `scripts/ontology_offline_checks.py`. 입력: KG JSON 사본, 골든 typed 시험지, (O1 이후) 등록부. 출력:
  1. 브랜드 인식 범위: KG 브랜드와 골든 엔티티 중 연결기가 인식하는 비율
  2. 골드 엣지 도달률: 술어별로 KG·런타임 방출로 표현 가능한 비율
  3. KG 정합성 위반 수: 도메인·범위, 대소문자 중복, 가짜 브랜드, 비대칭 대칭관계
  4. 골든 엔티티를 넣었을 때의 규칙 일치율 오프라인값(지난 기록: 연결기 25/32 vs 골드 29/32)
- **게이트**: 재사용 기준선의 새 지표값과 무비용 점검 기준값이 `docs/experiments/ontology_activation_2026-09.md`(신규)에 기록되어 있을 것. 테스트 통과.

### O1 — 온톨로지 원본과 로더 (API $0)

- 소유: `config/ontology/*`, `src/ontology/ontology.py`(신규), `tests/unit/ontology/test_ontology_core.py`(신규).
- **내용**:
  - **클래스**: Brand(하위: AmorepacificBrand, KBeautyBrand, 세그먼트 클래스 Luxury/Premium/Mass/…), CorporateGroup, Category(계층은 `category_hierarchy.json`), Product, Metric.
  - **술어 정의**: `ownedByGroup`(정식 이름, 별칭 `ownedBy`), `ownsBrand`(역관계), `siblingBrand`(대칭), `hasSegment`, `originatesFrom`, `acquiredIn`, `hasProduct`/`hasBrand`(역관계), `belongsToCategory`(도메인 Product), `rankedIn`(도메인 Brand), `hasSoS`/`hasHHI`/`hasPricePosition`(각각 분리, 수치는 `as_of` 필수), `competesWith`(대칭).
  - **브랜드 등록부**: `config/brands.json`, `tracked_competitors.json`, `entity_linker.KNOWN_BRANDS`, KG 브랜드 83개를 합쳐 만든다.
    - 각 항목: 정식 이름, 별칭(대소문자·기호 변형: `e.l.f.`/`elf`), 그룹, 세그먼트·티어(어휘 통일), 원산지.
    - 가짜 항목(`unknown`, `fresh`, `chi`)은 `is_placeholder: true`로 표시한다.
    - 원산지를 모르는 브랜드는 비워 두고 "모름"으로 처리한다. **추정으로 채우지 않는다.**
  - **로더 API**(모두 순수 함수, I/O는 로드 시 1회): `normalize_brand`, `is_a`, `instances_of(class)`, `brands_in_group`, `siblings`, `segment_of`, `origin_of`, `category_ancestors`/`descendants`, `predicate_spec`, `canonical_predicate`, `validate_triple`.
- **게이트**:
  - 폐포 계산 테스트: 전이·역관계·대칭, 닫힌 세계 판정(OE4), 포함 관계의 수치 합산 금지(OE3).
  - 등록부 자체 정합성 위반 0.
  - O0-B의 브랜드 인식 범위를 등록부 기준으로 다시 잰 값 기록.

### O2 — 인식·정규화 (API $0, O1 병합 후)

- 소유: `src/rag/entity_linker.py`, `src/rag/entity_tags.py`, `src/core/tool_registry.py`의 `resolve_entity`, 관련 테스트.
- **내용**:
  - 연결기 사전을 등록부에서 만든다. 하드코딩 목록은 제거하거나 등록부로 대체한다.
  - "아모레퍼시픽 브랜드", "K-Beauty", "프리미엄 브랜드" 같은 **클래스 언급**을 인식해 `classes` 엔티티로 넘긴다.
  - 대소문자·별칭을 정규화한다.
  - 스크레이퍼의 브랜드 오귀속(`amazon_scraper.py`)은 **고치지 않는다**. 매일 크롤에 영향을 주므로 범위 밖이며 FUTURE_WORK에 남긴다. 조회 쪽에서 `is_placeholder` 브랜드만 거른다.
- **게이트**:
  - 골든 L1 엔티티 연결 F1(오프라인 재계산)이 기준선 대비 개선.
  - 골든 미인식 브랜드 목록(IT Cosmetics, Jouer, Almay, Charlotte Tilbury, COVERGIRL 등) 해소를 확인.
  - 오프라인 규칙 일치율 재측정.

### O3 — 질의 경로 추론 (플래그 `ontology.use_class_reasoning`, 기본 OFF, O1 병합 후)

- 소유: `src/rag/hybrid_retriever.py`(`_query_knowledge_graph`·조회 범위 부분), `src/rag/evidence_adapters.py`, `src/rag/metric_facts.py`, `src/infrastructure/feature_flags.py`, `config/feature_flags.json`, 관련 테스트.
- **내용** (플래그 ON일 때):
  1. **그룹 전개**: 질의가 그룹이나 클래스를 가리키면 소속 브랜드 집합으로 넓힌다. `MetricFactsProvider`의 브랜드 3개 제한은 전개된 집합에 한해 늘리되 상한을 둔다(예: 12). 넘으면 잘랐다는 사실을 카드로 남긴다.
  2. **정적 사실 카드**(OE5): 세그먼트, 원산지, 그룹, 자매 브랜드, 인수 연도. 닫힌 세계 부정 판정 카드(OE4)도 싣는다.
  3. **읽을 때 술어 정식화**: `ownedByGroup`/`ownedBy`를 통일하고, `original_predicate`로 `hasPosition`을 다시 나눈다.
  4. **카테고리 포함 확장**(OE3): 조회 범위만 넓힌다. 수치 카드의 카테고리는 원래 값을 유지한다.
- **게이트**:
  - 무비용 골드 엣지 도달률(술어별)이 개선. 예: `ownedByGroup` 0/29 → ?, `hasSegment` 0/14 → ?
  - 스모크 5문항(그룹·세그먼트·원산지·포함·부정 문항 각 1)에서 새 카드가 실제로 프롬프트에 실리는지 확인.
  - 플래그 OFF에서 기존 테스트와 기존 출력이 불변(특성화 테스트).

### O4 — 규칙 입력 연결 (O3 병합 후, 같은 플래그)

- 소유: `src/ontology/rule_contracts.py`, `src/ontology/rules/*.py`의 입력 선언 부분, 관련 테스트.
- **내용**:
  - 소유 검증 규칙에 원산지·세그먼트·인수 입력을 연결한다(지금은 KG에 있는데 "입력 없음"으로 표시됨).
  - 가격 규칙에 세그먼트·티어를 연결한다.
  - `is_target`을 `laneige` 고정에서 "질의가 겨냥한 브랜드 또는 클래스"로 일반화할지 검토한다. **rule 골드의 정의를 바꾸면 안 된다.** 골드는 LANEIGE 기준이므로, 일반화는 골드 영향이 0임을 오프라인으로 확인한 경우에만 한다.
- **게이트**: rule 32문항 오프라인 일치율이 기준선보다 나빠지지 않음. 새로 발화 가능해진 규칙 수를 기록.

### O5 — KG 쓰기 정합성 (O1 병합 후, O2·O3와 병렬)

- 소유: `src/ontology/kg_enricher.py`, `src/ontology/kg_updater.py`, `src/ontology/knowledge_graph.py`의 쓰기 검증 부분, `scripts/migrate_kg_ontology.py`(신규), 관련 테스트.
- **내용**:
  - KG에 쓸 때 `validate_triple`과 정규화를 적용한다: 정식 술어, 브랜드 정식 이름, 도메인·범위, 가짜 브랜드 차단, 수치 `as_of`.
  - 타입 트리플이나 `entity_metadata.type`을 기록한다.
  - `hasPosition`에 합치지 않고 술어를 분리해 저장한다.
  - 마이그레이션 스크립트는 **입력 KG 사본 → 정리된 KG 사본**을 만들고, 변경 목록(추가·삭제·수정 건수, 사유별)을 출력한다. `--dry-run`을 기본값으로 한다.
  - 평가용 새 스냅샷 `eval_output/evidence-2026-09/eval_data_snapshot_ontology`를 태그 스냅샷 + 마이그레이션 결과로 만든다. 원본 스냅샷은 그대로 둔다(비교용).
- **게이트**:
  - 무비용 KG 정합성 위반 수가 기준선에서 0에 가깝게 줄어듦(남은 것은 사유와 함께 기록).
  - 매일 크롤 경로의 기존 테스트 통과.
  - **운영 KG 적용은 하지 않고, 적용 명령과 변경 목록을 최종 보고에서 소유자에게 제시한다.**

### O6 — OWL 모듈 정리·플래그 이름·문서 (O3 병합 후)

- 소유: OWL 관련 파일(OE8), `feature_flags.py`의 이름 변경 부분, 문서.
- **내용**:
  - OE8에 따라 삭제 또는 통합한다. 삭제 전에 호출처 확인표를 만든다.
  - OWL 내보내기 스크립트를 둔다.
  - 플래그 이름을 바로잡는다(OE7, 옛 이름 별칭 유지).
  - 검토 보고서 §2.1-6의 잘못된 서술을 정정한다: CLAUDE.md, AGENTS.md, `retrieval_strategy.py` 주석, `docs/portfolio/amore_architecture_evidence.md`(`[2026-09 사후]` 표시).
- **게이트**: 전체 테스트 통과. 삭제한 코드 목록과 호출처 확인표 기록.

### O7 — 측정과 판정 (예상 $3~4)

- **스냅샷 두 가지로 나눠 잰다**(효과 분리):
  1. **태그 스냅샷**(운영과 같은 KG): 코드 변경(O2·O3·O4) 효과
  2. **온톨로지 스냅샷**(O5 마이그레이션 KG): KG 정리까지 포함한 효과
- **구성** (모두 `run_eval3.sh`):

  | 구성 | 스냅샷 | 시험지 | 반복 |
  |---|---|---|---|
  | (a) 플래그 OFF | 태그 | 54문항(multihop+relation) | 3회 |
  | (b) 플래그 ON | 태그 | 54문항 | 3회 |
  | (c) 플래그 ON | 온톨로지 | 54문항 | 3회 |
  | rule on/off 영향 확인 | 태그 | rule 42문항 | 각 2회 |
  | 회귀 확인 (b 구성) | 태그 | 233문항 | 1회 |

- **판정 지표**:
  - 술어별 L3 recall(골드 엣지 있는 문항만)
  - L4 두 지표(O0에서 의미가 생긴 것)
  - 규칙 정답 일치율
  - L5 근거성·관련성·토큰 F1·수치 정확도
  - 그룹·세그먼트·원산지·부정 문항 부분집합의 통과 수
- **판정 기준**(지난 작업과 동일): 노이즈 기준(종합·검색 0.01, 근거성·관련성 0.03, 수치 0.05, 통과 5건)과 실행 간 폭 중 큰 값 이상 + 범위 비겹침. 이를 (a) 대비와 기준 6회 대비로 함께 본다.
  - 온톨로지 문항 부분집합에서 개선되고 다른 유형에 회귀가 없으면 `ontology.use_class_reasoning` 기본 ON.
  - 아니면 OFF를 유지하고 원인을 분류한다.
- **비용·지연 증가도 함께 기록한다.** 그룹 전개로 카드 수가 늘면 프롬프트 토큰이 는다. 5단계 한계인 "질의마다 카드 약 70장"을 악화시키는지 카드 수로 확인한다.
- **게이트**: 판정 결과와 근거가 결정표(`docs/plans/ontology-activation-decisions-2026-09.md`, 신규)에 있을 것.

---

## 7. 마무리

- **문서**:
  - 실험 문서 `docs/experiments/ontology_activation_2026-09.md`에 단계별 결과를 쓴다.
  - 결정표를 갱신한다.
  - FUTURE_WORK 해소 표시와 새 항목을 적는다.
  - CLAUDE.md·README·AGENTS.md·포트폴리오 근거 문서에 `[2026-09 사후]`로 반영한다.
- **전체 테스트 최종 1회**: 실행 전후 `data/` sha256을 비교한다. `chroma.sqlite3`의 `acquire_write` 테이블만 달라지는 현상은 알려진 것이다(S3-2).
- **최종 보고에 넣을 것**:
  - 단계별 커밋과 게이트 근거
  - 기준선 → 최종 표(평균·범위·판정)
  - 플래그 판정
  - 삭제한 코드
  - 운영 KG 적용 명령과 변경 목록(소유자 승인 대기)
  - 누적 비용
  - 남긴 항목과 이유

---

## 8. 하지 말 것

- push, 배포, Railway 설정 변경, 외부 크롤링
- 기존 baseline 파일 수정·삭제
- `laneige_golden_v2.jsonl`과 typed 시험지 **골드 수정**. 골드가 틀렸다고 판단되면 기록만 하고 사용자에게 묻는다.
- 운영 `data/` 수정(OE6)
- 비밀키·스프레드시트 ID 출력
- `* 2.py` 수정·삭제
- 원산지·그룹을 **추정으로 채우기**
- 평가 결과를 유리하게 다듬기
- 서비스 런타임에서 Java·외부 추론기 호출(OE2). 개발·검증 스크립트의 HermiT 사용은 허용(OE2-추가)

---

## 9. 새 세션 시작 문구 (복사해서 쓰기)

```
docs/plans/ontology-activation-plan-2026-09-18.md(이하 계획서)의 O0~O7과 §7 마무리를 끝까지 수행해라.
먼저 docs/analysis/ontology-review-2026-09-18.md(검토 보고서)와 계획서 §0~§5를 읽어라.
브랜치는 feat/ontology-activation-2026-09. `git log -1`이 계획서 커밋인지 확인하고 시작해라.
시작 전에 OE1(온톨로지 원본 형식: JSON 스키마 권장 vs OWL 파일)만 나에게 한 번 확인받고, 나머지 설계 OE2~OE9는 다시 묻지 말고 따를 것.
토큰을 효율적으로 써라: 구현은 서브에이전트에 목적·소유 파일·금지 파일·테스트·완료 조건을 주어 맡기고, 리드는 리뷰·병합·게이트 판정만 한다.
결과 보고는 한국어로, 쉬운 설명은 초등 5학년 수준으로.
```
