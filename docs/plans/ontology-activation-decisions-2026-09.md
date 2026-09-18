# 온톨로지 작동 — 결정표 (2026-09) [2026-09 사후]

> 지시서: `docs/plans/ontology-activation-plan-2026-09-18.md`(설계 OE1~OE10)
> 근거: `docs/analysis/ontology-review-2026-09-18.md`
> 결과 기록: `docs/experiments/ontology_activation_2026-09.md`
> 브랜치: `feat/ontology-activation-2026-09` (시작 HEAD `e7be5ed`, push 안 함)
> 공모전 이후 작업이다. 여기의 어떤 결정도 공모전 당시 동작을 말하지 않는다.

## 사전 승인 (계획서 §0)

| 항목 | 값 |
|---|---|
| 유료 API 상한 | $10 (경보선 $8). 지난 작업 누적 ≈ $26.1은 포함하지 않음 |
| 작업 브랜치 | `feat/ontology-activation-2026-09` |

## 결정

| ID | 단계 | 쟁점 | 결정 | 근거 |
|---|---|---|---|---|
| OA-1 | 시작 | OE1 온톨로지 원본 형식 | **JSON 스키마 + Python 로더** (사용자 확인 2026-09-18. 첫 선택은 "OWL 파일"이었으나 바로 "잘못 눌렀다, JSON으로"로 정정) | 계획서 OE1 권장안. OE8은 JSON 쪽(OWL 모듈 삭제 + OWL 내보내기·Pellet 검증 스크립트)을 따른다 |
| OA-2 | O1 | 카테고리 계층의 원본 | `config/category_hierarchy.json`을 그대로 원본으로 쓰고 `config/ontology/`에는 복사하지 않는다. 로더가 참조해 Category 개체·포함 관계를 만든다 | 계획서 OE1 표 "categories: 기존 파일 참조". 크롤러 URL 설정도 같은 파일이라 두 벌을 만들면 어긋난다 |
| OA-3 | O1 | 정의 클래스(AmorepacificBrand·KBeautyBrand·세그먼트 클래스)의 표현 | `schema.json`의 클래스에 `defined_by: {predicate, value}`(OWL `hasValue` 제한과 같은 뜻)를 둔다. Python 폐포는 이 형식과 `subClassOf` 전이·`inverseOf`·`symmetric`·`transitive`만 계산하고, 그 밖의 공리 형식이 원본에 있으면 로드 시 오류를 낸다 | 폐포가 원본을 "전부" 이해한다는 것을 보장해야 Pellet 교차 검증(OE2-추가)이 의미가 있다 |
| OA-4 | O1 | 기존 `.owl`의 SWRL 규칙 3개(SoS 구간 분류) | 옮기지 않는다. SoS 구간 분류는 Python 규칙(`market_rules`)이 이미 하고, 날짜가 있는 수치는 정적 온톨로지 원본에 넣지 않는다 | 검토 보고서 §5-1. 온톨로지 원본은 정적 정의 사실만 담는다(OE5) |
| OA-5 | O1 | 등록부 작성 판단 (O1 트랙 보고) | 등록부 119개(실제 116 + 가짜 3 `unknown`·`fresh`·`chi`는 `PlaceholderBrand`), 그룹 31(전부 아모레퍼시픽)·세그먼트 51·원산지 28·인수 연도 1(COSRX 2024). KG의 AP 브랜드 30개 "Korea" 원산지는 `kg_updater.load_brand_ownership`의 **기본값**이라 원본 진술이 아니므로 쓰지 않음 → IOPE·primera 등은 원산지 모름(K-Beauty 클래스에 안 들어감). `ownedBy`는 `ownedByGroup`의 별칭(`relations.py` 주석은 Product→Brand라 충돌 — 스키마에 기록). 세그먼트 "Makeup" id는 카테고리 `makeup`과 겹쳐 `makeup_line` | 계획서 §8 "원산지·그룹을 추정으로 채우지 않는다". Pellet 교차 검증: 클래스 15·개체 160·불일치 0(0.47초). 대조 실험으로 Pellet 없이 비교하면 불일치 9건이 나와 비교가 실제로 무언가를 검사함을 확인 |
| OA-6 | O2·O3 | 연결기(O2)의 새 인식도 플래그 뒤에 둘지 | **둔다.** `ontology.use_class_reasoning` OFF이면 `extract_entities` 출력이 지금과 같다. ON이면 등록부 사전·클래스/그룹 언급(`classes`·`groups` 키)·가짜 브랜드 제외가 켜진다 | O7 (a)/(b) 비교가 "O2·O3·O4 코드 효과"를 재려면 OFF가 현재 동작과 같아야 한다(OE7) |
| OA-7 | O5 | KG 쓰기 검증을 켜는 방식 | 문자열 플래그 `kg.write_validation`(`off`/`warn`/`enforce`), 기본 **`warn`**(위반을 로그로만 남기고 저장 내용은 그대로). 술어 분리 저장·가짜 브랜드 차단은 `enforce`에서만 | 매일 크롤이 운영 KG에 쓰는 경로라 기본값이 저장 내용을 바꾸면 OE6(운영 데이터 불변)과 어긋난다. 정리는 마이그레이션 스크립트 + 평가 스냅샷 사본에서 한다 |
| OA-8 | O5 | 마이그레이션에서 날짜 없는 수치 엣지 | `as_of`를 만들어 넣지 않는다. 333건(hasSoS 151·hasPricePosition 116·hasHHI 66)은 "날짜 없음"으로 남긴다. `created_at`은 엣지가 다시 쓰여도 값만 바뀌므로 값의 날짜가 아니다 | 계획서 §0 "추측은 추측이라고", §8 결과 다듬기 금지. `enforce` 전환 전 `dashboard_exporter`가 `as_of`를 넘겨야 함(FUTURE_WORK) |
| OA-9 | O5 | KG 정식 브랜드 표기 | 등록부 표시 이름의 소문자(`laneige`, `e.l.f.`). 등록부 id(`elf`)가 아님 | 질의 경로는 연결기가 낸 문자열의 소문자 변형을 찾는다. id를 쓰면 기존 조회가 못 찾는다. 소유 밖 파일 `relations.py`에 enum 3개(`hasSoS`·`hasHHI`·`hasPricePosition`) 추가는 분리 술어 KG를 읽기 위해 필요해 승인 |
| OA-10 | O7 | `ontology.use_class_reasoning` 기본값 | **ON** (`config/feature_flags.json`만 `true`, 코드 기본값은 키가 없을 때 False 유지) | 계획서 O7 기준 "온톨로지 문항 부분집합 개선 + 다른 유형 회귀 없음" 충족. 54문항 3회((a) OFF vs (b) ON, 기준 6회 대비도 같은 방향): 종합 0.692→0.745, L3 골드 엣지 recall(정식화) 0.301→0.582, 근거성 0.867→0.968, 관련성 0.912→0.960, 토큰 F1 0.199→0.225, 수치 정확도 0.601→0.733 모두 有. 온톨로지 부분집합 25문항 종합 0.697→0.806 有, 나머지 29문항 無(회귀 없음). rule 42문항 규칙 일치율 0.781→0.906 有(반대 방향 0문항). 233문항 1회에서 노이즈를 넘은 하락 3건은 확인 측정으로 해소: numeric 31문항 OFF×2·ON×2에서 토큰 F1 0.196→0.206, 수치 정확도 −0.03(노이즈 0.05 이내) 無. multihop 관련성은 54문항 3회에서 −0.027 無. 유형 외 문항 수치 정확도 −0.083은 수치 채점 문항이 소수라 판정 보류(기록) |
| OA-11 | O7 | (c) KG 마이그레이션 스냅샷 효과 | **운영 KG 적용을 권하지 않음(효과 없음)**. (c)와 (b)는 카드 수 외에 차이 없음 | 질의 경로가 읽을 때 정식화(O3)를 하므로 저장 형식 정리의 추가 효과가 이번 측정에서 보이지 않았다. 적용 명령은 실험 기록 §O5에 남기고 소유자 판단에 맡김 |
| OA-12 | O7 | 카드 수·비용 증가 | 기록하고 수용 | 프롬프트 카드 평균 58.1→69.9(5단계 한계 약 70장에 닿음), 70장 초과 문항 54문항 중 25~26→32~33, 233문항 중 102→131. 파이프라인 비용 +9%, 프롬프트 토큰 +417/문항(54), 지연 차이 無(54), 233은 +0.8초. 카드 상한 조정은 FUTURE_WORK |
| OA-13 | O7 | L4 규칙 위반율 상승(0.116→0.228) | 원인 수리 후 재측정 없이 오프라인 재채점 | 원인: `reasoner.py` `InferenceRule.apply()`가 결측 입력의 빈 문자열을 `related_entities`로 옮김(가격 규칙 등). `3421ef8`에서 빈 문자열·가짜 브랜드 제거. o7r-on-run1 트레이스 재채점 0.228→0.0, 발화 규칙·결론 130건 동일 |
