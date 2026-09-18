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
