# 증거 카드 · 규칙 추론 · ReAct 통합 — 결정표 (2026-09)

> 지시서: `docs/plans/evidence-react-ontology-kickoff-prompt-2026-09-17.md`(커밋 `de90c05`의 실행 지시서, 설계 E1~E11 확정)
> 결과 기록: `docs/experiments/evidence_pipeline_2026-09.md`
> 브랜치: `feat/evidence-react-ontology-2026-09` (시작 HEAD `de90c05`, push 안 함)
> 공모전 이후 작업이다. 여기의 어떤 결정도 공모전 당시 동작을 말하지 않는다.

## 사전 승인 (지시서 §0)

| 항목 | 값 |
|---|---|
| 유료 API 총 상한 | $40 (경보선 $32). 직전 작업 누적 ≈ $11.1은 포함하지 않음 |
| `data/chroma` 787청크 제거 | 승인 — 백업 후 제거 |
| 작업 브랜치 | `feat/evidence-react-ontology-2026-09` |

## 결정

| ID | 단계 | 쟁점 | 결정 | 근거 |
|---|---|---|---|---|
| S0-1 | 0 | `data/chroma` 복구 | `data/chroma_backup_2026-09-17-pre-restore/`로 전체 복사(1,145청크 확인) 후 `chroma_added_ids.json`의 787개만 삭제. 삭제 전 787개 전부 존재 확인, 삭제 후 `amore_docs`=358, 남은 358청크 메타데이터 키 6종 동일, 다른 컬렉션 `amore_docs_all_MiniLM_L6_v2`(82) 미변경 | 지시서 §0 승인 범위 |
| S0-2 | 0 | 초기화 읽기 전용(E9)과 Railway 배포 | `scripts/start.py`가 uvicorn 시작 전 같은 프로세스에서 색인 명령을 1회 실행한다 | Railway 컨테이너의 `./data/chroma`는 볼륨 밖이라 배포마다 비어 있고, 지금까지는 서버 초기화가 색인해 왔다. 초기화만 읽기 전용으로 바꾸면 배포 검색이 깨진다. 색인은 "전용 명령"에서만 한다는 E9를 지키면서 동작을 유지하는 방법 |
| S0-3 | 0 | 0-D(경로 관측)와 0-B(오류 가시화)가 모두 `eval/brain_adapter.py`·`eval/schemas.py`를 고쳐야 함 | 0-D는 코어(`query_graph.py`·`graph_state.py`·`Response`)만 먼저, 평가 리포트 연결은 0-B 병합 뒤 순차 | 지시서 §4 "같은 파일을 두 트랙이 동시에 고치지 않는다" |
| S0-4 | 0 | 0-C(비용)와 0-B가 `eval/schemas.py`·`eval/report.py`를 함께 고침 | 0-C는 비용 관련 클래스·함수만 수정하도록 제한하고 병합 시 충돌을 리드가 해소 | 수정 범위가 겹치지 않는 영역. 순차로 돌리면 0단계가 길어짐 |
| S0-5 | 0 | 실제 API를 부르는 테스트 재발 방지(§5 "LLM 호출 경로에 가드") | `tests/conftest.py` 전역 가드(더미 키 + 닫힌 base URL) — 0-E | 개별 테스트 수정만으로는 새 누수를 막지 못함. 새는 호출이 과금 없이 즉시 실패하게 |
| S2-0 | 2(사전) | SoS 스케일 | DB `brand_metrics.sos`는 **퍼센트(0~100)**(예: 2026-08-31 skin_care MEDICUBE 13.54), `market_metrics.hhi`는 **0~1**(lip_care 0.0681). 규칙 임계(`sos_above(0.15)`)는 0~1. 증거 카드 정본은 0~1이며 변환은 어댑터 한 곳에서 | 2026-09-17 읽기 전용 조회. 전 기간 일관성은 1-B 트랙이 확인 |
