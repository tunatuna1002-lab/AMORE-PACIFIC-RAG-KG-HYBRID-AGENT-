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
| S0-6 | 0 | 서브에이전트 worktree 기준 커밋 사고 | 첫 묶음(0-A·0-B·0-C·0-D·0-E) worktree가 로컬 작업 브랜치가 아니라 `origin/main`(`a46ce76`, 2026-08-30, 작업 브랜치보다 28커밋 뒤)에서 생성됐다(reflog "Created from origin/main"). 0-D 완료 보고 리뷰에서 발견해 네 트랙은 `cf4c1ec`로 reset 후 재수행, 0-D는 cherry-pick 후 재검증하게 했다. 이후 모든 worktree 지시에 "시작 시 `git log -1`이 지정 해시인지 확인, 아니면 reset" 단계를 넣는다. 병합 전 리뷰에서 `git merge-base --is-ancestor <작업 브랜치 HEAD> <트랙 브랜치>`를 확인한다 | 1-A/1-B(`de90c05`)·2-A(`cf4c1ec`)는 올바른 기준에서 생성됨. 원인은 확인 못 함(생성 시점에 따라 달랐음) |
| S0-7 | 0 | 전체 테스트가 운영 `data/`를 수정함 (승인 범위 밖 수정이 **발생**) | 발견: `data/knowledge_graph.json` `saved_at` 2026-09-17T21:12:02(리드가 `tests/test_llm_integration.py`를 원본 저장소에서 실행한 시각). 이 테스트가 기본 경로 `KnowledgeGraph()`로 실제 KG를 로드해 `add_relation` 속성 병합 후 자동 저장한다. 트리플 수 3,500 불변, 오늘 생성된 트리플 0, 속성 `product_name·rank·category`만 가진 `LANEIGE hasProduct` 트리플 6개(생성 08-30·09-06 — 과거 전체 테스트 실행이 만든 것으로 보임, 대문자 `LANEIGE` 주어). 같은 실행류가 `data/llm_insight_result.json`·`orchestrator_state.json`·`latest_crawl_result.json`·`competitor_products.json`도 갱신. 변경 전 KG 사본이 없어 이번 실행이 바꾼 속성 값(rank 등)은 특정 불가. **조치**: 추가 수정 방지 — 트랙 0-F로 테스트 쓰기 격리, 그 전까지 전체 테스트는 `data/` 복사본이 있는 worktree에서만 실행. 테스트가 만든 KG 트리플의 정리는 승인 범위 밖이라 하지 않고 최종 보고에서 소유자에게 묻는다 | 지시서 §0 "`data/` 수정 금지"는 의도적 수정 금지이며, 발생 사실을 숨기지 않고 기록 |
| S1-1 | 1 | 생성 문항의 정답 수치 출처: 지표 테이블 vs raw_data 독립 재계산 | **지표 테이블(brand_metrics·market_metrics)을 정본**, raw 독립 계산은 교차 확인 값으로 문항에 남기고 두 출처의 **결론이 다른 문항은 제외**. 테이블에 값이 없어 시스템이 접근할 수 없는 수치로는 정답을 만들지 않는다 | 1-B가 2026-08-31 지표 테이블이 raw_data보다 먼저(08-30 18:06 UTC) 계산돼 값이 다름을 발견(lip_care HHI raw 0.0637 / 테이블 0.0681, lip_makeup 21/21 브랜드 SoS 불일치). 기존 골든 snapshot 문항은 모두 테이블 값(lg049 0.0681 등)이라 같은 사실에 두 정답이 섞이면 안 되고, 이 시험지는 "시스템의 기록 원천 증거로 규칙을 맞게 적용하는가"를 잰다. 테이블의 시점 불일치는 데이터 파이프라인 결함으로 FUTURE_WORK에 기록 |
| S2-1 | 2 | 기준선 이후 운영 `data/`가 바뀜 → 단계 간 비교 오염 위험 | **평가는 실행마다 지정 커밋의 별도 worktree + 동결 데이터 스냅샷 복사본에서만** 한다(`eval_output/evidence-2026-09/run_eval2.sh`). 스냅샷 = 기준선 실행 시점과 같은 해시의 DB(`7a0dce00`)·`dashboard_data.json`(`b57af9d4`)·KG(`876ea40f`)·Chroma(358). 실행 전후 KG 해시를 로그에 남긴다 | 22:00 소유자 LaunchAgent `com.amore.daily-crawl`(`scripts/daily_crawl.py`)이 22:03 DB·`dashboard_data.json`·`raw_products/2026-09-17.json`을 갱신(정상 운영 활동). 22:23 이 저장소 cwd에서 리드가 띄우지 않은 `pytest tests/` 프로세스(별도 세션으로 추정)가 실행돼 22:25 KG·`competitor_products.json`·외부 신호 파일을 갱신. 리드의 22:1x 부분 테스트 실행(tests/unit/rag·core·agents·integration·eval)도 격리 병합(0-F) 전이라 쓰기 원인일 수 있어 구분 불가. 기준선(21:37~21:59) 동안에는 DB 13:23·KG 21:12 이후 쓰기가 없어 기준선 데이터는 스냅샷과 같다. 스냅샷 원천: 게이트 worktree의 DB·dashboard 복사본(테스트가 바꾸지 않은 것을 sha256으로 확인), KG는 iCloud 충돌 사본으로 보이는 `data/knowledge_graph 3.json`(해시가 기준선 KG와 일치) |
