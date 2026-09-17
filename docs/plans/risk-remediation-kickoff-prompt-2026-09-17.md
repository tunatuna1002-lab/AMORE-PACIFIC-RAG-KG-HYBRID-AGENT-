# 위험 지점 결정 확정 + 보완 세션 킥오프 프롬프트

> 새 Claude Code 세션에 아래 블록을 그대로 붙여넣기.
> 배경: `docs/portfolio/amore_architecture_evidence.md`(2026-09-17 근거 감사)가 찾은 위험 지점 4종을, 과거에 내렸던 결정을 회수해 확정한 뒤 실제로 보완한다.
> 붙여넣기 전에 고칠 곳: **API 비용 상한($5)**, 필요하면 브랜치 이름.

---

docs/portfolio/amore_architecture_evidence.md 를 먼저 읽어라. 이 문서가 지적한 위험 지점 4종을 보완하려 한다. 예전에 Claude와 함께 이 문제들의 처리 방향을 정리한 적이 있는데 무엇으로 결정했는지 내가 기억하지 못한다. 이번 세션의 목표는 (1) 과거 결정을 회수하고 (2) 빈 곳을 채워 결정을 확정·기록하고 (3) 확정된 대로 실행하는 것이다.

대상 위험 지점:
- R1. ReAct·OWL 미연결 — `src/core/brain.py:380`의 존재하지 않는 import 경로, `OWLRetrievalStrategy` 생성자에 없는 `docs_path` 인자(`brain.py:326-330`, `src/infrastructure/container.py:170-174`), ReAct `ALLOWED_ACTIONS` 중 실행기 미등록 3종(`brain.py:903-907`에 5종만 등록), `src/core/react_agent.py:244-245`가 채워지지 않는 `observation`으로 최종 답을 만드는 문제, 예외를 debug/info 로그로 삼키는 침묵 실패.
- R2. 평가 경로 불일치 — 평가 하니스는 v1 `HybridChatbotAgent`를 측정(`eval/cli.py:717-721`)하는데 대시보드는 v4 Brain 경로(`/api/v4/chat/stream`)를 쓴다.
- R3. KG·온톨로지 효과 미입증 — 유효 ablation이 `requires_kg=false` 30문항·1회뿐이다.
- R4. 문서-코드 불일치 — CLAUDE.md, README.md, AGENTS.md, PORTFOLIO_FACTS.md, `src/api/routes/chat.py`·`src/api/dashboard_api.py`의 docstring.

## Phase 0 — 과거 결정 회수와 확정 (여기서 한 번 멈춘다)

1. 과거 기록을 찾는다. 찾을 곳:
   - 저장소: `docs/plans/`, `docs/eval/`, `docs/experiments/`, `docs/dev/FUTURE_WORK.md`, `docs/portfolio/`, `PORTFOLIO_FACTS.md`, `.sisyphus/`, `.omc/`(notepad·project-memory·plans), 그리고 미추적 파일 `docs/plans/refactoring-plan-2026-08-31.md`, `docs/plans/refactoring-kickoff-prompt.md`.
   - 이전 Claude 세션: `~/.claude/projects/-Users-leedongwon-Desktop-AMORE-RAG-ONTOLOGY-HYBRID-AGENT/` 아래의 세션 기록(*.jsonl)과 `memory/`. 세션 기록은 용량이 크니 통째로 읽지 말고 키워드 grep으로 해당 대목만 뽑아라. 키워드 예: `react_agent`, `ReAct`, `OWLRetrievalStrategy`, `docs_path`, `미연결`, `requires_kg`, `no-kg`, `v4 경로`, `HybridChatbotAgent`, `PORTFOLIO_FACTS`, `결정`, `확정`, `기각`, `보류`. 세션 검색 도구(`search_session_transcripts`)가 있으면 그것을 먼저 써라.
   - `refactoring-kickoff-prompt.md`에 "`src/rag/retrieval_strategy.py`와 그 테스트는 다른 세션이 작업 중"이라는 문구가 있다. 그 작업이 어디까지 갔는지(커밋됐는지, 브랜치·stash·미추적 파일 `src/rag/retrieval_strategy 2.py`에 남았는지) 확인하고 R1의 OWL 수리와 충돌하지 않게 해라. `* 2.py` 파일은 읽기만 하고 수정·삭제하지 마라.
2. 찾은 것을 결정표로 정리한다. 열: 결정 ID | 쟁점 | 과거에 정한 것(출처: 파일:줄 또는 세션 날짜) | 출처를 못 찾음 여부 | 이번 권고안 | 근거. 과거 기록끼리 충돌하면 둘 다 적고 최신 것을 표시해라. 기록에 없는 것을 과거 결정처럼 쓰지 마라.
3. 이미 확정된 것: **R1은 "삭제"나 "서술만 교정"이 아니라 실제로 연결·수리한다.** 나머지 쟁점은 아래 권고안을 기본값으로 두되 과거 기록이 다르면 그 차이를 표에 드러내라.
   - D1(R1 세부): 수리 후 ReAct·OWL을 기본 ON으로 둘지. 권고 — 피처 플래그 뒤에 두고, 테스트 통과 + 평가에서 회귀 없음이 확인되면 ON, 아니면 OFF로 두고 문서에 "연결됨, 기본 비활성"으로 적는다.
   - D2(R1 세부): ReAct 도구 중 실행기가 없는 `query_data`·`query_knowledge_graph`·`calculate_metrics`를 구현해 등록할지, `ALLOWED_ACTIONS`를 등록된 도구에 맞출지. 권고 — 읽기 전용인 앞의 둘은 구현(SQLite 읽기 전용 조회, KG 조회), `calculate_metrics`는 기존 `MetricCalculator` 호출로 얇게 감싼다. 크롤 같은 부작용 도구는 추가하지 않는다.
   - D3(R2): 평가 대상 경로. 권고 — eval에 v4 Brain 경로 어댑터를 추가해 `--target v1|v4`로 고르게 하고, 이후 기준선은 v4로 새로 시작한다(v1 기준선과 직접 비교하지 않는다). v1/v4 이중 경로 자체의 통합은 이번 범위 밖.
   - D4(R3): 실험 설계. 권고 — `laneige_golden_v2.jsonl`의 `requires_kg=true` 문항 전체, 구성 3종(full / KG off / 규칙 추론 off), 구성당 3회(실행 간 분산: 통과 수 ±4, 근거성 ±0.023이 이미 측정돼 있다), D3에서 정한 경로로 측정. 결과가 무효과나 음의 효과여도 그대로 기록한다.
   - D5(R4): PORTFOLIO_FACTS.md 처리. 권고 — 상단에 "기준 커밋 2b04aa1, 이후 내용은 docs/portfolio/amore_architecture_evidence.md가 대체" 고지를 달고 본문은 보존. CLAUDE.md·README·AGENTS.md·docstring은 현재 코드에 맞게 교정.
4. 결정표를 `docs/plans/risk-remediation-decisions-2026-09-17.md`로 저장하고, **표를 나에게 보여 준 뒤 멈춰서 확정을 받아라.** 이 지점이 이번 세션에서 유일하게 멈추는 곳이다. 내가 확정하면 그 문서에 확정 일시와 최종값을 기록하고 Phase 1부터 끝까지 진행해라.

## Phase 1 — R1 연결·수리

- 현재 브랜치에서 새 브랜치(`fix/wire-react-owl-2026-09`)를 만들어 작업한다.
- 먼저 "지금은 실패하는" 테스트를 쓴다: Brain 초기화 후 `_react_agent`가 None이 아닌지, OWL 전략이 생성되는지, 복잡 질의가 실제로 ReAct 경로를 타는지, ReAct의 `final_answer`가 빈 문자열이 아닌지, 등록되지 않은 도구를 고르면 어떻게 되는지. mock으로 버그를 가리는 테스트를 만들지 마라 — 이 저장소는 `1cd4307`에서 "테스트가 버그를 정상으로 박제"한 전례가 있다. LLM 호출만 가짜로 두고 import·생성자·배선은 실제 객체로 검증해라.
- 침묵 실패를 없앤다: 선택 컴포넌트 초기화 실패는 warning 이상으로 남기고, `/api/v4/brain/status`(또는 health) 응답에 react·owl 활성 여부를 노출해라.
- 수리 후 `src/api/routes/chat.py`의 docstring("모든 판단을 LLM이 수행", "ReAct + OWL 지원")을 실제 동작에 맞게 고쳐라.

## Phase 2 — R2 평가 경로

- D3 확정안대로 구현한다. v4 어댑터는 SSE가 아닌 `brain.process_query` 비스트림 경로를 호출하고, 기존 지표가 요구하는 트레이스(검색 청크, KG 엣지, 추론 결과)를 v4 컨텍스트에서 뽑아 같은 스키마로 넘겨라. 요청별 컨텍스트를 결과에 동봉하는 기존 방식(`5b64536`)을 따라 동시 실행 오염을 피해라.
- 먼저 5문항 스모크로 트레이스가 채워지는지 확인한 뒤 전량 1회를 돌려 v4 기준선을 `eval/baselines/`에 저장한다.

## Phase 3 — R3 KG 효과 실험

- D4 확정안대로 실행한다. 실행 전에 예상 비용을 계산해 로그에 남겨라(참고: 172문항 1회 ≈ $0.38).
- 결과 문서 `docs/experiments/kg_ablation_2026-09.md`에는 평가일, 대상 커밋, 데이터셋·문항 수, 구성, 실행 횟수, 지표별 평균·범위, 실행 간 분산 대비 유의한지, 한계를 적는다. 분산 범위 안의 차이는 "차이 없음"으로 적어라. ReAct·OWL을 켠 구성도 비용 상한 안에서 가능하면 1종 추가해라.

## Phase 4 — R4 문서 교정

- D5 확정안대로. 최소 교정 목록: 임베딩 모델(실제는 OpenAI `text-embedding-3-small`), reranker 기본 OFF, 골든셋 172문항, KG 트리플 수(실측), 커버리지 게이트(`fail_under = 0`), `AUTO_START_SCHEDULER` 기본 false, CLAUDE.md 구조도의 없는 파일(`src/core/batch_workflow.py`, `true_hybrid_insight_agent.py`, `crawl_workflow.py`, `config/competitors.json`), Sheets/SQLite는 "교체"가 아니라 병행. 수치는 직접 세어서 적어라.
- 마지막에 `docs/portfolio/amore_architecture_evidence.md`를 갱신한다: §0·§3.3·§3.4·§5.5·§7·§10에서 이번 작업으로 바뀐 사실을 고치되, **수정 시점이 2026-09이고 공모전 이후 사후 분석에서 발견해 고친 것임을 명시**해라. 공모전 당시에 동작했던 것처럼 읽히는 문장을 만들지 마라.

## 작업 규칙

- 테스트는 `.venv/bin/python -m pytest`(없으면 `python3 -m pytest`). Phase마다 관련 테스트를 돌리고 Phase 단위로 커밋한다. 커밋 메시지는 기존 컨벤션(`fix(scope): 한글 요약` + 본문). 전체 테스트는 시작 시 기준선 1회, 전체 완료 후 1회만.
- **유료 API 총 상한은 $5다.** 누적 비용을 추적하고, 상한의 80%에 닿으면 남은 실험을 줄여 상한 안에서 끝내라. 상한 때문에 축소한 내용은 보고서에 적어라.
- 하지 말 것: push, 배포, Railway 설정 변경, 외부 크롤링, 기존 baseline 파일 수정·삭제, `data/`의 DB·KG 파일 직접 수정, 비밀키·스프레드시트 ID 출력.
- 이번 범위 밖에서 발견한 문제는 고치지 말고 `docs/dev/FUTURE_WORK.md`에 한 줄만 추가해라.

요청한 것을 요청한 범위대로 수행해라. Phase 0의 확정 지점 외에는 멈추지 말고, 모호한 부분은 신중한 동료가 하듯 스스로 판단하되 해석에 따라 결과물이 실질적으로 달라질 때만 질문해라. 계획이 잘못됐거나 더 나은 방법이 있다고 판단되면 한 문장으로 밝히고 요청받은 대로 계속 진행해라 — 조용히 범위를 좁히거나 넓히지 마라. 전체를 완료해라. 정말 끝낼 수 없는 항목이 있으면 나머지를 다 한 뒤 무엇이 왜 빠졌는지 명시해라.

서브에이전트는 과거 세션 기록 검색처럼 독립적이고 규모가 큰 탐색에만 써라. 파일 몇 개 읽기나 몇 군데 수정은 직접 해라.

문서는 과제에 필요한 만큼만 써라. 실험 결과를 유리하게 보이도록 다듬지 마라 — 이 산출물은 면접에서 검증당할 근거로 쓴다.

완료 후 최종 보고에는 다음만 담아라: 확정된 결정표(과거 결정을 찾은 것/못 찾은 것 구분), Phase별 커밋 해시, R1 수리 전후의 동작 차이와 그것을 증명하는 테스트 이름, v4 기준선 수치, KG 실험 결과와 해석 한계, 누적 API 비용, 남긴 항목.
