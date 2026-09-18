# 증거 카드 · 규칙 추론 · ReAct 통합 — 이어가기 지시서 #2 (2026-09-18 07:10)

> 새 Claude Code 세션에 `---` 아래를 그대로 붙여넣기. 0~5단계는 끝났고 **6단계와 마무리**만 남았다.

---

`docs/plans/evidence-react-ontology-kickoff-prompt-2026-09-17.md`(이하 **지시서**)의 6단계와 §7 마무리를 끝까지 수행해라. 지시서의 원칙(§0 정확함 > 속도, §4 병렬 규칙, §5 작업 규칙, §8 하지 말 것)과 확정 설계 E1~E11은 그대로 유효하다. 다시 묻지 말고 따를 것.

**토큰을 효율적으로 써라.** 큰 문서는 필요한 절만 읽고, 구현은 서브에이전트에 목적·소유 파일·금지 파일·테스트·완료 조건을 짧고 정확하게 주어 맡긴다. 리드는 diff 리뷰·검증·병합·게이트 판정만 한다. 이미 확인된 사실(아래 §2)은 재조사하지 말 것. 결과 보고는 한국어로.

## 1. 먼저 읽을 것 (이것만)

1. 지시서 §0, §2 노이즈 기준, §3의 E7·E8, §6의 6단계, §7, §8
2. `docs/experiments/evidence_pipeline_2026-09.md`의 **3·4·5단계 절**(0~2단계는 읽지 않아도 된다)
3. `docs/plans/evidence-react-ontology-decisions-2026-09.md`의 S3-1·S4-1·S4-2·S4-3·S5-1
4. `eval_output/evidence-2026-09/notes/5b_confidence_calibration.md`(신뢰도 보정 근거·한계)

## 2. 현재 상태 (2026-09-18 07:10 KST)

- 브랜치 `feat/evidence-react-ontology-2026-09`, HEAD `8a9de37`. **push 안 함**(지시서 §8).
- 전체 테스트(`ba714eb`): **6,004 passed / 7 skipped / 0 failed**, 커버리지 69.10%.
- 0~5단계 트랙 전부 병합 완료. 단계별 결과·게이트 판정은 실험 문서에 기록돼 있다.
- **누적 유료 API 비용 ≈ $20.0** (상한 $40, 경보선 $32). 장부 `eval_output/risk-remediation-2026-09-17/cost_ledger.md`.
- 운영 `data/`는 건드리지 않았다. 테스트가 만든 KG 트리플 20개(hasProduct 6 + competesWith 14)는 사용자 승인으로 이미 삭제됨(결정 S2-2, 3,500→3,480).

### 지금 시스템이 하는 일 (병합된 것)

| 영역 | 상태 |
|---|---|
| 증거 카드 | KG 관계·DB 수치·문서 청크·규칙 추론·도구 관찰이 한 모델(`src/domain/entities/evidence.py`). 답변 프롬프트·judge 컨텍스트·출처·평가 트레이스가 이 카드만 읽는다 |
| 규칙 추론 | 카드 입력으로 발화(`src/ontology/rule_contracts.py`, `HybridRetriever._evaluate_rules`). 발화율 0.568, 규칙 정답 일치율 0.779(기준선 0.469) |
| 수치 검증 | `src/core/numeric_verifier.py`, 플래그 `response.numeric_verification_mode` **기본 annotate**(답변 불변, 기록만) |
| 도구 | 읽기 전용 5종 레지스트리 `src/core/tool_registry.py`(resolve_entity·kg_neighbors·get_metrics·apply_rules·search_docs). DecisionMaker·ReAct 공용, 네이티브 function calling |
| 색인 | `python -m src.rag.build_index`(`--retag` = 메타데이터만, 무비용). 조회는 읽기 전용. 엔티티 태그는 **가산점**(필터 아님) |
| 분기 | QueryGraph 하나. 스트리밍도 같은 그래프(`brain.process_query_stream`) |
| 신뢰도 | 적합도 기반 `0.60×엔티티 충족도 + 0.40×증거 종류 충족 ± 0.05×검색 분포`, 임계 0.95/0.61/0.60 |
| 라우터 | 홉 수 기반 `src/core/router.py`. 2홉 이상 → ReAct. holdout 18문항 경로 18/18 |
| ReAct | function calling, 토큰 예산 기본 12,000, 플래그 `agents.use_react_agent` **기본 OFF**, 그림자 모드 `agents.react_shadow_mode` **기본 OFF** |
| 삭제 | OWL 검색 전략·`use_owl_strategy`, `unified_reasoner.py`, `llm_orchestrator.py`, `query_processor.py`, SPARQL 계층, `src/core/tools.py` |

### 5단계 게이트에서 남은 사실 (6단계 판정에 영향)

- route 분포: direct 185 / decide 31 / clarify 16 (기준선 direct 232). 홉 분포 1홉 178·2홉 44·3홉 10.
- **신뢰도 HIGH 실패율 0.75 vs LOW 0.74 — 게이트 조건 미충족**을 그대로 기록했다. 원인은 임계값이 아니라 **증거 선별**(233문항 중 184문항이 엔티티·종류 충족도 만점, 질의마다 카드 ~70장이 거의 같게 실림). 6단계에서 이 한계를 다시 확인하되, 증거 선별 개선은 범위 밖으로 두고 FUTURE_WORK에 남길지 판단해라.

## 3. 평가 실행 방법 (그대로 따를 것)

- **4단계 이후 측정은 `eval_output/evidence-2026-09/run_eval3.sh`만 쓴다** (태그 스냅샷 `eval_data_snapshot_tagged` 사용 — 결정 S4-3). 0~3단계 비교용 `run_eval2.sh`(태그 없는 `_base`)와 섞지 마라.
- 사용법: `zsh eval_output/evidence-2026-09/run_eval3.sh <이름> <데이터셋> <커밋> [ENV=VAL ...]`. 추적되지 않는 데이터셋은 절대경로로. 실행마다 지정 커밋의 worktree + 동결 데이터 사본이 만들어진다.
- 동시 3개까지, 시작 간격 45초. **세션이 끊겨도 살아남게 분리 실행할 것**:
  ```
  nohup python3 -c "
  import os,subprocess,sys
  if os.fork(): sys.exit(0)
  os.setsid()
  subprocess.run(['zsh','<스크립트 경로>'], stdout=open('<로그>','w'), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
  " >/dev/null 2>&1 &
  ```
  (macOS에는 `setsid` 명령이 없다. 앞선 세션에서 측정 3회가 세션 종료와 함께 죽어 약 $2를 버렸다.)
- 유형별 재집계: `.venv/bin/python scripts/typed_eval_summary.py --config a=<report.json>,... --config b=...`. 전체 233문항 시험지가 아닌 부분 시험지 리포트는 이 스크립트가 거부하므로, 부분 집합 비교는 리포트를 직접 읽어 집계해라(3단계 규칙 on/off 비교 때 그렇게 했다).
- 22:00 전후는 소유자 크롤 LaunchAgent가 `data/`를 갱신하니 전체 테스트·측정을 피해라.

## 4. 할 일

### 4-1. 6단계 — ReAct 비교와 켜기 판정 (예상 $6~8)

지시서 §6 6단계 그대로. 측정 대상은 ④multihop 27문항 + ②relation 27문항(`eval/data/golden/typed/multihop.jsonl`, `relation.jsonl` — 절대경로로 넘길 것).

세 구성 × 각 3회:
- (a) 파이프라인만: 기본 플래그
- (b) 그림자 모드: `FF_AGENTS_REACT_SHADOW_MODE=true`
- (c) ReAct ON: `FF_AGENTS_USE_REACT_AGENT=true`

측정 전에 스모크 5문항으로 (b)·(c) 경로가 실제로 도는지 확인해라(앞 단계에서 스모크가 문제를 먼저 잡아 준 적이 있다). 그림자 모드 토큰은 `react_shadow.token_usage`에 따로 있으니 파이프라인 비용과 분리해 집계해라 — **이 집계는 아직 평가 리포트에 연결돼 있지 않다. 리드가 작은 트랙으로 연결하거나, 리포트의 `trace.route_trace.react_shadow`를 직접 읽어 집계해라.**

판정(지시서 §2 노이즈 기준: 종합·검색 0.01, 근거성·관련성 0.03, 수치 정확도 0.05, 통과 수 5건, 그리고 실행 간 폭 중 큰 값 + 범위 비겹침):
- (c)가 (a) 대비 근거성·토큰 F1·수치 정확도·규칙 일치율에서 기준 이상 개선되고 다른 유형 회귀가 없으면 `agents.use_react_agent` 기본 ON, 아니면 OFF 유지 + 원인 분류(도구 선택 실수·단계 초과·인용 누락 등 로그 기준).
- 비용·지연 증가를 반드시 함께 기록.
- **E8 수치 검증기 enforce 기본값 여부도 여기서 판정**한다. 근거: 5단계 실측에서 검사 수치 1,963개 중 verified 609 / mismatch 148 / no_citation 1,205(그중 1,217건은 인용만 없을 뿐 다른 프롬프트 카드에는 있던 값) / unknown_card 1. enforce로 올리면 mismatch·unknown_card가 "확인되지 않음"으로 바뀌는데, 그중에는 카드 값으로 **계산한** 수치(차이·배수·계단)가 섞여 있다. 표본을 직접 읽고(리포트 `trace.numeric_verification.details`) 계산값 비율을 세어 판정 근거로 삼아라.

비용이 경보선($32)에 닿으면 지시서 §3 축소 순서를 적용하고 보고만 해라(① 6단계 반복 3회→2회 ...).

### 4-2. 마무리 (지시서 §7)

- `docs/experiments/evidence_pipeline_2026-09.md`에 6단계 절 추가, `docs/plans/evidence-react-ontology-decisions-2026-09.md`에 6단계 결정(ReAct 기본값, enforce 기본값) 기록.
- 문서 갱신은 대부분 끝났다(커밋 `97a751b`~`ba714eb`: CLAUDE.md·README.md·AGENTS.md·docs/architecture·FUTURE_WORK 해소 표시·포트폴리오 근거 문서 `[2026-09-18 사후]`). **남은 것**: 6단계 결과 반영, 그리고 문서 트랙이 범위 밖으로 남긴 항목(`docs/architecture.md`·`docs/SYSTEM_ARCHITECTURE.md`의 이전부터 있던 낡은 서술, `docs/architecture/LLM_ORCHESTRATOR_DESIGN.md`, `docs/CORE_ARCHITECTURE_DEEP_DIVE.md`, `docs/guides/react_agent_guide.md`) — 삭제된 모듈을 가리키는 부분만 고치고 나머지는 FUTURE_WORK에 한 줄.
- 전체 테스트 최종 1회(실행 전후 `data/` sha256 비교).
- 최종 보고: 지시서 §7 목록 + KG 테스트 트리플 20개 삭제 사실(S2-2) + `eval_output/evidence-2026-09/notes/0c_recalc.md`의 범위 밖 발견을 `docs/dev/FUTURE_WORK.md`에 반영했는지(9.9·9.9.1에 반영 완료 — 6단계에서 새로 나온 것만 추가하면 된다).

### 4-3. 최종 보고에 반드시 넣을 수치 (이미 확정된 것)

| 지표 | 기준선(0b39e02) | 최종(ba714eb) |
|---|---|---|
| 규칙 발화 문항 비율 | 0 | 0.572 |
| 규칙 정답 일치율 | 0.469 | 0.781 |
| L5 수치 정확도(전체) | 0.029 | 0.567 |
| L5 근거성(전체) | 0.696 | 0.939 ※judge 컨텍스트 변경이 섞임 |
| 통과 수(233문항) | 10.7 | 54~55 |
| route 분포 | direct 232 | direct 185 / decide 31 / clarify 16 |
| 전체 테스트 | 5,488 | 6,004 |

## 5. 앞 세션에서 배운 운영 규칙 (반드시)

1. **서브에이전트에게 "작은 단위로 자주 커밋"을 반드시 지시해라.** 세션이 끊기면 미커밋 작업물은 사라진다(이번에 두 트랙이 그렇게 날아갔다).
2. worktree 기준 커밋 확인: 지시 첫 단계에 "`git log -1 --format=%h`가 지정 해시인지 확인, 아니면 `git reset --hard <해시>`"를 넣어라. 병합 전 `git merge-base --is-ancestor`로 확인.
3. 병합은 리드가 `git cherry-pick`으로 하나씩. RED를 건너뛴 트랙은 새 테스트만 변경 전 코드에 돌려 실패를 확인하고 병합.
4. 트랙이 서로의 기대값을 깨면(예: 5-B가 5-A의 특성화 테스트를 깨뜨림) **테스트를 고치기 전에 "왜 값이 바뀌었는지" 수치로 설명하게 해라.** 이번에 그 과정에서 신뢰도 식의 구조적 결함(S5-1)을 잡았다.
5. `KnowledgeGraph()` 기본 생성은 로드 중 자동 저장으로 KG 파일을 다시 쓴다. 스크립트·테스트는 `persist_path`=임시, `auto_save=False`.
6. 서브에이전트 셸 가드가 `eval` 문자열을 막을 수 있다 — Read/Edit 도구로 우회하도록 지시.
7. `* 2.py`(iCloud 충돌 사본)는 수정·삭제 금지. 소스 전수 검사 테스트를 쓸 때는 이 파일을 제외해라.
8. pre-commit의 ruff 버전(v0.8.0)과 PATH ruff(0.14.x)가 달라 포맷이 서로 핑퐁한다. 훅이 고친 파일은 다시 스테이징해 커밋하고, 무관한 재포맷은 커밋에 넣지 마라.

## 6. 알려진 미해결 항목 (6단계에서 새로 고치지 말 것 — FUTURE_WORK에 있음)

- 엔티티 추출기가 골든 브랜드 일부를 인식하지 못한다(IT Cosmetics·Jouer·Almay·Charlotte Tilbury·COVERGIRL). 규칙 일치율 격차의 대부분이 여기서 온다(오프라인 A 25/32 vs 골드 엔티티 주입 B 29/32).
- judge 호출 60초 시간 초과로 문항이 채점에서 빠진다(lg155는 반복 실패). 재시도 간격 없음.
- 검색이 질의마다 카드 ~70장을 비슷하게 담아 와 신뢰도 신호의 변별력이 낮다(증거 선별 문제).
- `scripts/refresh_golden_snapshot_values.py:711`이 Python 3.11에서 파싱되지 않는다(Dockerfile은 3.11-slim).
- `ToolCoordinator`의 FALLBACK 경로가 존재하지 않는 `cache.get_tool_result()`를 부른다.
