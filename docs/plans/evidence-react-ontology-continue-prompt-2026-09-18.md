# 증거 카드 · 규칙 추론 · ReAct 통합 — 이어가기 지시서 (2026-09-18)

> 새 Claude Code 세션에 `---` 아래를 그대로 붙여넣기. 앞 세션(2026-09-17)이 사용 한도로 끊겨 이어간다.

---

`docs/plans/evidence-react-ontology-kickoff-prompt-2026-09-17.md`(이하 **지시서**)의 작업을 이어서 끝까지 수행해라. 지시서의 원칙(§0 정확함 > 속도, §4 병렬 규칙, §5 작업 규칙, §8 하지 말 것)과 확정 설계 E1~E11은 그대로 유효하다. 다시 묻지 말고 따를 것.

**토큰을 효율적으로 써라.** 큰 문서·파일은 필요한 절만 읽고, 구현은 서브에이전트에 목적·소유 파일·금지 파일·테스트·완료 조건을 짧고 정확하게 주어 맡긴다. 리드는 diff 핵심부 리뷰·검증·병합·게이트 판정만 한다. 이미 확인된 사실(아래 §2)은 재조사하지 말 것. 결과 보고는 한국어로.

## 1. 먼저 읽을 것 (이것만)

1. 지시서 §0, §2 표(F1~F15), §3(E1~E11), §6의 3~6단계, §7, §8
2. `docs/plans/evidence-react-ontology-decisions-2026-09.md` — 결정 S0-1~S2-2 (특히 S0-6, S0-7, S1-1, S2-1, S2-2)
3. `docs/experiments/evidence_pipeline_2026-09.md` — 0·1단계 결과(기준선 표)
4. `eval_output/evidence-2026-09/notes/0c_recalc.md`(트랙별 범위 밖 발견 모음), `notes/stage45_map_summary.txt`(4·5단계 코드 지도 요약 — 함정 목록 포함)

## 2. 현재 상태 (2026-09-18 01:50 KST)

- 브랜치 `feat/evidence-react-ontology-2026-09`, HEAD = 이 문서를 추가한 커밋(그 직전 `919b9e6`). push 안 함.
- **사용자 승인**: 유료 API 상한 $40 유지하며 계획된 측정 진행 승인. 누적 추정 ≈ $8~9(2단계 3회 비용 미집계 — 아래 3-1에서 기입). 장부 `eval_output/risk-remediation-2026-09-17/cost_ledger.md`. 단가는 0-C에서 수리됨(리포트 비용 × 1.1만 보정).
- **병합 완료**: 0단계 전부(0-A 색인 읽기 전용·`python -m src.rag.build_index`, 0-B 검색 오류 가시화, 0-C 단가, 0-D/0-D2 route_trace, 0-E API 가드, 0-F 테스트가 `data/`를 쓰지 않게 격리), 1단계(유형별 시험지 `eval/data/golden/typed/`, 통합 `combined_v1.jsonl` 233문항, 기준선 `eval/baselines/brain-v4-typed-1.0-2026-09-17`), 2-A 증거 카드 모델, 2-B 카드로 프롬프트 조립(`src/rag/evidence_assembly.py`), 2-C judge 컨텍스트=프롬프트 카드, 3-A 규칙 입력 계약(`src/ontology/rule_contracts.py`), 3-C 규칙 관측 리포트 필드.
- **2단계 측정 완료, 분석 안 함**: `eval_output/evidence-2026-09/s2-run{1,2,3}/report.json`(커밋 `b7b3ba5`, 동결 스냅샷). 로그상 통과 43/46/50(기준선 9/12/11), run2·run3에 인프라 실패 각 1건(lg161, lg155) — 원인 미확인.
- **중단된 트랙 2개(미커밋, 기준 `b7b3ba5`)** — 사용 한도로 에이전트가 끊김. 작업물은 두 곳에 있다:
  - 3-B: worktree `.claude/worktrees/agent-a62b7fde7422e68c6`(변경이 인덱스에 스테이징됨), 패치 사본 `eval_output/evidence-2026-09/wip/track-3b.patch`(15 files)
  - 2-D: worktree `.claude/worktrees/agent-a3d57bb41e772ffb8`, 패치 사본 `eval_output/evidence-2026-09/wip/track-2d.patch`(9 files)
- 운영 `data/knowledge_graph.json`에서 테스트가 만든 트리플 20개(hasProduct 6 + competesWith 14)를 사용자 승인으로 삭제 완료(S2-2, 3,500→3,480). 동결 평가 스냅샷 KG는 그대로.

## 3. 할 일 (순서대로)

### 3-1. 2단계 분석·게이트
- `python3 scripts/typed_eval_summary.py --config base=eval_output/evidence-2026-09/base-run1/report.json,...run2...,...run3... --config s2=eval_output/evidence-2026-09/s2-run1/report.json,...` 로 유형별 판정표.
- lg161·lg155 인프라 실패 원인을 `s2-run2.log`·`s2-run3.log`와 리포트 `error_item_ids`로 확인(재현 필요 시 해당 문항만 스모크).
- 게이트(지시서 2단계): ① numeric의 수치 정확도·통과 수가 노이즈 이상 개선 또는 원인 분석, 다른 유형 근거성 회귀 없음. 프롬프트가 약 3배 길어진 비용·지연 증가도 기록.
- `docs/experiments/evidence_pipeline_2026-09.md`에 "## 2단계" 절(병합 기록·표·해석), 장부에 2단계 3회 실제 비용 기입.

### 3-2. 중단된 트랙 재개
각각 서브에이전트에 맡기되 **먼저 기존 worktree의 스테이징된 변경을 읽고 이어서 완성**하게 한다(처음부터 다시 쓰지 말 것). worktree가 사라졌으면 `b7b3ba5`에서 새 worktree를 만들고 `git apply --index <patch>` 후 이어간다. 완성 후 리드가 현재 HEAD에 cherry-pick(충돌 해소)하고, 새 테스트를 **변경 전 코드에 돌려 RED 확인**한 뒤 병합한다.
- **3-B 요구사항(요약)**: 규칙 추론 입력을 metric/relation 카드에서 `rule_contracts`의 `EvidenceBinding`으로 조립(`build_rule_context`), (브랜드×카테고리) 조합 상한·카테고리 규칙 중복 제거, `retrieve`에서 카드 조립을 추론 **전**으로 이동, `evaluate_all` 결과를 `context.inferences`+inference 카드(`derived_from`=입력 카드 id)로, `current_metrics`를 추론 입력으로 쓰지 않음, 추론 off 플래그 게이트 유지, `metadata["rule_evaluation"] = {"combinations","evaluated","fired","non_fire_top","non_fire_counts_by_kind"}`(3-C가 이 모양을 읽음 — 변경 금지), `reasoner.apply`가 결론의 `position`·`market_structure` 등을 보존, `MetricFactsProvider._share_facts`에 `cpi`·`avg_rating_gap`·`brand_avg_rank` 추가, 대소문자 비교 결함(brand_ownership_verification) 수정. 게이트 테스트: DB 픽스처(sos 18%, hhi 0.10)로 실제 `HybridRetriever.retrieve` → `market_dominance_fragmented` 발화 + `derived_from`. 관찰: 동결 스냅샷 복사본으로 rule 생성 문항 32개 LLM 없이 규칙 정답 일치율.
- **2-D 요구사항(요약)**: `src/core/numeric_verifier.py` — 답변 문장의 수치를 인용 카드 value(렌더러 `format_value` 표시 규칙)와 대조, 분류 verified / no_citation / mismatch / unknown_card, 날짜·연도·분기·Top 100 제외. 플래그 `response.numeric_verification_mode` = off|annotate|enforce, **기본 annotate**(답 불변, `response.metadata["numeric_verification"]` 기록), enforce는 mismatch·unknown_card 수치를 "확인되지 않음"으로 치환. `ResponsePipeline`의 모든 생성 경로에 적용, 검증 대상은 `context.prompt_evidence`. `CITATION_INSTRUCTION`을 R/D/I 카드 인용도 유도하게 수정. 이후 리드가 평가 리포트에 검증 건수 집계를 연결(작은 트랙).

### 3-3. 2·3단계 게이트 전체 테스트 → 3단계 측정
- 전체 테스트: 0-F 병합으로 테스트가 `data/`를 쓰지 않게 됐지만, 실행 전후 `data/` sha256을 비교해 확인한다. **22:00 전후는 소유자 크롤 LaunchAgent가 `data/`를 갱신하니 피할 것.**
- 3단계 측정: `eval_output/evidence-2026-09/run_eval2.sh <이름> eval/data/golden/typed/combined_v1.jsonl <커밋>` 3회(동시 3, 45초 간격). 검증기는 annotate. 게이트: DB 픽스처 통합 테스트 발화, ③ rule 3회에서 발화율 > 0과 규칙 정답 일치율, "규칙 off"(`FF_REASONER_USE_OWL_REASONER=false FF_REASONER_USE_UNIFIED_REASONER=false`) 대비 판정 — 규칙 off 측정 횟수는 비용을 보고 지시서 §3 축소 순서에 따라 정한다.

### 3-4. 4·5·6단계와 마무리
지시서 §6 4~6단계, §7 그대로. `notes/stage45_map_summary.txt`의 함정(테스트가 private 이름·소스 문자열에 결합, patch 경로, 평가 어댑터가 `retriever.retrieve`/`react.run`을 감쌈, 순환 import, SSE `type`/`content` 형식, `kg_triples` 없는 필드, ReAct `sources` dict 버그, 신뢰도 스케일 혼재, `PromptGuard`의 `get_brand_status` 문자열 의존, 스트림에 캐시 적용 시 부작용)을 서브에이전트 지시에 포함한다. 6단계에서 검증기 enforce 기본값 여부도 판정한다.

## 4. 앞 세션에서 배운 운영 규칙 (반드시)

1. **worktree 기준 커밋**: `isolation: "worktree"`가 오래된 `origin/main`(`a46ce76`)에서 생성되는 일이 반복됐다. 모든 서브에이전트 지시 첫 단계에 "`git log -1`이 지정 해시인지 확인, 아니면 `git reset --hard <해시>`(a46ce76은 조상이라 안전)"를 넣고, 병합 전 `git merge-base --is-ancestor`로 확인. 에이전트가 끝나면 worktree가 자동 정리될 수 있어, 재개가 필요한 트랙은 리드가 `git worktree add -b <브랜치> <scratchpad 경로> <해시>`로 전용 worktree를 만들어 준다.
2. **병합**: 트랙 커밋을 `git cherry-pick`으로 한 번에 하나씩. RED를 건너뛴 트랙은 새 테스트 파일만 변경 전 코드에 돌려 실패를 확인하고 병합.
3. **평가는 `run_eval2.sh`로만**: 실행마다 지정 커밋 worktree + 동결 스냅샷(`eval_output/evidence-2026-09/eval_data_snapshot_base`, DB 7a0dce00·KG 876ea40f·dashboard b57af9d4·Chroma 358) 복사본. 추적되지 않는 데이터셋은 **절대경로**로. 동시 3개·45초 간격. 문서 작업은 평가 중에도 가능(평가는 worktree에서 돈다).
4. **`KnowledgeGraph()` 기본 생성은 로드 중 자동 저장으로 KG 파일을 다시 쓴다**(0-F 근본 원인). 스크립트·테스트는 `persist_path`=임시, `auto_save=False`. `src` 수정 제안은 FUTURE_WORK에.
5. 서브에이전트 셸 가드가 `eval` 문자열을 막을 수 있다 — Read/Edit 도구로 우회하도록 지시.
6. 대용량 파일 훅(5MB): 기준선 `report.json`은 들여쓰기 없는 JSON으로 저장.
7. 사용 한도에 대비해, 긴 트랙은 중간 커밋을 지시하고 리드는 단계마다 결과를 문서에 먼저 적는다.

## 5. 최종 보고에 반드시 포함

지시서 §7 목록 + KG 테스트 트리플 20개 삭제 사실(S2-2) + `eval_output/evidence-2026-09/notes/0c_recalc.md`의 범위 밖 발견을 `docs/dev/FUTURE_WORK.md`에 반영했는지.
