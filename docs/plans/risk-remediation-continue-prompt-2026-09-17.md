# 위험 지점 보완 — 이어받기 프롬프트 (세션 2)

> 새 Claude Code 세션에 아래 블록을 그대로 붙여넣기.
> 배경: `docs/plans/risk-remediation-kickoff-prompt-2026-09-17.md`로 시작한 세션 1이 Phase 0~2와 Phase 3 실행 대부분, Phase 4 문서 교정 일부를 마쳤다. 이 프롬프트는 남은 일만 넘긴다.

---

`docs/plans/risk-remediation-kickoff-prompt-2026-09-17.md`(원래 지시서)와 `docs/plans/risk-remediation-decisions-2026-09-17.md`(확정된 결정표)를 먼저 읽어라. 결정 D0~D6은 확정됐으니 다시 묻지 말고 따른다. 이 세션은 그 지시서의 **Phase 3 마무리 → Phase 4 → 최종 보고**만 한다. 작업 규칙(테스트·커밋 컨벤션·하지 말 것)은 원래 지시서를 그대로 따르되, 아래 변경분이 우선한다.

## 바뀐 조건

- **유료 API 총 상한은 $20이다**(경보선 $16). 세션 1 종료 시 누적 추정 **약 $11.1**(재실행 3건 포함 완료).
- **비용은 리포트 값 × 2.667로 계산해라.** `eval/cost_tracker.py:34`가 gpt-4.1-mini를 $0.15/$0.60 per 1M으로 적어 두었는데 공시가는 $0.40/$1.60이다. 질의 확장·임베딩은 리포트에 잡히지 않아 +10%를 더해 왔다. 장부: `eval_output/risk-remediation-2026-09-17/cost_ledger.md`에 이어서 적어라.
- 남은 일에 새 유료 실행은 원칙적으로 필요 없다. 아래 "재실행이 죽었으면"의 경우만 예외다.

## 현재 상태 (세션 1 종료 시점)

브랜치 `fix/wire-react-owl-2026-09` (push 안 함). 커밋:

| 커밋 | 내용 |
|---|---|
| `422ee6c` | 결정표 확정 |
| `26cd8e6` | Phase 1 — ReAct·OWL 연결·수리(플래그 기본 OFF), 읽기 전용 도구 3종, 상태 노출, chat.py docstring |
| `3c5af51` | Phase 2 — eval `--target v1\|v4`, `eval/brain_adapter.py` |
| `0669b75` | OWL 전략이 HybridRetriever의 문서 검색기를 공유(아래 사고의 재발 방지), `DocumentRetriever.initialize` 멱등 |
| `65970f1` | 결정표에 상한 $20·단가 과소 집계 경위 기록 |

아직 커밋하지 않은 산출물(전부 Phase 3 커밋에 넣을 것):
- `eval/baselines/brain-v4-1.0-2026-09-17/` — v4 기준선(172문항, 커밋 `3c5af51`, 종합 0.626, 통과 16, 근거성 0.705, 수치 정확도 0.029)
- `docs/experiments/eval_v4_baseline_2026-09-17.md` — 기준선 기록과 한계
- `eval/data/golden/subset_requires_kg_v2.jsonl` — requires_kg=true 130문항(원본 172문항에서 필터)
- `eval_output/risk-remediation-2026-09-17/kg_ablation_compare.py` → `scripts/kg_ablation_compare.py`로 옮겨 커밋(`subset`·`compare` 서브커맨드, 판정 규칙은 docstring)
- `eval_output/risk-remediation-2026-09-17/judge_context_swap.py` → 결과 해석에 썼으므로 `scripts/`로 옮겨 커밋

Phase 4 문서 교정 일부가 **별도 브랜치** `docs/risk-r4-2026-09`의 커밋 `1f8a688`에 있다(CLAUDE.md·README·AGENTS.md·PORTFOLIO_FACTS 동결 고지·dashboard_api.py docstring). 세션 1의 worktree 경로는 사라졌을 수 있으니 `git worktree prune` 후 `git cherry-pick 1f8a688`로 가져와라.

### Phase 3 실행 현황 (130문항, `--target v4 --data-as-of 2026-08-31 --judge llm --semantic-similarity --concurrency 4`)

| 구성 | 유효 실행 | 위치 |
|---|---|---|
| full | 3회 | `eval_output/kg-full-run1`(v4 기준선에서 130문항 재집계), `kg-full-run2`, `kg-full-run3` |
| KG off (`FF_ONTOLOGY_USE_ONTOLOGY_KG=false`) | 2회 + 재실행 1 | `kg-nokg-run1`, `kg-nokg-run2`, **`kg-nokg-run3`(재실행)** |
| 규칙 추론 off (`FF_REASONER_USE_OWL_REASONER=false FF_REASONER_USE_UNIFIED_REASONER=false`) | 2회 + 재실행 1 | `kg-norules-run1`, `kg-norules-run2`, **`kg-norules-run3`(재실행)** |
| ReAct·OWL ON (`FF_AGENTS_USE_REACT_AGENT=true FF_RETRIEVER_USE_OWL_STRATEGY=true`) | 1회(완료, 아래 사실 10) | `kg-reactowl-run1` |

`*-invalid-chroma` 디렉터리·로그는 아래 사고로 무효가 된 실행이다. 분석에 넣지 말고 문서에 제외 사유만 적어라.

**재실행 3건은 세션 1 안에서 모두 끝났고 검증도 마쳤다**(`CHROMA_PERSIST_DIR`=358청크로 되돌린 복사본, 커밋 `65970f1`, 검색 실패·시맨틱 색인 0건, KG off·규칙 off 3회차는 청크 0개 문항 0). KG off 3회차 종합 0.613·통과 2, 규칙 off 3회차 0.637·통과 7. 아래 1~3은 파일이 그대로인지만 확인하는 용도다:
1. `ps -ax | grep "eval.cli run"`로 아직 도는지, `eval_output/kg-{nokg,norules}-run3/report.json`·`kg-reactowl-run1/report.json`이 생겼는지.
2. 각 로그에서 `grep -c "Hybrid retrieval failed"`와 `grep -c "semantic chunks"`가 0인지, report.json의 `config.git_commit`이 `-dirty` 없이 `0669b75` 또는 `65970f1`인지. 코드가 같은 커밋이면 둘 다 유효하다.
3. **재실행이 죽었으면:** `eval_output/risk-remediation-2026-09-17/run_kg.sh`로 해당 구성만 다시 돌려라. 복사본이 없으면(재부팅 등으로 /private/tmp가 비었으면) `data/chroma`를 스크래치 디렉터리로 **복사**한 뒤, 복사본에서만 `chroma_added_ids.json`의 ID를 지워 `amore_docs`가 358개인지 확인하고 `CHROMA_PERSIST_DIR`로 지정해라. 사용자가 이미 `data/chroma`를 복구했다면(읽기 전용으로 count=358 확인) 복사 없이 돌려도 된다. 실행은 한 번에 하나씩 시작 시각을 45초 이상 벌려라.

### 세션 1에서 확인한 사실 (문서에 그대로 반영할 것)

1. **`data/chroma` 오염 사고.** 17:01 ReAct·OWL 실행 초기화 때 `OWLRetrievalStrategy`가 기본값인 시맨틱 청킹 DocumentRetriever를 따로 만들어 `amore_docs`에 청크 787개를 추가 색인했다(358 → 1,145). 증분 색인이라 기존 358개는 남아 있고, 추가분 ID는 `eval_output/risk-remediation-2026-09-17/chroma_added_ids.json`에 있다(기존 청킹이 만드는 ID와 교집합 0). 동시에 돌던 KG off·규칙 off 3회차는 검색이 "Error finding id"로 68·70문항 실패했는데, `HybridRetriever`가 예외를 삼켜 하니스가 인프라 실패로 분류하지 못했다. 원본 `data/chroma` 삭제는 권한 거부로 하지 않았고 **복구 여부는 사용자 결정**이다 — 네가 `data/`를 수정하지 마라. 재발 방지는 `0669b75`.
2. **규칙 추론은 모든 구성·모든 실행에서 추론 0건**이었다(130문항 전부 `l4_ontology.inferences` 비어 있음). 따라서 "규칙 추론 off"는 이 데이터셋·경로에서 full과 사실상 같은 구성이고, **이 실험으로는 규칙 추론의 효과를 측정할 수 없다.** 근거 문서 §5.3의 코드 기반 추정(추론 입력 키 불일치)과 일치한다. 규칙 off와 full의 차이는 실행 간 분산으로 읽어야 한다.
3. **KG off 근거성 −0.21 분해**(`eval_output/judge_swap_full1_nokg1.json`, full 1회차 × KG off 1회차, 130문항): full답변×full컨텍스트 0.733, KG off답변×KG off컨텍스트 0.521, KG off답변×full컨텍스트 0.550, full답변×KG off컨텍스트 0.608. 컨텍스트를 고정해도 KG off 답변이 0.09~0.18 낮고(답변 자체의 변화), 답변을 고정해도 컨텍스트에서 KG 사실을 빼면 0.03~0.13 낮다(채점 컨텍스트 효과). 1회차 한 쌍이고 judge 재호출 노이즈(사이클 9 기준 근거성 폭 0.023)가 섞여 있다.
4. KG off 답변은 "제공된 데이터·확인할 수 없" 류 표현이 적다(1·2회차 39·43건 vs full 57~62건) — KG 사실이 없을 때 일반 지식으로 채운다는 해석의 **정황**이지 검증이 아니다.
5. **L3 엣지 Recall과 통과 수는 KG off에서 기계적으로 떨어진다**(트레이스에 KG 사실이 없으니 엣지를 셀 수 없고, requires_kg 문항의 게이트가 L3를 본다). 종합 점수에도 L2·L3 가중 0.35가 들어간다. 답변 품질의 증거로는 L5(근거성·관련성·토큰 F1·수치 정확도)를 따로 보고, L3·통과 수·종합 점수의 하락을 "KG가 답변을 좋게 만든다"로 쓰지 마라.
6. v4 답변 프롬프트에는 DB 수치 사실(`metric_facts`)이 실리지 않는다(v1 `ContextBuilder`만 렌더링). judge 컨텍스트에는 들어간다. FUTURE_WORK 9.8에 기록됨.
7. OWL 전략 경로는 레거시 경로의 DB 지표 사실·BM25/RRF·KG 사실 조회를 쓰지 않는다(`retrieval_strategy.py` `retrieve`). ReAct·OWL 구성의 결과를 해석할 때 두 변경(ReAct, OWL 검색)이 섞여 있다는 점을 적어라.
8. v1 평가 명령의 `FF_AGENTS_USE_EXTERNAL_SIGNALS=false`, `LLM_TEMPERATURE=0.1`은 v4에 작용하지 않는다(외부 신호 미사용, 답변 온도 0.3·결정 0.1 코드 고정).
9. 첫 5문항 스모크 실측: 오류 0, 추적 비용 $0.0087.
10. **ReAct·OWL 재실행(`kg-reactowl-run1`, 커밋 `65970f1`)은 세션 1 종료 직전에 끝났다**: 종합 0.452, 통과 0, 근거성 0.404, 관련성 0.756, 리포트 비용 $0.695(실제 ≈ $2.0). 로그의 검색 실패·시맨틱 색인은 0건이다. 하지만 **130문항 중 124문항의 검색 청크가 0개**이고, **ReAct가 발동한 문항은 0개**다(`query_type == "react"` 0). 즉 이 1회는 "ReAct 효과"가 아니라 "OWL 검색 경로가 문서를 거의 못 가져온다"를 측정했다. 원인(예: `_ontology_guided_search`의 `top_k // len(queries)`가 0이 되는지, `_matches_filters`가 전부 거르는지, ReAct 발동 조건인 MEDIUM/LOW·복잡 질의 판정)을 코드와 트레이스로 확인해 문서에 적되, **고치지는 말고** FUTURE_WORK에 남겨라. D1 판정 규칙상 ReAct 발동 0건이면 ON 조건을 충족하지 못한다.

## 남은 일

### Phase 3 마무리
1. 재실행 3건 검증(위 1~2). `scripts/kg_ablation_compare.py compare --config full=... --config nokg=... --config norules=... --config reactowl=...`로 표를 만든다. `reactowl`은 1회라 범위가 없다.
2. `docs/experiments/kg_ablation_2026-09.md` 작성. 원래 지시서의 필수 항목(평가일, 대상 커밋, 데이터셋·문항 수, 구성, 실행 횟수, 지표별 평균·범위, 분산 대비 유의 여부, 한계)에 더해 위 사실 1~8과 무효 실행 제외 사유, 비용 축소·복원 경위(ReAct·OWL을 한때 생략했다가 상한 $20으로 되살림)를 적는다. 분산 범위 안의 차이는 "차이 없음". 결과를 유리하게 다듬지 마라.
3. **D1 판정**: ReAct·OWL 1회가 full 3회 범위 대비 노이즈 기준(종합 −0.01, 근거성 −0.03, 통과 수 −5, 수치 정확도 −0.05)을 넘게 떨어지지 않았고 ReAct가 1문항 이상 발동했으면(`l5_answer.query_type == "react"` 수) 두 플래그를 ON으로, 아니면 OFF 유지 + 문서에 "연결됨, 기본 비활성". 1회 측정·두 변경 혼재·L3 기계적 영향을 판정 근거와 함께 적고, 결정표 §확정 기록에 D1 최종값을 추가한다. ON으로 바꾸면 `config/feature_flags.json`, `test_default_config_keeps_react_and_owl_off`와 `test_feature_flags.py`의 기본값 테스트, chat.py·README·CLAUDE.md·AGENTS.md의 "기본 OFF" 서술을 함께 고쳐라.
4. `docs/dev/FUTURE_WORK.md` 9.8에 한 줄씩 추가: cost_tracker 단가 과소, 검색 예외를 삼켜 평가가 인프라 실패로 못 잡는 문제, 여러 평가 프로세스의 동시 초기화, `OWLRetrievalStrategy` 생성자 기본값이 별도 시맨틱 청킹 색인기를 만든다는 점, 규칙 추론 0건(추론 입력 키 불일치 — 근거 문서 §5.3).
5. Phase 3 커밋: 위 미커밋 산출물 + 스크립트 2개 + 실험 문서 + 결정표 갱신.

### Phase 4
1. `git cherry-pick 1f8a688`. D1이 ON이면 그 커밋의 "기본 OFF" 서술을 이어서 고친다.
2. `docs/portfolio/amore_architecture_evidence.md` 갱신: §0·§3.3·§3.4·§5.5·§7·§10에서 이번 작업으로 바뀐 사실을 고친다. **수정 시점이 2026-09이고 공모전 이후 사후 분석에서 발견해 고친 것임을 명시**하고, 공모전 당시 동작했던 것처럼 읽히는 문장을 만들지 마라. 추가로 §3.3의 "`9c32aba`에서 유입"을 "`9c32aba`(02-08) 당시 `docs_path`는 `get_true_hybrid_retriever`의 유효 인자였고 `f049cb8`(02-15) 리팩터가 `OWLRetrievalStrategy`로 바꾸며 인자만 남겨 깨짐"으로 정정하고, §7에 v4 기준선(v1과 비교 불가)과 KG 실험 결론, §4.3 또는 적절한 곳에 `data/chroma` 오염 사고와 현재 상태(사용자 복구 여부를 읽기 전용으로 확인해 358/1,145 중 무엇인지)를 적는다. §10.2의 "ReAct·OWL" 대안 문구도 D1 결과에 맞춘다.
3. Phase 4 커밋.

### 마무리
- 전체 테스트 1회(`.venv/bin/python -m pytest tests/ -q --no-cov`). 세션 1 시작 기준선은 5,458 passed / 7 skipped / 0 failed(376초). `test_cache.py::test_cleanup_expired_removes_old`는 간헐 실패(FUTURE_WORK 9.8)이고, `test_react_agent.py::test_react_run`은 실제 OpenAI를 부른다.
- 최종 보고에는 원래 지시서가 요구한 것만 담는다: 확정된 결정표(과거 결정을 찾은 것/못 찾은 것 구분 — 찾은 것은 D0와 D6뿐), Phase별 커밋 해시(위 표 + 이번 세션 커밋), R1 수리 전후 동작 차이와 증명 테스트 이름(`tests/unit/core/test_react_owl_wiring.py`, `test_react_loop_fixes.py`, `test_react_tools.py`, `tests/eval/test_brain_adapter.py`), v4 기준선 수치, KG 실험 결과와 해석 한계, 누적 API 비용(실제 단가 기준), 남긴 항목(`data/chroma` 복구 여부 포함).

하지 말 것: push, 배포, Railway 설정 변경, 외부 크롤링, 기존 baseline 파일 수정·삭제, `data/`의 DB·KG·Chroma 수정, 비밀키·스프레드시트 ID 출력, `* 2.py` 파일 수정·삭제.

요청한 것을 요청한 범위대로 수행해라. 멈추지 말고 끝까지 진행하되, 해석에 따라 결과물이 실질적으로 달라질 때만 질문해라. 끝낼 수 없는 항목이 있으면 나머지를 다 한 뒤 무엇이 왜 빠졌는지 명시해라. 문서는 필요한 만큼만 쓰고, 실험 결과를 유리하게 보이도록 다듬지 마라 — 면접에서 검증당할 근거로 쓴다.
