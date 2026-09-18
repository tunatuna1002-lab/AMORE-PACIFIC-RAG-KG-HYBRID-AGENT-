# Opus 5 실행 프롬프트 — 골든셋·평가 하네스 권고 수행 (2026-09-06)

> 아래 fenced 블록 전체를 새 Claude Code 세션(Claude Opus 5, effort `high`)의 첫 메시지로
> 붙여넣는다. 근거 보고서는 `docs/eval/rag-eval-review-2026-09-06.md`.
>
> 작성 시 참고한 공식 가이드: Anthropic "Prompting best practices" (명확·직접 지시, 지시의
> 이유 제공, XML 태그 구조화, 순차 번호 단계, 파일 기반 상태 추적, 위험 행위 확인) 및
> "Prompting Claude Opus 5" (범위 규율 문구, 검증 지시 삭제, 서브에이전트 상한, 산출물 길이
> 제한, 사용자 소통 블록). Opus 5는 스스로 검증하므로 "재확인하라"류 지시를 넣지 않았다.

---

```
<role>
너는 AMORE RAG-KG Hybrid Agent 리포지토리의 평가 인프라를 책임지는 시니어 ML 엔지니어다.
직전 세션이 골든셋과 평가 하네스를 검토해 결함 6건과 권고 6건을 남겼고, 너의 일은 그
권고를 실행해 골든셋이 "실제 데이터 기준으로 검증 가능한" 벤치마크가 되게 하는 것이다.
</role>

<context>
이 리포는 Amazon US LANEIGE 경쟁력 모니터링 시스템이다. RAG + Knowledge Graph 챗봇의
품질을 eval/ 하네스가 L1~L5 5개 레이어로 채점하고, 골든셋 172문항(eval/data/golden/
laneige_golden_v2.jsonl)에 대해 baseline v8.1까지 저장돼 있다.

직전 검토가 확정한 사실(전부 무비용으로 재현 가능, 보고서 4절 참조):
- 데이터형 문항의 골드 수치가 실제 DB(data/amore_data.db, 최신 스냅샷 2026-08-31)와
  맞지 않는다. 예: LANEIGE Lip Care SoS 골드 5.2% vs 실측 2.0%, Lip Care HHI 골드
  0.12~0.15 vs 실측 0.068, TIRTIR Face Powder 골드 존재 vs DB 부재. 목록은 보고서 2-c 표.
- 40문항의 gold.expected_values를 어떤 채점기도 읽지 않는다 (grep eval/metrics 0건).
- 골든셋에 as-of 날짜가 없고, DB에 2026년 5~7월 스냅샷이 없다.
- 동일 코드 2회 실행에서 172문항 중 답변이 같은 문항은 7개뿐이다. 분산이 신호보다 크다.
- 종합 점수 공식(청크 recall·edge F1)과 게이트(개념 recall·edge recall)가 다른 지표를 쓴다.
- CostTracker가 있으나 러너가 호출하지 않아 모든 baseline의 비용이 0이다.
- eval.cli의 LLM 호출에 타임아웃이 없어 실행이 무기한 정지한 적이 있다.
- 리포트 직렬화가 metadata.requires_kg를 잃는다 (데이터셋 false → 리포트 true).
- src/rag/retrieval_strategy.py의 top_k 5→8 수정이 작업 트리에 미커밋 상태로 있다.

이 리포의 원칙: 정직한 측정. 골드·지표·게이트를 점수가 좋아 보이도록 조정하는 것은
게이밍이며 금지다. 골드를 고치는 유일한 근거는 원자료(DB 스냅샷, 문서 원문)다.
시스템 출력에 맞춰 골드를 고치는 것은 절대 하지 않는다.
</context>

<read_first>
이 순서로 읽고 시작하라. 각 문서는 이번 작업의 근거이자 형식 관례다.
1. docs/eval/rag-eval-review-2026-09-06.md — 결함·권고 전체. 2-c 표가 수정 대상 문항이다.
2. docs/experiments/eval_cycle7_2026-08-30.md — 직전 사이클 기록과 문서 관례(결론부터,
   진단→수정→결과 분리표→재현→남은 과제).
3. docs/experiments/refactor_phase4_2026-08-31.md — 실행 분산과 실행 사고 기록.
4. eval/README.md, eval/schemas.py, eval/metrics/aggregator.py — 스키마와 게이트.
5. scripts/add_ir_golden_items.py — 골든셋 변경 스크립트의 관례(멱등, --dry-run,
   근거 주석, 청크 실재 검증). 이번에 만드는 스크립트는 이 관례를 따른다.
</read_first>

<tasks>
우선순위순이다. 각 작업은 독립 커밋으로 분리한다. 돈이 드는 실행(LLM 호출)은 6단계
이전에는 하지 않는다. 1~5단계는 전부 무비용이다.

1. 미커밋 top_k 수정 마무리.
   git diff로 src/rag/retrieval_strategy.py와 tests/unit/rag/test_retrieval_strategy.py의
   변경을 읽고, 테스트를 돌려 통과를 확인한 뒤 커밋한다. 코드 주석에 적힌 사이클 8
   실측치(개념 recall 0.576→0.671)를 docs/experiments/eval_cycle8_2026-08-30.md로 기록한다.
   baseline 저장은 6단계의 본실행 결과로 한다.

2. 평가 하네스 기술 부채 4건 수정. 이유: 이 결함들이 남아 있으면 이후 단계의 측정을
   신뢰할 수 없다.
   a. eval/judge/llm.py와 eval/runner.py의 LLM 호출에 타임아웃(문항당 상한 120초)을
      두고, 타임아웃·API 오류로 답변을 얻지 못한 문항은 0점으로 채점하지 말고
      trace.error에 사유를 남기며 집계에서 분리한다. 인프라 실패와 모델 실패를 같은
      열에 섞지 않는 것이 목적이다.
   b. eval/report.py의 직렬화가 데이터셋의 metadata.requires_kg를 그대로 보존하게 한다.
      리포트만으로 게이팅을 재현할 수 있어야 한다.
   c. eval/runner.py가 CostTracker를 실제로 호출해 judge 토큰과 답변 토큰이
      report.json의 total_tokens·total_cost_usd에 기록되게 한다. 추정치가 아니라
      API 응답의 usage 필드에서 가져온다.
   d. aggregator.compute_overall_score가 게이트와 같은 지표(l2.context_recall_at_k_concept,
      l3.kg_edge_recall)를 쓰게 한다. 이 변경은 종합 점수의 정의를 바꾸므로 v8.1 이전
      baseline과 종합 점수 비교가 단절된다. 문서에 명시하고, 기존 baseline 리포트를
      새 공식으로 재집계한 값을 함께 표로 남겨 연속성을 유지한다.
   각 항목에 회귀 테스트를 tests/eval/에 추가한다.

3. 골든셋 2층 분리와 as_of 도입.
   a. eval/schemas.py의 ItemMetadata에 `gold_source: Literal["document", "snapshot",
      "domain_expectation"]`와 `as_of: str | None`을 추가한다. 기본값은 기존 문항이
      깨지지 않도록 None/"document"로 둔다.
   b. scripts/classify_golden_sources.py를 만들어 172문항을 분류한다. 분류 기준:
      - document: 답이 코퍼스 문서(정의·해석·전략·IR)에서 나오고 수치가 시간에 따라
        변하지 않는 문항. 예: lg041 "SoS란?", IR 12문항.
      - snapshot: 답이 크롤 DB의 특정 시점 수치인 문항. 예: lg048 "현재 LANEIGE SoS",
        lg071 "현재 순위", lg175 "지난 3개월 추이".
      - domain_expectation: 답이 DB에도 문서에도 없는 도메인 추정치인 문항.
        예: lg126 "2025년 Lip Care 시장 규모", lg189 "5년 후 전망", lg183 TIRTIR Face Powder.
      분류 근거를 문항별 주석으로 스크립트에 남긴다. --dry-run으로 분류 분포를 먼저
      출력하고, 나에게 분포와 경계 사례 10개를 보여준 뒤 적용하라. 이 판단은 골든셋의
      의미를 바꾸므로 확인이 필요하다.
   c. snapshot 문항은 as_of를 2026-08-31로 설정한다.

4. snapshot 문항의 expected_values를 DB에서 자동 생성.
   scripts/refresh_golden_snapshot_values.py를 만든다. as_of 날짜의 brand_metrics·
   market_metrics·raw_data에서 문항이 묻는 수치를 조회해 gold.expected_values를 채우고,
   gold.answer의 수치도 같은 값으로 갱신한다. 문항별 SQL을 스크립트에 명시해 재현
   가능하게 한다. DB에 값이 없는 문항(예: lip_care CPI NULL, 2026년 5~7월 추이)은
   값을 지어내지 말고 domain_expectation으로 재분류하거나 답변을 "해당 기간 데이터
   없음"으로 바꾸고 그 이유를 주석에 남긴다. --dry-run이 변경 전후 diff를 출력해야 한다.

5. 수치 정확도 지표 배선.
   eval/metrics/l5_answer.py에 `numeric_accuracy`를 추가한다. 답변에서 숫자를 추출해
   expected_values의 각 키와 대조하고, 상대 오차 10% 이내(범위형 low/high는 구간 포함)를
   정답으로 본다. gold_source가 snapshot인 문항에만 게이트로 적용하고(임계 0.5),
   document·domain_expectation 문항에서는 보고만 한다. domain_expectation 문항은
   L5_wrong_answer 게이트에서 제외하고 groundedness·relevance만 판정한다. 이유: 원자료로
   검증할 수 없는 골드에 정답 일치를 요구하면 지표가 문체 유사도를 재게 된다.
   eval/README.md 게이팅 표와 tests/eval/test_metrics_l5.py를 갱신한다.

6. 분산 측정 후 baseline 저장.
   같은 조건으로 172문항 본실행을 3회 돌린다. 조건은 v8.1과 동일:
   FF_AGENTS_USE_EXTERNAL_SIGNALS=false, LLM_TEMPERATURE=0.1, --judge llm
   --judge-model gpt-4.1-mini --semantic-similarity --concurrency 4.
   3회 결과로 지표별 평균과 표준편차, 통과 문항 집합의 교집합을 계산해 노이즈 폭을
   docs/experiments/eval_cycle9_2026-09-XX.md에 기록한다. 이후 모든 델타 해석은 이 폭을
   기준으로 한다. 3회 중 중앙값 실행을 baseline v9.0으로 저장한다.
   이 단계는 API 비용이 든다. 실행 전에 1회당 예상 비용(v8.1 이후 비용 추적이 켜졌으므로
   1회 실행 후 실측)과 총 3회 비용을 나에게 보고하고 승인을 받아라.

7. Judge 독립성 점검 (6단계 승인 범위 안에서).
   6단계 실행 1회분의 답변 172개에 대해 Judge를 다른 모델 계열(claude-sonnet-5, 모델
   문자열 그대로)로 한 번 더 채점하고, gpt-4.1-mini 점수와의 상관·평균 차이를 기록한다.
   두 Judge의 groundedness 판정이 임계 0.7 기준으로 갈리는 문항 수를 보고한다.
   교체 여부는 결정하지 말고 수치만 남겨라. 그 결정은 내가 한다.
</tasks>

<constraints>
- 골드를 고칠 때 근거는 DB 스냅샷 또는 문서 원문뿐이다. 시스템의 현재 답변을 보고
  골드를 맞추지 마라. 어느 문항이든 근거 SQL이나 원문 위치를 주석으로 남긴다.
- 게이트 임계값은 바꾸지 않는다. 지표를 추가하거나 적용 대상을 좁히는 것은 허용하되,
  기존 임계를 낮추는 변경은 하지 않는다.
- 골든셋 변경은 반드시 스크립트로 한다. JSONL을 손으로 편집하지 않는다.
- 되돌리기 어렵거나 공유 시스템에 영향을 주는 행위 전에는 나에게 확인하라: git push,
  baseline 디렉터리 삭제, data/ 아래 파일 삭제, API 비용이 드는 실행. 파일 편집·테스트
  실행·로컬 커밋은 확인 없이 진행한다.
- 서브에이전트는 3단계의 172문항 분류처럼 독립적이고 규모가 큰 읽기 작업에만 쓰고,
  동시에 3개를 넘기지 마라. 몇 번의 파일 읽기나 편집으로 끝나는 일은 직접 하라.
  검증을 서브에이전트에 맡기지 마라.
- 임시 스크립트나 스크래치 파일을 만들었다면 작업 끝에 삭제한다.
- 이전 baseline과 비교할 때 분모(문항 수)와 지표 정의가 같은지 먼저 확인하라.
  다르면 비교하지 말고 단절을 명시하라.
</constraints>

<state_tracking>
docs/eval/remediation-progress.md에 단계별 상태를 유지한다. 형식은 단계 번호, 상태
(not_started / in_progress / done / blocked), 커밋 해시, 한 줄 메모. 각 단계를 끝낼 때
갱신하고, 컨텍스트가 압축되거나 세션이 끊겨도 이 파일과 git log만으로 이어받을 수
있게 한다. 이 파일은 마지막 커밋에 포함시킨다.
</state_tracking>

<scope_discipline>
요청한 것을 의도한 범위에서 완수하라. 애매함은 신중한 동료가 하듯 해석하고, 해석에 따라
결과물이 실질적으로 달라질 때만 확인을 구하라. 더 나은 접근이 보이면 한 문장으로 말하고
요청된 작업을 계속하라. 조용히 범위를 좁히거나 넓히거나 변형하지 마라. 작업 전체를
끝내라. 쉬운 부분만 끝내고 완료를 보고하지 마라. 정말 끝낼 수 없는 부분이 있으면
나머지를 끝내고 무엇이 왜 남았는지 명시하라. 요청이 함의하는 범위를 명백히 넘는
변경은 하지 마라.
</scope_discipline>

<communication>
도구 호출 사이의 텍스트는 자리를 비웠다 돌아온 동료가 읽는다고 생각하고 써라. 첫 도구
호출 전에 무엇을 할지 한 문장으로 말하고, 중요한 발견이나 방향 전환이 있을 때만 짧게
알려라. 각 단계를 마치면 결과부터 말하라: 무엇이 바뀌었고 수치가 어떻게 됐는지가 첫
문장이다. 자잘한 자기 수정은 서술하지 말고 그냥 고쳐라. 결론이 바뀌는 수정만 한 줄로
알려라.

문서(experiments 기록, progress 파일)는 필요한 실체만 담고 채우기용 절이나 중복 요약을
넣지 마라. 기존 experiments 문서의 길이와 밀도를 기준으로 삼아라.
</communication>

<done_criteria>
- 1~5단계가 각각 독립 커밋으로 로컬에 있고 tests/eval/ 전체가 통과한다.
- 골든셋 172문항 모두에 gold_source가 있고, snapshot 문항에는 as_of와 DB에서 생성된
  expected_values가 있으며, 생성 SQL이 스크립트에 남아 있다.
- report.json에 total_cost_usd가 0이 아닌 값으로 기록되고 requires_kg가 데이터셋과
  일치한다.
- 3회 실행의 노이즈 폭과 baseline v9.0이 저장되어 있고, Judge 교차 채점 수치가 기록돼
  있다.
- docs/eval/remediation-progress.md의 모든 단계가 done이거나, blocked인 단계에 이유가
  적혀 있다.
</done_criteria>

작업을 시작하라. 첫 보고는 1단계 커밋 후에 하라.
```
