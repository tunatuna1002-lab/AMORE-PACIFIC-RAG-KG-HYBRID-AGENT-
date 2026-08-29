# 프롬프트 실험 기록 — 2026-08

## 목적
챗봇 시스템 프롬프트(`prompts/agents/chatbot_system.txt`)의 4개 변경안이 근거성(Groundedness)·관련성(Relevance)·정답 F1·지연·비용에 미치는 영향을 골든셋 하네스로 측정한다.

## 1단계 — 하네스 실행 가능성 점검 (2026-08-19, API 미사용 환경)

| 점검 | 명령 | 결과 |
|---|---|---|
| 데이터셋 검증 | `python eval/cli.py run --dataset eval/data/golden/laneige_golden_v2.jsonl --dry-run` | 최초 160 중 109만 로드. 51건 스키마 불일치 → 수정 후 160 로드 |
| 하네스 단위테스트 | `pytest tests/eval -q` | 357 통과 / 2 실패(아래) |
| 에이전트 import | `HybridChatbotAgent()` 직접 생성 | 성공 (MockAgent 폴백 아님) |
| 에이전트 실행 | `--judge none` 1문항 | OPENAI_API_KEY 없음 → "Vector search is required" 오류로 중단(예상된 동작). 벡터 검색이 OpenAI 임베딩에 의존 |

발견·수정한 결함 3건 (`eval/schemas.py`):
1. `EvalConfig.judge_model: str` → `--judge none/nli` 시 None 전달로 크래시. `str | None`으로 수정.
2. `ItemMetadata.domain` Literal이 `market/brand/product/metric/general`만 허용 → 골든셋 v2의 `multi_hop`(20)·`edge`(15)·`time`(15) 50건 탈락. Literal 확장.
3. `gold.constraints: list[str]` → 골든셋 7건이 dict 형태 제약 사용. `list[str | dict]`로 확장.

미해결 실패 2건(환경 요인, 코드 결함 아님으로 추정):
- `test_judges.py::TestStubJudge::test_stats_tracking` — 이벤트 루프 없음(RuntimeError). pytest-asyncio 설정 관련으로 추정, 실험과 무관.
- `test_semantic.py::test_similar_text` — sentence-transformers 모델 다운로드가 네트워크 차단으로 실패해 유사도 0. 인터넷 환경에서 재확인 필요.

## 2단계 — 실험 (실행 예정)

고정 조건: 데이터셋 `eval/data/golden/subset_nokg.jsonl` (requires_kg=false 30문항; 크롤링 DB 복원 시 160문항으로 확장) / 모델 gpt-4.1-mini / temperature 0.1 / top-k 8 / Judge gpt-4.1-mini / 버전당 1회

| 버전 | 변경 변수 | 가설 |
|---|---|---|
| v0 | baseline | 기준 |
| v1 | 근거 인용 강제 + "데이터 없음" 명시 | Groundedness ↑ |
| v2 | 단계적 내부 추론 후 결론만 출력 | 어려운 문항 F1 ↑, 지연·비용 ↑ |
| v3 | 출력 스키마 고정(결론/근거3/유의사항) | Relevance ↑, 토큰 ↓ |
| v4 | 압축(길이 약 40%) | 점수 유지 시 비용 ↓ |

실행: `bash scripts/run_prompt_experiment.sh` → `eval_output/exp_*/RESULTS.md` 생성 → 이 문서 아래에 결과표를 붙여 넣는다.

## 결과 (2026-08-30 실행)

**설계 변경**: 2026-08 현행 `chatbot_system.txt`에 이미 v1 계열 근거 규칙 일부가 반영되어
(커밋 9474e82), 원안의 v0 기반 비교 대신 **현행(control) vs 현행+v1 인용 강제 블록(v1b, treatment)**
A/B로 진행. 조건: subset_nokg 30문항 / gpt-4.1-mini / temp 0.1 / top-k 8 / LLM Judge /
semantic-similarity ON / 외부신호 OFF / concurrency 4 / 버전당 1회.

| 지표 | control (현행) | treatment (v1b) | Δ |
|---|---|---|---|
| Groundedness (Judge) | 0.286 (≥0.7: 4/30) | **0.326 (≥0.7: 6/30)** | **+0.040 (+14% 상대)** |
| Answer Relevance | 0.855 | 0.853 | -0.002 (동일 수준) |
| Semantic Similarity | 0.657 | 0.656 | 동일 |
| Answer token-F1 | 0.103 | 0.109 | +0.006 |
| L2 Context Recall | 0.267 | 0.267 | 동일 |
| Overall (가중) | 0.469 | 0.481 | +0.012 |
| 게이트 통과 | 1/30 | 1/30 | 동일 |
| 평균 지연 | 13.1s | 13.6s | +0.5s (+4%) |

## 대표 실패 사례
- grounding 실패 26→24건으로 소폭 개선되었으나 여전히 최대 실패 태그.
  실패 문항 다수는 검색 컨텍스트에 해당 근거 자체가 없는 경우
  (L2 recall 0.267 — 골드 청크가 top-8에 없음) → 프롬프트로 해결 불가.

## 결론·채택
- 채택 버전: **v1b** (현행 + 근거 인용 규칙 강화) → `chatbot_system.txt` 반영
- 이유: relevance·semantic 손실 없이 groundedness 개선 방향 확인, 지연 비용 +4%로 수용 가능
- 트레이드오프: n=30 단회 실행이라 개선폭(+0.04)은 노이즈 범위일 수 있음.
  **주 병목은 프롬프트가 아니라 검색 컨텍스트 부족**(코퍼스 271청크, L2 recall 0.27)으로 판정.

## 다음 실험
- RAG 코퍼스 확충(지표 가이드·경쟁 분석 문서 추가) 후 groundedness 재측정
- L2 recall 개선: 골드 청크가 top-8에 들도록 검색 가중치/쿼리 확장 튜닝
- v2(단계적 추론)·v3(출력 스키마)는 병목 해소 후 재평가
