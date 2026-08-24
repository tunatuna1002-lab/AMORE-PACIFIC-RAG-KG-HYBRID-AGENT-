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

## 결과
(RESULTS.md 붙여넣기)

## 대표 실패 사례
-

## 결론·채택
- 채택 버전:
- 이유:
- 트레이드오프:

## 다음 실험
-
