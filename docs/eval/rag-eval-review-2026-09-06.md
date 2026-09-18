# RAG 정량평가 방식과 골든셋 검토 (2026-09-06)

기준: `eval/baselines/v8.1-2026-08-30/` (172문항), `eval/data/golden/laneige_golden_v2.jsonl`,
`docs/experiments/eval_cycle*.md`, 로컬 DB `data/amore_data.db` (최신 스냅샷 2026-08-31).

**결론부터.** 평가 하네스와 개선 사이클 기록은 잘 설계되어 있고 측정 결함을 스스로
찾아 고친 이력이 정직하게 남아 있다. 다만 골든셋의 **데이터형 문항은 정답 수치가 실제
크롤 데이터와 맞지 않고**, 40문항에 달린 `expected_values`는 **어떤 채점기도 읽지 않는다**.
따라서 정답 일치 계열 지표(token F1, 의미 유사도, `L5_wrong_answer` 97건)는 데이터형
문항에서 문체 유사도만 재고 있다. 근거성(groundedness)은 답변을 검색 컨텍스트와
대조하므로 이 결함과 무관하게 유효하다.

---

## 1. 정량평가가 이뤄진 방식

### 1-a. 하네스 구조

`eval/`는 실제 `HybridChatbotAgent`를 문항마다 호출하고, 파이프라인 중간 산출물을
5개 레이어의 트레이스로 잡아 채점한다.

| 레이어 | 측정 대상 | 핵심 지표 | 게이트 |
|---|---|---|---|
| L1 질의 이해 | 브랜드·카테고리·개념 추출 | Entity F1, Concept F1, Constraint F1 | 0.50 / 0.30 |
| L2 문서 검색 | ChromaDB top-8 | 개념 단위 Recall, MRR, Precision | 개념 Recall 0.80 (requires_kg=False만) |
| L3 KG | 엔티티·엣지 회수 | Hits@8, Edge Recall, Edge Precision | 0.80 / 0.50 (requires_kg=True만) |
| L4 온톨로지 | 제약 위반, 타입 일관성 | 위반율, 일관성 | 0.05 / 0.90 |
| L5 답변 | 최종 답변 | Token F1, 의미 유사도, LLM Judge 근거성·관련성 | F1 0.50 또는 유사도 0.65, 근거성 0.70, 관련성 0.70 |

- 종합 점수 가중: L5 45%, L2·L3 35%, L1 10%, L4 10% (`eval/metrics/aggregator.py`).
- 통과 판정: 게이트 10개를 모두 넘어야 pass. 하나라도 미달이면 fail 태그를 붙인다.
- Judge: gpt-4.1-mini, RAGAS 방식 프롬프트(claim 단위 근거 판정, JSON 출력). `eval/judge/llm.py`.
- 의미 유사도: `paraphrase-multilingual-MiniLM-L12-v2` (한국어 골드 때문에 다국어 모델 필수).
- 실행 조건 고정: 외부신호 OFF, 답변 온도 0.1, 질의 확장 온도 0, top_k 8, 코퍼스 358청크.
- 회귀 비교: `python -m eval.cli compare --baseline-name v8.1-2026-08-30 --report <report.json>`.
- 부속 도구: ablation(`eval/ablation.py`), 프롬프트 실험, 포트폴리오 리포트 생성기.

### 1-b. 추이 (2026-08-30 하루 7사이클)

| 지표 | v1.0 | v4.1 | v7.1 | v8.1 (현행 기준) |
|---|---|---|---|---|
| 문항 수 | 160 | 160 | 160 | 172 |
| 종합 점수 | 0.428 | 0.507 | 0.562 | 0.551 |
| 게이트 통과 | 0 | 3 | 17 | 16 (9.3%) |
| L5 근거성 | 0.365 | 0.650 | 0.678 | 0.691 |
| L5 관련성 | 0.768 | 0.786 | 0.792 | 0.765 |
| L5 Token F1 | 0.100 | 0.099 | 0.106 | 0.105 |
| L2 개념 Recall | - | - | 0.603 | 0.579 |
| L3 Hits@8 | 0.650 | 0.725 | 0.775 | 0.797 |
| L1 Entity / Concept F1 | 0.357 / 0.127 | 0.487 / 0.334 | 0.606 / 0.357 | 0.636 / 0.351 |
| 평균 지연 | 12.7s | 8.3s | 7.3s | 8.1s |

상승분의 상당 부분은 시스템 개선이 아니라 **측정 결함 수정**이다. 동시 실행 트레이스
오염(사이클 3), dict 제약으로 7문항 채점 크래시(사이클 4), 청크 라벨 재매핑(사이클 2·6),
엣지 지표 F1→Recall 전환(사이클 4)이 그것이며, 각 문서에 라벨 변경분과 시스템 개선분의
분리표가 남아 있다. 실제 시스템 개선은 리랭커 제거·RRF 융합(사이클 6), 코퍼스 증분 색인과
base64 잡음 제거(사이클 5·6), 컨텍스트 상한 3→8(사이클 2), IR 라우터 패턴(사이클 7)이다.

v8.1 도메인별: edge 0.658, metric 0.601(통과 9/30), brand 0.556(0), multi_hop 0.551,
product 0.550, time 0.527, market 0.502(0), ir 0.413(0).
fail 태그: `L5_wrong_answer` 97, `L5_grounding_fail` 84, `L3_edge_fail` 67, `L1_concept_fail` 65.

### 1-c. 평가 방식의 한계

1. **실행 간 분산이 신호보다 크다.** 8월 31일 Phase 4 검증(`docs/experiments/refactor_phase4_2026-08-31.md`)에서
   동일 코드 재실행 시 172문항 중 답변이 같은 문항은 7개, 통과는 12→9로 흔들렸다.
   모든 baseline이 1회 실행이라 0.03 이하 델타는 해석할 수 없다.
2. **Token F1 0.10은 무의미하다.** 골드는 1~2문장, 에이전트는 장문 마크다운이다.
   의미 유사도 게이트로 우회 중이지만 지표 자체는 여전히 보고된다.
3. **Judge와 답변 모델이 같다** (gpt-4.1-mini). 자기 선호 편향 가능성이 있고, 검증된 적 없다.
4. **종합 점수 공식과 게이트가 다른 지표를 쓴다.** 공식은 청크 Recall·Edge F1, 게이트는
   개념 Recall·Edge Recall. 이중 기준이라 종합 점수가 게이트 개선을 반영하지 못한다.
5. **비용 추적이 0으로 기록된다.** 모든 baseline에서 `total_tokens`, `total_cost_usd`가 0.
   `CostTracker`는 있으나 러너가 호출하지 않는다.
6. 기록된 미해결 결함: LLM 호출 타임아웃 없음(2차 실행이 135/172에서 무기한 정지),
   리포트의 `metadata.requires_kg` 직렬화 유실, 기준선과 현재의 L1 concept 임계 불일치.
7. **인텐트 top_k 5 vs 평가 k 8 불일치**는 작업 트리에 수정되어 있으나 **미커밋** 상태다
   (`src/rag/retrieval_strategy.py`, 사이클 8 실측: 개념 recall 0.576→0.671). 기록 문서와
   baseline이 아직 없다.

---

## 2. 골든셋 평가

### 2-a. 구성

| 항목 | 값 |
|---|---|
| 문항 수 | 172 (v1 40문항과 ID 불겹침, v1은 미사용 레거시) |
| 도메인 | metric 30, product 30, brand 25, market 25, multi_hop 20, edge 15, time 15, ir 12 |
| 난이도 | easy 33, medium 95, hard 44 |
| requires_kg | True 130, False 42 |
| 언어 | 한국어 167, 영어 5 |
| 골드 채움 | answer 172, doc_chunk_groups 149, kg_entities 170, kg_edges 105, expected_values 40, constraints 7 |
| 중복 ID·질문 | 0 |
| 골드 답변 길이 | 70~337자, 평균 187자 |

### 2-b. 잘 된 점

- 레이어별 골드 필드가 분리되어 있어 어느 층에서 실패했는지 진단 가능하다.
- L2 골드를 개념당 청크 **집합**으로 재설계해 라벨 입도 문제를 해결했다(사이클 6).
- IR 12문항은 원문 수치를 그대로 인용하고 청크 실재를 스크립트가 검증한다.
- 오타·범위 밖 질문을 다루는 edge 도메인이 있다.
- 골든셋 생성·재매핑 스크립트가 멱등이고 dry-run을 지원한다.

### 2-c. 문제점

**1. 데이터형 문항의 정답 수치가 실제 DB와 불일치한다.** 골드는 도메인 지식으로 작성된
추정치이고 크롤 데이터와 대조된 적이 없다. 최신 스냅샷(2026-08-31) 대조:

| 문항 | 골드 주장 | DB 실측 |
|---|---|---|
| lg048, lg050, lg164, lg169 | LANEIGE Lip Care SoS 5.2% | 2.0% (2025-12 5.0%에서 하락, 월평균 4.4→2.5→2.0→2.9) |
| lg049, lg149 | Lip Care HHI 0.12~0.15 | 0.068 (전기간 최대 0.15) |
| lg148 | Face Powder HHI 0.15~0.20 | 0.053 |
| lg148 | Skin Care HHI 0.08~0.10 | 0.067 |
| lg183 | TIRTIR Face Powder SoS 2→9% | face_powder에 TIRTIR 데이터 없음 (face_makeup·makeup에만) |
| lg071, lg162 | Lip Sleeping Mask Top 5 | 7위 |
| lg076 | 리뷰 100,000건 | 37,380건 |
| lg051, lg181 | CPI 1.2 | lip_care CPI NULL, face_powder 111.1 (스케일 자체가 다름) |
| lg070 | MEDICUBE Skin Care SoS 2~4% | 13.5% |
| lg066 | Beauty of Joseon Skin Care SoS 3~5% | 데이터 없음 |
| lg175 | 최근 3개월 SoS 5.2→6.1% 상승 | 해당 기간(2025-11~2026-01) 5.0→4.4% 하락 |
| lg161 | 1위 브랜드와 격차 10~13%p | 1위 eos 9.0%, 격차 7%p |

사이클 4에서 골드 엣지의 46%가 KG에 없다고 확인된 것과 같은 성격이다.

**2. `expected_values`가 채점에 쓰이지 않는다.** 40문항에 수치 기대값이 있으나
`eval/metrics/` 어디에서도 참조하지 않는다(grep 0건). 수치 정확도 검증이 실질적으로 없다.

**3. 시점 정보가 없다.** 20문항이 특정 기간이나 Top N 기준을 답변에 박아 두었고
12문항은 "현재/최근" 수치를 단정한다. 골든셋에 as-of 날짜 필드가 없어 데이터가 바뀌면
자동으로 틀린다. DB에도 2026년 5~7월 스냅샷이 없어(월별 스냅샷 수: 12월 8, 1월 30, 2월 14,
3월 6, 4월 13, 8월 18, 9월 5) 추이 문항 다수는 애초에 답할 데이터가 없다.

**4. 분포 편향.** LANEIGE 엔티티 등장 문항 126/172(73%). 영어 질의 5문항(3%).
L2 골드 청크 참조 1,516건 중 1,089건이 `laneige_strategy_2026` 한 문서. L2 게이트는
requires_kg=False 42문항(24%)에만 적용된다.

**5. 규모.** 도메인당 12~30문항이라 도메인별 델타의 신뢰구간이 넓다. IR은 12문항 중
통과 0건이라 아직 벤치마크로 기능하지 못한다.

---

## 3. 권고 (우선순위순)

1. **골든셋 2층 분리.** 정의·해석형(문서 근거, 정적)과 데이터형(DB 스냅샷 기준)으로 나누고,
   데이터형은 `as_of` 날짜를 달아 `expected_values`를 DB에서 스크립트로 자동 생성한다.
   그 위에 수치 허용오차 지표를 L5에 배선한다.
2. **분산 정량화.** 동일 조건 3회 반복 실행으로 노이즈 폭을 먼저 잰다. 그 전까지 소폭
   델타를 개선·회귀로 판정하지 않는다.
3. **불일치 골드 처리.** 위 표의 문항은 DB 기준으로 재작성하거나 "도메인 기대치" 태그로
   분리해 정답 일치 채점에서 제외한다. 골드를 시스템 출력에 맞춰 고치는 것은 게이밍이므로
   금지. 기준은 원자료(DB·문서)뿐이다.
4. **기술 부채 4건.** LLM 호출 타임아웃, `requires_kg` 직렬화, 비용 추적 0, 종합 점수
   공식과 게이트 지표 정합.
5. **Judge 독립성.** 답변 모델과 다른 계열의 Judge로 교차 검증하거나 최소 1회 사람 라벨과
   일치율을 잰다.
6. **미커밋 top_k 수정 마무리.** 사이클 8 기록 문서와 baseline 저장 후 커밋.

실행 지시 프롬프트: `docs/eval/opus5-eval-remediation-prompt-2026-09-06.md`.

---

## 4. 재현

```bash
# 골든셋 분포 프로파일 (무비용)
python3 -c "
import json,collections
rows=[json.loads(l) for l in open('eval/data/golden/laneige_golden_v2.jsonl') if l.strip()]
print(len(rows), collections.Counter(r['metadata']['domain'] for r in rows))
print('expected_values:', sum(1 for r in rows if r['gold'].get('expected_values')))"

# expected_values 사용처 확인 (0건이어야 문제 재현)
grep -rn expected_values eval/metrics/ eval/runner.py

# 골드 vs DB 대조 예시
sqlite3 data/amore_data.db "select snapshot_date,sos from brand_metrics where lower(brand) like '%laneige%' and category_id='lip_care' order by snapshot_date desc limit 3"

# 회귀 비교
python3 -m eval.cli compare --baseline-name v8.1-2026-08-30 --report <report.json>
```
