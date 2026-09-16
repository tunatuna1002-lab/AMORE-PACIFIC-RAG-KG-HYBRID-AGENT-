# Future Work

> 폴더 구조 리팩토링(Phase 0~5) 이후 남은 작업.
> 최종 업데이트: 2026-09-16 · 기준 커밋: `claude/trusting-babbage-v1cfrw`
>
> 전체 계획은 `docs/plans/folder-structure-refactor-plan-2026-09-01.md`,
> 진행 로그는 그 문서 §9.

---

## 0. 이 문서에서 걷어낸 것 (이미 해결)

기록용으로만 남긴다. 다시 열지 말 것.

| 과거 항목 | 결과 |
|-----------|------|
| 순환 의존성 23건 → 0 | 해결. `tests/unit/test_import_graph.py` 가 정적으로 고정 |
| Application Layer 120 LOC | 해결. `workflows/` 2개 + `services/` 12개 |
| `dashboard_api.py` 3,236줄 | 해결. 195줄 진입점 + `routes/` 13개 |
| `ontology/business_rules.py` 1,540줄 | 해결. `ontology/rules/` 6개로 분할 후 shim 삭제 |
| `knowledge_graph.py` 1,514줄 | 해결. 586줄 |
| 커버리지 43% (목표 60%) | 해결. 실측 79%대, `fail_under = 75` 게이트 |
| `@app.on_event("startup")` | 해결. `lifespan` 전환 (`dashboard_api.py`) |
| `knowledge_graph.json` 동시 쓰기 | 해결. `threading.Lock` + tmp → `replace()` 원자적 쓰기 |
| `TestResult` 이름 충돌 | 해결. 해당 클래스 없음 |
| `competitors.json` dead config 의심 | **오진**. 소비자 5곳(크롤러·라우트·스크래퍼·익스포터·KG 빌더). 유지 |
| VULN-006/007/008/011/012 | 해결. `hmac.compare_digest`, 챗 `verify_api_key`, `prompt_guard.py`, `middleware/{csrf,security_headers}.py`, `uuid4`(os.urandom 기반) |

---

## 1. 동작을 바꾸므로 미룬 항목 (각각 별도 커밋 + RED 테스트 선행)

직전 세션이 근거를 코드 docstring 에 남기고 의도적으로 보류한 것들이다.
현재 동작은 등가·특성화 테스트가 고정하고 있으므로, 고치려면 그 핀을
같은 커밋에서 `CHANGED (Fx):` 근거 주석과 함께 갱신해야 한다.

| # | 위치 | 내용 | 성격 |
|---|------|------|------|
| 1 | `src/rag/fusion/rrf.py` | dedup 키 폴백이 호출처 3곳마다 다름 (`""` / `str(rank)` / content 해시). `""` 는 id 없는 문서를 전부 하나로 병합하고, `str(rank)` 는 서로 다른 리스트의 동순위 문서를 잘못 병합 | **잠복 결함** — 검색 결과가 바뀜 |
| 2 | `src/rag/context_render.py` | KG 사실 렌더러 문구가 두 갈래. 수치는 동일, 문구만 다름. 승자는 `ContextBuilder`(출처를 등록하는 유일한 렌더러) | 프롬프트 텍스트 변경 → 골든셋 실행 동반 |
| 3 | `src/rag/fusion/hybrid_search.py` | BM25+RRF 이중 실행. `DocumentRetriever.search` 가 RRF 융합한 결과를 `HybridRetriever._hybrid_search` 가 다른 점수 공간에서 또 융합 | 중복 연산 + 점수 왜곡 |
| 4 | `src/rag/hybrid_retriever.py` | doc-type 필터가 BM25 레그에 도달하지 않음 | **잠복 결함** (테스트로 고정됨) |
| 5 | `src/rag/search_cache.py` | 캐시 키에 임베딩 모델명 누락 → 모델 교체 시 stale hit | **잠복 결함** |
| 6 | `config/retrieval_weights.json` | `weights` 블록은 사문(프로덕션은 항상 `_INTENT_STRATEGY_MAP` 가중치를 넘김). **단 같은 파일의 `freshness` 와 `max_context_items` 는 살아 있다** — 특성화 테스트가 `rag_chunks: 8` 을 여기서 가져오는 것을 고정. 파일을 지우지 말고 `weights` 만 정리 | 죽은 설정 정리 |
| 7 | `tests/integration/test_rag_integration.py` | `src.rag.hybrid_retriever` 에서 `QueryIntent`/`get_doc_type_filter` 를 import. `src.core.intent` 로 돌리면 `src/rag/legacy_intent.py` 삭제 가능 | shim 제거 |
| 8 | `src/api/dashboard_shape.py` | API 계층에 있는 순수 어댑터라 `application/` 도 `tools/` 도 import 불가. `application/services/` 로 옮기면 뷰모델도 서비스에서 조립 가능 | 계층 위치 교정 |

---

## 2. 사용자 확인 대기 (계획 문서 Q4)

삭제 여부를 확정받지 못해 손대지 않았다.

- [ ] 동기 `/api/export/docx` (비동기 `/api/export/async/start` 와 중복)
- [ ] `/api/v3/*` 레거시 엔드포인트
- [ ] `/api/chat/memory/*`

> **주의**: `JobType.EXPORT_DOCX` 는 **삭제 대상이 아니다.**
> `/api/export/async/start` 가 여전히 이 `job_type` 값을 받고 여러 테스트가 그것을 쓴다.
> 사라진 것은 핸들러(`export_handlers.handle_export_docx`)뿐이다.
> (과거 서브에이전트가 "핸들러가 없으니 지워도 된다"고 잘못 보고한 항목 — grep 으로 직접 검증할 것.)

---

## 3. 골든셋 회귀 게이트 활성화 (환경 의존)

`tests/eval/test_golden_replay_gate.py` 는 기록 파일
`eval/baselines/replay/subset_nokg.jsonl` 이 없어 **skip 된다.**

활성화 절차 — `OPENAI_API_KEY` 가 있고 OpenAI 아웃바운드가 열린 환경에서 1회:

```bash
python3 scripts/record_golden_replay.py
git add eval/baselines/replay/subset_nokg.jsonl && git commit
```

기록을 커밋하면 이후 **오프라인 CI 에서 자동 동작**한다.
리팩토링 컨테이너는 키도 아웃바운드도 없어 여기서는 만들 수 없다.

---

## 4. 남은 분할 후보

1,000줄 이상. 급하지 않으며, 건드릴 때 특성화 테스트를 먼저 깔 것.

| 파일 | 줄 수 | 비고 |
|------|-------|------|
| `src/tools/exporters/dashboard_exporter.py` | 1,595 | |
| `src/tools/scrapers/amazon_scraper.py` | 1,561 | 셀렉터 폴백 로직이 대부분 |
| `src/tools/collectors/external_signal_collector.py` | 1,338 | 소스별 분리 후보 |
| `src/application/workflows/batch_workflow.py` | 1,319 | 단계별 분리 후보 |
| `src/agents/hybrid_insight_agent.py` | 1,312 | |
| `src/tools/storage/sqlite_storage.py` | 1,274 | |
| `src/core/brain.py` | 1,218 | 스케줄러는 이미 `brain_scheduler.py` 로 분리됨 |
| `src/rag/entity_linker.py` | 1,166 | |
| `src/agents/period_insight_agent.py` | 1,113 | |
| `src/ontology/owl_reasoner.py` | 1,109 | 배치 전용 |
| `src/ontology/reasoner.py` | 1,004 | |

`src/api/routes/alerts.py` 의 인라인 HTML 분리는 완료됐다(737 → 589줄,
`src/api/templates/` 로 이동). 남은 589줄은 라우트 로직이라 더 줄이려면
엔드포인트를 서비스로 더 밀어내야 한다.

---

## 5. 남은 직접 import (Protocol DI 전환 후보)

| 파일 | 직접 import 대상 |
|------|-----------------|
| `src/agents/hybrid_insight_agent.py` | `ExternalSignalCollector`, `MarketIntelligenceEngine`, `InsightSourceBuilder` |
| `src/agents/period_insight_agent.py` | `PeriodAnalyzer` |
| `src/api/routes/signals.py` | `ExternalSignalCollector` |

---

## 6. 기타

- [ ] **VULN-005 (HIGH)**: Dockerfile 에 non-root `USER` 추가 — **미해결**
      (현재 root 로 실행. `/data` 볼륨 권한과 함께 바꿔야 해서 배포 검증 필요)
- [ ] **`google-adk` 삭제 검토** — `requirements.txt:6` 에 선언돼 있으나
      `src/` `scripts/` `main.py` 어디에서도 import 하지 않는다
      (`grep -rn "google.adk\|google_adk"` → 0건). 무거운 패키지라 삭제 이득이 크지만,
      전이 의존으로만 들어오던 `aiosqlite` 를 직접 선언으로 고정한 것처럼
      다른 전이 의존이 더 있는지 클린 설치로 확인한 뒤 지울 것
- [ ] Pydantic 스키마 기반 config 검증 (`config/*.json`)
- [ ] 커버리지가 얇은 순서: `api/routes` → `ontology/rules` → `tools/notifications`
- [ ] `tests/unit/tools/test_sqlite_storage.py::test_update_products_sync` 의
      `except TypeError: pytest.skip(...)` 제거. 현재는 통과하지만(스킵 분기 미도달),
      시그니처가 바뀌면 실패 대신 조용히 스킵되는 구조다
