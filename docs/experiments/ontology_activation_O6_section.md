## §O6 — OWL 모듈 정리·플래그 이름·문서 (2026-09-18, API $0)

기준 커밋 `e0ffac7`. 결정 OA-1(JSON 원본)에 따라 OE8의 JSON 쪽(OWL 모듈 삭제)을 따른다.

### O6-1. 호출처 확인표 (삭제 전)

조사: `grep -rn` (src·scripts·tests·eval·examples·config·docs 코드 블록, `* 2.*` 제외) + `git log -S "OntologyKnowledgeGraph(" -- src`(도입 커밋 `7ca17ce` 1건뿐 — 서비스 코드가 생성한 적 없음).

| 삭제 대상 심볼 | 호출처 | 종류 | 처리 |
|---|---|---|---|
| `OWLReasoner`·`ConsistencyReport`·`get_owl_reasoner`·`OWLREADY2_AVAILABLE` (`src/ontology/owl_reasoner.py`) | `src/ontology/ontology_knowledge_graph.py` | 같이 삭제되는 모듈 | 삭제 |
| 〃 | `tests/unit/ontology/test_owl_reasoner.py`, `test_owl_consistency.py` | OWL 전용 테스트 | 삭제 |
| 〃 | `tests/integration/test_sprint9_integration.py` `TestOWLConsistencyIntegration`(3개) | OWL 전용 테스트 클래스 | 그 클래스만 삭제 |
| 〃 | `tests/unit/rag/test_owl_strategy_removed.py::test_owl_reasoner_module_is_kept_as_vocabulary` | "어휘 용도로 남는다" 고정 테스트 | "모듈이 삭제됐다"로 반전 |
| `owl_reasoner` 인자·속성 (`src/rag/entity_linker.py` `EntityLinker.__init__`, `get_entity_linker`) | 저장만 하고 읽지 않음. 생성처(src·tests·examples) 모두 인자 없이 호출 | 죽은 인자 | 제거 |
| `OntologyKnowledgeGraph`·`OWL_CLASS_MAPPING` (`src/ontology/ontology_knowledge_graph.py`) | `src/ontology/__init__.py` export만 | export | export 제거 |
| 〃 | `tests/unit/ontology/test_ontology_knowledge_graph.py`, `test_ontology_kg.py` | OKG 전용 테스트 | 삭제 |
| 〃 | `scripts/migrate_kg_to_ontology.py` | 개발 스크립트(OKG 전용) | 삭제 |
| `cosmetics_ontology.owl` | 로드하는 코드 없음(`owl_reasoner.py`도 이 파일을 쓰지 않음, 81행 `<` 때문에 파싱 불가) | 정적 파일 | 삭제 |
| 문서 코드 예시 | `src/rag/README_ENTITY_LINKER.md`(OWLReasoner 통합 예시), `examples/entity_linker_integration.py` docstring | 문서 | 정정 |

**서비스 호출처: 0건.** `src/core/brain.py`·`container.py`에도 없다. `tests/unit/ontology/test_iri_migration.py`는 `KnowledgeGraph`의 IRI 기능 테스트라 OWL 전용이 아니므로 **남긴다.**

owlready2 import 위치(삭제 후): `scripts/export_ontology_owl.py`, `scripts/check_ontology_owl.py`(개발 전용, 유지)와 그 테스트(`importorskip`). 런타임(`src/`) import 0건 → `requirements.txt`에서 `requirements-dev.txt`로 옮긴다. Dockerfile은 `requirements.txt`만 설치한다.

이름만 OWL인 플래그(모듈과 무관, 삭제하지 않고 이름만 바로잡음 — O6-3):

| 호출처 | 옛 이름 | 실제 뜻 |
|---|---|---|
| `hybrid_retriever.py` 규칙 판정 분기, `tool_registry.py` `apply_rules` 가용성 | `reasoner.use_owl_reasoner` (+ `use_unified_reasoner`, OR) | 규칙 추론 on/off |
| `hybrid_retriever.py` KG 조회, `tool_registry.py` `kg_neighbors` | `ontology.use_ontology_kg` | 일반 KG 조회 on/off |
| `eval/ablation.py`, 실험 스크립트 | `FF_REASONER_USE_OWL_REASONER`, `FF_ONTOLOGY_USE_ONTOLOGY_KG` | 위와 같음 — 별칭으로 계속 동작해야 함 |

### O6-2. 삭제한 코드 (`7e50cf1`)

| 파일 | 줄 |
|---|---|
| `src/ontology/owl_reasoner.py` | 1,247 |
| `src/ontology/ontology_knowledge_graph.py` | 398 |
| `src/ontology/cosmetics_ontology.owl` | 770 |
| `scripts/migrate_kg_to_ontology.py` | 229 |
| `tests/unit/ontology/test_owl_reasoner.py`·`test_owl_consistency.py`·`test_ontology_knowledge_graph.py`·`test_ontology_kg.py` | 648 + 228 + 771 + 47 |
| `tests/integration/test_sprint9_integration.py` `TestOWLConsistencyIntegration` | 약 60 |

합계: 소스·스크립트·OWL 2,644줄, 테스트 약 1,750줄(커밋 기준 −4,423 / +33). `entity_linker`의 죽은 `owl_reasoner` 인자 제거. owlready2는 `requirements-dev.txt`로 이동(`src/` import 0건, Dockerfile은 `requirements.txt`만 설치). `scripts/export_ontology_owl.py`·`check_ontology_owl.py`는 유지.

### O6-3. 플래그 이름 (`7f2a3f9`)

| 새 이름 (ENV) | 옛 이름 (ENV) | 기본 |
|---|---|---|
| `reasoner.enabled` (`FF_REASONER_ENABLED`) | `reasoner.use_owl_reasoner` (`FF_REASONER_USE_OWL_REASONER`) | True |
| `kg.enabled` (`FF_KG_ENABLED`) | `ontology.use_ontology_kg` (`FF_ONTOLOGY_USE_ONTOLOGY_KG`) | True |

- 해석 순서: ENV 새 > ENV 옛 > JSON 새 > JSON 옛 > 기본. ENV가 JSON보다 앞선다는 기존 규칙을 지켜, 평가의 규칙 off 명령(`FF_REASONER_USE_OWL_REASONER=false FF_REASONER_USE_UNIFIED_REASONER=false`)이 `config/feature_flags.json`의 새 키(`reasoner.enabled: true`)보다 우선한다.
- 옛 이름이 실제로 값을 정하면 프로세스당 1회 `WARNING` 로그("deprecated; use … instead").
- 규칙 판정 조건은 `use_unified_reasoner() or reasoner_enabled()`로 이전과 같은 뜻. `use_owl_reasoner()`·`use_ontology_kg()` 메서드는 새 메서드로 위임하는 deprecated 별칭으로 남겼다.
- `config/feature_flags.json`은 새 이름으로 바꿨다(`reasoner.enabled`, `kg.enabled`). 테스트: `tests/unit/infrastructure/test_feature_flags_aliases.py`(두 플래그 × 옛/새 × JSON/ENV 조합, 경고 1회, 규칙 off 명령, 저장소 설정 키).

### O6-4. 배포 쪽 온톨로지 상태 (OE10, `471f1ea`)

- `GET /api/v4/brain/status` 응답에 `ontology` 추가: `{version, as_of, class_count, brand_count, use_class_reasoning, kg_write_validation}`. `get_ontology()` 캐시를 쓰고, 로드 실패는 상태 조회를 깨뜨리지 않고 `error` 필드로 싣는다(Brain 초기화 실패 응답에도 포함).
- `scripts/start.py`가 uvicorn 전에 `get_ontology()`를 한 번 불러, 원본 형식 오류면 로그를 남기고 `exit 1`. Dockerfile·Railway 설정은 바꾸지 않았다(선택 과제인 Docker 다단계 Pellet 검증은 하지 않음, FUTURE_WORK 9.10).

### O6-5. 문서 정정 (`35ba26f`, `0e2079f`)

"OWL은 카테고리 계층 어휘로만 사용"(검토 보고서 §2.1-6)을 `[2026-09 사후]` 표시로 정정: `CLAUDE.md`(기술 스택·트리·모듈 표·엔드포인트), 루트 `AGENTS.md`, `src/ontology/AGENTS.md`, `README.md`, `src/rag/retrieval_strategy.py` 모듈 주석, `docs/portfolio/amore_architecture_evidence.md`(원문 보존, 정정 덧붙임), `src/rag/README_ENTITY_LINKER.md`·`examples/entity_linker_integration.py`(삭제 모듈 참조). FUTURE_WORK 9.10 신설.

### O6 게이트

`.venv/bin/python -m pytest tests/unit -q --no-cov` — 5,601 통과, 7 skip (알려진 다른 트랙 실패 `test_ontology_offline_checks.py::test_real_retriever_filters_are_found`는 제외하고 실행). `tests/integration/test_sprint9_integration.py` 15 통과.
