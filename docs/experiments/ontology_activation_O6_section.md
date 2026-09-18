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
