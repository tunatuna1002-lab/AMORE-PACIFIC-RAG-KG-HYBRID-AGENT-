"""
색인 청크 엔티티 태그 + 온톨로지 재정렬 보너스 (트랙 4-B, 설계 E9·E10)
====================================================================

색인 시점(`python -m src.rag.build_index`)에 청크마다 EntityLinker 단어사전으로
브랜드·카테고리·지표를 뽑아 Chroma 메타데이터에 저장하고, 조회 시점에는 질의
엔티티와 태그가 겹치는 청크에 **가산 보너스**만 준다. 필터가 아니므로 후보 집합은
줄지 않고 순서만 바뀐다.

결함 F9: 기존 OWL 전략의 엔티티 필터(`EntityLinker.get_ontology_filters`)는
색인에 없는 키(`brand`·`category`)와 canonical id가 아닌 표기(`LANEIGE`·`lip care`)로
`$or` 조건을 만들었고, 매칭 함수는 `$or`를 메타데이터 키로 취급했다. 그래서
엔티티가 연결된 질의는 문서를 0건 받았다.

메타데이터 형식 (Chroma 메타데이터는 리스트를 저장하지 못한다):
    brands      = "|laneige|cosrx|"   (canonical id, 앞뒤 구분자 포함 → "|laneige|" 부분
    categories  = "|lip_care|"          문자열 매칭이 정확 일치가 된다. 없으면 "")
    metrics     = "|sos|hhi|"
    entity_tags_version = 1             (태그가 붙은 청크 표시. 없으면 미태깅 청크)

canonical id는 `EntityLinker.extract_entities`의 단어사전(config/entities.json +
클래스 사전)을 그대로 쓰므로 `evidence_adapters.normalize_brand/normalize_category`와
같은 값이다 (`LANEIGE`·`라네즈` → `laneige`, `Lip Care`·`립케어` → `lip_care`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Any

TAG_DELIMITER = "|"
TAG_VERSION_KEY = "entity_tags_version"
TAG_VERSION = 1

# 메타데이터 키 → EntityLinker.extract_entities 결과 키
TAG_FIELDS: dict[str, str] = {
    "brands": "brands",
    "categories": "categories",
    "metrics": "indicators",
}

# 보너스 대상 필드와 기본 가중치 (config/retrieval_weights.json `entity_tag_bonus`로 덮어씀).
# RRF 점수 척도: 1/(60+rank+1) — 상위권 한 칸 차이가 약 0.00026이므로 0.001은 한 목록 안에서
# 대략 4칸 끌어올리는 크기다. 두 목록(dense·BM25)에 모두 든 청크(≈0.033)와 한 목록에만 든
# 청크(≈0.016) 사이의 간격은 넘지 못한다 — 보너스가 검색 점수를 압도하지 않게 한다.
DEFAULT_BONUS_WEIGHTS: dict[str, float] = {"brands": 0.001, "categories": 0.001}

_RRF_K = 60


@lru_cache(maxsize=1)
def _linker() -> Any:
    from src.rag.entity_linker import EntityLinker

    return EntityLinker(use_spacy=False)


def encode_tags(values: Iterable[str]) -> str:
    """["laneige", "cosrx"] → "|laneige|cosrx|" (중복 제거·정렬). 빈 목록은 ""."""
    cleaned = sorted({str(v).strip().lower() for v in values if v and str(v).strip()})
    cleaned = [v for v in cleaned if TAG_DELIMITER not in v]
    if not cleaned:
        return ""
    return f"{TAG_DELIMITER}{TAG_DELIMITER.join(cleaned)}{TAG_DELIMITER}"


def decode_tags(value: Any) -> list[str]:
    """ "|laneige|cosrx|" → ["laneige", "cosrx"]. 문자열이 아니면 빈 목록."""
    if not isinstance(value, str) or not value:
        return []
    return [v for v in value.split(TAG_DELIMITER) if v]


def extract_tags(text: str) -> dict[str, list[str]]:
    """텍스트 → {"brands": [...], "categories": [...], "metrics": [...]} (canonical id)."""
    entities = _linker().extract_entities(text or "")
    return {field: list(entities.get(source, [])) for field, source in TAG_FIELDS.items()}


def build_tag_metadata(title: str, content: str) -> dict[str, Any]:
    """청크 제목+본문으로 Chroma 메타데이터 태그 필드를 만든다."""
    tags = extract_tags(f"{title or ''}\n{content or ''}")
    metadata: dict[str, Any] = {field: encode_tags(values) for field, values in tags.items()}
    metadata[TAG_VERSION_KEY] = TAG_VERSION
    return metadata


def read_tags(metadata: Mapping[str, Any] | None) -> dict[str, list[str]] | None:
    """메타데이터 → 태그. 태그가 없는(미태깅) 청크면 None."""
    if not metadata or TAG_VERSION_KEY not in metadata:
        return None
    return {field: decode_tags(metadata.get(field)) for field in TAG_FIELDS}


def query_tag_targets(entities: Mapping[str, Any] | None) -> dict[str, set[str]]:
    """질의 엔티티(extract_entities 형식) → 보너스 대상 {필드: canonical id 집합}.

    대상이 없는 필드는 빠진다. 보너스 필드(brands·categories)만 본다.
    """
    if not entities:
        return {}
    targets: dict[str, set[str]] = {}
    for field in DEFAULT_BONUS_WEIGHTS:
        values = entities.get(TAG_FIELDS[field]) or []
        normalized = {str(v).strip().lower() for v in values if v and str(v).strip()}
        if normalized:
            targets[field] = normalized
    return targets


def result_chunk_id(result: Mapping[str, Any]) -> str | None:
    """검색 결과의 청크 id. dense 결과는 `id`, BM25 결과는 metadata의 `chunk_id`/`id`."""
    rid = result.get("id")
    if rid:
        return str(rid)
    metadata = result.get("metadata") or {}
    rid = metadata.get("chunk_id") or metadata.get("id")
    return str(rid) if rid else None


def resolve_bonus_weights(config: Mapping[str, Any] | None) -> dict[str, float]:
    """`entity_tag_bonus` 설정 → 필드별 가중치 (알 수 없는 키·숫자 아님은 무시)."""
    weights = dict(DEFAULT_BONUS_WEIGHTS)
    if isinstance(config, Mapping):
        for field in DEFAULT_BONUS_WEIGHTS:
            value = config.get(field)
            if isinstance(value, int | float) and not isinstance(value, bool):
                weights[field] = float(value)
    return weights


def apply_entity_bonus(
    results: list[dict[str, Any]],
    tags_by_id: Mapping[str, Mapping[str, list[str]]],
    targets: Mapping[str, set[str]],
    weights: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """RRF 병합 결과에 엔티티 태그 보너스를 더해 재정렬한다 (필터 아님).

    score' = base + Σ_f w_f · 1[질의 f ∩ 청크 태그 f ≠ ∅]   (f ∈ brands, categories)
    base   = 결과의 `rrf_score`(모든 결과에 있을 때) 또는 순위 기반 1/(60+rank+1).

    - 결과 개수·구성은 그대로다. 보너스를 받은 결과에만 `entity_bonus`·
      `entity_tag_matches` 키를 붙인다 (원본 dict는 바꾸지 않고 복사한다).
    - 보너스를 받은 결과가 하나도 없으면 입력 순서를 그대로 반환한다.

    Returns:
        (재정렬된 결과, {"candidates", "tagged_candidates", "bonus_candidates"})
    """
    stats = {"candidates": len(results), "tagged_candidates": 0, "bonus_candidates": 0}
    if not results:
        return results, stats

    use_rrf = all(
        isinstance(r.get("rrf_score"), int | float) and not isinstance(r.get("rrf_score"), bool)
        for r in results
    )

    scored: list[tuple[float, dict[str, Any]]] = []
    for rank, result in enumerate(results):
        base = float(result["rrf_score"]) if use_rrf else 1.0 / (_RRF_K + rank + 1)
        chunk_id = result_chunk_id(result)
        tags = tags_by_id.get(chunk_id) if chunk_id else None
        bonus = 0.0
        matches: dict[str, list[str]] = {}
        if tags is not None:
            stats["tagged_candidates"] += 1
            for field, wanted in targets.items():
                weight = float(weights.get(field, 0.0))
                if weight <= 0:
                    continue
                hit = sorted(wanted & set(tags.get(field, [])))
                if hit:
                    bonus += weight
                    matches[field] = hit
        if bonus > 0:
            stats["bonus_candidates"] += 1
            result = {**result, "entity_bonus": bonus, "entity_tag_matches": matches}
        scored.append((base + bonus, result))

    if stats["bonus_candidates"] == 0:
        return results, stats

    # 안정 정렬 — 점수가 같으면 원래 RRF 순서를 유지한다
    reordered = [r for _, r in sorted(scored, key=lambda item: -item[0])]
    return reordered, stats
