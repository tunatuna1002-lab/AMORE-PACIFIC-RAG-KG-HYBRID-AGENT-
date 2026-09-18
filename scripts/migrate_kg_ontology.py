#!/usr/bin/env python3
"""
KG 온톨로지 정리 마이그레이션 (트랙 O5, OE6) [2026-09 사후]

입력 KG JSON **사본**에 ``src/ontology/kg_write_validation.py``의 enforce 규칙을 적용해
정리된 KG JSON을 만든다. 운영 ``data/``는 건드리지 않는다(OE6).

사용:
    # 기본은 dry-run: 변경 목록·건수만 출력하고 아무것도 쓰지 않는다
    .venv/bin/python scripts/migrate_kg_ontology.py --in <kg.json> --out <kg.json>
    # --apply: --out과 변경 기록(--log, 기본 <out>.changes.json)만 쓴다
    .venv/bin/python scripts/migrate_kg_ontology.py --in <kg.json> --out <kg.json> --apply

--apply는 ``--out``이 ``--in``과 같거나, 저장소(워크트리 포함)의 ``data/`` 아래이거나,
``/data`` 아래이면 거부한다. 운영 KG에 적용하려면 사본으로 만든 뒤 소유자가 직접 복사한다.

연산(순서 고정, 결정적 — 현재 시각을 쓰지 않는다):
1. 트리플마다 enforce 정규화: 술어 정식화(``hasPosition`` 분리·``rankedIn``·``ownedBy``),
   브랜드 정식 표기(등록부 표시 이름 소문자), 가짜 브랜드 트리플 삭제, 도메인·범위·리터럴
   위반 삭제, 나눌 수 없는 ``hasPosition`` 삭제. 수치 트리플의 ``as_of``는 속성의 날짜 키
   (``as_of``·``snapshot_date``·``collected_at``)나 ``valid_from``에서만 채운다.
   ``created_at``은 쓰지 않는다 — enricher는 같은 (s,p,o)를 다시 쓸 때 속성만 덮어쓰고
   ``created_at``은 처음 값을 유지하므로, ``created_at``은 값의 관측 날짜가 아니다.
   채울 수 없으면 남기고 ``undated_numeric``으로 센다(지어내지 않는다).
2. 정규화 후 같은 (s,p,o)는 합친다(``KnowledgeGraph.add_relation``과 같은 규칙: 먼저 나온
   트리플을 남기고 나중 속성으로 update, confidence는 큰 값).
3. 대칭 술어(스키마 ``symmetric``: ``competesWith``·``siblingBrand``)의 빠진 역방향을 추가한다.
   스키마가 대칭으로 선언했으므로 새 사실을 지어내는 것이 아니다. 추가된 트리플은
   ``properties.inferred_by = "symmetric:<술어>"``로 표시한다. (공동 출현 기반 competesWith의
   품질 문제 자체는 고치지 않는다 — 검토 보고서 §3.2.)
4. ``entity_metadata``: 브랜드 키를 정식 표기로 합치고(기존 값 우선), 온톨로지가 아는 개체에
   ``type``·``ontology_types``를 없을 때만 넣는다.

출력: 사유별 건수(removed/modified/added/metadata/info), 정합성 위반 수 전후 비교,
변경 기록 JSON(--apply 시).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ontology.kg_write_validation import (  # noqa: E402
    canonical_brand_string,
    check_triple,
    entity_types,
    normalize_triple,
    record_types,
    resolve_predicate,
)
from src.ontology.ontology import get_ontology  # noqa: E402

VIOLATION_CODES = (
    "non_canonical_predicate",
    "non_canonical_brand",
    "placeholder_brand",
    "domain_violation",
    "range_violation",
    "literal_violation",
    "missing_as_of",
    "unresolvable_legacy_predicate",
)


@dataclass
class MigrationResult:
    data: dict[str, Any]
    changes: list[dict[str, Any]] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)


def _key(t: dict[str, Any]) -> tuple[str, str, str]:
    return (t["subject"], t["predicate"], str(t["object"]))


def _spec_for(onto: Any, t: dict[str, Any]) -> Any:
    canonical, _ = resolve_predicate(onto, t["predicate"], t.get("properties"))
    return onto.predicate_spec(canonical) if canonical else None


def _brand_positions(onto: Any, t: dict[str, Any]) -> list[str]:
    spec = _spec_for(onto, t)
    if spec is None:
        return []
    out = []
    if spec.domain == "Brand":
        out.append(t["subject"])
    if spec.kind == "object" and spec.range == "Brand":
        out.append(t["object"])
    return [v for v in out if isinstance(v, str)]


def _typed_entities(onto: Any, triples: list[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    types: dict[str, tuple[str, ...]] = {}
    for t in triples:
        spec = _spec_for(onto, t)
        if spec is None:
            continue
        values = [t["subject"]] + ([t["object"]] if spec.kind == "object" else [])
        for v in values:
            if v not in types:
                found = entity_types(onto, v)
                if found:
                    types[v] = found
    return types


def _stats(triples: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(t["predicate"] for t in triples)
    return {
        "total_triples": len(triples),
        "unique_subjects": len({t["subject"] for t in triples}),
        "unique_objects": len({t["object"] for t in triples}),
        "relation_types": dict(by_type),
    }


# ----------------------------------------------------------------------
# Consistency report
# ----------------------------------------------------------------------


def consistency_report(data: dict[str, Any], onto: Any = None) -> dict[str, int]:
    """KG 정합성 위반 수 (LLM 호출 없음).

    - ``violation:<code>``: 트리플별 ``check_triple`` 사유 (온톨로지 밖 술어는 제외)
    - ``case_duplicate_brands``: 브랜드 자리에서 표기가 둘 이상인 등록부 브랜드 수
    - ``asymmetric_symmetric_edges``: 대칭 술어인데 역방향이 없는 트리플 수
    - ``untyped_entities``: 온톨로지가 아는 개체인데 ``entity_metadata.ontology_types``가 없는 수
    - ``outside_ontology_triples``: 정보용
    """
    onto = onto or get_ontology()
    triples = data.get("triples", [])
    report: dict[str, int] = {f"violation:{c}": 0 for c in VIOLATION_CODES}
    outside = 0
    surfaces: dict[str, set[str]] = defaultdict(set)
    keys = {_key(t) for t in triples}
    asym = 0
    for t in triples:
        codes = check_triple(onto, t["subject"], t["predicate"], t["object"], t.get("properties"))
        if codes == ["outside_ontology"]:
            outside += 1
            continue
        for code in codes:
            report[f"violation:{code}"] = report.get(f"violation:{code}", 0) + 1
        for value in _brand_positions(onto, t):
            bid = onto.normalize_brand(value)
            if bid is not None:
                surfaces[bid].add(value)
        spec = _spec_for(onto, t)
        if spec is not None and spec.symmetric:
            if (t["object"], t["predicate"], t["subject"]) not in keys:
                asym += 1
    meta = data.get("entity_metadata", {}) or {}
    untyped = sum(
        1
        for entity in _typed_entities(onto, triples)
        if "ontology_types" not in (meta.get(entity) or {})
    )
    report["case_duplicate_brands"] = sum(1 for s in surfaces.values() if len(s) > 1)
    report["asymmetric_symmetric_edges"] = asym
    report["untyped_entities"] = untyped
    report["outside_ontology_triples"] = outside
    report["total_violations"] = (
        sum(v for k, v in report.items() if k.startswith("violation:"))
        + report["case_duplicate_brands"]
        + asym
        + untyped
    )
    return report


# ----------------------------------------------------------------------
# Migration
# ----------------------------------------------------------------------


def migrate(
    data: dict[str, Any], onto: Any = None, input_sha256: str | None = None
) -> MigrationResult:
    """enforce 규칙을 기존 KG dict에 적용한 새 dict (입력은 바꾸지 않는다)."""
    onto = onto or get_ontology()
    src = copy.deepcopy(data)
    changes: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()
    modified: Counter[str] = Counter()
    added: Counter[str] = Counter()
    info: Counter[str] = Counter()
    metadata_counts: Counter[str] = Counter()

    # 1. per-triple normalization
    staged: list[tuple[int, dict[str, Any], list[str]]] = []
    for idx, t in enumerate(src.get("triples", [])):
        before = [t["subject"], t["predicate"], t["object"]]
        as_of = str(t["valid_from"]) if t.get("valid_from") else None
        norm = normalize_triple(
            onto,
            t["subject"],
            t["predicate"],
            t["object"],
            t.get("properties"),
            as_of=as_of,
            on_missing_as_of="keep",
        )
        for note in norm.notes:
            info[note] += 1
        if norm.blocked:
            removed[norm.blocked] += 1
            changes.append(
                {"action": "removed", "reason": norm.blocked, "index": idx, "triple": before}
            )
            continue
        new_t = dict(t)
        new_t.update(
            subject=norm.subject,
            predicate=norm.predicate,
            object=norm.object,
            properties=norm.properties,
        )
        staged.append((idx, new_t, list(norm.changes)))

    # 2. merge duplicates (add_relation semantics)
    kept: dict[tuple[str, str, str], dict[str, Any]] = {}
    kept_changes: dict[tuple[str, str, str], tuple[int, list[str], list[Any]]] = {}
    original = src.get("triples", [])
    for idx, t, reasons in staged:
        key = _key(t)
        if key in kept:
            existing = kept[key]
            existing["properties"].update(t["properties"])
            existing["confidence"] = max(existing.get("confidence", 1.0), t.get("confidence", 1.0))
            removed["merged_duplicate"] += 1
            changes.append(
                {
                    "action": "removed",
                    "reason": "merged_duplicate",
                    "index": idx,
                    "triple": [
                        original[idx]["subject"],
                        original[idx]["predicate"],
                        original[idx]["object"],
                    ],
                    "merged_into": list(key),
                }
            )
            continue
        kept[key] = t
        o = original[idx]
        kept_changes[key] = (idx, reasons, [o["subject"], o["predicate"], o["object"]])

    for key, (idx, reasons, before) in kept_changes.items():
        if reasons:
            for r in reasons:
                modified[r] += 1
            changes.append(
                {
                    "action": "modified",
                    "reasons": reasons,
                    "index": idx,
                    "before": before,
                    "after": list(key),
                }
            )

    triples = list(kept.values())

    # 3. symmetric closure
    keys = set(kept)
    extra: list[dict[str, Any]] = []
    for t in triples:
        spec = _spec_for(onto, t)
        if spec is None or not spec.symmetric:
            continue
        rev = (t["object"], t["predicate"], t["subject"])
        if rev in keys:
            continue
        keys.add(rev)
        new_t = dict(t)
        new_t.update(
            subject=t["object"],
            object=t["subject"],
            properties={
                **t["properties"],
                "inferred_by": f"symmetric:{t['predicate']}",
                "inferred_from": list(_key(t)),
            },
            source="migrate_kg_ontology",
        )
        extra.append(new_t)
        added["symmetric_closure"] += 1
        changes.append({"action": "added", "reason": "symmetric_closure", "triple": list(rev)})
    triples.extend(extra)

    # 4. entity metadata: canonical brand keys, then type records
    meta_in = src.get("entity_metadata", {}) or {}
    meta: dict[str, dict[str, Any]] = {}
    for entity in meta_in:
        canon = canonical_brand_string(onto, entity)
        target = entity if canon is None or onto.is_placeholder(entity) else canon
        if target != entity:
            metadata_counts["brand_key_canonicalized"] += 1
            changes.append(
                {
                    "action": "metadata",
                    "reason": "brand_key_canonicalized",
                    "entity": entity,
                    "into": target,
                }
            )
        bucket = meta.setdefault(target, {})
        for k, v in meta_in[entity].items():
            bucket.setdefault(k, v)
    types = _typed_entities(onto, triples)
    for entity in sorted(types):
        if record_types(meta, {entity: types[entity]}):
            metadata_counts["type_record_added"] += 1
            changes.append({"action": "metadata", "reason": "type_record_added", "entity": entity})

    out: dict[str, Any] = {
        "version": src.get("version", "2.0"),
        "triples": triples,
        "entity_metadata": meta,
        "stats": _stats(triples),
        "saved_at": src.get("saved_at"),
        "ontology_migration": {
            "script": "scripts/migrate_kg_ontology.py",
            "ontology_version": getattr(onto, "version", ""),
            "ontology_as_of": getattr(onto, "as_of", ""),
            "input_sha256": input_sha256,
            "input_triples": len(original),
        },
    }
    counts = {
        "removed": dict(sorted(removed.items())),
        "modified": dict(sorted(modified.items())),
        "added": dict(sorted(added.items())),
        "metadata": dict(sorted(metadata_counts.items())),
        "info": dict(sorted(info.items())),
    }
    return MigrationResult(out, changes, counts)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _protected_dirs() -> list[Path]:
    dirs = [Path("/data")]
    for ancestor in (_SCRIPT_DIR, *_SCRIPT_DIR.parents):
        if (ancestor / ".git").exists():
            dirs.append((ancestor / "data").resolve())
    return dirs


def _refusal(in_path: Path, out_path: Path) -> str | None:
    out_r = out_path.resolve()
    if out_r == in_path.resolve():
        return f"--out must differ from --in ({out_r})"
    for d in _protected_dirs():
        if out_r == d or out_r.is_relative_to(d):
            return f"--out {out_r} is under protected directory {d} (production data, OE6)"
    return None


def _dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _print_counts(result: MigrationResult, before: dict[str, int], after: dict[str, int]) -> None:
    print("== change counts by reason ==")
    for group, values in result.counts.items():
        for reason, n in values.items():
            print(f"  {group:9s} {reason:32s} {n}")
    print("== consistency (before -> after) ==")
    for key in sorted(set(before) | set(after)):
        print(f"  {key:40s} {before.get(key, 0):6d} -> {after.get(key, 0):6d}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--in", dest="in_path", required=True, help="input KG JSON (read only)")
    parser.add_argument("--out", dest="out_path", required=True, help="output KG JSON")
    parser.add_argument("--apply", action="store_true", help="write --out and the change log")
    parser.add_argument(
        "--log", dest="log_path", help="change log path (default <out>.changes.json)"
    )
    parser.add_argument("--verbose", action="store_true", help="print every change")
    args = parser.parse_args(argv)

    in_path, out_path = Path(args.in_path), Path(args.out_path)
    log_path = Path(args.log_path) if args.log_path else Path(str(out_path) + ".changes.json")
    reason = _refusal(in_path, out_path) or (_refusal(in_path, log_path) if args.apply else None)
    if reason:
        print(f"REFUSED: {reason}", file=sys.stderr)
        return 2

    raw = in_path.read_bytes()
    input_sha = hashlib.sha256(raw).hexdigest()
    content = raw.decode("utf-8").rstrip("\x00")
    data, _ = json.JSONDecoder().raw_decode(content)

    onto = get_ontology()
    before = consistency_report(data, onto)
    result = migrate(data, onto, input_sha256=input_sha)
    after = consistency_report(result.data, onto)

    print(f"input  {in_path} sha256={input_sha} triples={len(data.get('triples', []))}")
    print(f"output triples={len(result.data['triples'])} ({'apply' if args.apply else 'dry-run'})")
    _print_counts(result, before, after)
    shown = result.changes if args.verbose else result.changes[:20]
    print(f"== changes ({len(result.changes)} total, showing {len(shown)}) ==")
    for c in shown:
        print("  " + json.dumps(c, ensure_ascii=False))

    if not args.apply:
        print("dry-run: nothing written (use --apply)")
        return 0

    text = _dumps(result.data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    output_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    log = {
        "input": str(in_path),
        "output": str(out_path),
        "input_sha256": input_sha,
        "output_sha256": output_sha,
        "counts": result.counts,
        "consistency_before": before,
        "consistency_after": after,
        "changes": result.changes,
    }
    log_path.write_text(_dumps(log), encoding="utf-8")
    print(f"written {out_path} sha256={output_sha}")
    print(f"change log {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
