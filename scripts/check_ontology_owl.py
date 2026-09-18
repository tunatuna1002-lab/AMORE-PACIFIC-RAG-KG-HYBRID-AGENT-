#!/usr/bin/env python3
"""
JSON 온톨로지를 OWL로 내보내고 Pellet으로 검증한다. 개발 전용 (OE2-추가) [2026-09 사후]

1. 일관성: 모순 클래스 0개, ``OwlReadyInconsistentOntologyError`` 없음
2. 교차 검증: Pellet 추론 결과가 Python 폐포(``src/ontology/ontology.py``)와 같은지
   - 모든 클래스의 소속 개체 집합
   - 모든 object 술어의 (주어, 목적어) 쌍 — 단언 + Pellet 추론 값을 owlready2로 읽는다
     (owlready2는 읽을 때 owl:inverseOf·SymmetricProperty를 적용한다)

내보낸 파일을 새 ``World``에 다시 읽어 검증하므로 파일 자체가 검증 대상이다.
추론기는 Pellet만 쓴다(HermiT는 쓰지 않음). owlready2 0.50의 Pellet은 Java 25 이상이 필요하다.

종료 코드: 0 통과, 1 불일치·모순, 3 Java 25 이상 없음(SKIPPED — 통과가 아님)

Usage:
    .venv/bin/python scripts/check_ontology_owl.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from src.ontology.ontology import Ontology, load_ontology  # noqa: E402

MIN_JAVA = 25
SKIP_MESSAGE = f"SKIPPED: Pellet requires Java >= {MIN_JAVA}"


def parse_java_major(text: str) -> int | None:
    """``java -version`` 출력에서 주 버전을 꺼낸다 (``1.8`` → 8)."""
    m = re.search(r'version "(\d+)(?:\.(\d+))?', text)
    if not m:
        return None
    major = int(m.group(1))
    if major == 1 and m.group(2) is not None:
        return int(m.group(2))
    return major


def java_major_version() -> int | None:
    java = shutil.which("java")
    if java is None:
        return None
    try:
        proc = subprocess.run(
            [java, "-version"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_java_major(proc.stderr + proc.stdout)


def diff_maps(
    label: str,
    expected: Mapping[str, set[Hashable]],
    actual: Mapping[str, set[Hashable]],
) -> list[str]:
    """키별로 (Python 폐포 = expected) vs (Pellet = actual) 차이를 사람이 읽는 줄로."""
    out: list[str] = []
    for key in sorted(set(expected) | set(actual)):
        exp = expected.get(key, set())
        act = actual.get(key, set())
        missing = sorted(exp - act, key=repr)
        extra = sorted(act - exp, key=repr)
        if missing:
            out.append(f"{label} {key}: in Python closure but not Pellet: {missing}")
        if extra:
            out.append(f"{label} {key}: in Pellet but not Python closure: {extra}")
    return out


@dataclass
class CheckResult:
    consistent: bool
    inconsistent_classes: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)
    class_count: int = 0
    individual_count: int = 0
    object_property_count: int = 0
    pellet_seconds: float = 0.0
    owl_path: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.consistent and not self.inconsistent_classes and not self.mismatches


def _expected(onto: Ontology) -> tuple[dict[str, set[Hashable]], dict[str, set[Hashable]]]:
    classes = {c: set(onto.instances_of(c)) for c in onto.classes}
    props: dict[str, set[Hashable]] = {}
    for name in onto.predicates:
        spec = onto.predicate_spec(name)
        if spec is not None and spec.kind == "object":
            props[name] = set(onto.relations(name))
    return classes, props


def _actual(
    owl_onto: Any, onto: Ontology
) -> tuple[dict[str, set[Hashable]], dict[str, set[Hashable]]]:
    import owlready2 as owl

    named = {c.name: c for c in owl_onto.classes()}
    classes: dict[str, set[Hashable]] = {c: set() for c in onto.classes}
    for ind in owl_onto.individuals():
        for direct in ind.is_a:
            if not isinstance(direct, owl.ThingClass):
                continue
            for anc in direct.ancestors():
                if anc.name in classes and named.get(anc.name) is anc:
                    classes[anc.name].add(ind.name)
    props: dict[str, set[Hashable]] = {}
    for name in onto.predicates:
        spec = onto.predicate_spec(name)
        if spec is None or spec.kind != "object":
            continue
        prop = getattr(owl_onto, name)
        props[name] = {(s.name, o.name) for s, o in prop.get_relations()}
    return classes, props


def run_check(out_dir: Path | str | None = None, onto: Ontology | None = None) -> CheckResult:
    """내보내기 → 새 World에 로드 → Pellet → 일관성·교차 검증."""
    import owlready2 as owl
    from export_ontology_owl import export_owl

    onto = onto if onto is not None else load_ontology()
    base = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp(prefix="amore_owl_"))
    owl_path = export_owl(base / "amore_ontology.owl", onto)

    world = owl.World()
    owl_onto = world.get_ontology(owl_path.as_uri()).load()
    result = CheckResult(consistent=True, owl_path=str(owl_path))
    result.class_count = len(list(owl_onto.classes()))
    result.individual_count = len(list(owl_onto.individuals()))
    result.object_property_count = len(list(owl_onto.object_properties()))

    started = time.perf_counter()
    try:
        owl.sync_reasoner_pellet(world, infer_property_values=True, debug=0)
    except owl.OwlReadyInconsistentOntologyError as e:
        result.consistent = False
        result.error = f"Pellet: ontology is inconsistent ({e})"
    finally:
        result.pellet_seconds = time.perf_counter() - started
    if not result.consistent:
        return result

    result.inconsistent_classes = sorted(c.name for c in world.inconsistent_classes())
    exp_classes, exp_props = _expected(onto)
    act_classes, act_props = _actual(owl_onto, onto)
    result.mismatches = diff_maps("class", exp_classes, act_classes) + diff_maps(
        "property", exp_props, act_props
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pellet check of the JSON ontology")
    parser.add_argument("--out-dir", type=Path, default=None, help="where to write the .owl")
    args = parser.parse_args(argv)

    java = java_major_version()
    if java is None or java < MIN_JAVA:
        print(f"{SKIP_MESSAGE} (found: {java if java is not None else 'no java'})")
        return 3

    onto = load_ontology()
    result = run_check(args.out_dir, onto)
    print(f"ontology {onto.version} (as_of {onto.as_of}) -> {result.owl_path}")
    print(
        f"classes={result.class_count} individuals={result.individual_count} "
        f"object_properties={result.object_property_count} java={java} "
        f"pellet_seconds={result.pellet_seconds:.2f}"
    )
    if not result.consistent:
        print(f"FAIL: {result.error}")
        return 1
    print(f"consistent=yes inconsistent_classes={len(result.inconsistent_classes)}")
    for name in result.inconsistent_classes:
        print(f"  inconsistent class: {name}")
    print(f"mismatches={len(result.mismatches)}")
    for line in result.mismatches:
        print(f"  {line}")
    if not result.ok:
        print("FAIL")
        return 1
    checked = len(onto.classes)
    print(f"OK: Pellet class membership for {checked} classes and object property values match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
