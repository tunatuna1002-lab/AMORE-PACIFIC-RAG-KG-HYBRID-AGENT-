#!/usr/bin/env python3
"""
JSON 단일 원본 온톨로지를 OWL(RDF/XML)로 내보낸다. 개발 전용 (OE1, OE8) [2026-09 사후]

서비스 질의 경로는 이 파일을 쓰지 않는다. Protégé 같은 표준 도구로 보거나
``scripts/check_ontology_owl.py``로 Pellet 교차 검증할 때만 쓴다.

매핑:
- 클래스 → owl:Class (subClassOf), 정의 클래스 → ``equivalent_to = [Base & pred.value(ind)]``
- 술어 → owl:ObjectProperty / owl:DatatypeProperty (inverseOf, Symmetric, Transitive, domain, range)
- 개체 → 브랜드·가짜 브랜드·그룹·카테고리·세그먼트·국가·지표 (원본에 적힌 타입만 단언)
- 사실 → 원본 사실만 단언. 역관계·전이 결과는 단언하지 않는다(추론기가 도출해야 함).
  자매 브랜드는 그룹에서 도출한 쌍을 한 방향(a < b)만 단언한다(반대 방향은 대칭성으로 도출).
- ``disjoint_sets`` → AllDisjoint, 모든 개체 → AllDifferent

Usage:
    .venv/bin/python scripts/export_ontology_owl.py [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ontology.ontology import Ontology, load_ontology  # noqa: E402

DEFAULT_OUT = Path(tempfile.gettempdir()) / "amore_ontology" / "amore_ontology.owl"
_DATA_DIR = (PROJECT_ROOT / "data").resolve()
_XSD_TO_PY: dict[str, type] = {"xsd:integer": int, "xsd:decimal": float, "xsd:string": str}


def _check_out_path(out: Path) -> Path:
    resolved = out.resolve()
    if resolved == _DATA_DIR or _DATA_DIR in resolved.parents:
        raise ValueError(f"refusing to write under production data/ directory: {resolved}")
    return resolved


def _class_order(onto: Ontology) -> list[str]:
    """상위 클래스가 먼저 오도록 정렬 (결정적)."""
    done: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        for sup in sorted(onto.class_spec(name).super_classes):
            visit(sup)
        done.append(name)

    for name in onto.classes:
        visit(name)
    return done


def build_owl(onto: Ontology, world: Any = None) -> tuple[Any, Any]:
    """owlready2 ``World`` 안에 온톨로지를 만든다. (world, ontology)를 돌려준다."""
    import owlready2 as owl

    world = world if world is not None else owl.World()
    ont = world.get_ontology(onto.iri)
    with ont:
        classes: dict[str, Any] = {}
        for name in _class_order(onto):
            spec = onto.class_spec(name)
            bases = tuple(classes[s] for s in spec.super_classes) or (owl.Thing,)
            cls = types.new_class(name, bases)
            cls.label = [spec.label]
            classes[name] = cls

        props: dict[str, Any] = {}
        for name in onto.predicates:
            spec = onto.predicate_spec(name)
            assert spec is not None
            if spec.kind == "object":
                bases_p: tuple[type, ...] = (owl.ObjectProperty,)
                if spec.symmetric:
                    bases_p += (owl.SymmetricProperty,)
                if spec.transitive:
                    bases_p += (owl.TransitiveProperty,)
                prop = types.new_class(name, bases_p)
                prop.range = [classes[spec.range]]
            else:
                prop = types.new_class(name, (owl.DataProperty,))
                prop.range = [_XSD_TO_PY[spec.range]]
            prop.domain = [classes[spec.domain]]
            props[name] = prop
        for name in onto.predicates:
            spec = onto.predicate_spec(name)
            assert spec is not None
            if spec.inverse_of and name < spec.inverse_of:
                props[name].inverse_property = props[spec.inverse_of]

        individuals: dict[str, Any] = {}
        for iid in onto.individuals:
            asserted = onto.asserted_types(iid)
            if not asserted:
                continue
            ind = classes[asserted[0]](iid, namespace=ont)
            for extra in asserted[1:]:
                ind.is_a.append(classes[extra])
            label = onto.label_of(iid)
            if label:
                ind.label = [label]
            individuals[iid] = ind

        for name in onto.classes:
            spec = onto.class_spec(name)
            if spec.defined_by is None:
                continue
            pred, value = spec.defined_by
            base = classes[spec.super_classes[0]]
            classes[name].equivalent_to.append(base & props[pred].value(individuals[value]))

        for s, p, o in onto.asserted_facts():
            getattr(individuals[s], p).append(individuals[o])
        for a, b in onto.relations("siblingBrand"):
            if a < b:
                individuals[a].siblingBrand.append(individuals[b])
        for s, p, v in onto.data_facts():
            getattr(individuals[s], p).append(v)

        for group in onto.disjoint_sets:
            owl.AllDisjoint([classes[c] for c in group])
        owl.AllDifferent([individuals[i] for i in sorted(individuals)])
    return world, ont


def export_owl(out: Path | str = DEFAULT_OUT, onto: Ontology | None = None) -> Path:
    """OWL(RDF/XML) 파일을 쓰고 경로를 돌려준다. ``data/`` 아래에는 쓰지 않는다."""
    target = _check_out_path(Path(out))
    onto = onto if onto is not None else load_ontology()
    _world, ont = build_owl(onto)
    target.parent.mkdir(parents=True, exist_ok=True)
    ont.save(file=str(target), format="rdfxml")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output .owl path")
    args = parser.parse_args(argv)
    onto = load_ontology()
    path = export_owl(args.out, onto)
    print(
        f"wrote {path} (ontology {onto.version}, {onto.class_count} classes, "
        f"{len(onto.predicates)} predicates, {len(onto.individuals)} individuals)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
