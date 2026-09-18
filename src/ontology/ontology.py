"""
단일 원본 온톨로지 로더 (트랙 O1, 결정 OA-1~OA-4) [2026-09 사후]

원본:
- ``config/ontology/schema.json``  클래스·술어·정적 개체(세그먼트·국가·지표)
- ``config/ontology/brands.json``  브랜드 등록부 + 기업 그룹
- ``config/category_hierarchy.json``  카테고리 계층 (OA-2: 복사하지 않고 참조)

로드 시 한 번만 폐포(closure)를 계산하고, 이후 모든 조회는 I/O 없는 순수 함수다.
폐포가 이해하는 공리 형식은 다음뿐이다(OA-3). 원본에 다른 형식이 있으면
``UnsupportedAxiomError``로 로드를 거부한다.

- ``subClassOf`` 전이
- ``defined_by: {predicate, value}`` (OWL ``Base and (predicate value v)``, hasValue 제한)
- ``disjoint_sets`` (서로소 클래스 — 위반 시 ``OntologyError``)
- 술어의 ``inverse_of`` / ``symmetric`` / ``transitive`` / ``domain`` / ``range``
- 그룹 소속에서 자매 브랜드(``siblingBrand``) 도출 (같은 그룹, 자기 자신 제외)

런타임에는 Java·외부 추론기를 부르지 않는다(OE2). OWL 의미론과 같은지는
``scripts/check_ontology_owl.py``가 Pellet으로 개발 시점에 교차 검증한다(OE2-추가).

카테고리 포함 관계는 조회 범위를 넓히는 데만 쓴다(OE3). 이 모듈은 수치(SoS·HHI·
순위)를 상위 카테고리로 합산·전파하는 API를 제공하지 않는다.
"""

from __future__ import annotations

import json
import re
import threading
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ONTOLOGY_DIR = _REPO_ROOT / "config" / "ontology"
DEFAULT_CATEGORY_PATH = _REPO_ROOT / "config" / "category_hierarchy.json"

_ASIN_RE = re.compile(r"^B0[A-Z0-9]{8}$")

_SCHEMA_TOP_KEYS = frozenset(
    {
        "version",
        "as_of",
        "iri",
        "description",
        "classes",
        "disjoint_sets",
        "predicates",
        "kg_legacy_predicates",
        "category_inclusion",
        "individuals",
        "segment_vocabulary_note",
    }
)
_CLASS_KEYS = frozenset({"label", "comment", "subClassOf", "defined_by"})
_DEFINED_BY_KEYS = frozenset({"predicate", "value"})
_PREDICATE_KEYS = frozenset(
    {
        "kind",
        "domain",
        "range",
        "inverse_of",
        "symmetric",
        "transitive",
        "static",
        "numeric",
        "requires_as_of",
        "value_attrs",
        "allowed_values",
        "aliases",
        "comment",
    }
)
_INDIVIDUAL_KEYS = frozenset({"label", "kind", "source_values", "comment"})
_REGISTRY_TOP_KEYS = frozenset({"version", "as_of", "_meta", "groups", "brands"})
_BRAND_KEYS = frozenset(
    {
        "id",
        "name",
        "aliases",
        "group",
        "segment",
        "origin",
        "acquired",
        "is_placeholder",
        "sources",
        "notes",
    }
)
_GROUP_KEYS = frozenset({"id", "name", "aliases", "sources", "notes"})
_XSD_RANGES = frozenset({"xsd:integer", "xsd:decimal", "xsd:string"})

BRAND_CLASS = "Brand"
PLACEHOLDER_CLASS = "PlaceholderBrand"
GROUP_CLASS = "CorporateGroup"
CATEGORY_CLASS = "Category"
PRODUCT_CLASS = "Product"
SEGMENT_CLASS = "Segment"
COUNTRY_CLASS = "Country"
METRIC_CLASS = "Metric"


class OntologyError(ValueError):
    """온톨로지 원본이 잘못되었거나 모순일 때."""


class UnsupportedAxiomError(OntologyError):
    """폐포 계산이 이해하지 못하는 공리 형식이 원본에 있을 때 (OA-3)."""


@dataclass(frozen=True)
class PredicateSpec:
    """술어 정의 (schema.json ``predicates`` 항목)."""

    name: str
    kind: str
    domain: str
    range: str
    inverse_of: str | None
    symmetric: bool
    transitive: bool
    static: bool
    numeric: bool
    requires_as_of: bool
    value_attrs: tuple[str, ...]
    allowed_values: tuple[str, ...]
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ClassSpec:
    """클래스 정의. ``defined_by``가 있으면 정의 클래스(hasValue 제한)."""

    name: str
    label: str
    super_classes: tuple[str, ...]
    defined_by: tuple[str, str] | None


def normalize_key(text: str) -> str:
    """대소문자·악센트·기호에 무관한 비교 키.

    ``"e.l.f."``·``"ELF"`` → ``"elf"``, ``"La Roche-Posay"`` → ``"la roche posay"``.
    공백은 의미를 가진다(``"Amore Pacific"`` 브랜드 ≠ ``"AMOREPACIFIC"`` 그룹).
    """
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    t = unicodedata.normalize("NFC", t).casefold()
    t = re.sub(r"[-_/&+]", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise OntologyError(f"ontology source not found: {path}") from e
    except json.JSONDecodeError as e:
        raise OntologyError(f"invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise OntologyError(f"{path}: top level must be an object")
    return data


def _reject_unknown(keys: Iterable[str], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(keys) - allowed)
    if unknown:
        raise UnsupportedAxiomError(
            f"{where}: unsupported key(s) {unknown} — the Python closure only understands "
            f"{sorted(allowed)} (decision OA-3)"
        )


class Ontology:
    """로드 후 불변인 온톨로지. 모든 공개 메서드는 I/O 없는 순수 함수다."""

    def __init__(
        self,
        schema: Mapping[str, Any],
        registry: Mapping[str, Any],
        category_hierarchy: Mapping[str, Any],
    ) -> None:
        self.version: str = str(schema.get("version", ""))
        self.as_of: str = str(schema.get("as_of", ""))
        self.registry_version: str = str(registry.get("version", ""))
        self.iri: str = str(schema.get("iri", "http://amore.ai/ontology/core#"))

        self._classes: dict[str, ClassSpec] = {}
        self._predicates: dict[str, PredicateSpec] = {}
        self._pred_alias: dict[str, str] = {}
        self._legacy: dict[str, dict[str, Any]] = {}
        self._disjoint_sets: list[tuple[str, ...]] = []
        self._individual_labels: dict[str, str] = {}
        self._segment_kind: dict[str, str] = {}

        self._asserted_types: dict[str, set[str]] = {}
        self._facts: set[tuple[str, str, str]] = set()
        self._data_facts: dict[tuple[str, str], Any] = {}

        self._brand_key: dict[str, str] = {}
        self._group_key: dict[str, str] = {}
        self._category_key: dict[str, str] = {}
        self._brand_names: dict[str, str] = {}
        self._placeholders: set[str] = set()
        self._category_parent: dict[str, str | None] = {}

        self._problems: list[str] = []

        self._load_schema(schema)
        self._load_categories(category_hierarchy)
        self._load_registry(registry)
        if self._problems:
            raise OntologyError("ontology source violations: " + "; ".join(self._problems))
        self._compute_closure()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_schema(self, schema: Mapping[str, Any]) -> None:
        _reject_unknown(schema.keys(), _SCHEMA_TOP_KEYS, "schema.json")

        classes = schema.get("classes") or {}
        for name, spec in classes.items():
            _reject_unknown(spec.keys(), _CLASS_KEYS, f"class {name}")
            supers = tuple(spec.get("subClassOf") or [])
            defined = spec.get("defined_by")
            defined_pair: tuple[str, str] | None = None
            if defined is not None:
                _reject_unknown(defined.keys(), _DEFINED_BY_KEYS, f"class {name} defined_by")
                if set(defined.keys()) != _DEFINED_BY_KEYS:
                    raise UnsupportedAxiomError(
                        f"class {name} defined_by must have exactly 'predicate' and 'value'"
                    )
                if len(supers) != 1:
                    raise UnsupportedAxiomError(
                        f"class {name}: a defined class needs exactly one subClassOf base"
                    )
                defined_pair = (str(defined["predicate"]), str(defined["value"]))
            self._classes[name] = ClassSpec(
                name=name,
                label=str(spec.get("label", name)),
                super_classes=supers,
                defined_by=defined_pair,
            )
        for c in self._classes.values():
            for s in c.super_classes:
                if s not in self._classes:
                    raise OntologyError(f"class {c.name}: unknown superclass {s}")
        self._check_class_cycles()

        for group in schema.get("disjoint_sets") or []:
            for c in group:
                if c not in self._classes:
                    raise OntologyError(f"disjoint_sets: unknown class {c}")
            self._disjoint_sets.append(tuple(group))

        preds = schema.get("predicates") or {}
        for name, spec in preds.items():
            _reject_unknown(spec.keys(), _PREDICATE_KEYS, f"predicate {name}")
            kind = spec.get("kind")
            if kind not in ("object", "data"):
                raise UnsupportedAxiomError(f"predicate {name}: kind must be 'object' or 'data'")
            domain, rng = spec.get("domain"), spec.get("range")
            if domain not in self._classes:
                raise OntologyError(f"predicate {name}: unknown domain {domain}")
            if kind == "object" and rng not in self._classes:
                raise OntologyError(f"predicate {name}: unknown range {rng}")
            if kind == "data" and rng not in _XSD_RANGES:
                raise UnsupportedAxiomError(f"predicate {name}: unsupported datatype {rng}")
            numeric = bool(spec.get("numeric", False))
            transitive = bool(spec.get("transitive", False))
            symmetric = bool(spec.get("symmetric", False))
            if kind == "data" and (transitive or symmetric or spec.get("inverse_of")):
                raise UnsupportedAxiomError(
                    f"predicate {name}: data predicates cannot be transitive/symmetric/inverse"
                )
            if numeric and transitive:
                raise OntologyError(
                    f"predicate {name}: numeric predicates cannot be transitive (OE3)"
                )
            self._predicates[name] = PredicateSpec(
                name=name,
                kind=kind,
                domain=domain,
                range=rng,
                inverse_of=spec.get("inverse_of"),
                symmetric=symmetric,
                transitive=transitive,
                static=bool(spec.get("static", False)),
                numeric=numeric,
                requires_as_of=bool(spec.get("requires_as_of", False)),
                value_attrs=tuple(spec.get("value_attrs") or ()),
                allowed_values=tuple(spec.get("allowed_values") or ()),
                aliases=tuple(spec.get("aliases") or ()),
            )
        for p in self._predicates.values():
            if p.inverse_of is not None:
                inv = self._predicates.get(p.inverse_of)
                if inv is None:
                    raise OntologyError(f"predicate {p.name}: unknown inverse {p.inverse_of}")
                if inv.inverse_of not in (None, p.name):
                    raise OntologyError(f"predicate {p.name}: inverse_of is not mutual")
                if (inv.domain, inv.range) != (p.range, p.domain):
                    raise OntologyError(f"predicate {p.name}: inverse domain/range mismatch")
                if inv.transitive != p.transitive:
                    raise OntologyError(f"predicate {p.name}: inverse transitivity mismatch")
            if p.symmetric and p.domain != p.range:
                raise OntologyError(f"predicate {p.name}: symmetric needs domain == range")
            for alias in (p.name, *p.aliases):
                if alias in self._pred_alias and self._pred_alias[alias] != p.name:
                    raise OntologyError(f"predicate alias {alias} used twice")
                self._pred_alias[alias] = p.name

        for c in self._classes.values():
            if c.defined_by is not None and c.defined_by[0] not in self._predicates:
                raise OntologyError(
                    f"class {c.name}: defined_by unknown predicate {c.defined_by[0]}"
                )

        legacy = dict(schema.get("kg_legacy_predicates") or {})
        legacy.pop("description", None)
        for name, spec in legacy.items():
            _reject_unknown(
                spec.keys(), frozenset({"split_by", "map", "default"}), f"kg_legacy {name}"
            )
            for target in [*spec.get("map", {}).values(), spec.get("default")]:
                if target is not None and target not in self._predicates:
                    raise OntologyError(f"kg_legacy_predicates {name}: unknown target {target}")
            self._legacy[name] = spec

        inclusion = schema.get("category_inclusion") or {}
        _reject_unknown(
            inclusion.keys(),
            frozenset({"use", "numeric_propagation", "comment"}),
            "category_inclusion",
        )
        if inclusion.get("numeric_propagation", False) is not False:
            raise OntologyError(
                "category_inclusion.numeric_propagation must be false (OE3: category inclusion "
                "only widens retrieval scope; metrics are never aggregated to parents)"
            )
        if inclusion.get("use", "scope_expansion_only") != "scope_expansion_only":
            raise OntologyError("category_inclusion.use must be 'scope_expansion_only' (OE3)")

        individuals = schema.get("individuals") or {}
        for cls, members in individuals.items():
            if cls not in self._classes:
                raise OntologyError(f"individuals: unknown class {cls}")
            for iid, spec in members.items():
                _reject_unknown(spec.keys(), _INDIVIDUAL_KEYS, f"individual {iid}")
                self._declare(iid, cls, str(spec.get("label", iid)))
                if cls == SEGMENT_CLASS:
                    self._segment_kind[iid] = str(spec.get("kind", ""))

    def _check_class_cycles(self) -> None:
        for start in self._classes:
            seen: set[str] = set()
            stack = list(self._classes[start].super_classes)
            while stack:
                c = stack.pop()
                if c == start:
                    raise OntologyError(f"subClassOf cycle through {start}")
                if c not in seen:
                    seen.add(c)
                    stack.extend(self._classes[c].super_classes)

    def _declare(self, iid: str, cls: str, label: str) -> None:
        if not iid or not re.fullmatch(r"[a-z][a-z0-9_]*", iid):
            self._problems.append(f"individual id {iid!r} is not a lowercase slug")
        existing = self._asserted_types.get(iid)
        if existing is not None and cls not in existing:
            self._problems.append(
                f"individual id {iid} declared as {sorted(existing)} and {cls}; ids must be unique"
            )
        self._asserted_types.setdefault(iid, set()).add(cls)
        self._individual_labels.setdefault(iid, label)

    def _load_categories(self, hierarchy: Mapping[str, Any]) -> None:
        cats = hierarchy.get("categories") or {}
        for cid, spec in cats.items():
            self._declare(cid, CATEGORY_CLASS, str(spec.get("name", cid)))
            self._category_parent[cid] = spec.get("parent_id")
            for text in (cid, spec.get("name")):
                if text:
                    self._register_key(
                        self._category_key, normalize_key(str(text)), cid, "category"
                    )
        for cid, parent in self._category_parent.items():
            if parent is None:
                continue
            if parent not in self._category_parent:
                self._problems.append(f"category {cid}: unknown parent_id {parent}")
                continue
            self._facts.add((cid, "subCategoryOf", parent))

    def _register_key(self, table: dict[str, str], key: str, iid: str, kind: str) -> None:
        if not key:
            return
        other = table.get(key)
        if other is not None and other != iid:
            self._problems.append(f"{kind} alias key {key!r} maps to both {other} and {iid}")
            return
        table[key] = iid

    def _load_registry(self, registry: Mapping[str, Any]) -> None:
        _reject_unknown(registry.keys(), _REGISTRY_TOP_KEYS, "brands.json")
        for g in registry.get("groups") or []:
            _reject_unknown(g.keys(), _GROUP_KEYS, f"group {g.get('id')}")
            gid = str(g["id"])
            self._declare(gid, GROUP_CLASS, str(g.get("name", gid)))
            for text in (gid, g.get("name"), *(g.get("aliases") or [])):
                if text:
                    self._register_key(self._group_key, normalize_key(str(text)), gid, "group")

        declared = self._asserted_types
        for b in registry.get("brands") or []:
            _reject_unknown(b.keys(), _BRAND_KEYS, f"brand {b.get('id')}")
            bid = str(b["id"])
            placeholder = bool(b.get("is_placeholder", False))
            self._declare(bid, PLACEHOLDER_CLASS if placeholder else BRAND_CLASS, str(b["name"]))
            self._brand_names[bid] = str(b["name"])
            if placeholder:
                self._placeholders.add(bid)
            if not b.get("sources"):
                self._problems.append(f"brand {bid}: no sources")
            for text in (bid, b.get("name"), *(b.get("aliases") or [])):
                if text:
                    self._register_key(self._brand_key, normalize_key(str(text)), bid, "brand")

            for field_name, pred, cls in (
                ("group", "ownedByGroup", GROUP_CLASS),
                ("segment", "hasSegment", SEGMENT_CLASS),
                ("origin", "originatesFrom", COUNTRY_CLASS),
            ):
                value = b.get(field_name)
                if value is None:
                    continue
                if cls not in declared.get(str(value), set()):
                    self._problems.append(
                        f"brand {bid}: {field_name} {value!r} is not a declared {cls}"
                    )
                    continue
                if placeholder:
                    self._problems.append(f"placeholder {bid} must not carry {field_name}")
                    continue
                self._facts.add((bid, pred, str(value)))
            acquired = b.get("acquired")
            if acquired is not None:
                if isinstance(acquired, bool) or not isinstance(acquired, int):
                    self._problems.append(f"brand {bid}: acquired must be an integer year")
                else:
                    self._data_facts[(bid, "acquiredIn")] = acquired

        # brand keys must not collide with group keys (a string names one thing)
        for key, gid in self._group_key.items():
            if key in self._brand_key:
                self._problems.append(
                    f"alias key {key!r} names both brand {self._brand_key[key]} and group {gid}"
                )

    # ------------------------------------------------------------------
    # Closure (one-time, deterministic)
    # ------------------------------------------------------------------

    def _compute_closure(self) -> None:
        facts = set(self._facts)
        preds = self._predicates
        while True:
            before = len(facts)
            # group membership -> siblingBrand (same group, a != b)
            members: dict[str, set[str]] = {}
            for s, p, o in facts:
                if p == "ownedByGroup":
                    members.setdefault(o, set()).add(s)
                elif p == "ownsBrand":
                    members.setdefault(s, set()).add(o)
            for group_members in members.values():
                for a in group_members:
                    for b in group_members:
                        if a != b:
                            facts.add((a, "siblingBrand", b))
            # inverse / symmetric
            for s, p, o in list(facts):
                spec = preds[p]
                if spec.inverse_of:
                    facts.add((o, spec.inverse_of, s))
                if spec.symmetric:
                    facts.add((o, p, s))
            # transitive
            for name, spec in preds.items():
                if not spec.transitive:
                    continue
                succ: dict[str, set[str]] = {}
                for s, p, o in facts:
                    if p == name:
                        succ.setdefault(s, set()).add(o)
                for s in list(succ):
                    stack, seen = list(succ[s]), set()
                    while stack:
                        x = stack.pop()
                        if x in seen:
                            continue
                        seen.add(x)
                        stack.extend(succ.get(x, ()))
                    for x in seen:
                        facts.add((s, name, x))
            if len(facts) == before:
                break

        # types: asserted + domain/range + superclasses + defined classes
        types: dict[str, set[str]] = {k: set(v) for k, v in self._asserted_types.items()}
        for s, p, o in facts:
            spec = preds[p]
            types.setdefault(s, set()).add(spec.domain)
            types.setdefault(o, set()).add(spec.range)
        for s, p in self._data_facts:
            types.setdefault(s, set()).add(preds[p].domain)
        ancestors = {c: self._ancestors(c) for c in self._classes}
        defined = [c for c in self._classes.values() if c.defined_by is not None]
        changed = True
        while changed:
            changed = False
            for ind, ts in types.items():
                for t in list(ts):
                    new = ancestors[t] - ts
                    if new:
                        ts |= new
                        changed = True
                for c in defined:
                    if c.name in ts:
                        continue
                    pred, value = c.defined_by  # type: ignore[misc]
                    if c.super_classes[0] in ts and (ind, pred, value) in facts:
                        ts.add(c.name)
                        changed = True

        for ind, ts in sorted(types.items()):
            for group in self._disjoint_sets:
                hit = sorted(set(group) & ts)
                if len(hit) > 1:
                    raise OntologyError(
                        f"inconsistent: {ind} is a member of disjoint classes {hit}"
                    )

        self._closed_facts: frozenset[tuple[str, str, str]] = frozenset(facts)
        self._types: dict[str, frozenset[str]] = {k: frozenset(v) for k, v in types.items()}
        by_sp: dict[tuple[str, str], set[str]] = {}
        by_p: dict[str, set[tuple[str, str]]] = {}
        for s, p, o in facts:
            by_sp.setdefault((s, p), set()).add(o)
            by_p.setdefault(p, set()).add((s, o))
        self._by_sp = {k: tuple(sorted(v)) for k, v in by_sp.items()}
        self._by_p = {k: tuple(sorted(v)) for k, v in by_p.items()}
        instances: dict[str, set[str]] = {c: set() for c in self._classes}
        for ind, ts in types.items():
            for t in ts:
                instances[t].add(ind)
        self._instances = {c: tuple(sorted(v)) for c, v in instances.items()}

    def _ancestors(self, cls: str) -> frozenset[str]:
        out: set[str] = set()
        stack = list(self._classes[cls].super_classes)
        while stack:
            c = stack.pop()
            if c not in out:
                out.add(c)
                stack.extend(self._classes[c].super_classes)
        return frozenset(out)

    # ------------------------------------------------------------------
    # Status (OE10)
    # ------------------------------------------------------------------

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple(sorted(self._classes))

    @property
    def class_count(self) -> int:
        return len(self._classes)

    @property
    def brand_count(self) -> int:
        """등록부의 실제 브랜드 수 (가짜 브랜드 제외)."""
        return len(self._instances[BRAND_CLASS])

    @property
    def predicates(self) -> tuple[str, ...]:
        return tuple(sorted(self._predicates))

    @property
    def individuals(self) -> tuple[str, ...]:
        return tuple(sorted(self._types))

    @property
    def numeric_propagation_allowed(self) -> bool:
        """OE3: 항상 False. 수치는 카테고리 포함 관계를 따라 전파하지 않는다."""
        return False

    @property
    def disjoint_sets(self) -> tuple[tuple[str, ...], ...]:
        return tuple(self._disjoint_sets)

    def class_spec(self, cls: str) -> ClassSpec:
        self._require_class(cls)
        return self._classes[cls]

    def label_of(self, iid: str) -> str | None:
        return self._individual_labels.get(iid)

    def segment_kind(self, segment: str) -> str | None:
        return self._segment_kind.get(segment)

    def data_facts(self) -> tuple[tuple[str, str, Any], ...]:
        return tuple(sorted((s, p, v) for (s, p), v in self._data_facts.items()))

    def asserted_facts(self) -> tuple[tuple[str, str, str], ...]:
        """폐포 전 원본 사실 (OWL 내보내기용)."""
        return tuple(sorted(self._facts))

    def asserted_types(self, iid: str) -> tuple[str, ...]:
        return tuple(sorted(self._asserted_types.get(iid, ())))

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_brand(self, text: str | None) -> str | None:
        """브랜드 문자열 → 등록부 id. 모르면 None. 가짜 브랜드도 id를 돌려준다
        (``is_placeholder``로 구분)."""
        if not text:
            return None
        return self._brand_key.get(normalize_key(str(text)))

    def normalize_group(self, text: str | None) -> str | None:
        if not text:
            return None
        return self._group_key.get(normalize_key(str(text)))

    def normalize_category(self, text: str | None) -> str | None:
        if not text:
            return None
        return self._category_key.get(normalize_key(str(text)))

    def brand_name(self, brand: str) -> str | None:
        bid = self.normalize_brand(brand)
        return self._brand_names.get(bid) if bid else None

    def is_placeholder(self, brand: str) -> bool:
        bid = self.normalize_brand(brand)
        return bid is not None and bid in self._placeholders

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------

    def _require_class(self, cls: str) -> None:
        if cls not in self._classes:
            raise OntologyError(f"unknown class {cls}")

    def superclasses(self, cls: str) -> tuple[str, ...]:
        self._require_class(cls)
        return tuple(sorted(self._ancestors(cls)))

    def _resolve(self, entity: str) -> str | None:
        if entity in self._types:
            return entity
        for fn in (self.normalize_brand, self.normalize_group, self.normalize_category):
            iid = fn(entity)
            if iid is not None:
                return iid
        return None

    def types_of(self, entity: str) -> tuple[str, ...]:
        """폐포 후 소속 클래스 전체(정렬). 등록부 밖 개체는 빈 튜플."""
        iid = self._resolve(entity)
        return tuple(sorted(self._types.get(iid, ()))) if iid else ()

    def is_a(self, entity: str, cls: str) -> bool:
        self._require_class(cls)
        return cls in self.types_of(entity)

    def instances_of(self, cls: str) -> tuple[str, ...]:
        self._require_class(cls)
        return self._instances[cls]

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def object_values(self, subject: str, predicate: str) -> tuple[str, ...]:
        """폐포 후 (subject, predicate, ?) 목적어들 (정렬)."""
        p = self.canonical_predicate(predicate)
        iid = self._resolve(subject)
        if p is None or iid is None:
            return ()
        return self._by_sp.get((iid, p), ())

    def relations(self, predicate: str) -> tuple[tuple[str, str], ...]:
        """폐포 후 술어의 (subject, object) 쌍 전체 (정렬)."""
        p = self.canonical_predicate(predicate)
        return self._by_p.get(p, ()) if p else ()

    def brands_in_group(self, group: str) -> tuple[str, ...]:
        gid = self.normalize_group(group)
        return self._by_sp.get((gid, "ownsBrand"), ()) if gid else ()

    def group_of(self, brand: str) -> str | None:
        bid = self.normalize_brand(brand)
        if bid is None:
            return None
        groups = self._by_sp.get((bid, "ownedByGroup"), ())
        return groups[0] if groups else None

    def siblings(self, brand: str) -> tuple[str, ...]:
        """같은 그룹 브랜드(자기 자신 제외). 그룹이 없으면 빈 튜플."""
        bid = self.normalize_brand(brand)
        return self._by_sp.get((bid, "siblingBrand"), ()) if bid else ()

    def segment_of(self, brand: str) -> str | None:
        return self._single(brand, "hasSegment")

    def origin_of(self, brand: str) -> str | None:
        """원산지. 어떤 원본에도 적혀 있지 않으면 None(모름)."""
        return self._single(brand, "originatesFrom")

    def acquired_in(self, brand: str) -> int | None:
        bid = self.normalize_brand(brand)
        if bid is None:
            return None
        value = self._data_facts.get((bid, "acquiredIn"))
        return int(value) if value is not None else None

    def _single(self, brand: str, predicate: str) -> str | None:
        bid = self.normalize_brand(brand)
        if bid is None:
            return None
        values = self._by_sp.get((bid, predicate), ())
        return values[0] if values else None

    def closed_world_member(self, brand: str, group: str) -> bool | None:
        """OE4 닫힌 세계 판정.

        - 등록부 브랜드이고 그 그룹 소속 → True
        - 등록부 브랜드인데 그룹이 없거나 다른 그룹 → False
        - 등록부 밖 브랜드·가짜 브랜드·모르는 그룹 → None (알 수 없음)
        """
        bid = self.normalize_brand(brand)
        gid = self.normalize_group(group)
        if bid is None or gid is None or bid in self._placeholders:
            return None
        return self.group_of(bid) == gid

    # ------------------------------------------------------------------
    # Categories (OE3: scope expansion only)
    # ------------------------------------------------------------------

    def category_ancestors(self, category: str) -> tuple[str, ...]:
        """상위 카테고리, 가까운 것부터. 조회 범위 확장 전용이며 수치를 옮기지 않는다."""
        cid = self.normalize_category(category)
        out: list[str] = []
        parent = self._category_parent.get(cid) if cid else None
        while parent is not None:
            out.append(parent)
            parent = self._category_parent.get(parent)
        return tuple(out)

    def category_descendants(self, category: str) -> tuple[str, ...]:
        """하위 카테고리 전체(정렬). 조회 범위 확장 전용이며 수치를 합산하지 않는다."""
        cid = self.normalize_category(category)
        return self._by_sp.get((cid, "hasSubCategory"), ()) if cid else ()

    # ------------------------------------------------------------------
    # Predicates
    # ------------------------------------------------------------------

    def canonical_predicate(self, name: str | None) -> str | None:
        """별칭 → 정식 술어 이름. 모르거나 분리가 필요한 KG 술어(hasPosition)는 None."""
        if not name:
            return None
        return self._pred_alias.get(name)

    def predicate_spec(self, name: str) -> PredicateSpec | None:
        canonical = self.canonical_predicate(name)
        return self._predicates.get(canonical) if canonical else None

    def resolve_kg_predicate(
        self, predicate: str, original_predicate: str | None = None
    ) -> str | None:
        """KG 술어 + ``properties.original_predicate`` → 정식 술어.

        ``hasPosition``은 original_predicate 없이 정할 수 없어 None.
        """
        legacy = self._legacy.get(predicate)
        if legacy is not None:
            mapping = legacy.get("map", {})
            if original_predicate in mapping:
                return str(mapping[original_predicate])
            default = legacy.get("default")
            return str(default) if default else None
        return self.canonical_predicate(predicate)

    def _entity_types_for_check(self, entity: str) -> frozenset[str] | None:
        iid = self._resolve(entity)
        if iid is not None:
            return self._types.get(iid, frozenset())
        if _ASIN_RE.match(entity):
            return frozenset({PRODUCT_CLASS})
        return None  # outside the ontology: cannot be checked

    def validate_triple(
        self,
        subject: str,
        predicate: str,
        obj: Any,
        attrs: Mapping[str, Any] | None = None,
    ) -> list[str]:
        """트리플의 온톨로지 위반 목록(정렬). 빈 목록이면 위반 없음.

        확인 항목: 모르는 술어, 분리가 필요한 KG 술어, 도메인·범위 불일치(개체 타입을 알 때만),
        가짜 브랜드, 허용 값·데이터 타입, 수치 술어의 ``as_of`` 누락.
        """
        attrs = attrs or {}
        out: list[str] = []
        if predicate in self._legacy and self.canonical_predicate(predicate) is None:
            return [
                f"legacy KG predicate {predicate!r} must be split by original_predicate "
                f"({sorted(self._legacy[predicate].get('map', {}))})"
            ]
        spec = self.predicate_spec(predicate)
        if spec is None:
            return [f"unknown predicate {predicate!r}"]

        for role, entity in (("subject", subject), ("object", obj)):
            if isinstance(entity, str) and self.is_placeholder(entity):
                out.append(f"{role} {entity!r} is a placeholder brand, not a real brand")

        s_types = self._entity_types_for_check(str(subject))
        if s_types is not None and spec.domain not in s_types:
            out.append(
                f"domain violation: {spec.name} expects {spec.domain}, "
                f"subject {subject!r} is {sorted(s_types)}"
            )
        if spec.kind == "object":
            o_types = self._entity_types_for_check(str(obj))
            if o_types is not None and spec.range not in o_types:
                out.append(
                    f"range violation: {spec.name} expects {spec.range}, "
                    f"object {obj!r} is {sorted(o_types)}"
                )
        else:
            out.extend(self._check_literal(spec, obj))

        if spec.requires_as_of and not attrs.get("as_of"):
            out.append(f"{spec.name} is numeric/time-dependent and requires an as_of attribute")
        return sorted(out)

    @staticmethod
    def _check_literal(spec: PredicateSpec, value: Any) -> list[str]:
        if spec.allowed_values and str(value) not in spec.allowed_values:
            return [
                f"{spec.name} value {value!r} not in allowed values {list(spec.allowed_values)}"
            ]
        if spec.range == "xsd:integer":
            if isinstance(value, bool) or not re.fullmatch(r"-?\d+", str(value)):
                return [f"{spec.name} value {value!r} is not an xsd:integer"]
        if spec.range == "xsd:decimal":
            try:
                float(str(value))
            except ValueError:
                return [f"{spec.name} value {value!r} is not an xsd:decimal"]
        return []

    # ------------------------------------------------------------------
    # Self-consistency
    # ------------------------------------------------------------------

    def self_check(self) -> list[str]:
        """등록부·스키마 정합성 위반(정렬). 로드에 성공했다면 비어 있어야 한다."""
        out: list[str] = list(self._problems)
        for key, bid in self._brand_key.items():
            if self.normalize_brand(key) != bid:
                out.append(f"brand alias key {key!r} does not normalize to {bid}")
        for bid in self._brand_names:
            if bid in self._placeholders:
                continue
            for pred in ("ownedByGroup", "hasSegment", "originatesFrom"):
                if len(self._by_sp.get((bid, pred), ())) > 1:
                    out.append(f"brand {bid} has more than one {pred}")
        for s, p, o in self._closed_facts:
            spec = self._predicates[p]
            if spec.domain not in self._types.get(s, ()):
                out.append(f"{s} {p} {o}: subject not typed {spec.domain}")
            if spec.range not in self._types.get(o, ()):
                out.append(f"{s} {p} {o}: object not typed {spec.range}")
        return sorted(out)


# ----------------------------------------------------------------------
# Loading and module-level cache
# ----------------------------------------------------------------------


def load_ontology(
    path: str | Path | None = None,
    category_path: str | Path | None = None,
) -> Ontology:
    """원본을 읽어 폐포까지 계산한 ``Ontology``를 만든다.

    Args:
        path: ``schema.json``·``brands.json``이 있는 디렉터리 (기본 ``config/ontology``).
        category_path: 카테고리 계층 파일 (기본 ``config/category_hierarchy.json``).
    """
    base = Path(path) if path is not None else DEFAULT_ONTOLOGY_DIR
    cat = Path(category_path) if category_path is not None else DEFAULT_CATEGORY_PATH
    return Ontology(
        schema=_read_json(base / "schema.json"),
        registry=_read_json(base / "brands.json"),
        category_hierarchy=_read_json(cat),
    )


_cache_lock = threading.Lock()
_cached: Ontology | None = None


def get_ontology() -> Ontology:
    """기본 원본의 ``Ontology``를 한 번만 로드해 돌려준다 (스레드 안전)."""
    global _cached
    onto = _cached
    if onto is not None:
        return onto
    with _cache_lock:
        if _cached is None:
            _cached = load_ontology()
        return _cached


def reset_ontology_cache() -> None:
    """테스트용: 캐시를 비워 다음 ``get_ontology()``가 다시 로드하게 한다."""
    global _cached
    with _cache_lock:
        _cached = None
